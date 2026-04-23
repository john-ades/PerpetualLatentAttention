import torch
import transformers
from typing import Optional
from dataclasses import dataclass, field
from datasets import load_dataset
import warnings

def preprocess_function(examples, tokenizer, seq_len):
    model_inputs = {"input_ids": [[]]}
    acc_len = 0
    for message in examples['text']:
        # Add EOS to the very end of the document
        message_ids = tokenizer.encode(message, add_special_tokens=False) + [tokenizer.eos_token_id]
        
        # Chunk it into seq_len blocks
        input_ids_list = [message_ids[i:i + seq_len] for i in range(0, len(message_ids), seq_len)]
        
        for input_ids in input_ids_list:
            if acc_len + len(input_ids) > seq_len:
                model_inputs["input_ids"].append([input_ids])
                acc_len = len(input_ids)
            else:
                model_inputs["input_ids"][-1].append(input_ids)
                acc_len += len(input_ids)
    return model_inputs

@dataclass
class DataCollatorWithFlattening(transformers.DefaultDataCollator):
    """
    Data collator used for padding free approach. Does the following:

    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """

    def __init__(self, *args, return_position_ids=True, separator_id=-100, max_len=8192, pad_token_id=128001, label_ignore_id=-100, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids
        self.separator_id = separator_id
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.label_ignore_id = label_ignore_id
        warnings.warn(
            "Using `DataCollatorWithFlattening` will flatten the entire mini batch into single long sequence."
            "Make sure your attention computation is able to handle it!"
        )

    def __call__(self, features, return_tensors=None, separator_id=None):
        def padding_ret(ret):
            padding_len = self.max_len - len(ret["input_ids"])
            if self.return_position_ids:
                padded_position_ids = list(range(padding_len))
                ret["position_ids"] += padded_position_ids
            ret["input_ids"] += [self.pad_token_id] * padding_len
            ret["labels"] += [self.label_ignore_id] * padding_len
            ret["input_ids"] = ret["input_ids"][:self.max_len]
            ret["labels"] = ret["labels"][:self.max_len]
            return ret
        
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id

        rets = []
        for idx in range(0, len(features)):
            ret = {"input_ids": [], "labels": []}
            if self.return_position_ids:
                ret.update({"position_ids": []})
            for f_input_ids in features[idx]["input_ids"]:
                ret["input_ids"] += f_input_ids
                ret["labels"] += [separator_id] + f_input_ids[1:]
                if self.return_position_ids:
                    ret["position_ids"] += list(range(len(f_input_ids)))
            rets.append(padding_ret(ret))

        return transformers.default_data_collator(rets, return_tensors)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The dataset name to use (e.g. sample-10BT)."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging, truncate the number of training examples."})
    attn_implementation : Optional[str] = field(default="sdpa")
    seq_len: int = field(default=2048,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    train_m6_adapter: bool = field(default=False, metadata={"help": "Use M6 TBPTT Trainer"})
    freeze_backbone: bool = field(default=False, metadata={"help": "Freeze TransMLA backbone except M6 adapter"})

parser = transformers.HfArgumentParser(TrainingArguments)
training_args = parser.parse_args_into_dataclasses()[0]

model = transformers.AutoModelForCausalLM.from_pretrained(
    training_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    attn_implementation=training_args.attn_implementation, 
    trust_remote_code=True,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.model_name_or_path, fix_mistral_regex=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if training_args.data_path == "mixed_healing":
    from datasets import interleave_datasets
    from datasets.distributed import split_dataset_by_node
    import os

    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).select_columns(["text"])
    cosmopedia = load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True).select_columns(["text"])
    open_web_math = load_dataset("open-web-math/open-web-math", "default", split="train", streaming=True).select_columns(["text"])
    python_edu = load_dataset("flytech/python-codes-25k", split="train", streaming=True).select_columns(["text"]) 
    
    def format_so(x):
        return {"text": str(x.get("title", "")) + "\n\n" + str(x.get("body", ""))}
    stackoverflow = load_dataset("pacovaldez/stackoverflow-questions", split="train", streaming=True).map(format_so).select_columns(["text"])

    train_dataset = interleave_datasets(
        [fineweb, cosmopedia, open_web_math, python_edu, stackoverflow],
        probabilities=[0.70, 0.15, 0.08, 0.06, 0.01],
        seed=42,
    )

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_dataset = split_dataset_by_node(train_dataset, rank=rank, world_size=world_size)

    if training_args.max_train_samples is not None:
        train_dataset = train_dataset.take(training_args.max_train_samples)

    processed_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1024,
        remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer, "seq_len": training_args.seq_len}
    )
else:
    if training_args.max_train_samples is not None:
        train_dataset = load_dataset(training_args.data_path, name=training_args.dataset_name, split="train", streaming=True).take(training_args.max_train_samples)
        from datasets import Dataset
        train_dataset = Dataset.from_generator(lambda: (yield from train_dataset))
    else:
        train_dataset = load_dataset(training_args.data_path, name=training_args.dataset_name, split="train")

    processed_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=1024,
        remove_columns=train_dataset.column_names,
        num_proc=128,
        fn_kwargs={"tokenizer": tokenizer, "seq_len": training_args.seq_len * 2} # Double size for chunks!
    )

from transmla.m6_adapter import M6LatentAdapter

if training_args.train_m6_adapter:
    # Initialize M.6 Adapter
    kv_lora_rank = getattr(model.config, "kv_lora_rank", 512)
    model.memory_adapter = M6LatentAdapter(kv_lora_rank).to(model.dtype).to(model.device)

    # Freeze TransMLA backbone conditionally
    if training_args.freeze_backbone:
        for name, param in model.named_parameters():
            if "memory_adapter" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

class M6TBPTTTrainer(transformers.Trainer):
    def __init__(self, tbptt_chunks=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tbptt_chunks = tbptt_chunks

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Split inputs according to chunks (e.g. 2 chunks)
        chunk_size = inputs["input_ids"].size(1) // self.tbptt_chunks
        
        P_state = None
        total_loss = 0
        
        for i in range(self.tbptt_chunks):
            # 1. Extract Chunk
            chunk_inputs = {
                k: v[:, i * chunk_size : (i + 1) * chunk_size].contiguous() if v.dim() > 1 else v
                for k, v in inputs.items()
            }
            
            # VRAM Staircase Test Block 1
            if i == 0 and self.state.global_step < 10:
                print(f"Step {self.state.global_step} Chunk 1 memory allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                
            # 2. Add hook to capture k_pass of the first layer
            captured_k_pass = []
            def hook(module, inp, out):
                if isinstance(out, tuple): out = out[0]
                k_pass = out[..., :model.memory_adapter.kv_lora_rank]
                captured_k_pass.append(k_pass)
                
            handle = model.model.layers[0].self_attn.kv_a_proj_with_mqa.register_forward_hook(hook)
            
            # 3. Forward Pass & Loss Computation
            if P_state is not None:
                chunk_inputs["memory_latents"] = P_state
                
            loss = self.compute_loss(model, chunk_inputs, return_outputs=False)
            
            # 4. Backpropagate the Chunk
            total_loss += loss.detach() / self.tbptt_chunks
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                self.accelerator.backward(loss)
                
            handle.remove()
            
            # VRAM Staircase Test Block 2
            if i == 1 and self.state.global_step < 10:
                print(f"Step {self.state.global_step} Chunk 2 memory allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
                # Gradients test
                if hasattr(model.memory_adapter.W_a, "weight") and model.memory_adapter.W_a.weight.grad is not None:
                    print(f"W_a grad norm: {model.memory_adapter.W_a.weight.grad.norm().item()}")
                q_proj = None
                for name, module in model.named_modules():
                    if "q_proj" in name or "q_a_proj" in name:
                        q_proj = module
                        break
                if q_proj is not None and hasattr(q_proj, "weight") and q_proj.weight.grad is not None:
                    print(f"WARNING: q_proj grad is NOT None! Backbone is leaking gradients!")
                else:
                    print(f"Backbone securely frozen (q_proj grad is None)")
            
            # 5. Eviction Trigger
            # Sever the computational graph for TransMLA backbone 
            k_pass_evicted = captured_k_pass[0].detach() 
            
            # Update Memory Bank
            P_new = model.memory_adapter.write(k_pass_evicted)
            
            # Sever the graph of P to prevent OOM across chunks
            P_state = P_new.detach() 
            # In-place update registered buffer safely for ZeRO-3
            model.memory_adapter.P.copy_(P_state)
            
        return total_loss

if training_args.train_m6_adapter:
    trainer = M6TBPTTTrainer(
        tbptt_chunks=2,
        args=training_args,
        model=model,
        train_dataset=processed_dataset,
        data_collator=DataCollatorWithFlattening(max_len=training_args.seq_len * 2, pad_token_id=tokenizer.pad_token_id, return_position_ids=False)
    )
else:
    trainer = transformers.Trainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset,
        data_collator=DataCollatorWithFlattening(max_len=training_args.seq_len, pad_token_id=tokenizer.pad_token_id, return_position_ids=False)
    )

trainer.train()
trainer.save_state()
trainer.save_model(output_dir=training_args.output_dir)