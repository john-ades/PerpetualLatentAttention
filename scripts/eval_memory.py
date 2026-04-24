import torch
import sys
import os
import math
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transmla.m6_adapter import M6LatentAdapter

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_memory.py <model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "perpetual-latent-attention"),
        job_type="evaluation",
        name="evaluation"
    )
    
    print(f"Loading tokenizer and model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except ValueError:
        import json
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Could not find tokenizer files or config.json in {model_path}")

        with open(config_path, "r") as f:
            base_model_path = json.load(f).get("_name_or_path")
            
        if not base_model_path:
            raise ValueError(f"Cannot infer base tokenizer: '_name_or_path' missing in {config_path}.")
            
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()

    # Load Adapter
    kv_lora_rank = getattr(model.config, "kv_lora_rank", 512)
    num_hidden_layers = getattr(model.config, "num_hidden_layers", 32)
    model.memory_adapter = M6LatentAdapter(kv_lora_rank, num_hidden_layers=num_hidden_layers).to(model.dtype).to(model.device)

    # --- CRITICAL FIX: Recover weights discarded by HuggingFace initialization ---
    from safetensors.torch import load_file
    import glob
    
    st_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if st_files:
        adapter_sd = {}
        # Iterate through all shards
        for st_file in st_files:
            sd = load_file(st_file)
            shard_sd = {k.replace("memory_adapter.", ""): v for k, v in sd.items() if "memory_adapter" in k}
            adapter_sd.update(shard_sd)
            
        if adapter_sd:
            model.memory_adapter.load_state_dict(adapter_sd, strict=False)
            print("✅ Successfully loaded trained M.6 Adapter weights!")
        else:
            print("⚠️ WARNING: No adapter weights found in any safetensors shards.")
    
    print("Loading test dataset for Streaming PPL test...")
    # Load a long document text
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(ds["text"])
    
    # Tokenize and slice
    input_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)
    total_tokens = input_ids.shape[1]
    
    # We want a document at least 4k tokens long
    if total_tokens < 4096:
        print("Dataset text too short! Found tokens:", total_tokens)
        return
        
    chunk_1_ids = input_ids[:, :2048]
    chunk_2_ids = input_ids[:, 2048:4096]
    
    print("--- Running Baseline (No Memory) ---")
    
    with torch.no_grad():
        # Baseline Chunk 1
        loss_function = torch.nn.CrossEntropyLoss()
        
        # We need attention mask
        attention_mask_1 = torch.ones_like(chunk_1_ids)
        outputs_1 = model(input_ids=chunk_1_ids, attention_mask=attention_mask_1, use_cache=False)
        shift_logits_1 = outputs_1.logits[:, :-1, :].contiguous()
        shift_labels_1 = chunk_1_ids[:, 1:].contiguous()
        loss_1 = loss_function(shift_logits_1.view(-1, shift_logits_1.size(-1)), shift_labels_1.view(-1))
        ppl_1 = math.exp(loss_1.item())
        print(f"Chunk 1 PPL: {ppl_1:.2f}")

        # Baseline Chunk 2 (Stateless)
        attention_mask_2 = torch.ones_like(chunk_2_ids)
        outputs_base_2 = model(input_ids=chunk_2_ids, attention_mask=attention_mask_2, use_cache=False)
        shift_logits_base_2 = outputs_base_2.logits[:, :-1, :].contiguous()
        shift_labels_2 = chunk_2_ids[:, 1:].contiguous()
        loss_base_2 = loss_function(shift_logits_base_2.view(-1, shift_logits_base_2.size(-1)), shift_labels_2.view(-1))
        ppl_base_2 = math.exp(loss_base_2.item())
        print(f"Chunk 2 PPL (No Memory): {ppl_base_2:.2f}")
        
    print("\n--- Running Memory Injected Evaluation ---")
    
    with torch.no_grad():
        # Hook to capture k_pass exactly like training
        captured_k_pass = []
        def hook(module, inp, out):
            if isinstance(out, tuple): out = out[0]
            k_pass = out[..., :model.memory_adapter.kv_lora_rank]
            captured_k_pass.append(k_pass)

        # Register hook to first layer
        handle = model.model.layers[0].self_attn.kv_a_proj_with_mqa.register_forward_hook(hook)
        
        # Rerun Chunk 1 to capture its k_pass!
        model(input_ids=chunk_1_ids, attention_mask=attention_mask_1, use_cache=False)
        handle.remove()
        
        k_pass_evicted = captured_k_pass[0].detach()
        
        # ❌ REMOVE THIS: P is a persistent global bank, do not wipe it!
        # model.memory_adapter.P.data.normal_(mean=0.0, std=0.02)
        
        # Create P_1
        P_1 = model.memory_adapter.write(k_pass_evicted)
        
        # ✅ FIX: Map P_1 into layer coordinates using the ZeRO-3 safe forward pass
        memory_latents_1, memory_gate_1 = model.memory_adapter(P_1)
        
        # Run Chunk 2 WITH memory injected
        outputs_mem_2 = model(
            input_ids=chunk_2_ids, 
            attention_mask=attention_mask_2, 
            use_cache=False, 
            memory_latents=memory_latents_1,
            memory_gate=memory_gate_1
        )
        shift_logits_mem_2 = outputs_mem_2.logits[:, :-1, :].contiguous()
        loss_mem_2 = loss_function(shift_logits_mem_2.view(-1, shift_logits_mem_2.size(-1)), shift_labels_2.view(-1))
        ppl_mem_2 = math.exp(loss_mem_2.item())
        
        print(f"Chunk 2 PPL (With Memory): {ppl_mem_2:.2f}")

    print("\n--- Summary ---")
    print(f"Memory improved PPL on Chunk 2 from {ppl_base_2:.2f} to {ppl_mem_2:.2f}")

    if wandb.run is not None:
        wandb.log({
            "eval/chunk_1_ppl": ppl_1,
            "eval/chunk_2_base_ppl": ppl_base_2,
            "eval/chunk_2_mem_ppl": ppl_mem_2
        })
        wandb.finish()

if __name__ == "__main__":
    main()
