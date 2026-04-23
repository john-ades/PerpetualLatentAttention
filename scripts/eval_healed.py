import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add transmla to path so we can import your utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'transmla')))
from utils import get_dataset, prepare_test_dataloader, evaluate_ppl

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_healed.py <model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    print(f"\n[Eval] Loading tokenizer and model from {model_path}...")
    
    try:
        from vllm import ModelRegistry
        from transmla.vllm_registry.deepseek import DeepseekV2ForCausalLM
        ModelRegistry.register_model("LlamaMLAForCausalLM", DeepseekV2ForCausalLM)
        ModelRegistry.register_model("Gemma2MLAForCausalLM", DeepseekV2ForCausalLM)
    except ImportError:
        pass  # If vLLM is not installed, continue with Transformers

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except ValueError:
        import json
        import os
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

    # device_map="auto" gracefully distributes the model across your available GPUs without DeepSpeed overhead
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print("--- Perplexity Evaluation (Wikitext-2) ---")
    dataset = get_dataset("wikitext2")
    test_loader = prepare_test_dataloader(dataset["test"], tokenizer, batch_size=2)
    
    ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, "Evaluating Healed Model")
    print(f"\n✅ Final Healed Perplexity on Wikitext-2: {ppl:.4f}")

    import wandb
    wandb.init(project=os.environ.get("WANDB_PROJECT", "perpetual-latent-attention"), name="eval_healed", job_type="eval")
    wandb.log({"eval/perplexity_wikitext2": ppl})
    wandb.finish()


if __name__ == "__main__":
    main()
