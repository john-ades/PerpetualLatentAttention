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
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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

if __name__ == "__main__":
    main()
