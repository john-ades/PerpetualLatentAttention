# Perpetual Latent Attention (PLA) - Run Pipeline

This repository contains an automated pipeline for converting and finetuning models to use Perpetual Latent Attention.

## Running the Pipeline

The primary entry point for executing the pipeline is the `scripts/run_experiment.sh` script.

### Prerequisites
- Bash environment. Use `tmux` or `screen` manually if you want to protect your session against SSH connection drops.
- A functional Python environment with dependencies like `accelerate` and `deepspeed` configured.
- A Hugging Face token exported in your environment to access models and datasets. Set this by running:
  ```bash
  export HF_TOKEN="your_token_here"
  ```

### Usage

To execute the pipeline, run the `scripts/run_experiment.sh` script and pass the number of GPUs you want to use for the training phase.

```bash
./scripts/run_experiment.sh <num_gpus>
```

**Example:** Run the pipeline using 4 GPUs:
```bash
./scripts/run_experiment.sh 4
```

### What does the script do?

The `scripts/run_experiment.sh` script performs the following automated steps:

1. **Extraction & Conversion (Step 1)**: Extracts activations using calibration data (`wikitext-2`) and converts the base model (`meta-llama/Llama-3.2-1B`) to an MLA format using `transmla/converter.py`. This calibration step runs on a single GPU (`cuda:0`).
2. **Finetuning & Healing (Step 2)**: Finetunes the newly converted model using the `HuggingFaceFW/fineweb-edu` dataset (`sample-10BT` subset). It runs via `accelerate` and `DeepSpeed Zero-3` utilizing the exact number of GPUs specified. 

### Monitoring the Run

The script runs sequentially in your current terminal. Standard output and errors will be printed directly to your console.
