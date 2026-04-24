#!/bin/bash
set -e

# ==============================================================================
# Cleanup
# ==============================================================================
cleanup() {
    if [ -f "training/zero3.yaml.bak" ]; then
        mv training/zero3.yaml.bak training/zero3.yaml
    fi
}
trap cleanup EXIT

# ==============================================================================
# Input Validation
# ==============================================================================
if [ -z "$1" ]; then
  if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l | awk '{print $1}')
  else
    NUM_GPUS=$(uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
  fi
  if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
  fi
  echo "No GPU count provided. Automatically detected $NUM_GPUS GPUs."
else
  NUM_GPUS=$1
fi
MODEL_NAME="meta-llama/Llama-3.1-8B"

# Directories for the converted and healed models
CONVERTED_PATH="outputs/llama3.1-8B-mla"
FINETUNED_PATH="outputs/llama3.1-8B-mla-healed"

# Finetuning dataset
DATASET="mixed_healing"

echo "====================================================================="
echo "🚀 Starting Perpetual Latent Attention Pipeline with $NUM_GPUS GPUs"
echo "📦 Base Model: $MODEL_NAME"
echo "====================================================================="

# ==============================================================================
# Step 1: Extract Activations & Convert Model to MLA
# ==============================================================================
echo -e "\n[1/4] Extracting activations on wikitext-2 & Converting Model to MLA..."

# We use --freqfold 4 as defined in your scripts/llama3.2-1B.sh. 
# This calibration step is mapped to a single GPU (cuda:0).
uv run python transmla/converter.py \
    --model-path "$MODEL_NAME" \
    --save-path "$CONVERTED_PATH" \
    --cal-dataset wikitext2 \
    --cal-nsamples 128 \
    --cal-max-seqlen 256 \
    --freqfold 4 \
    --device cuda:0

# ==============================================================================
# Step 2: Finetune / Heal Model
# ==============================================================================
echo -e "\n[2/4] Healing model on $DATASET using DeepSpeed & Accelerate..."

# fineweb-edu is a massive 1.3TB dataset. We pass 'sample-10BT' subset and 
# max_train_samples below to avoid multi-hour downloads and disk quota exceeded issues.

# Patch the zero3.yaml configuration file to use the specified number of GPUs
sed -i.bak "s/num_processes: .*/num_processes: $NUM_GPUS/g" training/zero3.yaml

# Run the training script with accelerate. 
# --max_steps 500 is used here to ensure a quick sanity check of the pipeline.
# Remove it when you want to run the full training loop.
export WANDB_PROJECT="perpetual-latent-attention"
uv run accelerate launch \
    --config_file training/zero3.yaml \
    training/train.py \
    --model_name_or_path "$CONVERTED_PATH" \
    --data_path "$DATASET" \
    --dataset_name "sample-10BT" \
    --output_dir "$FINETUNED_PATH" \
    --bf16 \
    --num_train_epochs 1 \
    --seq_len 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --max_steps 10 

# ==============================================================================
# Step 3: Train M.6 Persistent Memory Adapter (TBPTT)
# ==============================================================================
echo -e "\n[3/4] Freezing TransMLA Backbone and Training M.6 Adapter..."
MEMORY_ADAPTER_PATH="outputs/llama3.1-8B-mla-m6"

# Pass the FINETUNED_PATH (healed model) as the base model
uv run accelerate launch \
    --config_file training/zero3.yaml \
    training/train.py \
    --model_name_or_path "$FINETUNED_PATH" \
    --output_dir "$MEMORY_ADAPTER_PATH" \
    --data_path "$DATASET" \
    --dataset_name "sample-10BT" \
    --train_m6_adapter True \
    --freeze_backbone True \
    --bf16 \
    --num_train_epochs 1 \
    --seq_len 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_steps 10

# ==============================================================================
# Step 4: Evaluation (Streaming PPL)
# ==============================================================================
echo -e "\n[4/4] Evaluating the M.6 persistent memory model setup..."

uv run python scripts/eval_memory.py "$MEMORY_ADAPTER_PATH"

echo "====================================================================="
echo "✅ Pipeline Completed Successfully!"
echo "Transformed, Healed, and Memorized MLA Model saved at: $MEMORY_ADAPTER_PATH"
echo "====================================================================="