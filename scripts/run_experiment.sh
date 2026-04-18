#!/bin/bash

# ==============================================================================
# Input Validation
# ==============================================================================
if [ -z "$1" ]; then
  echo "Usage: $0 <num_gpus>"
  echo "Example: $0 4"
  exit 1
fi

NUM_GPUS=$1

# ==============================================================================
# Tmux Encapsulation (Protect against SSH drops)
# ==============================================================================
# If INSIDE_TMUX_RUN is not set, we are outside the encapsulated session
if [ -z "$INSIDE_TMUX_RUN" ]; then
  if ! command -v tmux &> /dev/null; then
    echo "❌ Error: tmux is not installed. Please install it (e.g., 'sudo apt install tmux' or 'brew install tmux')."
    exit 1
  fi

  SESSION_NAME="pla_pipeline"
  LOG_FILE="pipeline_run.log"
  SCRIPT_PATH=$(realpath "$0" 2>/dev/null || echo "$PWD/$0")

  echo "====================================================================="
  echo "🛡️  Encapsulating execution in a tmux session to protect against SSH drops..."
  echo "📌 Session Name: $SESSION_NAME"
  echo "💡 If your connection drops, reconnect and run: tmux attach-session -t $SESSION_NAME"
  echo "====================================================================="
  sleep 3
  
  # Start the script inside a detached tmux session, tee output to a log, and wait for user input at the end
  env INSIDE_TMUX_RUN=1 tmux new-session -d -s "$SESSION_NAME" \
    "bash \"$SCRIPT_PATH\" \"$NUM_GPUS\" 2>&1 | tee \"$LOG_FILE\"; echo ''; echo '✅ Script finished. Press [ENTER] to close this window.'; read"

  # Attach your current terminal to the secure session
  tmux attach-session -t "$SESSION_NAME"
  
  # Exit the parent shell once detached or closed
  exit 0
fi

# Exit immediately if a command exits with a non-zero status inside tmux
set -e

# ==============================================================================
# Robust Cleanup on Exit / Failure
# ==============================================================================
cleanup() {
    echo -e "\n🧹 Running cleanup..."
    [ -f training/train.py.bak ] && mv training/train.py.bak training/train.py
    [ -f training/zero3.yaml.bak ] && mv training/zero3.yaml.bak training/zero3.yaml
    [ -f benchmark_and_test.py ] && rm -f benchmark_and_test.py
}
trap cleanup EXIT

# ==============================================================================
# Configuration Variables
# ==============================================================================
# Base model to use
MODEL_NAME="meta-llama/Llama-3.2-1B" 

# Directories for the converted and healed models
CONVERTED_PATH="outputs/llama3.2-1B-mla"
FINETUNED_PATH="outputs/llama3.2-1B-mla-healed"

# Finetuning dataset
DATASET="HuggingFaceFW/fineweb-edu"

echo "====================================================================="
echo "🚀 Starting Perpetual Latent Attention Pipeline with $NUM_GPUS GPUs"
echo "📦 Base Model: $MODEL_NAME"
echo "====================================================================="

# ==============================================================================
# Step 1: Extract Activations & Convert Model to MLA
# ==============================================================================
echo -e "\n[1/3] Extracting activations on wikitext-2 & Converting Model to MLA..."

# We use --freqfold 4 as defined in your scripts/llama3.2-1B.sh. 
# This calibration step is mapped to a single GPU (cuda:0).
python transmla/converter.py \
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
echo -e "\n[2/3] Healing model on $DATASET using DeepSpeed & Accelerate..."

# fineweb-edu is a massive 1.3TB dataset. We dynamically patch train.py 
# to load the 'sample-10BT' subset to avoid multi-hour downloads/OOM issues.
sed -i.bak 's/load_dataset(training_args.data_path, split="train")/load_dataset(training_args.data_path, name="sample-10BT", split="train")/g' training/train.py

# Patch the zero3.yaml configuration file to use the specified number of GPUs
sed -i.bak "s/num_processes: .*/num_processes: $NUM_GPUS/g" training/zero3.yaml

# Run the training script with accelerate. 
# --max_steps 100 is used here to ensure a quick sanity check of the pipeline.
# Remove it when you want to run the full training loop.
accelerate launch \
    --config_file training/zero3.yaml \
    training/train.py \
    --model_name_or_path "$CONVERTED_PATH" \
    --data_path "$DATASET" \
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
    --report_to "none" \
    --max_steps 100 

# Restore original files via explicit call to prevent duplicate actions later
mv training/train.py.bak training/train.py
mv training/zero3.yaml.bak training/zero3.yaml

# Clean up done natively via trap function.

echo "====================================================================="
echo "✅ Pipeline Completed Successfully!"
echo "Transformed & Healed MLA Model saved at: $FINETUNED_PATH"
echo "Logs have been saved to: pipeline_run.log"
echo "====================================================================="