#!/bin/bash

# Train only the ZIP-RC output head (leaving the base model frozen).
# Simplified configuration for general use.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

MODEL_ID="Qwen/Qwen3-1.7B"
DATA_PATH="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_thinking_with_joint_values.parquet"
WEIGHTS_PATH="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_freeze_baseline_test"
DISTRIBUTION_TOKEN_ID=151669
LEARNING_RATE=3e-5
LABEL_COLUMN="value"
KL_COEFFICIENT=0.0
VISUALIZATION_FREQ=10
MAX_STEPS=100
LOG_DIR="logs"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
cd "$REPO_ROOT"

mkdir -p "$LOG_DIR"
BASE_NAME=$(basename "$DATA_PATH" .parquet)_no_kl
LOG_FILE="$LOG_DIR/train_${BASE_NAME}.log"

if [ "$LABEL_COLUMN" = "value" ]; then
    REWARD_VALUES_ARG=""
else
    REWARD_VALUES_ARG="--reward_values 0.0 1.0"
fi

echo "=================================================="
echo "Starting ZIP Head-Only Training"
echo "=================================================="
echo "  Model: $MODEL_ID"
echo "  Data: $DATA_PATH"
echo "  Output: $WEIGHTS_PATH"
echo "  Distribution start token ID: $DISTRIBUTION_TOKEN_ID"
echo "  Learning rate: $LEARNING_RATE"
echo "  Label column: $LABEL_COLUMN"
echo "  KL coefficient: $KL_COEFFICIENT"
echo "  Start time: $(date)"
echo "=================================================="

python3 -u src/train_ziprc_head_only.py \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" \
    --weights_path "$WEIGHTS_PATH" \
    --distribution_token_id "$DISTRIBUTION_TOKEN_ID" \
    --learning_rate "$LEARNING_RATE" \
    --label_column "$LABEL_COLUMN" \
    $REWARD_VALUES_ARG \
    --visualization_freq "$VISUALIZATION_FREQ" \
    --max_steps "$MAX_STEPS" \
    --dist-backend "ddp" 2>&1 | tee -a "$LOG_FILE"

exit_code=${PIPESTATUS[0]}
echo "=================================================="
echo "Training completed with exit code: $exit_code at $(date)"
if [ -d "$WEIGHTS_PATH" ]; then
  echo "Model size: $(du -sh "$WEIGHTS_PATH" | cut -f1)"
fi
echo "=================================================="

exit "$exit_code"
