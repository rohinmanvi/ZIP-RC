#!/bin/bash

# Score rollouts with a trained joint head to produce per-step joint values.
# Edit the configuration block below; the script is scheduler-agnostic.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

JOINT_MODEL="/home/rohin/ZIP/models/zip_joint_distribution_qwen_17b_non_thinking_hallucination_no_kl_correctness"
IN_PARQUET="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_non_thinking.parquet"
OUT_PARQUET="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_non_thinking_with_joint_values.parquet"
DISTRIBUTION_TOKEN_ID=151669
NUM_LENGTH_BINS=8
REWARD_VALUES="0.0 1.0"
LOG_DIR="logs"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
cd "$REPO_ROOT"

mkdir -p "$LOG_DIR" "$(dirname "$OUT_PARQUET")"
BASE_NAME=$(basename "$IN_PARQUET" .parquet)
LOG_FILE="$LOG_DIR/label_joint_values_${BASE_NAME}.log"

echo "=================================================="
echo "Labeling with joint-distribution model"
echo "=================================================="
echo "  Model:   $JOINT_MODEL"
echo "  Input:   $IN_PARQUET"
echo "  Output:  $OUT_PARQUET"
echo "  Dist token id: $DISTRIBUTION_TOKEN_ID"
echo "  Length bins:   $NUM_LENGTH_BINS"
echo "  Reward values: $REWARD_VALUES"
echo "  Time:    $(date)"
echo "=================================================="

python3 -u src/score_with_ziprc_joint_head.py \
  --model "$JOINT_MODEL" \
  --in-parquet "$IN_PARQUET" \
  --out-parquet "$OUT_PARQUET" \
  --distribution-token-id "$DISTRIBUTION_TOKEN_ID" \
  --num-length-bins "$NUM_LENGTH_BINS" \
  --reward-values $REWARD_VALUES \
  --last-k 512 \
  --batch-size 1 \
  --num-workers 2 \
  --dtype bfloat16 \
  --pos-chunk-size 512 \
  --log-every 25 2>&1 | tee -a "$LOG_FILE"

exit_code=${PIPESTATUS[0]}
echo "=================================================="
echo "Labeling completed with exit code: $exit_code at $(date)"
if [ -f "$OUT_PARQUET" ]; then
    echo "Output file size: $(du -sh "$OUT_PARQUET" | cut -f1)"
fi
echo "=================================================="

exit "$exit_code"
