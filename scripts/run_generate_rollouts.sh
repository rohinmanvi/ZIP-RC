#!/bin/bash

# Minimal launcher for generating ZIP-RC rollouts. Replace the configuration
# variables below as needed; SBATCH directives are intentionally omitted so the
# script can be run locally or wrapped in your own scheduler template.

set -euo pipefail
export PYTHONUNBUFFERED=1

# --- Configuration ----------------------------------------------------------
MODEL="Qwen/Qwen3-1.7B"
DATASET="rohinm/adaptivemath"
PROMPT_COLUMN="problem"
OUTPUT_FILE="data/zip_training_adaptivemath_data_qwen17b_non_thinking_32_min_p_001.parquet"
THINKING_SAMPLES=0
NON_THINKING_SAMPLES=32
MAX_NUM_SEQS=32
MAX_PROMPTS=128000000
TEMPERATURE=1.0
MIN_P=0.1
LOG_DIR="logs"

# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
cd "$REPO_ROOT"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/data_$(basename "${OUTPUT_FILE}" .parquet).log"

echo "Starting ZIP training data generation:"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_FILE"
echo "  Prompts: $MAX_PROMPTS (${THINKING_SAMPLES} reasoning + ${NON_THINKING_SAMPLES} non-reasoning per prompt)"
echo "  Start time: $(date)"

python3 -u src/generate_ziprc_rollouts.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --prompt-column "$PROMPT_COLUMN" \
    --out "$OUTPUT_FILE" \
    --max-num-prompts "$MAX_PROMPTS" \
    --thinking-samples "$THINKING_SAMPLES" \
    --non-thinking-samples "$NON_THINKING_SAMPLES" \
    --dp-size 8 --tp-size 1 \
    --allow-partial-merge \
    --temperature "$TEMPERATURE" --min-p "$MIN_P" \
    --max-model-len 32768 --max-num-seqs "$MAX_NUM_SEQS" 2>&1 | tee -a "$LOG_FILE"

exit_code=${PIPESTATUS[0]}
echo "Data generation completed with exit code: $exit_code at $(date)"

if [ -f "$OUTPUT_FILE" ]; then
  echo "Generated file size: $(du -h "$OUTPUT_FILE" | cut -f1)"
fi

exit "$exit_code"
