#!/bin/bash

# Label rollouts with ground truth answers. Configuration is kept minimal for
# reproducibility; adjust the paths and task name as needed.

set -euo pipefail
export PYTHONUNBUFFERED=1

DATA_PATH="data/zip_training_adaptivemath_data_qwen17b_non_thinking_8_min_p_01.parquet"
TASK="correctness"
LOG_DIR="logs"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"
cd "$REPO_ROOT"

mkdir -p "$LOG_DIR"
BASE_NAME=$(basename "$DATA_PATH" .parquet)
LOG_FILE="$LOG_DIR/label_ground_truth_${BASE_NAME}.log"

echo "Starting evaluate_and_label_rollouts with ground truth:"
echo "  Data: $DATA_PATH"
echo "  Task: $TASK"
echo "  Start time: $(date)"
echo

python3 -u src/evaluate_and_label_rollouts.py \
    --data "$DATA_PATH" \
    --task "$TASK" \
    --show-examples \
    2>&1 | tee -a "$LOG_FILE"

exit_code=${PIPESTATUS[0]}
echo "Label and evaluate completed with exit code: $exit_code at $(date)"

if [ -f "$DATA_PATH" ]; then
  echo "Input file size: $(du -h "$DATA_PATH" | cut -f1)"
fi

exit "$exit_code"
