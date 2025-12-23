#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_ID="${ZIPRC_MODEL_ID:-Qwen/Qwen3-1.7B}"
GRADER_MODEL_ID="${ZIPRC_GRADER_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

DATASET_ID="${ZIPRC_DATASET_ID:-rohinm/adaptivemath}"
SPLIT="${ZIPRC_SPLIT:-train}"
PROMPT_COLUMN="${ZIPRC_PROMPT_COLUMN:-problem}"
ANSWER_COLUMN="${ZIPRC_ANSWER_COLUMN:-answer}"

MAX_NUM_PROMPTS="${ZIPRC_MAX_NUM_PROMPTS:-8}"
THINKING_SAMPLES="${ZIPRC_THINKING_SAMPLES:-0}"
NON_THINKING_SAMPLES="${ZIPRC_NON_THINKING_SAMPLES:-1}"
TEMPERATURE="${ZIPRC_TEMPERATURE:-1.0}"
MIN_P="${ZIPRC_MIN_P:-0.1}"

GEN_MAX_MODEL_LEN="${ZIPRC_GEN_MAX_MODEL_LEN:-4096}"
GRADER_MAX_MODEL_LEN="${ZIPRC_GRADER_MAX_MODEL_LEN:-8192}"

INTERMEDIATE_MAX_STEPS="${ZIPRC_INTERMEDIATE_MAX_STEPS:-50}"
FINAL_MAX_STEPS="${ZIPRC_FINAL_MAX_STEPS:-50}"
TRAIN_MAX_LENGTH="${ZIPRC_TRAIN_MAX_LENGTH:-4096}"

DISTRIBUTION_TOKEN_ID="${ZIPRC_DISTRIBUTION_TOKEN_ID:-151669}"
NUM_LENGTH_BINS="${ZIPRC_NUM_LENGTH_BINS:-8}"
REWARD_VALUES="${ZIPRC_REWARD_VALUES:-0.0 1.0}"
LAST_K="${ZIPRC_LAST_K:-64}"

DATA_DIR="${ZIPRC_DATA_DIR:-data}"
MODELS_DIR="${ZIPRC_MODELS_DIR:-models}"
ROLLOUTS_PATH="${ZIPRC_ROLLOUTS_PATH:-${DATA_DIR}/smoke_rollouts.parquet}"
VALUE_ROLLOUTS_PATH="${ZIPRC_VALUE_ROLLOUTS_PATH:-${DATA_DIR}/smoke_rollouts_with_value.parquet}"

mkdir -p "${DATA_DIR}" "${MODELS_DIR}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -n "${ZIPRC_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${ZIPRC_CUDA_VISIBLE_DEVICES}"
    echo "[smoke_test] CUDA_VISIBLE_DEVICES not set; using ZIPRC_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${GPU_COUNT}" -ge 1 ]]; then
      DEFAULT_MAX_GPUS="${ZIPRC_DEFAULT_NUM_GPUS:-8}"
      USE_GPUS="${GPU_COUNT}"
      if [[ "${USE_GPUS}" -gt "${DEFAULT_MAX_GPUS}" ]]; then
        USE_GPUS="${DEFAULT_MAX_GPUS}"
      fi
      export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((USE_GPUS - 1)))"
      echo "[smoke_test] CUDA_VISIBLE_DEVICES not set; using ${CUDA_VISIBLE_DEVICES} (detected ${GPU_COUNT} GPU(s))" >&2
    else
      export CUDA_VISIBLE_DEVICES="0"
      echo "[smoke_test] CUDA_VISIBLE_DEVICES not set; defaulting to ${CUDA_VISIBLE_DEVICES}" >&2
    fi
  else
    export CUDA_VISIBLE_DEVICES="0"
    echo "[smoke_test] CUDA_VISIBLE_DEVICES not set; defaulting to ${CUDA_VISIBLE_DEVICES}" >&2
  fi
fi

IFS=',' read -r -a _CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${ZIPRC_NUM_GPUS:-${#_CUDA_DEVICES[@]}}"

DP_SIZE="${ZIPRC_DP_SIZE:-${NUM_GPUS}}"
TP_SIZE="${ZIPRC_TP_SIZE:-1}"
GRADER_TP_SIZE="${ZIPRC_GRADER_TP_SIZE:-${NUM_GPUS}}"

echo "[smoke_test] model=${MODEL_ID}" >&2
echo "[smoke_test] grader_model=${GRADER_MODEL_ID}" >&2
echo "[smoke_test] dataset=${DATASET_ID} split=${SPLIT} prompt_column=${PROMPT_COLUMN} answer_column=${ANSWER_COLUMN}" >&2
echo "[smoke_test] gpus=${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})" >&2

python3 src/generate_ziprc_rollouts.py \
  --model "${MODEL_ID}" \
  --dataset "${DATASET_ID}" --split "${SPLIT}" --prompt-column "${PROMPT_COLUMN}" --answer-column "${ANSWER_COLUMN}" \
  --out "${ROLLOUTS_PATH}" \
  --max-num-prompts "${MAX_NUM_PROMPTS}" \
  --thinking-samples "${THINKING_SAMPLES}" --non-thinking-samples "${NON_THINKING_SAMPLES}" \
  --temperature "${TEMPERATURE}" --min-p "${MIN_P}" \
  --max-model-len "${GEN_MAX_MODEL_LEN}" \
  --dp-size "${DP_SIZE}" --tp-size "${TP_SIZE}"

python3 src/evaluate_and_label_rollouts.py \
  --data "${ROLLOUTS_PATH}" \
  --model "${GRADER_MODEL_ID}" \
  --tensor-parallel-size "${GRADER_TP_SIZE}" \
  --max-model-len "${GRADER_MAX_MODEL_LEN}"

python3 src/train_ziprc_joint_head.py \
  --model-id "${MODEL_ID}" \
  --data-path "${ROLLOUTS_PATH}" \
  --weights-path "${MODELS_DIR}/smoke_joint_correct_no_kl" \
  --distribution-token-id "${DISTRIBUTION_TOKEN_ID}" \
  --label-column correct \
  --kl-coefficient 0.0 \
  --max-steps "${INTERMEDIATE_MAX_STEPS}" \
  --max-length "${TRAIN_MAX_LENGTH}"

python3 src/score_with_ziprc_joint_head.py \
  --model "${MODELS_DIR}/smoke_joint_correct_no_kl" \
  --in-parquet "${ROLLOUTS_PATH}" \
  --out-parquet "${VALUE_ROLLOUTS_PATH}" \
  --distribution-token-id "${DISTRIBUTION_TOKEN_ID}" \
  --num-length-bins "${NUM_LENGTH_BINS}" \
  --reward-values ${REWARD_VALUES} \
  --last-k "${LAST_K}" \
  --max-length "${TRAIN_MAX_LENGTH}"

python3 src/train_ziprc_joint_head.py \
  --model-id "${MODEL_ID}" \
  --data-path "${VALUE_ROLLOUTS_PATH}" \
  --weights-path "${MODELS_DIR}/smoke_ziprc_value_with_kl" \
  --distribution-token-id "${DISTRIBUTION_TOKEN_ID}" \
  --label-column value \
  --kl-coefficient 10.0 \
  --max-steps "${FINAL_MAX_STEPS}" \
  --max-length "${TRAIN_MAX_LENGTH}"

echo "[smoke_test] Done." >&2
