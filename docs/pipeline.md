# ZIP-RC: Training pipeline and script reference

This document contains the detailed, flag-level instructions for generating rollouts, labeling them, and training ZIP-RC predictors.

Notes:
- Most commands require GPUs. Rollout generation and grading use `vllm`; training uses `transformers` + (optionally) `flash_attention_2`.
- You will need access to download Hugging Face datasets/models.

## Setup
Create the conda environment:

```bash
conda env create -f environment.yml
conda activate zip
```

## End-to-end pipeline for training ZIP-RC

ZIP-RC is trained on on-policy rollouts from a training dataset (e.g. `rohinm/adaptivemath`).

1. Label rollouts with (noisy) `correct` using an eval/grader model.
2. Train an intermediate joint head on `correct` without KL (not used for inference).
3. Use that intermediate model to write a denoised scalar `value` label back onto the rollouts.
4. Train the final ZIP-RC model on `value` with KL to preserve next-token behavior.

### Smoke test (end-to-end)
Runs the full pipeline on a tiny slice to verify your environment, model downloads, and GPU setup.

Preferred:

```bash
bash scripts/smoke_test.sh
```

The script is configurable via environment variables (e.g., `ZIPRC_MODEL_ID`, `ZIPRC_GRADER_MODEL_ID`, `ZIPRC_DATASET_ID`, `ZIPRC_MAX_NUM_PROMPTS`). If you have fewer GPUs than the default setup, you will likely want to set a smaller `ZIPRC_GRADER_MODEL_ID` and/or reduce `ZIPRC_GRADER_TP_SIZE`.

Equivalent manual commands (matches the default `scripts/smoke_test.sh` settings):

```bash
mkdir -p data models
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 1) Generate rollouts (tiny)
python3 src/generate_ziprc_rollouts.py \
  --model Qwen/Qwen3-1.7B \
  --dataset rohinm/adaptivemath --split train --prompt-column problem --answer-column answer \
  --out data/smoke_rollouts.parquet \
  --max-num-prompts 8 \
  --thinking-samples 0 --non-thinking-samples 1 \
  --temperature 1.0 --min-p 0.1 \
  --max-model-len 4096 \
  --dp-size 8 --tp-size 1

# 2) Label correctness with a grader (writes `correct` into the same parquet)
python3 src/evaluate_and_label_rollouts.py \
  --data data/smoke_rollouts.parquet \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 8 \
  --max-model-len 8192

# 3) Train intermediate joint head on `correct` (KL=0; not used for inference)
python3 src/train_ziprc_joint_head.py \
  --model-id Qwen/Qwen3-1.7B \
  --data-path data/smoke_rollouts.parquet \
  --weights-path models/smoke_joint_correct_no_kl \
  --distribution-token-id 151669 \
  --label-column correct \
  --kl-coefficient 0.0 \
  --max-steps 50 \
  --max-length 4096

# 4) Write denoised scalar `value` labels
python3 src/score_with_ziprc_joint_head.py \
  --model models/smoke_joint_correct_no_kl \
  --in-parquet data/smoke_rollouts.parquet \
  --out-parquet data/smoke_rollouts_with_value.parquet \
  --distribution-token-id 151669 \
  --num-length-bins 8 \
  --reward-values 0.0 1.0 \
  --last-k 64 \
  --max-length 4096

# 5) Train final ZIP-RC model on `value` (with KL)
python3 src/train_ziprc_joint_head.py \
  --model-id Qwen/Qwen3-1.7B \
  --data-path data/smoke_rollouts_with_value.parquet \
  --weights-path models/smoke_ziprc_value_with_kl \
  --distribution-token-id 151669 \
  --label-column value \
  --kl-coefficient 10.0 \
  --max-steps 50 \
  --max-length 4096
```

### 1) Generate rollouts (`src/generate_ziprc_rollouts.py`)
Generates multiple samples per prompt (optionally “thinking” and “non-thinking”) and writes a single parquet.

```bash
python3 src/generate_ziprc_rollouts.py \
  --model Qwen/Qwen3-1.7B \
  --dataset rohinm/adaptivemath --split train --prompt-column problem --answer-column answer \
  --out data/adaptivemath_rollouts.parquet \
  --max-num-prompts 100000 \
  --thinking-samples 0 --non-thinking-samples 2 \
  --temperature 1.0 --min-p 0.1 \
  --max-model-len 32768 \
  --dp-size 8 --tp-size 1
```

Output columns include `prompt`, `answer`, `response`, `input_ids`, `label_positions`, `finished`, and `reasoning_enabled`.
Increase `--dp-size` (workers) and `--tp-size` (GPUs per worker) to scale up generation.

### 2) Label rollouts for correctness (`src/evaluate_and_label_rollouts.py`)
Uses a grading model to compare each finished response to the verified answer and writes `correct` back into the same parquet.

```bash
python3 src/evaluate_and_label_rollouts.py \
  --data data/adaptivemath_rollouts.parquet \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --tensor-parallel-size 8
```

### 3) Train an intermediate joint head on `correct` (no KL) (`src/train_ziprc_joint_head.py`)
This stage is used to learn a denoised terminal value signal; it is not used for inference, so set KL to 0.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 src/train_ziprc_joint_head.py \
  --model-id Qwen/Qwen3-1.7B \
  --data-path data/adaptivemath_rollouts.parquet \
  --weights-path models/joint_correct_no_kl \
  --distribution-token-id 151669 \
  --label-column correct \
  --kl-coefficient 0.0
```

### 4) Write `value` labels using the intermediate model (`src/score_with_ziprc_joint_head.py`)
Runs the intermediate joint head and writes a new parquet with a `value` column in `[0,1]` (mean expected reward over the last-K labeled positions).

```bash
python3 src/score_with_ziprc_joint_head.py \
  --model models/joint_correct_no_kl \
  --in-parquet data/adaptivemath_rollouts.parquet \
  --out-parquet data/adaptivemath_rollouts_with_value.parquet \
  --distribution-token-id 151669 \
  --num-length-bins 8 \
  --reward-values 0.0 1.0
```

### 5) Train the final ZIP-RC model on `value` (with KL) (`src/train_ziprc_joint_head.py`)
This is the model used at inference time for zero-overhead reward/length prediction.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 src/train_ziprc_joint_head.py \
  --model-id Qwen/Qwen3-1.7B \
  --data-path data/adaptivemath_rollouts_with_value.parquet \
  --weights-path models/ziprc_value_with_kl \
  --distribution-token-id 151669 \
  --label-column value \
  --kl-coefficient 10.0
```

### (Optional) ZIP-RC-Lite (head-only) (`src/train_ziprc_head_only.py`)
Freezes the transformer body and trains only the LM head to predict the joint distribution.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 src/train_ziprc_head_only.py \
  --model-id Qwen/Qwen3-1.7B \
  --data-path data/adaptivemath_rollouts_with_value.parquet \
  --weights-path models/ziprc_lite_value \
  --distribution-token-id 151669 \
  --label-column value
```

## Reading ZIP-RC predictions at inference time
Given `distribution_token_id` and `num_bins = num_length_bins * num_reward_states`, the joint distribution logits live in the vocab slice:
`[distribution_token_id, distribution_token_id + num_bins)`.

To use ZIP-RC during decoding:
1. Compute `softmax` over that slice to get `p(reward_bin, length_bin | prefix)`.
2. Mask those token IDs before sampling the next token so they are never generated.
3. Keep `distribution_token_id`, `num_length_bins`, and reward binning consistent across training/scoring, and ensure the reserved slice exists: `distribution_token_id + num_bins <= vocab_size`.

This repo currently provides offline scoring (`src/score_with_ziprc_joint_head.py`); ZIP-RC sampling (adaptive meta-actions) is not yet included.
