# ZIP-RC: zero-overhead reward–cost prediction

This repository contains a minimal, reproducible path for training and evaluating ZIP-RC:
zero-overhead joint reward–cost prediction for adaptive sampling. The steps below replace
all helper scripts; copy/paste the commands and adjust paths to your setup.

## Environment
```bash
conda env create -f environment.yml -n ziprc
conda activate ziprc
```

## 1) Generate rollouts
Create training data by sampling a base model against a prompt set.
```bash
python -u src/generate_ziprc_rollouts.py \
  --model Qwen/Qwen3-1.7B \
  --dataset rohinm/adaptivemath \
  --prompt-column problem \
  --out data/zip_training.parquet \
  --max-num-prompts 1000 \
  --thinking-samples 0 \
  --non-thinking-samples 32 \
  --dp-size 8 --tp-size 1 \
  --max-model-len 32768 --max-num-seqs 32 \
  --temperature 1.0 --min-p 0.1
```

## 2) Label rollouts with ground truth
Attach correctness/value labels used for supervision.
```bash
python -u src/evaluate_and_label_rollouts.py \
  --data data/zip_training.parquet \
  --task correctness \
  --show-examples
```

## 3) Train the ZIP-RC joint head (full model)
Fine-tune the full model to predict the joint reward–cost grid. Distributed
training spawns one process per visible GPU.
```bash
python -u src/train_ziprc_joint_head.py \
  --model_id Qwen/Qwen3-1.7B \
  --data_path data/zip_training.parquet \
  --weights_path models/ziprc_joint \
  --distribution_token_id 151669 \
  --label-column correct \
  --learning_rate 3e-5 \
  --visualization_freq 10 \
  --max_steps 100000
```

## 4) Optional: head-only ZIP-RC baseline
Train only the output head while freezing the backbone.
```bash
python -u src/train_ziprc_head_only.py \
  --model_id Qwen/Qwen3-1.7B \
  --data_path data/zip_training.parquet \
  --weights_path models/ziprc_head_only \
  --distribution_token_id 151669 \
  --label_column correct \
  --learning_rate 1e-3 \
  --num_epochs 2
```

## 5) Score rollouts with a trained joint head
Add per-step joint predictions to a rollout file using the trained model.
```bash
python -u src/score_with_ziprc_joint_head.py \
  --model models/ziprc_joint \
  --in-parquet data/zip_training.parquet \
  --out-parquet data/zip_training_with_joint_values.parquet \
  --distribution-token-id 151669 \
  --num-length-bins 8 \
  --reward-values 0.0 1.0 \
  --last-k 512 --batch-size 1 --num-workers 2 \
  --dtype bfloat16 --pos-chunk-size 512
```

### Notes
- Adjust dataset paths, model IDs, and token IDs to match your checkpoints.
- All commands run without scheduler-specific wrappers; add your own launchers if
you need SLURM/torchrun integration.
- For larger jobs, set `CUDA_VISIBLE_DEVICES` and `MASTER_PORT/ADDR` as needed
before running the commands above.
