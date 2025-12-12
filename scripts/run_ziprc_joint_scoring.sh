#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=zip_joint_scoring
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=6
#SBATCH --output=/home/rohin/ZIP/logs/ziprc_joint_scoring.out
#SBATCH --error=/home/rohin/ZIP/logs/ziprc_joint_scoring.err
#SBATCH --account=liquidai

set -euo pipefail
export PYTHONUNBUFFERED=1

# NCCL configuration for robust distributed operations
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_SOCKET_IFNAME=bond0
export UCX_TLS=self,shm,tcp
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

cd $HOME/ZIP
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate
conda activate zip

# Choose which dataset to label (uncomment the one you want)

# === Math dataset (Qwen 0.6B) ===
# # Joint-distribution model trained on correctness
# JOINT_MODEL="/home/rohin/ZIP/models/zip_joint_distribution_qwen_06b_thinking_math_no_kl_correctness_06"

# # Training set
# IN_PARQUET="/home/rohin/ZIP/data/zip_training_math_data_qwen06b_thinking_06.parquet"
# OUT_PARQUET="/home/rohin/ZIP/data/zip_training_math_data_qwen06b_thinking_06_with_joint_values.parquet"

# Joint-distribution model trained on correctness
# JOINT_MODEL="/home/rohin/ZIP/models/zip_joint_distribution_qwen_17b_thinking_hallucination_no_kl_correctness"
JOINT_MODEL="/home/rohin/ZIP/models/zip_joint_distribution_qwen_17b_non_thinking_hallucination_no_kl_correctness"

# Training set
# IN_PARQUET="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_thinking.parquet"
# OUT_PARQUET="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_thinking_with_joint_values.parquet"
IN_PARQUET="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_non_thinking.parquet"
OUT_PARQUET="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_non_thinking_with_joint_values.parquet"

# Test set (uncomment if needed)
# IN_PARQUET="/home/rohin/ZIP/data/zip_training_math_data_qwen06b_thinking_test.parquet"
# OUT_PARQUET="/home/rohin/ZIP/data/zip_training_math_data_qwen06b_thinking_test_with_joint_values.parquet"

# === Hallucination dataset (Qwen 0.6B) === 
# Uncomment this block to use hallucination model instead
# JOINT_MODEL="/home/rohin/ZIP/models/zip_joint_distribution_qwen_06b_thinking_hallucination_no_kl_correctness"
# IN_PARQUET="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen06b_thinking_train.parquet"
# OUT_PARQUET="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen06b_thinking_train_with_joint_values.parquet"

# === For Qwen 4B models (if they exist) ===
# Adjust paths accordingly if you have 4B correctness models

# Model configuration (matching what was used during training)
DISTRIBUTION_TOKEN_ID=151669
NUM_LENGTH_BINS=8
REWARD_VALUES="0.0 1.0"   # model trained on binary correctness

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

mkdir -p "$(dirname "$OUT_PARQUET")"

base_name=$(basename "$IN_PARQUET" .parquet)

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
  --log-every 25 2>&1 | tee -a /home/rohin/ZIP/logs/label_joint_values_${base_name}.log

exit_code=${PIPESTATUS[0]}
echo "=================================================="
echo "Labeling completed with exit code: $exit_code at $(date)"
if [ -f "$OUT_PARQUET" ]; then
    echo "Output file size: $(du -sh "$OUT_PARQUET" | cut -f1)"
fi
echo "=================================================="

exit $exit_code