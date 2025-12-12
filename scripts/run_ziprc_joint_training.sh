#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=zip_joint_head_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/ZIP/logs/ziprc_joint_head_training.out
#SBATCH --error=/home/rohin/ZIP/logs/ziprc_joint_head_training.err
#SBATCH --account=liquidai
#SBATCH --exclude=liquid-gpu-[054]

# Network configuration
export PMI_DEBUG=1
export MPI_ROOT=/usr/mpi/gcc/openmpi-4.1.7a1/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_ROOT/lib
export OMPI_MCA_btl_tcp_if_include=bond0
export UCX_TLS=self,shm,tcp
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_13,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=1
export NCCL_SOCKET_IFNAME=bond0
export LC_CTYPE=en_US.UTF-8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

cd $HOME/ZIP
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate
conda activate zip

# Configurations

# === Configuration Options ===
# Uncomment the configuration you want to use

# Option 1: Train on hard correctness labels (original value model outputs)
# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen06b_thinking_train_with_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_hallucination_no_kl_correctness"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"

# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen06b_thinking.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_hallucination_no_kl_correctness"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-1.7B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_thinking.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_thinking_hallucination_no_kl_correctness"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-1.7B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_non_thinking.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_non_thinking_hallucination_no_kl_correctness"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-4B-Instruct-2507"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen4b_2507_thinking.parquet"
# weights_path="models/zip_joint_distribution_qwen_4b_2507_thinking_hallucination_no_kl_correctness"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-4B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen4b_06.parquet"
# weights_path="models/zip_joint_distribution_qwen_4b_hallucination_no_kl_correctness_06"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"

# Option 2: Train on hard correctness labels (math dataset)
# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_math_data_qwen06b_thinking_train_with_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_math_no_kl_correctness"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"

# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen06b_thinking.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_adaptivemath_no_kl_correctness"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-1.7B"
# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_thinking.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_no_kl_correctness"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-1.7B"
# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_non_thinking.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_no_kl_correctness"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-4B"
# data_path="/home/rohin/ZIP/data/zip_training_math_data_qwen4b_06.parquet"
# weights_path="models/zip_joint_distribution_qwen_4b_math_no_kl_correctness_06"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="correct"

# Option 3: Train on soft value labels from joint distribution model (hallucination)
# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen06b_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_hallucination_soft_values"
# distribution_token_id=151669
# learning_rate=3e-5
# label_column="value"
# kl_coefficient=10.0

# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen06b_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_hallucination_soft_values_no_kl"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="value"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen06b_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_adaptivemath_soft_values"
# distribution_token_id=151669
# learning_rate=3e-5
# label_column="value"
# kl_coefficient=10.0

# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen06b_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_adaptivemath_soft_values_no_kl"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="value"
# kl_coefficient=0.0


model_id="Qwen/Qwen3-1.7B"
data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_topk32_full_10"
weights_path="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_topk32_full_10_test"
distribution_token_id=151669
learning_rate=3e-5
label_column="value"
kl_coefficient=0.00000001


# model_id="Qwen/Qwen3-1.7B"
# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_no_kl"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="value"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-1.7B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_thinking_hallucination_soft_values"
# distribution_token_id=151669
# learning_rate=3e-5
# label_column="value"
# kl_coefficient=10.0

# model_id="Qwen/Qwen3-1.7B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_thinking_hallucination_soft_values_no_kl"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="value"
# kl_coefficient=0.0

# model_id="Qwen/Qwen3-1.7B"
# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_non_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values_no_kl"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="value"
# kl_coefficient=0.0


# model_id="Qwen/Qwen3-1.7B"
# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_non_thinking_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values_topk32_full_25"
# distribution_token_id=151669
# learning_rate=3e-5
# label_column="value"
# kl_coefficient=25.0


# Option 4: Train on soft value labels from joint distribution model (math)
# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_math_data_qwen06b_thinking_06_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_math_soft_values_06"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="value"

# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen06b_thinking_06_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_hallucination_soft_values_06"
# distribution_token_id=151669
# learning_rate=1e-4
# label_column="value"

# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_math_data_qwen06b_thinking_train_with_joint_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_06b_thinking_math_soft_values_with_kl"
# distribution_token_id=151669
# learning_rate=3e-5
# label_column="value"

# model_id="Qwen/Qwen3-4B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen4b_train_with_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_4b_hallucination_no_kl"
# distribution_token_id=151669
# learning_rate=3e-5

# model_id="Qwen/Qwen3-4B"
# data_path="/home/rohin/ZIP/data/zip_training_math_data_qwen4b_train_with_values.parquet"
# weights_path="models/zip_joint_distribution_qwen_4b_math_no_kl"
# distribution_token_id=151669
# learning_rate=3e-5


# Test with bigger model
# model_id="Qwen/Qwen3-0.6B"
# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen06b_thinking_06_with_joint_values.parquet"
# weights_path="models/test_qwen17b_thinking_hallucination_soft_values_with_kl"
# distribution_token_id=151669
# learning_rate=3e-5
# label_column="value"
# kl_coefficient=10.0


visualization_freq=10
max_steps=10000000

base_name=$(basename $data_path .parquet)_no_kl

echo "=================================================="
echo "Starting ZIP Joint Distribution Training (No KL)"
echo "=================================================="
echo "  Model: $model_id"
echo "  Data: $data_path"
echo "  Output: $weights_path"
echo "  Distribution start token ID: $distribution_token_id"
echo "  Learning rate: $learning_rate"
echo "  Label column: $label_column"
echo "  KL coefficient: $kl_coefficient (no reference model)"
echo "  Start time: $(date)"
echo "=================================================="

# Training with --full_model_training but kl_coefficient=0 means no reference model is loaded
# Set reward values based on label column
if [ "$label_column" = "value" ]; then
    # For soft values, use 7 bins as per train_ziprc_joint_head.py default
    reward_values_arg=""  # Let train_ziprc_joint_head.py use its default 7 bins for value column
else
    # For correctness, use binary
    reward_values_arg="--reward_values 0.0 1.0"
fi

python3 -u src/train_ziprc_joint_head.py \
    --model_id "$model_id" \
    --data_path "$data_path" \
    --weights_path "$weights_path" \
    --distribution_token_id $distribution_token_id \
    --learning_rate $learning_rate \
    --full_model_training \
    --label-column "$label_column" \
    $reward_values_arg \
    --visualization_freq $visualization_freq \
    --max_steps $max_steps \
    --kl_coefficient $kl_coefficient \
    --dist-backend "ddp" 2>&1 | tee -a /home/rohin/ZIP/logs/train_${base_name}.log

exit_code=${PIPESTATUS[0]}
echo "=================================================="
echo "Training completed with exit code: $exit_code at $(date)"
[ -d "$weights_path" ] && echo "Model size: $(du -sh $weights_path | cut -f1)"
echo "=================================================="

exit $exit_code