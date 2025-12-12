#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=zip_ground_truth_labeling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/ZIP/logs/zip_ground_truth_labeling.out
#SBATCH --error=/home/rohin/ZIP/logs/zip_ground_truth_labeling.err
#SBATCH --account=liquidai
#SBATCH --exclude=liquid-gpu-[054]

# -----------------------------
# Network / NCCL configuration
# -----------------------------
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

# -----------------------------
# Environment setup
# -----------------------------
cd $HOME/ZIP
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate
conda activate zip

# -----------------------------
# Configuration
# -----------------------------


# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen06b_thinking.parquet"
# task="hallucination"

# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen4b_2507_thinking.parquet"
# task="hallucination"

# data_path="/home/rohin/ZIP/data/zip_training_hallucination_data_qwen17b_non_thinking.parquet"
# task="hallucination"

# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_non_thinking.parquet"
# task="correctness"

# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen4b_2507_thinking.parquet"
# task="correctness"

# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_thinking_4_min_p_01.parquet"
# task="correctness"

# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_thinking_4_min_p_001.parquet"
# task="correctness"

data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_non_thinking_8_min_p_01.parquet"
task="correctness"

# data_path="/home/rohin/ZIP/data/zip_training_adaptivemath_data_qwen17b_non_thinking_8_min_p_001.parquet"
# task="correctness"


base_name=$(basename $data_path .parquet)

echo "Starting evaluate_and_label_rollouts with ground truth:"
echo "  Data: $data_path"
echo "  Task: $task"
echo "  Start time: $(date)"
echo

python3 -u src/evaluate_and_label_rollouts.py \
    --data "$data_path" \
    --task "$task" \
    --show-examples \
    2>&1 | tee -a "/home/rohin/ZIP/logs/label_ground_truth_${base_name}.log"

exit_code=${PIPESTATUS[0]}
echo "Label and evaluate completed with exit code: $exit_code at $(date)"

[ -f "$data_path" ] && echo "Input file size: $(du -h $data_path | cut -f1)"
exit $exit_code