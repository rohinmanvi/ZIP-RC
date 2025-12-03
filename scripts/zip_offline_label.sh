#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=zip_offline_label
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/ZIP/logs/offline_label.out
#SBATCH --error=/home/rohin/ZIP/logs/offline_label.err
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

# input_file="data/offline_generations_gsm8k_qwen06b_thinking_with_kl_repeat4_prelabels.parquet"
# output_file="data/offline_generations_gsm8k_qwen06b_thinking_with_kl_repeat4_prelabels_labeled_with_kl.parquet"


# input_file="data/offline_generations_amc2023_qwen06b_repeat32_prelabels.parquet"
# output_file="data/offline_generations_amc2023_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# input_file="data/offline_generations_aime2024_qwen06b_thinking_repeat32_prelabels.parquet"
# output_file="data/offline_generations_aime2024_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# input_file="data/offline_generations_math500_qwen06b_thinking_repeat8_prelabels.parquet"
# output_file="data/offline_generations_math500_qwen06b_thinking_repeat8_prelabels_adaptivemath_labeled.parquet"
# input_file="data/offline_generations_gsm8k_qwen06b_thinking_repeat4_prelabels.parquet"
# output_file="data/offline_generations_gsm8k_qwen06b_thinking_repeat4_prelabels_adaptivemath_labeled.parquet"


# input_file="data/offline_generations_amc2023_qwen17b_thinking_repeat32_prelabels.parquet"
# output_file="data/offline_generations_amc2023_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# input_file="data/offline_generations_aime2024_qwen17b_thinking_repeat32_prelabels.parquet"
# output_file="data/offline_generations_aime2024_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# input_file="data/offline_generations_math500_qwen17b_thinking_repeat8_prelabels.parquet"
# output_file="data/offline_generations_math500_qwen17b_thinking_repeat8_prelabels_adaptivemath_labeled.parquet"
# input_file="data/offline_generations_gsm8k_qwen17b_thinking_repeat4_prelabels.parquet"
# output_file="data/offline_generations_gsm8k_qwen17b_thinking_repeat4_prelabels_adaptivemath_labeled.parquet"


# input_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels.parquet"
# output_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_no_kl.parquet"
# input_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels.parquet"
# output_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_no_kl.parquet"
# input_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels.parquet"
# output_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_no_kl.parquet"
# input_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels.parquet"
# output_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_no_kl.parquet"


# input_file="data/offline_generations_amc2023_qwen17b_thinking_repeat32_prelabels.parquet"
# output_file="data/offline_generations_amc2023_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
# input_file="data/offline_generations_aime2024_qwen17b_thinking_repeat32_prelabels.parquet"
# output_file="data/offline_generations_aime2024_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
# input_file="data/offline_generations_math500_qwen17b_thinking_repeat8_prelabels.parquet"
# output_file="data/offline_generations_math500_qwen17b_thinking_repeat8_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
# input_file="data/offline_generations_gsm8k_qwen17b_thinking_repeat4_prelabels.parquet"
# output_file="data/offline_generations_gsm8k_qwen17b_thinking_repeat4_prelabels_adaptivemath_labeled_freeze_baseline.parquet"


# input_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels.parquet"
# output_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
# input_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels.parquet"
# output_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
# input_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels.parquet"
# output_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
# input_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels.parquet"
# output_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_freeze_baseline.parquet"


# input_file="data/offline_generations_truthfulqa_qwen17b_non_thinking_repeat8_prelabels.parquet"
# output_file="data/offline_generations_truthfulqa_qwen17b_non_thinking_repeat8_prelabels_labeled.parquet"
# input_file="data/offline_generations_truthfulqa_qwen17b_thinking_repeat8_prelabels.parquet"
# output_file="data/offline_generations_truthfulqa_qwen17b_thinking_repeat8_prelabels_kl_labeled.parquet"
# input_file="data/offline_generations_truthfulqa_qwen17b_thinking_with_kl_repeat8_prelabels.parquet"
# output_file="data/offline_generations_truthfulqa_qwen17b_thinking_with_kl_repeat8_prelabels_kl_labeled.parquet"


# teacher_model="models/zip_joint_distribution_qwen_06b_thinking_math_no_kl_correctness"
# teacher_model="models/zip_joint_distribution_qwen_06b_thinking_math_soft_values_with_kl"
# teacher_model="models/zip_joint_distribution_qwen_06b_thinking_adaptivemath_soft_values_no_kl" # This is now accidentally a 1.7B model since I put the wrong model name in the script
# teacher_model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_no_kl"
# teacher_model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values"
# teacher_model="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values_no_kl"
# teacher_model="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values"
# teacher_model="models/zip_joint_distribution_qwen_17b_thinking_hallucination_soft_values_no_kl"
# teacher_model="models/zip_joint_distribution_qwen_17b_thinking_hallucination_soft_values"
# teacher_model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_3"
# teacher_model="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values_2"
# teacher_model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_topk128_25"
# teacher_model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_topk32_full_25"
# teacher_model="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values_topk32_full_25"
# teacher_model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_no_kl_correctness"
teacher_model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_freeze_baseline"

dp_size=8

distribution_token_id=151669
# num_value_bins=2
num_value_bins=7
value_min=0.0
value_max=1.0
length_bins="0,256,512,1024,2048,4096,8192,16384,32768"

window_size=512        # causal smoothing window
update_interval=64    # how often to emit predictions
dtype="bfloat16"

echo "Starting offline ZIP labeling:"
echo "  Input: $input_file"
echo "  Output: $output_file"
echo "  Teacher: $teacher_model"
echo "  Window size: $window_size | Interval: $update_interval"
echo "  Start time: $(date)"

python3 -u src/zip_offline_label.py \
    --in-parquet "$input_file" \
    --out-parquet "$output_file" \
    --teacher-model "$teacher_model" \
    --distribution-token-id $distribution_token_id \
    --num-value-bins $num_value_bins \
    --value-min $value_min --value-max $value_max \
    --length-bins $length_bins \
    --window-size $window_size \
    --update-interval $update_interval \
    --dtype $dtype \
    --dp-size $dp_size \
    2>&1 | tee -a /home/rohin/ZIP/logs/offline_label_$(basename ${output_file} .parquet).log

exit_code=${PIPESTATUS[0]}
echo "Offline labeling completed with exit code: $exit_code at $(date)"

[ -f "$output_file" ] && echo "Generated file size: $(du -h $output_file | cut -f1)"
[ -f "${output_file%.parquet}_joint_probs.npz" ] && echo "Generated NPZ size: $(du -h ${output_file%.parquet}_joint_probs.npz | cut -f1)"
exit $exit_code