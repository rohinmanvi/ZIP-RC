#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=zip_offline_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/ZIP/logs/offline_gen.out
#SBATCH --error=/home/rohin/ZIP/logs/offline_gen.err
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

# Limit vLLM per-worker concurrent sequences to avoid OOM; override by exporting before submit
export ZIP_MAX_CONCURRENT="${ZIP_MAX_CONCURRENT:-32}"

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

# model="Qwen/Qwen3-0.6B"
# benchmark="amc2023"
# output_file="data/offline_generations_amc2023_qwen06b_thinking_repeat32.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=32

# model="Qwen/Qwen3-0.6B"
# benchmark="aime2024"
# output_file="data/offline_generations_aime2024_qwen06b_thinking_repeat32.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=32

# model="Qwen/Qwen3-0.6B"
# benchmark="math500"
# output_file="data/offline_generations_math500_qwen06b_thinking_repeat8.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=8

# model="Qwen/Qwen3-0.6B"
# benchmark="gsm8k"
# output_file="data/offline_generations_gsm8k_qwen06b_thinking_repeat4.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=4

# model="Qwen/Qwen3-1.7B"
# benchmark="amc2023"
# output_file="data/offline_generations_amc2023_qwen17b_thinking_repeat32.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=32

# model="Qwen/Qwen3-1.7B"
# benchmark="aime2024"
# output_file="data/offline_generations_aime2024_qwen17b_thinking_repeat32.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=32

# model="Qwen/Qwen3-1.7B"
# benchmark="math500"
# output_file="data/offline_generations_math500_qwen17b_thinking_repeat8.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=8

# model="Qwen/Qwen3-1.7B"
# benchmark="gsm8k"
# output_file="data/offline_generations_gsm8k_qwen17b_thinking_repeat4.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=4

# model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_topk32_full_25"
# benchmark="amc2023"
# output_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=32

# model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_topk32_full_25"
# benchmark="aime2024"
# output_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=32

# model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_topk32_full_25"
# benchmark="math500"
# output_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=8

# model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values_topk32_full_25"
# benchmark="gsm8k"
# output_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=4

# model="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values_topk32_full_25"
# benchmark="amc2023"
# output_file="data/offline_generations_amc2023_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32.parquet"
# num_thinking_samples=0
# num_nonthinking_samples=8
# repeat_factor=32

# model="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values_topk32_full_25"
# benchmark="aime2024"
# output_file="data/offline_generations_aime2024_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32.parquet"
# num_thinking_samples=0
# num_nonthinking_samples=8
# repeat_factor=32

# model="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values_topk32_full_25"
# benchmark="math500"
# output_file="data/offline_generations_math500_qwen17b_non_thinking_with_kl_topk32_full_25_repeat8.parquet"
# num_thinking_samples=0
# num_nonthinking_samples=8
# repeat_factor=8

model="models/zip_joint_distribution_qwen_17b_non_thinking_adaptivemath_soft_values_topk32_full_25"
benchmark="gsm8k"
output_file="data/offline_generations_gsm8k_qwen17b_non_thinking_with_kl_topk32_full_25_repeat4.parquet"
num_thinking_samples=0
num_nonthinking_samples=8
repeat_factor=4

# model="models/zip_joint_distribution_qwen_06b_thinking_math_soft_values_with_kl"
# benchmark="amc2023"
# output_file="data/offline_generations_amc2023_qwen06b_thinking_with_kl_repeat32.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=32

# model="models/zip_joint_distribution_qwen_06b_thinking_math_soft_values_with_kl"
# benchmark="aime2024"
# output_file="data/offline_generations_aime2024_qwen06b_thinking_with_kl_repeat32.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=32

# model="models/zip_joint_distribution_qwen_06b_thinking_math_soft_values_with_kl"
# benchmark="math500"
# output_file="data/offline_generations_math500_qwen06b_thinking_with_kl_repeat8.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=8

# model="models/zip_joint_distribution_qwen_06b_thinking_math_soft_values_with_kl"
# benchmark="gsm8k"
# output_file="data/offline_generations_gsm8k_qwen06b_thinking_with_kl_repeat4.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=4

# model="Qwen/Qwen3-4B"
# benchmark="amc2023"
# output_file="data/offline_generations_amc2023_qwen4b_06_repeat32.parquet"
# num_thinking_samples=0
# num_nonthinking_samples=8


# model="Qwen/Qwen3-1.7B"
# benchmark="truthfulqa"
# output_file="data/offline_generations_truthfulqa_qwen17b_thinking_repeat8.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=8

# model="models/zip_joint_distribution_qwen_17b_thinking_adaptivemath_soft_values"
# benchmark="truthfulqa"
# output_file="data/offline_generations_truthfulqa_qwen17b_thinking_with_kl_repeat8.parquet"
# num_thinking_samples=8
# num_nonthinking_samples=0
# repeat_factor=8

# model="Qwen/Qwen3-1.7B"
# benchmark="truthfulqa"
# output_file="data/offline_generations_truthfulqa_qwen17b_non_thinking_repeat8.parquet"
# num_thinking_samples=0
# num_nonthinking_samples=8
# repeat_factor=8


max_prompts=1000000   # adjust as needed

temperature=1.0
top_p=0.95
min_p=0.1
top_k=50
# top_k=8

echo "Starting offline ZIP generation:"
echo "  Model: $model"
echo "  Benchmark: $benchmark"
echo "  Output: $output_file"
echo "  Prompts: $max_prompts (x${repeat_factor} repeats, ${num_thinking_samples}+${num_nonthinking_samples} samples per repeat)"
echo "  Start time: $(date)"

python3 -u src/zip_offline_generate.py \
    --model "$model" \
    --benchmark "$benchmark" \
    --out "$output_file" \
    --max-num-prompts $max_prompts \
    --repeat-factor $repeat_factor \
    --num-thinking-samples $num_thinking_samples \
    --num-nonthinking-samples $num_nonthinking_samples \
    --dp-size 8 --tp-size 1 \
    --temperature $temperature --min-p $min_p --top-p $top_p --top-k $top_k \
    --max-model-len 32768 --max-tokens 32768 \
    2>&1 | tee -a /home/rohin/ZIP/logs/offline_gen_$(basename ${output_file} .parquet).log

exit_code=${PIPESTATUS[0]}
echo "Offline generation completed with exit code: $exit_code at $(date)"

[ -f "$output_file" ] && echo "Generated file size: $(du -h $output_file | cut -f1)"
exit $exit_code