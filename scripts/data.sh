#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=zip_data_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/ZIP/logs/data.out
#SBATCH --error=/home/rohin/ZIP/logs/data.err
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

cd $HOME/ZIP
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate
conda activate zip

# Configuration

# model="Qwen/Qwen3-4B"
# dataset="allenai/llama-3.1-tulu-3-8b-preference-mixture"
# prompt_column="prompt"
# output_file="data/zip_training_hallucination_data_qwen4b.parquet"
# thinking_samples=0
# non_thinking_samples=1

# model="Qwen/Qwen3-4B"
# dataset="agentica-org/DeepScaleR-Preview-Dataset"
# prompt_column="problem"
# output_file="data/zip_training_math_data_qwen4b.parquet"
# thinking_samples=0
# non_thinking_samples=2

# model="Qwen/Qwen3-0.6B"
# dataset="allenai/llama-3.1-tulu-3-8b-preference-mixture"
# prompt_column="prompt"
# output_file="data/zip_training_hallucination_data_qwen06b_thinking.parquet"
# thinking_samples=1
# non_thinking_samples=0

# model="Qwen/Qwen3-0.6B"
# dataset="agentica-org/DeepScaleR-Preview-Dataset"
# prompt_column="problem"
# output_file="data/zip_training_math_data_qwen06b_thinking.parquet"
# thinking_samples=2
# non_thinking_samples=0

# model="Qwen/Qwen3-0.6B"
# dataset="rohinm/adaptivemath"
# prompt_column="problem"
# output_file="data/zip_training_adaptivemath_data_qwen06b_thinking.parquet"
# thinking_samples=2
# non_thinking_samples=0

# model="Qwen/Qwen3-1.7B"
# dataset="rohinm/adaptivemath"
# prompt_column="problem"
# output_file="data/zip_training_adaptivemath_data_qwen17b_thinking.parquet"
# thinking_samples=2
# non_thinking_samples=0

model="Qwen/Qwen3-1.7B"
dataset="rohinm/adaptivemath"
prompt_column="problem"
output_file="data/zip_training_adaptivemath_data_qwen17b_non_thinking_32_min_p_001.parquet"
# output_file="data/zip_training_adaptivemath_data_qwen17b_thinking_4_min_p_001.parquet"
thinking_samples=0
non_thinking_samples=32
max_num_seqs=32

# model="Qwen/Qwen3-4B-Instruct-2507"
# dataset="rohinm/adaptivemath"
# prompt_column="problem"
# output_file="data/zip_training_adaptivemath_data_qwen4b_2507_thinking.parquet"
# thinking_samples=0
# non_thinking_samples=2
# max_num_seqs=32

# model="Qwen/Qwen3-1.7B"
# dataset="allenai/llama-3.1-tulu-3-8b-preference-mixture"
# prompt_column="prompt"
# output_file="data/zip_training_hallucination_data_qwen17b_thinking.parquet"
# thinking_samples=1
# non_thinking_samples=0

# model="Qwen/Qwen3-1.7B"
# dataset="allenai/llama-3.1-tulu-3-8b-preference-mixture"
# prompt_column="prompt"
# output_file="data/zip_training_hallucination_data_qwen17b_non_thinking.parquet"
# thinking_samples=0
# non_thinking_samples=1
# max_num_seqs=32

# model="Qwen/Qwen3-4B-Instruct-2507"
# dataset="allenai/llama-3.1-tulu-3-8b-preference-mixture"
# prompt_column="prompt"
# output_file="data/zip_training_hallucination_data_qwen4b_2507_thinking.parquet"
# thinking_samples=0
# non_thinking_samples=1

max_prompts=128_000_000

temperature=1.0
min_p=0.1

echo "Starting ZIP training data generation:"
echo "  Model: $model"
echo "  Output: $output_file"
echo "  Prompts: $max_prompts (${thinking_samples} reasoning + ${non_thinking_samples} non-reasoning per prompt)"
echo "  Start time: $(date)"

python3 -u src/data.py \
    --model "$model" \
    --dataset "$dataset" \
    --prompt-column "$prompt_column" \
    --out "$output_file" \
    --max-num-prompts $max_prompts \
    --thinking-samples $thinking_samples \
    --non-thinking-samples $non_thinking_samples \
    --dp-size 8 --tp-size 1 \
    --allow-partial-merge \
    --temperature $temperature --min-p $min_p \
    --max-model-len 32768 --max-num-seqs $max_num_seqs 2>&1 | tee -a /home/rohin/ZIP/logs/data_$(basename ${output_file} .parquet).log

exit_code=${PIPESTATUS[0]}
echo "Data generation completed with exit code: $exit_code at $(date)"

[ -f "$output_file" ] && echo "Generated file size: $(du -h $output_file | cut -f1)"
exit $exit_code 