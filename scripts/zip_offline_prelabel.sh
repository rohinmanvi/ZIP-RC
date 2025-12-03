#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=zip_offline_prelabel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/ZIP/logs/offline_prelabel.out
#SBATCH --error=/home/rohin/ZIP/logs/offline_prelabel.err
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
export VLLM_USE_V1=0

# -----------------------------
# Environment setup
# -----------------------------
cd $HOME/ZIP
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate
conda activate zip
mkdir -p /home/rohin/ZIP/logs

# -----------------------------
# Configuration
# -----------------------------

# INPUT: parquet produced by zip_offline_generate.sh
# Adjust if you switch model/benchmark
# input_file="data/offline_generations_gsm8k_qwen06b_thinking_with_kl_repeat4.parquet"

input_file="data/offline_generations_aime2024_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32.parquet"
# input_file="data/offline_generations_amc2023_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32.parquet"
# input_file="data/offline_generations_math500_qwen17b_non_thinking_with_kl_topk32_full_25_repeat8.parquet"
# input_file="data/offline_generations_gsm8k_qwen17b_non_thinking_with_kl_topk32_full_25_repeat4.parquet"

# input_file="data/offline_generations_truthfulqa_qwen17b_non_thinking_repeat8.parquet"
# input_file="data/offline_generations_truthfulqa_qwen17b_thinking_repeat8.parquet"
# input_file="data/offline_generations_truthfulqa_qwen17b_thinking_with_kl_repeat8.parquet"




# OUTPUT: prelabel parquet (adds `correct` + `extracted_answer`)
prelabel_file="${input_file%.parquet}_prelabels.parquet"

# Grader model (same one you use in label_and_evaluate.py by default)
grader_model="Qwen/Qwen3-235B-A22B-Instruct-2507"

# vLLM runtime knobs (match your evaluation defaults)
tp_size=8
gpu_mem_util=0.90
max_model_len=32768
max_num_seqs=8
enforce_eager_flag=""   # set to "--enforce-eager" if you need eager mode

# Task: "correctness" (with gold answers) or "hallucination" (no gold)
task="correctness"

echo "Starting offline prelabeling:"
echo "  Input parquet:     $input_file"
echo "  Output parquet:    $prelabel_file"
echo "  Grader model:      $grader_model"
echo "  TP size / max_seqs: ${tp_size} / ${max_num_seqs}"
echo "  Max model len:     ${max_model_len}"
echo "  GPU mem util:      ${gpu_mem_util}"
echo "  Task:              ${task}"
echo "  Start time:        $(date)"

if [ ! -f "$input_file" ]; then
  echo "ERROR: input parquet not found: $input_file"
  exit 2
fi

python3 -u src/prelabel_dataset.py \
  --in "$input_file" \
  --out "$prelabel_file" \
  --model "$grader_model" \
  --task "$task" \
  --tensor-parallel-size $tp_size \
  --gpu-memory-utilization $gpu_mem_util \
  --max-model-len $max_model_len \
  --max-num-seqs $max_num_seqs \
  $enforce_eager_flag \
  2>&1 | tee -a /home/rohin/ZIP/logs/offline_prelabel_$(basename ${prelabel_file} .parquet).log

exit_code=${PIPESTATUS[0]}
echo "Offline prelabeling completed with exit code: $exit_code at $(date)"

if [ -f "$prelabel_file" ]; then
  echo "Prelabeled file size: $(du -h "$prelabel_file" | cut -f1)"
else
  echo "ERROR: expected output not found: $prelabel_file"
fi

exit $exit_code