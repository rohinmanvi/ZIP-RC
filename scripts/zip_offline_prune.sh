#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=zip_offline_prune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/rohin/ZIP/logs/offline_prune.out
#SBATCH --error=/home/rohin/ZIP/logs/offline_prune.err
#SBATCH --account=liquidai
#SBATCH --exclude=liquid-gpu-[054]

# -----------------------------
# Environment setup
# -----------------------------
cd $HOME/ZIP
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate
conda activate zip
# Use single-threaded math libs to avoid oversubscription/segfaults
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# -----------------------------
# Configuration
# -----------------------------
JOINT_TEMP=${JOINT_TEMP:-1.0}
EV_THRESHOLD=${EV_THRESHOLD:-0.0}   # default expected value threshold for early pruning
SELECTION_GEOM_ALPHA=${SELECTION_GEOM_ALPHA:-1.0}
JOINT_TOP_P=${JOINT_TOP_P:-1.0}     # 1.0 = no change; e.g., 0.95 to enable nucleus filtering

# labeled_file="data/offline_generations_amc2023_qwen06b_repeat32_labeled_prelabels.parquet"
# npz_file="data/offline_generations_amc2023_qwen06b_repeat32_labeled_joint_probs.npz"
# output_file="results/offline_results_amc2023_qwen06b_repeat32.parquet"
# labeled_file="data/offline_generations_amc2023_qwen06b_repeat32_prelabels_labeled.parquet"
# npz_file="data/offline_generations_amc2023_qwen06b_repeat32_prelabels_labeled_joint_probs.npz"
# output_file="results/offline_results_amc2023_qwen06b_repeat32_correctness.parquet"
# labeled_file="data/offline_generations_amc2023_qwen06b_repeat32_prelabels_labeled_soft_values_2.parquet"
# npz_file="data/offline_generations_amc2023_qwen06b_repeat32_prelabels_labeled_soft_values_2_joint_probs.npz"
# output_file="results/offline_results_amc2023_qwen06b_repeat32_soft_values_2.parquet"
# labeled_file="data/offline_generations_math500_qwen06b_thinking_repeat8_prelabels_labeled_soft_values_2.parquet"
# npz_file="data/offline_generations_math500_qwen06b_thinking_repeat8_prelabels_labeled_soft_values_2_joint_probs.npz"
# output_file="results/offline_results_math500_qwen06b_thinking_repeat8_soft_values_2.parquet"
# labeled_file="data/offline_generations_gsm8k_qwen06b_thinking_repeat4_prelabels_labeled_soft_values_2.parquet"
# npz_file="data/offline_generations_gsm8k_qwen06b_thinking_repeat4_prelabels_labeled_soft_values_2_joint_probs.npz"
# output_file="results/offline_results_gsm8k_qwen06b_thinking_repeat4_soft_values_2.parquet"
# labeled_file="data/offline_generations_gsm8k_qwen06b_thinking_with_kl_repeat4_prelabels_labeled_with_kl.parquet"
# npz_file="data/offline_generations_gsm8k_qwen06b_thinking_with_kl_repeat4_prelabels_labeled_with_kl_joint_probs.npz"
# output_file="results/offline_results_gsm8k_qwen06b_thinking_with_kl_repeat4_soft_values_with_kl.parquet"
# labeled_file="data/offline_generations_amc2023_qwen06b_thinking_06_repeat32_prelabels_labeled_soft_values.parquet"
# npz_file="data/offline_generations_amc2023_qwen06b_thinking_06_repeat32_prelabels_labeled_soft_values_joint_probs.npz"
# output_file="results/offline_results_amc2023_qwen06b_thinking_06_repeat32_soft_values.parquet"

# labeled_file="data/offline_generations_amc2023_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_amc2023_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/offline_results_amc2023_qwen06b_thinking_repeat32_adaptivemath.parquet"

# labeled_file="data/offline_generations_aime2024_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_aime2024_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/offline_results_aime2024_qwen06b_thinking_repeat32_adaptivemath.parquet"

# labeled_file="data/offline_generations_math500_qwen06b_thinking_repeat8_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_math500_qwen06b_thinking_repeat8_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/offline_results_math500_qwen06b_thinking_repeat8_adaptivemath.parquet"

# labeled_file="data/offline_generations_gsm8k_qwen06b_thinking_repeat4_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_gsm8k_qwen06b_thinking_repeat4_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/offline_results_gsm8k_qwen06b_thinking_repeat4_adaptivemath.parquet"



# labeled_file="data/offline_generations_amc2023_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_amc2023_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/z_offline_results_amc2023_qwen17b_thinking_repeat32_adaptivemath.parquet"

# labeled_file="data/offline_generations_aime2024_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_aime2024_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/z_offline_results_aime2024_qwen17b_thinking_repeat32_adaptivemath.parquet"

# labeled_file="data/offline_generations_math500_qwen17b_thinking_repeat8_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_math500_qwen17b_thinking_repeat8_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/z_offline_results_math500_qwen17b_thinking_repeat8_adaptivemath.parquet"

# labeled_file="data/offline_generations_gsm8k_qwen17b_thinking_repeat4_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_gsm8k_qwen17b_thinking_repeat4_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/z_offline_results_gsm8k_qwen17b_thinking_repeat4_adaptivemath.parquet"




# labeled_file="data/offline_generations_truthfulqa_qwen17b_thinking_repeat8_prelabels_labeled.parquet"
# npz_file="data/offline_generations_truthfulqa_qwen17b_thinking_repeat8_prelabels_labeled_joint_probs.npz"
# output_file="results/offline_results_truthfulqa_qwen17b_thinking_repeat8.parquet"


# labeled_file="data/offline_generations_aime2024_qwen17b_non_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_aime2024_qwen17b_non_thinking_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/offline_results_aime2024_qwen17b_non_thinking_repeat32_adaptivemath.parquet"

# labeled_file="data/offline_generations_amc2023_qwen17b_non_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_amc2023_qwen17b_non_thinking_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/offline_results_amc2023_qwen17b_non_thinking_repeat32_adaptivemath.parquet"

# labeled_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_repeat32_prelabels_adaptivemath_labeled.parquet"
# npz_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/offline_results_aime2024_qwen17b_thinking_with_kl_repeat32_adaptivemath.parquet"





thinking_labeled_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_no_kl.parquet"
thinking_npz_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_no_kl_joint_probs.npz"
output_file="results/B_offline_results_gsm8k_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat4_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_no_kl.parquet"
thinking_npz_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_no_kl_joint_probs.npz"
output_file="results/B_offline_results_math500_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat8_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_no_kl.parquet"
thinking_npz_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_no_kl_joint_probs.npz"
output_file="results/B_offline_results_amc2023_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_no_kl.parquet"
thinking_npz_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_no_kl_joint_probs.npz"
output_file="results/B_offline_results_aime2024_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"



thinking_labeled_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled.parquet"
non_thinking_labeled_file="data/offline_generations_gsm8k_qwen17b_non_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_joint_probs.npz"
non_thinking_npz_file="data/offline_generations_gsm8k_qwen17b_non_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/A_offline_results_gsm8k_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat4_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled.parquet"
non_thinking_labeled_file="data/offline_generations_math500_qwen17b_non_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_joint_probs.npz"
non_thinking_npz_file="data/offline_generations_math500_qwen17b_non_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/A_offline_results_math500_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat8_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
non_thinking_labeled_file="data/offline_generations_amc2023_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
non_thinking_npz_file="data/offline_generations_amc2023_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/A_offline_results_amc2023_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk64_repeat32_prelabels_adaptivemath_labeled.parquet"
non_thinking_labeled_file="data/offline_generations_aime2024_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
non_thinking_npz_file="data/offline_generations_aime2024_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/A_offline_results_aime2024_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"



thinking_labeled_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/G_offline_results_gsm8k_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat4_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/G_offline_results_math500_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat8_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/G_offline_results_amc2023_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/G_offline_results_aime2024_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"



thinking_labeled_file="data/offline_generations_gsm8k_qwen06b_thinking_repeat4_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_gsm8k_qwen06b_thinking_repeat4_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/H_offline_results_gsm8k_qwen06b_thinking_and_non_thinking_repeat4_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_math500_qwen06b_thinking_repeat8_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_math500_qwen06b_thinking_repeat8_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/H_offline_results_math500_qwen06b_thinking_and_non_thinking_repeat8_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_amc2023_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_amc2023_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/H_offline_results_amc2023_qwen06b_thinking_and_non_thinking_repeat32_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_aime2024_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_aime2024_qwen06b_thinking_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/H_offline_results_aime2024_qwen06b_thinking_and_non_thinking_repeat32_adaptivemath.parquet"



thinking_labeled_file="data/offline_generations_gsm8k_qwen17b_non_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_gsm8k_qwen17b_non_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/D_offline_results_gsm8k_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat4_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_math500_qwen17b_non_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_math500_qwen17b_non_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/D_offline_results_math500_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat8_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_amc2023_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_amc2023_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/D_offline_results_amc2023_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_aime2024_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_aime2024_qwen17b_non_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/D_offline_results_aime2024_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"



thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_gsm8k_lfm1.2b_thinking_adaptivemath_repeat4_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_gsm8k_lfm1.2b_thinking_adaptivemath_repeat4_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/E_offline_results_gsm8k_lfm1.2b_thinking_and_non_thinking_adaptivemath_repeat4.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_math500_lfm1.2b_thinking_adaptivemath_repeat8_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_math500_lfm1.2b_thinking_adaptivemath_repeat8_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/E_offline_results_math500_lfm1.2b_thinking_and_non_thinking_adaptivemath_repeat8.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_amc2023_lfm1.2b_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_amc2023_lfm1.2b_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/E_offline_results_amc2023_lfm1.2b_thinking_and_non_thinking_adaptivemath_repeat32.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_aime2024_lfm1.2b_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_aime2024_lfm1.2b_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/E_offline_results_aime2024_lfm1.2b_thinking_and_non_thinking_adaptivemath_repeat32.parquet"



thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_gsm8k_lfm350m_thinking_adaptivemath_repeat4_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_gsm8k_lfm350m_thinking_adaptivemath_repeat4_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/F_offline_results_gsm8k_lfm350m_thinking_and_non_thinking_adaptivemath_repeat4.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_math500_lfm350m_thinking_adaptivemath_repeat8_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_math500_lfm350m_thinking_adaptivemath_repeat8_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/F_offline_results_math500_lfm350m_thinking_and_non_thinking_adaptivemath_repeat8.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_amc2023_lfm350m_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_amc2023_lfm350m_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/F_offline_results_amc2023_lfm350m_thinking_and_non_thinking_adaptivemath_repeat32.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_aime2024_lfm350m_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_aime2024_lfm350m_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/F_offline_results_aime2024_lfm350m_thinking_and_non_thinking_adaptivemath_repeat32.parquet"



thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_hotpotqad_lfm2.6b_nonthinking_repeat1_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_hotpotqad_lfm2.6b_nonthinking_repeat1_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/I_offline_results_hotpotqad_lfm2.6b_thinking_and_non_thinking_adaptivemath_repeat1.parquet"










thinking_labeled_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/J_offline_results_gsm8k_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat4_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/J_offline_results_math500_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat8_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/J_offline_results_amc2023_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"

thinking_labeled_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
thinking_npz_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
output_file="results/J_offline_results_aime2024_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"


thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_gsm8k_lfm1.2b_thinking_adaptivemath_repeat4_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_gsm8k_lfm1.2b_thinking_adaptivemath_repeat4_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/K_offline_results_gsm8k_lfm1.2b_thinking_and_non_thinking_adaptivemath_repeat4.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_math500_lfm1.2b_thinking_adaptivemath_repeat8_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_math500_lfm1.2b_thinking_adaptivemath_repeat8_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/K_offline_results_math500_lfm1.2b_thinking_and_non_thinking_adaptivemath_repeat8.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_amc2023_lfm1.2b_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_amc2023_lfm1.2b_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/K_offline_results_amc2023_lfm1.2b_thinking_and_non_thinking_adaptivemath_repeat32.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_aime2024_lfm1.2b_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_aime2024_lfm1.2b_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/K_offline_results_aime2024_lfm1.2b_thinking_and_non_thinking_adaptivemath_repeat32.parquet"


thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_gsm8k_lfm350m_thinking_adaptivemath_repeat4_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_gsm8k_lfm350m_thinking_adaptivemath_repeat4_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/L_offline_results_gsm8k_lfm350m_thinking_and_non_thinking_adaptivemath_repeat4.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_math500_lfm350m_thinking_adaptivemath_repeat8_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_math500_lfm350m_thinking_adaptivemath_repeat8_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/L_offline_results_math500_lfm350m_thinking_and_non_thinking_adaptivemath_repeat8.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_amc2023_lfm350m_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_amc2023_lfm350m_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/L_offline_results_amc2023_lfm350m_thinking_and_non_thinking_adaptivemath_repeat32.parquet"

thinking_labeled_file="/home/tim/workspace/ZIP/data/offline_generations_aime2024_lfm350m_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values.parquet"
thinking_npz_file="/home/tim/workspace/ZIP/data/offline_generations_aime2024_lfm350m_thinking_adaptivemath_repeat32_prelabels_labeled_soft_values_joint_probs.npz"
output_file="results/L_offline_results_aime2024_lfm350m_thinking_and_non_thinking_adaptivemath_repeat32.parquet"



thinking_labeled_file="data/offline_generations_gsm8k_qwen17b_thinking_repeat4_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
thinking_npz_file="data/offline_generations_gsm8k_qwen17b_thinking_repeat4_prelabels_adaptivemath_labeled_freeze_baseline_joint_probs.npz"
output_file="results/ZB_offline_results_gsm8k_qwen17b_thinking_repeat4_adaptivemath_freeze_baseline.parquet"

thinking_labeled_file="data/offline_generations_math500_qwen17b_thinking_repeat8_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
thinking_npz_file="data/offline_generations_math500_qwen17b_thinking_repeat8_prelabels_adaptivemath_labeled_freeze_baseline_joint_probs.npz"
output_file="results/ZB_offline_results_math500_qwen17b_thinking_repeat8_adaptivemath_freeze_baseline.parquet"

thinking_labeled_file="data/offline_generations_amc2023_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
thinking_npz_file="data/offline_generations_amc2023_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled_freeze_baseline_joint_probs.npz"
output_file="results/ZB_offline_results_amc2023_qwen17b_thinking_repeat32_adaptivemath_freeze_baseline.parquet"

thinking_labeled_file="data/offline_generations_aime2024_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
thinking_npz_file="data/offline_generations_aime2024_qwen17b_thinking_repeat32_prelabels_adaptivemath_labeled_freeze_baseline_joint_probs.npz"
output_file="results/ZB_offline_results_aime2024_qwen17b_thinking_repeat32_adaptivemath_freeze_baseline.parquet"


thinking_labeled_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
thinking_npz_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_freeze_baseline_joint_probs.npz"
output_file="results/ZB2_offline_results_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_adaptivemath_freeze_baseline.parquet"

thinking_labeled_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
thinking_npz_file="data/offline_generations_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_prelabels_adaptivemath_labeled_freeze_baseline_joint_probs.npz"
output_file="results/ZB2_offline_results_math500_qwen17b_thinking_with_kl_topk32_full_25_repeat8_adaptivemath_freeze_baseline.parquet"

thinking_labeled_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
thinking_npz_file="data/offline_generations_amc2023_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_freeze_baseline_joint_probs.npz"
output_file="results/ZB2_offline_results_amc2023_qwen17b_thinking_and_with_kl_topk32_full_25_repeat32_adaptivemath_freeze_baseline.parquet"

thinking_labeled_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_freeze_baseline.parquet"
thinking_npz_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_freeze_baseline_joint_probs.npz"
output_file="results/ZB2_offline_results_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_adaptivemath_freeze_baseline.parquet"


# thinking_labeled_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled.parquet"
# thinking_npz_file="data/offline_generations_aime2024_qwen17b_thinking_with_kl_topk32_full_25_repeat32_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/G_offline_results_aime2024_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat32_adaptivemath.parquet"

# thinking_labeled_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled.parquet"
# thinking_npz_file="data/offline_generations_gsm8k_qwen17b_thinking_with_kl_topk32_full_25_repeat4_prelabels_adaptivemath_labeled_joint_probs.npz"
# output_file="results/J_offline_results_gsm8k_qwen17b_thinking_and_non_thinking_with_kl_topk32_full_25_repeat4_adaptivemath.parquet"




decode_model="Qwen/Qwen3-1.7B"
eval_mode="policy"
# eval_mode="no_prune"
no_prune_try_size=2

# c_s=0.0
# c_l=0.0
# c_s=0.005
# c_l=0.1
# c_s=0.002
# c_l=0.05
# c_s=0.001
# c_l=0.02

c_s=0.01
c_l=0.0
# c_s=0.005
# c_l=0.0

# c_s=0.005
# c_l=0.1
# c_s=0.001
# c_l=0.1
# c_s=0.001
# c_l=0.05s

# c_s=0.005
# c_l=0.05
# c_s=0.005
# c_l=0.02
# c_s=0.005
# c_l=0.01
# c_s=0.005
# c_l=0.0

STEP_SIZE=${STEP_SIZE:-64}

echo "Starting offline ZIP pruning simulation (CPU-only):"
echo "  Inputs (labeled parquet):"
echo "    - $thinking_labeled_file"
echo "    - $non_thinking_labeled_file"
echo "  Inputs (joint NPZ):"
echo "    - $thinking_npz_file"
echo "    - $non_thinking_npz_file"
echo "  Output results:          $output_file"
echo "  c_s: $c_s | c_l: $c_l | step_size: $STEP_SIZE"
echo "  EV threshold:         $EV_THRESHOLD"
echo "  Selection geom alpha: $SELECTION_GEOM_ALPHA"
echo "  Joint top-p:          $JOINT_TOP_P (temp=$JOINT_TEMP)"
echo "  Start time: $(date)"

# python3 -u src/zip_offline_prune_4.py \
#     --labeled-parquet "$thinking_labeled_file" "$non_thinking_labeled_file" \
#     --joint-npz "$thinking_npz_file" "$non_thinking_npz_file" \
#     --mode-tags thinking nonthinking \
#     --out "$output_file" \
#     --c-pt $c_s \
#     --c-seq $c_l \
#     --step-size $STEP_SIZE \
#     --eval-mode $eval_mode \
#     --no-prune-try-size $no_prune_try_size \
#     --drop-incomplete-tries \
#     --adaptive-tries \
#     --ev-threshold $EV_THRESHOLD \
#     --sample-limit 1000000 \
#     --joint-temp $JOINT_TEMP \
#     --joint-top-p $JOINT_TOP_P \
#     --selection-geom-alpha $SELECTION_GEOM_ALPHA \
#     --log-interval 1 \
#     --trace-jsonl /home/rohin/ZIP/logs/offline_prune_trace.jsonl \
#     --decode-model "$decode_model" \
#     --compute-metrics --use-consistency \
#     --jit --faulthandler-timeout 2000 --mmap \
#     --zip-logs --zip-logs-groups 1000 --zip-logs-sample-idx -1 \
#     --viz-max --viz-pruning --debug \
#     2>&1 | tee -a /home/rohin/ZIP/logs/offline_prune_$(basename ${output_file} .parquet).log

# python3 -u src/zip_offline_prune_4.py \
#     --labeled-parquet "$thinking_labeled_file" \
#     --joint-npz "$thinking_npz_file" \
#     --mode-tags thinking \
#     --out "$output_file" \
#     --c-pt $c_s \
#     --c-seq $c_l \
#     --step-size $STEP_SIZE \
#     --eval-mode $eval_mode \
#     --no-prune-try-size $no_prune_try_size \
#     --drop-incomplete-tries \
#     --adaptive-tries \
#     --ev-threshold $EV_THRESHOLD \
#     --sample-limit 1000000 \
#     --joint-temp $JOINT_TEMP \
#     --joint-top-p $JOINT_TOP_P \
#     --selection-geom-alpha $SELECTION_GEOM_ALPHA \
#     --log-interval 1 \
#     --trace-jsonl /home/rohin/ZIP/logs/offline_prune_trace.jsonl \
#     --decode-model "$decode_model" \
#     --compute-metrics --use-consistency \
#     --jit --faulthandler-timeout 2000 --mmap \
#     --zip-logs --zip-logs-groups 1000 --zip-logs-sample-idx -1 \
#     --viz-max --viz-pruning --debug \
#     2>&1 | tee -a /home/rohin/ZIP/logs/offline_prune_$(basename ${output_file} .parquet).log

# python3 -u src/zip_offline_prune_4.py \
#     --labeled-parquet "$thinking_labeled_file" \
#     --joint-npz "$thinking_npz_file" \
#     --mode-tags thinking \
#     --out "$output_file" \
#     --c-pt $c_s \
#     --c-seq $c_l \
#     --step-size $STEP_SIZE \
#     --eval-mode $eval_mode \
#     --no-prune-try-size $no_prune_try_size \
#     --drop-incomplete-tries \
#     --adaptive-tries \
#     --ev-threshold $EV_THRESHOLD \
#     --sample-limit 1000000 \
#     --joint-temp $JOINT_TEMP \
#     --joint-top-p $JOINT_TOP_P \
#     --selection-geom-alpha $SELECTION_GEOM_ALPHA \
#     --log-interval 1 \
#     --trace-jsonl /home/rohin/ZIP/logs/offline_prune_trace.jsonl \
#     --decode-model "$decode_model" \
#     --compute-metrics --use-consistency \
#     --jit --faulthandler-timeout 2000 --mmap \
#     2>&1 | tee -a /home/rohin/ZIP/logs/offline_prune_$(basename ${output_file} .parquet).log

# python3 -u src/zip_offline_prune_7.py \
#     --labeled-parquet "$thinking_labeled_file" \
#     --joint-npz "$thinking_npz_file" \
#     --mode-tags thinking \
#     --out "$output_file" \
#     --c-pt $c_s \
#     --c-seq $c_l \
#     --step-size $STEP_SIZE \
#     --k-max 8 \
#     --max-active 1 \
#     --switch-penalty 0.0 \
#     --sample-limit 1000000 \
#     --log-interval 1 \
#     --trace-jsonl /home/rohin/ZIP/logs/offline_prune_trace.jsonl \
#     --decode-model "$decode_model" \
#     --compute-metrics --use-consistency \
#     --jit --faulthandler-timeout 2000 --mmap \
#     --zip-logs --zip-logs-groups 1000 --zip-logs-sample-idx -1 \
#     --viz-max --viz-pruning --debug \
#     2>&1 | tee -a /home/rohin/ZIP/logs/offline_prune_$(basename ${output_file} .parquet).log

python3 -u src/zip_offline_prune_7.py \
    --labeled-parquet "$thinking_labeled_file" \
    --joint-npz "$thinking_npz_file" \
    --mode-tags thinking \
    --out "$output_file" \
    --c-pt $c_s \
    --c-seq $c_l \
    --step-size $STEP_SIZE \
    --k-max 8 \
    --max-active 8 \
    --switch-penalty 0.0 \
    --sample-limit 1000000 \
    --log-interval 1 \
    --trace-jsonl /home/rohin/ZIP/logs/offline_prune_trace.jsonl \
    --decode-model "$decode_model" \
    --compute-metrics --use-consistency \
    --jit --faulthandler-timeout 2000 --mmap \
    2>&1 | tee -a /home/rohin/ZIP/logs/offline_prune_$(basename ${output_file} .parquet).log

# python3 -u src/zip_offline_prune_4.py \
#     --labeled-parquet "$thinking_labeled_file" "$non_thinking_labeled_file" \
#     --joint-npz "$thinking_npz_file" "$non_thinking_npz_file" \
#     --mode-tags thinking nonthinking \
#     --out "$output_file" \
#     --c-pt $c_s \
#     --c-seq $c_l \
#     --step-size $STEP_SIZE \
#     --eval-mode $eval_mode \
#     --no-prune-try-size $no_prune_try_size \
#     --drop-incomplete-tries \
#     --adaptive-tries \
#     --ev-threshold $EV_THRESHOLD \
#     --sample-limit 1000000 \
#     --joint-temp $JOINT_TEMP \
#     --selection-geom-alpha $SELECTION_GEOM_ALPHA \
#     --log-interval 1 \
#     --trace-jsonl /home/rohin/ZIP/logs/offline_prune_trace.jsonl \
#     --decode-model "$decode_model" \
#     --compute-metrics --use-consistency \
#     --jit --faulthandler-timeout 2000 --mmap \
#     2>&1 | tee -a /home/rohin/ZIP/logs/offline_prune_$(basename ${output_file} .parquet).log

# python3 -u src/zip_offline_prune_4.py \
#     --labeled-parquet "$thinking_labeled_file" \
#     --joint-npz "$thinking_npz_file" \
#     --mode-tags thinking \
#     --out "$output_file" \
#     --eval-mode $eval_mode \
#     --no-prune-try-size $no_prune_try_size \
#     --drop-incomplete-tries \
#     --sample-limit 1000000 \
#     --log-interval 1 \
#     --trace-jsonl /home/rohin/ZIP/logs/offline_prune_trace.jsonl \
#     --decode-model "$decode_model" \
#     --compute-metrics --use-consistency \
#     --jit --faulthandler-timeout 2000 --mmap \
#     2>&1 | tee -a /home/rohin/ZIP/logs/offline_prune_$(basename ${output_file} .parquet).log

# python3 -u src/zip_offline_prune_4.py \
#     --labeled-parquet "$non_thinking_labeled_file" \
#     --joint-npz "$non_thinking_npz_file" \
#     --mode-tags nonthinking \
#     --out "$output_file" \
#     --eval-mode $eval_mode \
#     --no-prune-try-size $no_prune_try_size \
#     --drop-incomplete-tries \
#     --sample-limit 1000000 \
#     --log-interval 1 \
#     --trace-jsonl /home/rohin/ZIP/logs/offline_prune_trace.jsonl \
#     --decode-model "$decode_model" \
#     --compute-metrics --use-consistency \
#     --jit --faulthandler-timeout 2000 --mmap \
#     2>&1 | tee -a /home/rohin/ZIP/logs/offline_prune_$(basename ${output_file} .parquet).log

exit_code=${PIPESTATUS[0]}
echo "Offline pruning simulation completed with exit code: $exit_code at $(date)"

[ -f "$output_file" ] && echo "Generated results file size: $(du -h $output_file | cut -f1)"
# Metrics JSON is automatically saved next to --out by the Python script
exit $exit_code