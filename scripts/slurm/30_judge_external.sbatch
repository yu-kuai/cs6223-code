#!/bin/bash
#SBATCH -J judge_ext32b
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=xgpj0
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
activate_env
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/tmp/kuaiyu_vllm_cache
python src/judge_external.py --model Qwen/Qwen3-32B --tag qwen3-32b \
  results/fuzz/qwen3-1.7b_self/{fixed,inconsistency,full,no_fp,no_tp} \
  results/fuzz/qwen3-8b_self/{fixed,inconsistency,full,no_fp,no_tp}
