#!/bin/bash
#SBATCH -J judge_ext_vl32b
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=xgpj0
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
# External judge for the VQA runs of the 2B and 8B pairs: Qwen3-VL-32B, same prompt as the in-loop judge.
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
activate_env
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/tmp/kuaiyu_vllm_cache
python src/judge_external.py --task vqa --model Qwen/Qwen3-VL-32B-Instruct --tag qwen3-vl-32b \
  results/fuzz_vqa/qwen3-vl-2b_self/{fixed,inconsistency,full,no_fp,no_tp} \
  results/fuzz_vqa/qwen3-vl-8b_self/{fixed,inconsistency,full,no_fp,no_tp}
