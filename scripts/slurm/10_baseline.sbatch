#!/bin/bash
#SBATCH -J baseline
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclude=xgpj0
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
# usage: sbatch --gres=gpu:<type>:1 -J baseline_<tag> 10_baseline.sbatch <hf_model_id> <tag>
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
MODEL=$1; TAG=$2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
activate_env
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/tmp/kuaiyu_vllm_cache
python src/run_baseline.py --model "$MODEL" --out results/baseline/$TAG.jsonl --k 5
python src/summarize_baseline.py results/baseline/$TAG.jsonl
echo BASELINE_DONE
