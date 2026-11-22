#!/bin/bash
#SBATCH -J download
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
activate_env
python scripts/prepare_mmlu.py --out data/mmlu_seeds.jsonl --per-subject 20 --seed 0
for m in Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B; do
  echo "== $m"; hf download "$m" || huggingface-cli download "$m"
done
du -sh "$HF_HOME"
echo DOWNLOAD_DONE
