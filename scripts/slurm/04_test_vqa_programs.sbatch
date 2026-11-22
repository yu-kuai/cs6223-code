#!/bin/bash
#SBATCH -J test_vqa_prog
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
activate_env
python scripts/test_vqa_programs.py "$@"
