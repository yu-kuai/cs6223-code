#!/bin/bash
#SBATCH -J vision_check
#SBATCH -p long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
set -uo pipefail
cd /mnt/scratch/k/kuaiyu/cs6223/project
python3 src/vision_check.py --models x-test-2 x-test-1 x-test-4 --n 40
