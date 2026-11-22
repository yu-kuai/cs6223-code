#!/bin/bash
#SBATCH -J grade_api
#SBATCH -p long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
# Grade violations of finished text runs with a hosted judge (API). usage: sbatch 33_grade_api.sbatch <model-id> <tag> <run dirs...>
set -uo pipefail
cd /mnt/scratch/k/kuaiyu/cs6223/project
M=$1; T=$2; shift 2
python3 src/judge_api_grade.py --model $M --tag $T "$@"
