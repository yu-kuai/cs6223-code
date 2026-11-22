#!/bin/bash
#SBATCH -J build_audit
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
# usage: sbatch 40_build_audit.sbatch [build_audit.py args]
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
activate_env
python src/build_audit.py "$@"
echo AUDIT_DONE
