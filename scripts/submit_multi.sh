#!/bin/bash
# usage: scripts/submit_multi.sh <conf> [extra sbatch args]   -> job name = conf basename
set -euo pipefail
CONF=$(readlink -f "$1"); shift
GRES=$(awk '$1=="gres"{print $2}' "$CONF")
for v in $(env | grep -oE "^SLURM_[A-Z_]+"); do unset $v; done   # do not leak the env of the job we may be running in
sbatch "$@" --gres="$GRES" -J "m_$(basename "$CONF" .conf)" /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/21_fuzz_multi.sbatch "$CONF"
