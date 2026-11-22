#!/bin/bash
# usage: scripts/submit_fuzz.sh <1.7b|8b|32b|vl-2b|vl-8b|vl-32b> <strategy> <budget> [extra loop args...]
# vl-* = VQA task (ScienceQA images, Qwen3-VL Instruct)
# env RUN_SUFFIX (optional) -> separate run dir, e.g. RUN_SUFFIX=_smoke
set -euo pipefail
SIZE=$1; STRAT=$2; BUDGET=$3; shift 3
case $SIZE in
  1.7b) MODEL=Qwen/Qwen3-1.7B; GRES=gpu:a100-40:1 ;;
  8b)   MODEL=Qwen/Qwen3-8B;   GRES=gpu:a100-80:1 ;;
  32b)  MODEL=Qwen/Qwen3-32B;  GRES=gpu:h100-96:1 ;;
  vl-2b)  MODEL=Qwen/Qwen3-VL-2B-Instruct;  GRES=gpu:a100-40:1; TASK=vqa ;;
  vl-8b)  MODEL=Qwen/Qwen3-VL-8B-Instruct;  GRES=gpu:a100-80:1; TASK=vqa ;;
  vl-32b) MODEL=Qwen/Qwen3-VL-32B-Instruct; GRES=gpu:h100-96:1; TASK=vqa ;;
  *) echo "unknown size $SIZE"; exit 1 ;;
esac
export GRES RUN_SUFFIX=${RUN_SUFFIX:-} TASK=${TASK:-mcq}
sbatch ${SBATCH_EXTRA:-} --gres=$GRES -J fuzz_${SIZE}_${STRAT}${RUN_SUFFIX} --export=ALL \
  /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/20_fuzz.sbatch $MODEL qwen3-${SIZE} $STRAT $BUDGET "$@"
