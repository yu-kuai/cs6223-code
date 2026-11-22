#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=xgpj0
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
# External judge over every FINISHED run of a task (runs already judged are skipped by judge_external.py).
# usage: sbatch -J judge_all_<task> 32_judge_external_all.sbatch mcq|vqa
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
activate_env
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/tmp/kuaiyu_vllm_cache
if [ "$1" = vqa ]; then M=Qwen/Qwen3-VL-32B-Instruct; T=qwen3-vl-32b; RD=results/fuzz_vqa
else M=Qwen/Qwen3-32B; T=qwen3-32b; RD=results/fuzz; fi
RUNS=()
for d in $RD/*/*/; do d=${d%/}
  [[ $d == *smoke* ]] && continue
  grep -q '"done": true' $d/state.json 2>/dev/null && RUNS+=($d)
done
echo "finished runs: ${#RUNS[@]}"
python src/judge_external.py --task $1 --model $M --tag $T "${RUNS[@]}"
echo JUDGE_ALL_DONE
