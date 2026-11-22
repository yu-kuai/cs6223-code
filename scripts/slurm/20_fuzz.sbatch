#!/bin/bash
#SBATCH -J fuzz
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=xgpj0
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
# Self-testing run: one vLLM server plays fuzzer, target and judge.
# usage: sbatch --gres=gpu:<type>:1 -J fuzz_<tag>_<strategy> 20_fuzz.sbatch <hf_model> <tag> <strategy> <budget> [extra loop args...]
# Submit via scripts/submit_fuzz.sh (sets GRES). Resubmits itself (same gres/name) if the loop stops at the deadline before the budget is used up.
set -uo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
MODEL=$1; TAG=$2; STRAT=$3; BUDGET=$4; shift 4; EXTRA="$*"
TASK=${TASK:-mcq}   # mcq (MMLU text) | vqa (ScienceQA images, Qwen3-VL)
if [ "$TASK" = vqa ]; then
  RUN=$PROJ/results/fuzz_vqa/${TAG}_self/$STRAT${RUN_SUFFIX:-}; BASE=results/baseline_vqa/$TAG.jsonl
  SERVE_EXTRA=(--limit-mm-per-prompt '{"image": 2}')
else
  RUN=$PROJ/results/fuzz/${TAG}_self/$STRAT${RUN_SUFFIX:-}; BASE=results/baseline/$TAG.jsonl; SERVE_EXTRA=()
fi
mkdir -p "$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
activate_env
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/tmp/kuaiyu_vllm_cache
PORT=$((20000 + SLURM_JOB_ID % 20000))
vllm serve "$MODEL" --served-model-name m --port $PORT --max-model-len 12288 \
    --gpu-memory-utilization 0.85 --seed 0 "${SERVE_EXTRA[@]}" > "$RUN/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
URL=http://127.0.0.1:$PORT
for i in $(seq 360); do   # wait up to 30 min, but fail fast if the server dies
  curl -sf $URL/health >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM server died; see $RUN/vllm_${SLURM_JOB_ID}.log"; tail -5 "$RUN/vllm_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 5
done
python src/fuzz_loop.py --task $TASK --strategy $STRAT --run-dir "$RUN" --budget $BUDGET \
    --baseline $BASE \
    --fuzzer-url $URL --target-url $URL --judge-url $URL --deadline-min 155 $EXTRA
RC=$?
echo "loop exit code $RC"
if [ $RC -eq 3 ]; then
  echo "resubmitting to continue"
  sbatch --gres="$GRES" -J "$SLURM_JOB_NAME" --export=ALL $PROJ/scripts/slurm/20_fuzz.sbatch "$MODEL" "$TAG" "$STRAT" "$BUDGET" $EXTRA
fi
[ $RC -eq 0 ] && echo FUZZ_JOB_DONE
exit $RC
