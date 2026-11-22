#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --exclude=xgpj0
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
# Several fuzz loops in one GPU job: one vLLM server per model (co-located on the GPU), many loops share them.
# Covers cross-model pairs (fuzzer+judge = model F, target = model T) and repeat seeds of self runs.
# usage: scripts/submit_multi.sh <conf>     (conf lines below; gres is read from the conf)
#   task  mcq|vqa
#   gres  gpu:h200-141:1
#   model <tag> <hf_model> <gpu_mem_fraction>          (tag = baseline file name, e.g. qwen3-8b)
#   loop  <fuzzer_tag> <target_tag> <strategy> <seed>
# Run dirs: results/fuzz[_vqa]/<F>_self/<strategy>[_s<seed>]  or  <F>_to_<T>/<strategy>[_s<seed>]
# Loops already done are skipped; if any loop hits the deadline the job resubmits itself (loops resume from state.json).
set -uo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
CONF=$(readlink -f "$1")
TASK=$(awk '$1=="task"{print $2}' "$CONF"); GRES=$(awk '$1=="gres"{print $2}' "$CONF")
BUDGET=${BUDGET:-5000}
if [ "$TASK" = vqa ]; then
  RD=$PROJ/results/fuzz_vqa; BD=results/baseline_vqa; SERVE_EXTRA=(--limit-mm-per-prompt '{"image": 2}')
else
  RD=$PROJ/results/fuzz; BD=results/baseline; SERVE_EXTRA=()
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
activate_env
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/tmp/kuaiyu_vllm_cache
LOGDIR=$PROJ/results/multi_logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID; mkdir -p "$LOGDIR"
declare -A URL; PIDS=()
trap 'kill ${PIDS[@]} 2>/dev/null' EXIT
i=0
while read -r kw tag hf util; do
  PORT=$((20000 + (SLURM_JOB_ID % 9000) * 4 + i)); i=$((i + 1))
  echo "[$(date +%T)] starting $tag ($hf) util=$util port=$PORT"
  vllm serve "$hf" --served-model-name m --port $PORT --max-model-len 12288 --gpu-memory-utilization "$util" \
      --seed 0 "${SERVE_EXTRA[@]}" < /dev/null > "$LOGDIR/vllm_$tag.log" 2>&1 &
  PID=$!; PIDS+=($PID)
  for k in $(seq 360); do   # sequential start: each server sizes its KV cache after the previous one is up
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
    kill -0 $PID 2>/dev/null || { echo "vLLM $tag died"; tail -20 "$LOGDIR/vllm_$tag.log"; exit 1; }
    sleep 5
  done
  URL[$tag]=http://127.0.0.1:$PORT
done < <(awk '$1=="model"' "$CONF")
echo "[$(date +%T)] servers up after ${SECONDS}s"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
DEADLINE=$(( (172 * 60 - SECONDS) / 60 ))
LP=(); LD=()
while read -r kw F T S SEED; do
  FLAGS=""; SD=$S   # strategy "<s>+guard" -> --label-guard, run dir <s>_guard
  if [[ $S == *+guard ]]; then S=${S%+guard}; SD=${S}_guard; FLAGS="--label-guard"; fi
  if [ "$F" = "$T" ]; then D=$RD/${F}_self/$SD; else D=$RD/${F}_to_${T}/$SD; fi
  [ "$SEED" != 0 ] && D=${D}_s$SEED
  if [ -f "$D/state.json" ] && grep -q '"done": true' "$D/state.json"; then echo "skip done $D"; continue; fi
  mkdir -p "$D"
  CUDA_VISIBLE_DEVICES= python src/fuzz_loop.py --task "$TASK" --strategy "$S" --run-dir "$D" --budget $BUDGET \
      --baseline "$BD/$T.jsonl" --fuzzer-url "${URL[$F]}" --target-url "${URL[$T]}" --judge-url "${URL[$F]}" \
      --seed "$SEED" --deadline-min $DEADLINE $FLAGS < /dev/null > "$D/loop_$SLURM_JOB_ID.log" 2>&1 &
  LP+=($!); LD+=("$D")
done < <(awk '$1=="loop"' "$CONF")
echo "[$(date +%T)] launched ${#LP[@]} loops, deadline ${DEADLINE} min"
AGAIN=0
for j in "${!LP[@]}"; do
  wait "${LP[$j]}"; RC=$?
  echo "[$(date +%T)] rc=$RC ${LD[$j]} :: $(grep '^round' "${LD[$j]}/loop_$SLURM_JOB_ID.log" | tail -1)"
  [ $RC -eq 3 ] && AGAIN=1
  [ $RC -ne 0 ] && [ $RC -ne 3 ] && tail -5 "${LD[$j]}/loop_$SLURM_JOB_ID.log"
done
if [ $AGAIN -eq 1 ]; then
  echo "resubmitting to continue"
  sbatch --gres="$GRES" -J "$SLURM_JOB_NAME" $PROJ/scripts/slurm/21_fuzz_multi.sbatch "$CONF"
else
  echo MULTI_DONE
fi
