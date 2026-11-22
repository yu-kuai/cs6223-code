#!/bin/bash
#SBATCH -J autograde_v4
#SBATCH -p long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
# Wait for each run dir to finish (state.json done=true), then grade it with DeepSeek. usage: sbatch 35_autograde.sbatch <run dirs...>
set -uo pipefail
cd /mnt/scratch/k/kuaiyu/cs6223/project
left=("$@")
while [ ${#left[@]} -gt 0 ]; do
  next=()
  for d in "${left[@]}"; do
    if [ -f $d/state.json ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$d/state.json')).get('done') else 1)"; then
      echo "$(date) grading $d"; python3 src/judge_api_grade.py --model x-test-2 --tag deepseek $d
    else next+=("$d"); fi
  done
  left=("${next[@]+"${next[@]}"}"); [ ${#left[@]} -gt 0 ] && sleep 600
done
echo AUTOGRADE_DONE
