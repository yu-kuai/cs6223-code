#!/bin/bash
#SBATCH -J dl_vqa
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
# Part 2 (VQA): ScienceQA image questions + Qwen3-VL 2B/8B/32B (Instruct, matches thinking-off text setup).
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
activate_env
python -c "import vllm, transformers, PIL; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'PIL', PIL.__version__)"
python scripts/prepare_scienceqa.py --out data/scienceqa_img_seeds.jsonl --img-dir data/scienceqa_img
du -sh data/scienceqa_img
for m in Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-32B-Instruct; do
  echo "== $m"; hf download "$m"
done
du -sh "$HF_HOME"
echo DOWNLOAD_VQA_DONE
