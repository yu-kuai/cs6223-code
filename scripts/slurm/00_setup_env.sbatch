#!/bin/bash
#SBATCH -J setup_env
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
export CONDA_PKGS_DIRS=/tmp/kuaiyu_conda_pkgs PIP_CACHE_DIR=/tmp/kuaiyu_pip_cache PIP_PROGRESS_BAR=off
rm -rf "$ENV_PREFIX"
conda create -q -y -p "$ENV_PREFIX" python=3.11
conda activate "$ENV_PREFIX"
pip install -q -U pip
pip install -q vllm datasets pandas "huggingface_hub[cli]" sentence-transformers rapidfuzz tqdm
python -c "import vllm, torch, datasets; print('vllm', vllm.__version__, 'torch', torch.__version__)"
touch "$ENV_PREFIX/.stamp"
echo "packing env..."; tar -cf "$ENV_TAR.tmp" -C "$ENV_PREFIX" . && mv "$ENV_TAR.tmp" "$ENV_TAR"
ls -lh "$ENV_TAR"
rm -rf /tmp/kuaiyu_conda_pkgs /tmp/kuaiyu_pip_cache
echo SETUP_DONE
