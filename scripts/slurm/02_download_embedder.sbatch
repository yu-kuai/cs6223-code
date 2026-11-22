#!/bin/bash
#SBATCH -J dl_embedder
#SBATCH -p normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o /home/k/kuaiyu/slurm_logs/%x_%j.out
set -euo pipefail
source /mnt/scratch/k/kuaiyu/cs6223/project/scripts/slurm/common.sh
activate_env
hf download sentence-transformers/all-MiniLM-L6-v2
HF_HUB_OFFLINE=1 python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
e = m.encode(['Which is not prime?', 'Which number is not a prime?', 'The capital of France'], normalize_embeddings=True)
print('sim paraphrase', float(e[0]@e[1]), 'sim unrelated', float(e[0]@e[2]))"
echo EMB_DONE
