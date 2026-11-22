# Sourced by every sbatch job.
# Env is built on node-local /tmp and shipped as one tarball on scratch (NFS is slow with many small files).
# Always unpacked to the SAME path so the conda prefix stays valid.
export PROJ=/mnt/scratch/k/kuaiyu/cs6223/project
export HF_HOME=$PROJ/hf_cache
export ENV_PREFIX=/tmp/kuaiyu_cs6223_env
export ENV_TAR=$PROJ/env.tar
cd $PROJ
[ "$(uname -m)" = x86_64 ] || { echo "non-x86 node $(hostname) ($(uname -m)); env is x86_64-only"; exit 1; }
echo "[$(date)] host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION"

activate_env() {   # unpack env tarball to local /tmp if missing/outdated (flock: safe for concurrent jobs on a node)
  (
    flock 9
    if [ ! -f "$ENV_PREFIX/.stamp" ] || [ "$ENV_TAR" -nt "$ENV_PREFIX/.stamp" ]; then
      echo "unpacking env -> $ENV_PREFIX"; rm -rf "$ENV_PREFIX"; mkdir -p "$ENV_PREFIX"
      tar -xf "$ENV_TAR" -C "$ENV_PREFIX" && touch "$ENV_PREFIX/.stamp"
    fi
  ) 9>/tmp/kuaiyu_cs6223_env.lock
  export PATH="$ENV_PREFIX/bin:$PATH" CONDA_PREFIX="$ENV_PREFIX"
  # some GPU nodes have an old /usr/bin/nvcc -> FlashInfer JIT build fails; use PyTorch sampler, keep JIT caches node-local
  export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=/tmp/kuaiyu_flashinfer XDG_CACHE_HOME=/tmp/kuaiyu_cache
  echo "python=$(command -v python)"
}
