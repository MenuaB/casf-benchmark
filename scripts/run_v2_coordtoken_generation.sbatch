#!/bin/bash -l
#SBATCH --partition=research
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00

set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $(basename "$0") core|ref|plinder [num_shards] [batch_size] [samples_per_molecule]" >&2
  exit 2
fi

REPO_ROOT="${CASF_REPO_ROOT:-/mnt/weka/mbedrosian/code/casf-benchmark}"
MOLGEN_ROOT="${MOLGEN_ROOT:-/mnt/weka/vtarasov/code/3DMolGen}"
MOLGEN_PYTHON="${MOLGEN_PYTHON:-/mnt/weka/mbedrosian/code/3DMolGen/.venv/bin/python}"
COHORT="$1"
NUM_SHARDS="${2:-1}"
BATCH_SIZE="${3:-256}"
SAMPLES_PER_MOLECULE="${4:-1000}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

export PYTHONPATH="${MOLGEN_ROOT}/src:${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${REPO_ROOT}"
exec "${MOLGEN_PYTHON}" scripts/run_v2_coordtoken_generation.py generate "${COHORT}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --batch-size "${BATCH_SIZE}" \
  --samples-per-molecule "${SAMPLES_PER_MOLECULE}"
