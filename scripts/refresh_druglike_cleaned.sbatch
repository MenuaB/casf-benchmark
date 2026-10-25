#!/bin/bash
#SBATCH --job-name=druglike-cleaned
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=0
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/weka/mbedrosian/code/casf-benchmark/outputs/slurm_jobs/druglike_cleaned/%j.out
#SBATCH --error=/mnt/weka/mbedrosian/code/casf-benchmark/outputs/slurm_jobs/druglike_cleaned/%j.err

# Rebuild extended_druglike_* on the cleaned 22-molecule SDF reference.
# RMSD cache is separate from the PLINDER panel job.

set -euo pipefail

REPO_ROOT="${CASF_REPO_ROOT:-/mnt/weka/mbedrosian/code/casf-benchmark}"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "${REPO_ROOT}/.venv/bin/python" scripts/refresh_druglike_cleaned.py \
  --num-workers 32
