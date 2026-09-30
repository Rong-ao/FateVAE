#!/bin/bash
#SBATCH --job-name=fatevae
#SBATCH --partition=intel-sc3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=results/slurm_%x_%j.out
#SBATCH --error=results/slurm_%x_%j.err

# FateVAE Phase-1 training on the Dmel genital disc data.
# Usage: sbatch scripts/slurm_fatevae.sh [extra args to phase1_train_fatevae.py]
#   e.g. sbatch scripts/slurm_fatevae.sh --tag run1 --scale 1.0 --adult-anchor

set -euo pipefail
cd "$(dirname "$0")/.."
PY=${FATEVAE_PYTHON:-python3}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

$PY scripts/phase1_train_fatevae.py "$@"
