#!/bin/bash
#SBATCH --job-name=fv_ot
#SBATCH --partition=intel-sc3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=results/slurm_%x_%j.out
#SBATCH --error=results/slurm_%x_%j.err
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${FATEVAE_PYTHON:-python3}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
$PY "$@"
