#!/bin/bash
#SBATCH --job-name=fv_gpu
#SBATCH --partition=rt-2080ti-short
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm_%x_%j.out
#SBATCH --error=results/slurm_%x_%j.err
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${FATEVAE_PYTHON:-python3}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi || true
$PY -c "import torch; print('cuda available:', torch.cuda.is_available())"
$PY "$@"
