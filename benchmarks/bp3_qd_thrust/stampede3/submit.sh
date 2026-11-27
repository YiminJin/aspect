#!/bin/bash
#SBATCH -J bp3-60
#SBATCH -p icx
#SBATCH -N 2
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -o bp3-%j.out
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the extracted bp3 directory}"
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if type module >/dev/null 2>&1; then module list 2>&1; fi
exec python3 -u tools/run.py "$@"
