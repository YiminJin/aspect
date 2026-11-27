#!/bin/bash
#SBATCH -J bp3-thrust
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 80
#SBATCH --cpus-per-task=1
#SBATCH -t 04:00:00
set -euo pipefail
[[ $# == 2 && ( $1 == pilot || $1 == production ) && ( $2 == start || $2 == resume ) ]] ||
  { echo "Usage: stampede3.sh pilot|production start|resume" >&2; exit 2; }
cd -- "${BP3_RUN:?Set BP3_RUN to the directory created by prepare.sh}"
exec 9>.launch.lock
flock -n 9 || { echo "An allocation already owns this run directory" >&2; exit 1; }
sha256sum --check inputs.sha256
solver=$(cat executable.path)
mode=$1
if [[ $2 == start ]]; then
  [[ ! -e output ]] || { echo "Fresh start requires an unused output directory" >&2; exit 1; }
  resume=false
else
  [[ -s output/restart/last_good_checkpoint.txt ]] || { echo "Missing branch checkpoint" >&2; exit 1; }
  resume=true
fi
job_input=$(mktemp "launch-${SLURM_JOB_ID:-local}-$mode-XXXXXX.prm")
printf 'include %s.prm\nset Resume computation = %s\n' "$mode" "$resume" > "$job_input"
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
launcher=()
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  launcher=(ibrun)
  module list 2>&1
elif [[ ${BP3_LOCAL_RANKS:-1} != 1 ]]; then
  launcher=(mpiexec -n "${BP3_LOCAL_RANKS:?}")
fi
printf 'Working directory %s\nSDK %s\nExecutable %s\nPlugin %s\nVertex table %s\nInput %s\n' \
  "$PWD" "$(cat sdk.path)" "$solver" "$PWD/libbp3_qd.release.so" "$PWD/vertices.txt" "$PWD/$job_input"
"${launcher[@]}" "$solver" "$job_input" 2>&1 | tee "${job_input%.prm}.log"
