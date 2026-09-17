# Source from bash or zsh; no simulation is started.
#   source /path/to/aspect/benchmarks/reconstructed_fault/bp3/environment.sh [gmg|amg]
# Load the server's MPI/deal.II modules first. This file leaves those paths intact.
if [ -n "${BASH_VERSION:-}" ]; then
  _bp3_env_file=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ]; then
  eval '_bp3_env_file=${(%):-%x}'
else
  printf '%s\n' 'BP3 environment.sh must be sourced from bash or zsh.' >&2
  return 2
fi
_bp3_backend=${1:-gmg}
case "$_bp3_backend" in
  gmg|amg) ;;
  *) printf '%s\n' 'Expected gmg or amg.' >&2; return 2 ;;
esac
_bp3_repo=$(cd -- "$(dirname -- "$_bp3_env_file")/../../.." && pwd -P) || return

# Remove inherited investigation selectors, without altering the MPI runtime.
for _bp3_name in $(env | sed -n 's/^\(ASPECT_[A-Za-z0-9_]*\)=.*/\1/p'); do
  unset "$_bp3_name"
done
export ASPECT_SOURCE_DIR="$_bp3_repo"
export ASPECT_FAULT_EXPLICIT_B=1 ASPECT_FAULT_EXPLICIT_G=1
export ASPECT_FAULT_SURFACE_SOLVER=tridiagonal
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
if [ "$_bp3_backend" = gmg ]; then
  export ASPECT_FAULT_VELOCITY_GMG=1 ASPECT_FAULT_GMG_HIERARCHY=1
fi
unset _bp3_env_file _bp3_backend _bp3_repo _bp3_name
