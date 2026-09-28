# Source from bash or zsh; no simulation is started.
#   source /path/to/aspect/benchmarks/reconstructed_fault/bp3/environment.sh
# Select block AMG/GMG in Solver parameters / Stokes solver parameters in the PRM.
# Load the server's MPI/deal.II modules first. This file leaves those paths intact.
if [ -n "${BASH_VERSION:-}" ]; then
  _bp3_env_file=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ]; then
  eval '_bp3_env_file=${(%):-%x}'
else
  printf '%s\n' 'BP3 environment.sh must be sourced from bash or zsh.' >&2
  return 2
fi
# Obsolete backend selectors must not leak into an older executable either.
unset ASPECT_FAULT_SURFACE_SOLVER ASPECT_FAULT_COMPARE_SURFACE_INVERSE
unset ASPECT_FAULT_VELOCITY_GMG ASPECT_FAULT_GMG_HIERARCHY
unset ASPECT_COMPARE_COUPLING ASPECT_FAULT_COMPARE_COUPLING

export ASPECT_FAULT_EXPLICIT_B=1 ASPECT_FAULT_EXPLICIT_G=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
unset _bp3_env_file
