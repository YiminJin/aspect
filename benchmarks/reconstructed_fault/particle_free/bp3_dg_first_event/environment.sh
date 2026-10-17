# Source after loading the server's matching compiler/MPI/deal.II modules.
# Environment only: no module loading, build, job submission or run controller.
# Set DEAL_II_DIR, VORO_DIR and CXX to the matching server dependency stack.

: "${WORK:?Set WORK to the site work directory before sourcing this file.}"
: "${SCRATCH:?Set SCRATCH to the site scratch directory before sourcing this file.}"

for _bp3_dg_env_name in $(env | sed -n 's/^\(ASPECT_[A-Za-z0-9_]*\)=.*/\1/p'); do
  unset "$_bp3_dg_env_name"
done
unset _bp3_dg_env_name

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export OMP_PROC_BIND=false

# WORK and SCRATCH are supplied by the site; neither is reassigned here.
export BP3_DG_SOURCE="$WORK/build/aspect/pf-rsf-dg/source"
export BP3_DG_BUILD="$WORK/build/aspect/pf-rsf-dg/build"
export BP3_DG_RUN="$SCRATCH/bp3_dg"
