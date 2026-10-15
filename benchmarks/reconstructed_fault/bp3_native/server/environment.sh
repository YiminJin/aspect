# Source AFTER loading the server's matching compiler/MPI/deal.II toolchain.
# Module names and dependency locations are site-specific; see README.md.
# This file changes environment variables only. It never builds or launches.
# All model/library paths in parameters.prm are relative to $SCRATCH/bp3_native.

# Presence-based development selectors must be absent, including values of "0".
# No ASPECT_SOURCE_DIR override is needed by this self-contained job directory.
for _bp3_native_env_name in $(env | sed -n 's/^\(ASPECT_[A-Za-z0-9_]*\)=.*/\1/p'); do
  unset "$_bp3_native_env_name"
done
unset _bp3_native_env_name

# One compute thread per MPI rank; preserve the loaded module/library paths.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
