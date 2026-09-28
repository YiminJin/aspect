# Source from bash before launching. Remove inherited investigation selectors;
# these PRMs, not old ASPECT_* environment flags, define the physical fixture.
for bp5_filter_variable in ${!ASPECT_@}; do
  unset "$bp5_filter_variable"
done
unset bp5_filter_variable
export ASPECT_FAULT_EXPLICIT_B=1
export ASPECT_FAULT_EXPLICIT_G=1
export ASPECT_FAULT_SURFACE_SOLVER=tridiagonal
export ASPECT_FAULT_PERFORMANCE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
