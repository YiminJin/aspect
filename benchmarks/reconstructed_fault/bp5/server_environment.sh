# Source this file AFTER loading the same compiler/MPI/deal.II modules used
# to build the server executable and plugin. No simulation starts here.
# Usage: cd /path/to/job-directory; source ./environment.sh
if [ ! -f ./first_event.prm ] || [ ! -f ./fixture/fault.txt ]; then
  printf '%s\n' 'Change to the copied job submission directory before sourcing environment.sh.' >&2
  return 2
fi
# Discard inherited investigation selectors, retaining module/library paths.
for _bp5_name in $(env | sed -n 's/^\(ASPECT_[A-Za-z0-9_]*\)=.*/\1/p'); do
  unset "$_bp5_name"
done
export ASPECT_FAULT_EXPLICIT_B=1 ASPECT_FAULT_EXPLICIT_G=1
export ASPECT_FAULT_SURFACE_SOLVER=tridiagonal
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
unset _bp5_name
