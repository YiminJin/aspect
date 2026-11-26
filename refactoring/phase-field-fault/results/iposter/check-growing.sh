#!/usr/bin/env bash
# Compare the installed one-field example with the corrected nine-strip control at 5 ms.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: $0 ASPECT_INSTALL_PREFIX FRESH_DIRECTORY" >&2
  exit 2
fi
prefix=$(realpath "$1")
examples=$prefix/benchmarks/growing_fault
aspect=$prefix/bin/aspect-release
mpi=${MPIEXEC:-mpirun}
export DEAL_II_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Never replace a previous native frame. All three calculations start from their initial state.
mkdir -- "$2"
run=$(realpath "$2")
mkdir "$run"/{seed,control,affine}
cp "$examples/growing_seed.prm" "$run/seed/input.prm"
cp "$examples/growing_seed.prm" "$run/affine/input.prm"
cp "$examples/onecut.prm" "$run/control/input.prm"

cat >> "$run/control/input.prm" <<'PRM'

set End time = .005
subsection Postprocess
  set List of postprocessors = growing fault history
end
PRM
cat >> "$run/affine/input.prm" <<'PRM'

subsection Adiabatic conditions model
  subsection Function
    set Function constants = p0=200, dp_dz=50
  end
end
PRM
for case in seed control affine; do
  echo 'set Output directory = output' >> "$run/$case/input.prm"
  (
    cd "$run/$case"
    "$mpi" --bind-to core --map-by core -np 2 "$aspect" input.prm > run.log 2>&1
  )
done
python3 "$examples/check_seed_example.py" "$run/control/output" "$run/seed/output" "$run/affine/output" \
  | tee "$run/check-5ms.log"
