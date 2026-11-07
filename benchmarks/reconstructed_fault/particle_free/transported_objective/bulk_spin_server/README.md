# Fresh corrected-spin reusable input

This is the complete, locally tested `bulk-ld-half` synthetic mature-fault
configuration: 16², dt=.0125 s, end=.1 s, direct stress, bulk-slip-subtracted
spin, translation plus shear, constant G and a prescribed x-dependent viscosity.
It is not production BP3 or the stopped equilibrated-layer campaign. It starts
fresh; do not replace `Resume computation = false` with a total-spin checkpoint.

Build the current source and `transported_bulk_spin` target from the native
checker CMake project with a matching ASPECT installation. Place that executable
as `aspect-release` and its matching library as
`libtransported_bulk_spin.release.so` in this directory (or update only those
paths). From this directory, the reproducible command is:

```sh
OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun -n 4 ./aspect-release input.prm > run.log 2>&1
```

Keep required ASPECT/deal.II/MPI runtime libraries available. Record source and
binary hashes before running; do not assume a retained binary matches HEAD.
The native checker supplies the common fixture's initial traction/age fields
and validates committed-state/rotation/publication invariants. Those inputs
are documented in `native/material_checks.cc`, not a production material law.
Output is isolated in `output_bulk_spin/`; choose a new directory for repeats.

This package was prepared only; no server job was submitted. Its bounded
configuration has local evidence. Extending the horizon or changing boundaries
requires a separately selected comparison. The next scientific step proposed
in the report uses a fresh compatible exterior/top-loading background, which
is a different boundary benchmark, not an automatic continuation of this input.
