# Bounded pressure-complement / fresh-residual replay

`one.prm` and `two.prm` include the unchanged `convergence/space16_dt05.prm`.
Only their output directories differ. Production nonlinear tolerances, scales,
linear iteration budget, support and full I_h are unchanged.

```sh
cmake --build build-pf-cpdi --target aspect.exe.debug -j4
cmake --build benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build -j4
python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py \
  benchmarks/reconstructed_fault/uniform_shear/linear-correction/one.prm --timeout 2400
python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py \
  benchmarks/reconstructed_fault/uniform_shear/linear-correction/two.prm --ranks 2 --timeout 2400
```

Each runner saves the executable/plugin hashes, command, rank count, elapsed
time and peak child RSS. Runs overlapped focused tests, so their elapsed times
are not isolated performance measurements. MPI RSS is a peak child-process
measurement, not summed distributed memory.

`one/` uses the existing serial CSV schema. `two/` writes corresponding
`name_rankN_step.csv` files: owned cells/particles are rank-local; surface and
time data are replicated. Do not sum replicated surface/time rows. Cell/QP
row numbers refer to the accompanying rank-local bulk file. Initial phi/H
snapshots are shared only to keep the benchmark's freeze checks valid for
relevant constraint lines and migrating particles.

`summarize.py LOG` emits each recorded linear and nonlinear residual with its
timestep. `fresh` is the freshly evaluated pressure-complement residual of the
original operator; `raw` includes the independently checked null component.
Neither is an Arnoldi estimate. `target` is the original relative linear
tolerance times the unprojected RHS norm.

Intermediate failed attempts are retained with descriptive suffixes. See
`doc/reconstructed_fault/benchmarking/stage_K1_linear_correction_review.md`
for the outcome and distinctions between solver, test, and exporter defects.
K2 is not part of this replay.
