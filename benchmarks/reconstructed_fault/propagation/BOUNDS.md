# P1 one-dimensional reference and box-constrained prototype

Completed 2026-10-06. The sampled-H reference retains a negative equality tail.
The test-only bound solver passes the straight and padded-seed box/KKT and
controlled-ramp checks, with no resolved incremental healing. The production
equality solver remains unchanged and unqualified for phase bounds. No P2.
See the [report](../../../doc/reconstructed_fault/propagation/P1_bounds_report.md),
[summary](bounds_evidence/summary.json), [plot](bounds_evidence/reference-and-bounds.svg)
and [provenance manifest](bounds_evidence/manifest.json).

`reference_1d.h` provides independent weak integration and a tridiagonal Newton
solve; material functions are called through the native test friend.
`bound_diagnostic.cc` applies projected Newton to production scratch assembly
and CG/AMG, with explicit free-residual, dual-sign, complementarity and bounds
checks. It does not impose phi >= previous phi or alter H evolution. The padded
seed uses its rounded profile only as a starting guess; no seed constraint is
added. The accepted equilibrium is the baseline for the diagnostic trajectory.
Full simulator state is restored afterward.

Run from the repository root. The commands used to configure/build were:

```sh
cmake -S benchmarks/reconstructed_fault/propagation -B build-propagation-p1-bounds \
  -DAspect_DIR=/home/ein/repository/aspect-pf-rsf-refactor/build-refactor-post-r6-gcc12-unity \
  -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-propagation-p1-bounds --target bound_diagnostic -j2
python3 benchmarks/reconstructed_fault/propagation/run_bound_case.py reference32 \
  --build build-propagation-p1-bounds \
  --executable /home/ein/repository/aspect-pf-rsf-refactor/benchmarks/reconstructed_fault/refactoring_r7b/build/aspect-r7b-qualified \
  --evidence benchmarks/reconstructed_fault/propagation/bounds_evidence
```

Subsequent selected cases were `reference64`, `bounds_straight`, and
`bounds_padded`. Diagnostic fixes required two distinct recorded repeats:
`bounds_straight --label bounds_straight_final` and
`bounds_padded --label bounds_padded_final`. `startup_failure/` preserves the
initial parser/registration failure. The manifest maps earlier source and
binary hashes to preserved snapshots; the reference numerical implementation
was only reformatted after its runs. See the report for each correction.
The runner refuses output/evidence overwrites. Reproduction requires fresh
output labels/evidence paths or another task-owned build directory.

No new ASPECT runs are needed to check the retained results:

```sh
python3 benchmarks/reconstructed_fault/propagation/check_bound_evidence.py
python3 -m unittest discover -s benchmarks/reconstructed_fault/propagation -p test_bound_checks.py -v
MPLCONFIGDIR=/tmp/propagation-matplotlib python3 benchmarks/reconstructed_fault/propagation/plot_bounds.py
```

The checker independently verifies all numerical gates; it exits 2 if resolved
healing or inconsistent repeats disqualify the restricted prescribed-H result.
Neither a box-KKT pass nor this restricted result qualifies the mechanical H
update or the production solver. Each simulation is bounded to 120 s, 2 GiB RSS
and 10 MiB output; actual completed runs take about 5–6 s. Mechanics, RSF
evolution, transport, propagation, native restart and physical rollback remain
outside this diagnostic.
