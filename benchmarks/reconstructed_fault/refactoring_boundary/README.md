# R2b proposal 2: prescribed boundary completion

This work starts at the qualified geometry-preparation extraction, branch
`pf-rsf-refactor`, HEAD `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700` plus the
preserved local R1/limiter/R2a/R2b changes. It contains two distinguishable steps:

1. `apply_boundary_normalization_completion()` privately extracts the legacy
   completion body. `evidence/*.move`, `*.move.diff`,
   `move-source-verification.json` and `aspect-completion-move` preserve this
   step independently of the extension. Both short BP3 runs match the qualified
   R2b baseline exactly: 186 field groups.
2. The explicit `automatic prescribed` selector adds geometric per-contact
   detection, verified ghost-Q1 completion and paired endpoint mechanical
   continuation. This is a behavior extension, not move-only refactoring.

## Reproduction

Run from the repository root. Existing logs and checkpoint branches are never
replaced by the harness. Use a fresh evidence/output directory or new run labels
for reruns. The staging script regenerates inputs only; it does not alter the
preserved scientific worktree, reference binaries or checkpoints.

- `python3 benchmarks/reconstructed_fault/refactoring_boundary/stage_inputs.py`
- Build `build-refactor-boundary` and the BP3 plugin in this directory's
  `plugin-build`, and build the test target
  `phase_field_fault_boundary_completion` in `build-refactor-boundary/tests`.
  Exact configure/build commands and toolchain options are in the evidence JSON.
- `bash benchmarks/reconstructed_fault/refactoring_boundary/run_focused.sh`
- `bash benchmarks/reconstructed_fault/refactoring_boundary/run_automatic_bp3.sh`
- `bash benchmarks/reconstructed_fault/refactoring_boundary/run_legacy.sh`
- `python3 benchmarks/reconstructed_fault/refactoring_boundary/verify_comparisons.py`
- `python3 benchmarks/reconstructed_fault/refactoring_boundary/verify_scope.py`
- Registered regression: `ctest --test-dir build-refactor-boundary/tests -R
  '^phase_field_fault_boundary_completion$' --output-on-failure`.

The move-only executable is intentionally separate. `run_move.sh` and
`compare_move.py` qualify it against R2b with exact equality. Do not overwrite
that executable with the later automatic implementation.

GCC 12.4.0, OpenMPI 5.0.6, deal.II 9.6.2, Release, unity/PCH, two build jobs and
one execution thread per rank were used. MPI requires local socket access.
BP3 is the existing coarse 1,875-cell seven-observation (steps 0–6) functional
fixture, not a production earthquake-cycle test. Small fixtures use 1,024 cells.
The cross-rank restart branches accepted step 4 into a new directory, resumes
on two ranks and compares steps 5–6 with the uninterrupted trajectory.

## Checks and evidence

The final build/run records use the `qualified-` prefix. The final legacy
one-rank trajectory is exact. `comparison-summary.json` records contact identities, exterior correction
comparisons, small-case MPI/reversal checks, and free-endpoint K/B/G errors.
`automatic-state-comparison.json` covers the same 186 groups as the exact move
comparison. `mpi-restart-mechanics.json` additionally checks cross-rank and
restart particle/bulk states and the existing weak mechanical output tables.

Automatic/legacy arithmetic is not required to be bitwise identical: the new
C++ integration computes the ghost data that the legacy Python table supplies.
Both use Q1 ghost vertices, grid-edge panels and Gauss8/two-half-panel refinement
at 1e-11 relative panel accuracy. Comparisons require 1e-10 of the corresponding
field scale, report absolute differences at zero, and require identical solver
decisions. Near-zero residual diagnostic differences are reported in absolute
units, without changing solver convergence criteria. The move-only criterion
remains exact.

The focused plugin prescribes all physical Q1 phase DoFs to the same stationary
profile used for exterior continuation. It checks interior-wedge endpoint
association and centered directional differences of free-endpoint K, pressure
G and bulk B (tolerance 1e-8); it does not assume G=B transpose. Expected rejection
runs require an actionable message and nonzero exit, not a timeout. Unit tests
cover periodic-face exclusion, shared-facet deduplication, unrepresented
segment crossings, short curved terminals, overlapping/diffuse branch support,
and 3D rejection. Refinement invalidation is implemented but no adaptive-mesh
runtime qualification or evolving-phase boundary treatment is claimed.

Exploratory failures remain in `evidence/`: inherited postprocessor names in the
first input; an under-resolved new small-profile fixture producing a negative
consistent projection; an existing stationary-H `phi <= core_phi` assertion at
0.60000000000000009 vs 0.59999999999999998 for a 45-degree fixture; and the existing
normal-profile overlap rejection for two insufficiently separated faults.
The qualified fixtures resolve the profile and use independent support. No M2
assertion, projection algorithm, expected result or production tolerance was
changed. The new multiple-fault run also found and qualified the tolerance-
limited preservation of an already admitted association at s=0.
