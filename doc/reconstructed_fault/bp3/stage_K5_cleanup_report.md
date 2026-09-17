# K5 research configuration consolidation

## Preserved baseline and scope

The baseline is `fully-frictional-seven-local4`, initialization plus seven
real steps to 4.1901293117 yr. Before editing, the full evidence directory
was hashed and left in place; source, binaries, HEAD and tracked diffs were
saved in `benchmarks/reconstructed_fault/bp3/cleanup-checkpoint-20260916/`.
Unrelated working changes and earlier experimental outputs were preserved.
The recorded HEAD is `3335d3d26c298ff5aaeba77062b0a77c8d20f0b5`; the
snapshot, not HEAD alone, identifies the dirty tested implementation. All
262 original evidence files were subsequently rehashed without a mismatch.

This cleanup changes no equations or acceptance criteria of that run. The
supported research choice is explicitly **modified BP3**: fully frictional,
continuous Q1 V/Theta/slip, mature C=0, bulk-QP work measure, paired endpoint
normalization/source corrections, fixed prestress and outer plate loading.
Mechanics uses lagged committed Theta; exact nodal aging and Maxwell history
are published once after acceptance. Timestep zero retains supplied histories.

## Retained, consolidated, retired

- `bp3_modified_fully_frictional.prm` flattens the effective tested parameter
  chain. `Postprocess/BP3/Fault loading configuration = fully frictional`
  replaces the experimental environment switch. The default remains
  `prescribed deep slip`; `bp3_constrained_reference.prm` records that reference.
- `run_research.py` selects the unchanged saved clock, mesh and 1236-node fault;
  one fresh four-rank process has a 2400-s hard cap and no retry/overwrite.
- Source and all surface terms retain their common work measure and free
  endpoint dependence. The work-based assembly continues to sample frozen
  **working FE stress**, whereas published particle stress is the accepted
  next history. These diagnostics are not interchanged.
- The stable independent Theta audit, fixed-geometry/Ih/H checks, initial
  stress retention, first Maxwell update, fresh linear checks and K/G finite
  differences are retained. `work_checks.h` replaces the misleading
  frozen-cohesion helper name for the tests actually used by this fixture.
- Frozen-cohesion/alternate-state/independent-trace overrides were removed
  from the material model, manager and mechanical assembly. Candidate-state
  tangent fields and their unused asymmetric surface assembly plumbing were
  removed. Pivoted indefinite-capable surface factorization is unchanged.
- The forced noncommitting diagnostic stop was removed from the solver.
  Ordinary exception rollback and lifecycle tests remain. Old A/B callbacks,
  paired-process clock and experiment-only exports no longer run in the
  maintained plugin. Retired headers are under `bp3/investigations/`; historical
  scripts and source/binary snapshots preserve reproducibility. Old physics
  selectors fail explicitly rather than silently changing a new run.
- Read-only stress/history observations and useful profile/source diagnostics
  remain opt-in. Benchmark-specific straight/frozen endpoint setup and the
  mature choice were not promoted to unrestricted defaults.

The only added user-facing selector is the benchmark loading parameter.
The point-input/response diagnostic-only fields `diagnostic_nodal_rates`,
`diagnostic_state_tangent` and `evaluated_state` are removed after checking
their callers. The semantic surface solve/action APIs, normal bulk residual,
history-publication interface and standard friction parameters are unchanged.

## Verification definition

Build the Release executable and plugin with `-j4`, then run:

```
python3 benchmarks/reconstructed_fault/bp3/run_research.py
python3 benchmarks/reconstructed_fault/bp3/check_research_cleanup.py
python3 benchmarks/reconstructed_fault/bp3/analyze_fully_frictional.py \
  --run benchmarks/reconstructed_fault/bp3/fully-frictional-cleanup-local4
```

The cleanup comparison reuses the existing restart checker coefficient 1e-8
separately for each physical field, with exact clock, geometry, fixed profile,
background and mask comparisons. It compares every accepted V/Theta/slip and
current raw stress at matching QPs, separately from stable-ID particle stress.
The existing physical analysis independently verifies once-per-step aging,
slip accumulation, 1236 free/zero prescribed nodes and the weak balance.

## Outcome

The Release core/plugin builds and both parameter-file validations passed.
The focused two-rank command
`mpirun -np 2 build-pf-cpdi/aspect-release --test 'Stage-I*,[fault_surface_direct],[phase_field_fault_cohesive],*FaultFriction*'`
passed **1262 assertions in 21 cases on each rank**, including captured
bound contact/release/rollback, Armijo rejection/exhaustion and indefinite
surface inversion. Python compilation and `git diff --check` passed.

The actual resolved parameter JSON differs from the original successful run
only in output directory and the new explicit loading selector. Initialization
fault tables are bitwise identical; the first real history update and K/G
checks passed.

The single four-rank replay completed in **704.962 s**, versus the 2400-s cap,
with no retry. Initialization and exactly seven real steps reached
132230424.76671731 s (4.1901293117 yr), on the identical accepted clock.
Every accepted state has **1236 free, zero prescribed and zero lower-active
nodes**. All **71 fresh-linear checks** pass; total Krylov iterations are
**1360**, identical to the reference. Final normalized bulk/surface residuals
are 1.7842672e-13 and 2.0049684e-13, also unchanged.

The full saved-state comparison passes with **zero absolute/scaled differences**
in every compared V, Theta, slip, C, weak shear, raw current constitutive
pressure/stress/normal/shear traction, localization and strain component.
Stable-ID published particle stresses also have zero difference; H, geometry,
fixed phase/Ih and background agree exactly. All 16 available rank-local bulk
VTU files have identical serialized numerical arrays (including velocity),
ignoring generation timestamps; these are checked at exported Float32 precision,
not used to replace the full-precision stress/history comparison at every step.
The independent physical analyzer passes aging,
slip accumulation and work-row reconstruction; its maximum reconstruction
error remains 7.1182156e-6 Pa m, unchanged.

Peak child RSS is **1,522,124 KiB (1.452 GiB)**, not aggregate MPI memory.
This is a behavior-preservation replay, not a controlled performance comparison.
No historical campaign or longer earthquake-cycle run was performed. The
constrained parameter file was validated but its trajectory was not rerun.

Evidence is in `benchmarks/reconstructed_fault/bp3/fully-frictional-cleanup-local4/`:
`execution.json`, `cleanup_equivalence.json`, `analysis/summary.json`,
`run.log`, `provenance.json`, source/configuration snapshots and copied focused
test/build/validation logs. The original successful evidence remains unchanged.

## Remaining limitations

This remains a fixed, straight 2-D benchmark-specific deployment. The mature
work replay is fresh-start only; a general restart qualification for it is not
added by cleanup. Its input mesh, prestress and completion tables still live
in the preserved experiment directories and are hashed by the launcher.
The seven-step driver is intentionally bounded, not a first-event launcher.
Long-time, mesh and timestep accuracy and the 15–18 km features are not newly
qualified. General noncommitting residual APIs remain available, but archived
experiments require their matching archived implementation.
Only Release binaries were rebuilt for this task. Use the launcher's explicit
`aspect-release`/`libbp3.release.so` pair; the older Debug artifacts are not a
qualification of the cleaned source. No complete integration suite was run.
