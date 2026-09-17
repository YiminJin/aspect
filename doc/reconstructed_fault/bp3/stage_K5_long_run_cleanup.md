# Modified BP3: clean 300-km long-run configuration

The requested cleanup and bounded qualification are complete. No long run
was launched and no commit was made. The physical domain is deliberately
wider; constitutive equations, history timing, support, quadrature, solver
tolerances and residual acceptance checks are unchanged. This remains a
coarse modified-BP3 exploratory fixture, not an accuracy-qualified seismic cycle.

## Maintained inputs and geometry

- `benchmarks/reconstructed_fault/bp3/bp3_modified_long_run.prm`: one complete
  PRM without includes, abandoned replay selectors or inactive refinement rules.
- `bp3.cc`: mature fully frictional continuous-Q1 model, fixed background,
  work measure, paired endpoint corrections and split exact aging. Old Airy,
  prescribed-deep, uniform-sliding and noncommitting diagnostic branches are
  absent from its normal include/execution path.
- `run_long.py`: prepare-only by default, default GMG velocity preconditioner
  with its hierarchy enabled; AMG remains selectable. It emits a standalone
  resolved PRM and records input/binary hashes. No testing/profiling callbacks
  are enabled by the production launcher.
- `fixtures/modified_bp3_long_run_300km/`: maintained mesh, fault, prestress,
  completion and provenance. `prepare_300km_fixture.py` regenerates these inputs.

The box is [-100,200] x [0,100] km, centered at (50,50) km. Its current fault
still dips **60 degrees**, with the same physical midpoint, ell=400 m,
1156 free vertices and 97.65625-m fault-crossed spacing. The actual exported
mesh has 36,282 cells, 88 more than the ordinary 200-km long-run fixture.
Square side lengths are at most 12.5 km (the corresponding square diagonal
is 17.678 km). Eight old 25-km squares were subdivided; other old physical
cells and near-fault resolution were retained. Fault/prestress/completion
files are byte-identical to the previous ordinary fixture.

This prepares room for a later 30-degree model; it does not silently change
the dip. A 30-degree case needs consistent new geometry-dependent inputs.

## Logging and visualization

Removed repeated preparation begin/end chatter. Coarse timer-summary scopes
remain. Sparse B/G size messages require the existing performance opt-in.
Default coupled logs retain aligned nonlinear bulk/surface convergence,
line-search decisions and independently checked fresh linear residuals:

```text
      Fault linear solve: iterations=17, fresh=5.705870e-01, target=1.094592e+00
      Relative nonlinear residuals (bulk, fault) after nonlinear iteration  0: ...
```

The old detailed linear/nonlinear diagnostics remain opt-in through
`ASPECT_FAULT_NONLINEAR_DIAGNOSTIC`. Suppressing their printing does not
disable fresh residual, compatibility, constraint, exhaustion or rollback checks.
The design/specification now distinguish required checks from optional verbosity.

Native visualization enables `Write higher order output = true` with output
interpolation. Actual exported bulk VTUs contain nine-node
`VTK_LAGRANGE_QUADRILATERAL` cells (type 70), not just a changed parameter.
This does not change FE spaces: velocity is Q2, while pressure/phase retain
their existing spaces. Particle output remains points and fault output Q1.
Initial/step-2 heavy payloads were approximately 8.4/16.4 MB; a checkpoint
was approximately 104.9 MB. These short-run sizes are not long-run storage bounds.

## Preservation and file responsibilities

`reference_200km/` preserves the pre-cleanup plugin, headers, PRM and launcher.
Optional CMake target `bp3_research` builds that research plugin separately;
`run_research.py` explicitly selects it. Removed root legacy headers remain
recoverable there. Analytical Airy and historical termination tests use the
preserved headers, not the clean production include graph. The copied old
PRM/launcher are provenance, not automatically relocatable run inputs.

Core changes in this pass are log-only in material/manager preparation,
surface-system and Stokes B/G reporting, the coupled solver and frozen phase
announcement. Required timers and every numerical check remain. The working
tree also contains earlier approved long-run, GMG and investigation changes;
those unrelated changes were preserved, not reset or silently included in a commit.

## Verification

All simulation results below used four MPI ranks, Release and unchanged
GMG/coupled acceptance criteria. Evidence root:
`benchmarks/reconstructed_fault/bp3/long-run-cleanup/`.

| Check | Result | Elapsed | Peak child RSS |
|---|---|---:|---:|
| `unchanged-200km`, initialization + step 1 | Bitwise agreement with preserved ordinary-200-km reference | 58.241 s | 1,342,004 KiB |
| `clean-300km`, initialization + steps 1–2 | Genuine convergence and all history/physical guards pass | 146.730 s | 1,342,900 KiB |
| `resumed-300km`, checkpoint 02 through step 2 | Bitwise agreement with uninterrupted step 2 | 104.940 s | 1,380,168 KiB |

RSS is the launcher child-process measurement, not a summed four-rank memory
measurement. No claim of bitwise agreement is made between different widths.
The same-width cleanup comparison separates cleanup from the deliberate
physical boundary/far-field-mesh change.

The 300-km accepted times are 0, 2666075.0589767243 and 5332320.5409856578 s.
Newton/Krylov counts are (1,37), (1,34), (13,225). Maximum normalized
bulk/surface residuals are 3.2579632e-9, 3.9128371e-10 and 5.7715170e-10.
Surface strong-RMS residuals are 0.00451649, 0.000181055 and 0.000905540 Pa.
All 1156 nodes remain free; no prescribed or lower-active nodes. The final
independent Theta relative error is 2.2204460e-16. Final total normal stress
ranges from 49,983,543.476 to 50,015,530.495 Pa.

Restart comparison includes velocity/pressure, particle stress/history,
current constitutive stress/strain, V/Theta/slip/C/Ih, background, endpoint
source and weak loads, accepted timestep and native output indices. See
`resumed-300km/long_run_equivalence.json`; unchanged-mesh results are in
`unchanged-200km/long_run_equivalence.json`. Actual mesh and high-order cells
were checked with `verify_long_mesh_output.py`; both 300-km directories contain
`mesh_output_verification.json`.

Reproducible bounded commands (choose unused output directories):

```sh
python3 benchmarks/reconstructed_fault/bp3/run_long.py --output <fresh> \
  --purpose recurrence --end-years 1500 --wall-hours 0.5 \
  --verify-through-step 2 --verify-output --execute
python3 benchmarks/reconstructed_fault/bp3/run_long.py --output <resumed> \
  --resume-from <fresh> --checkpoint 2 --purpose recurrence \
  --end-years 1500 --wall-hours 0.5 --verify-through-step 2 --verify-output --execute
python3 benchmarks/reconstructed_fault/bp3/check_long_run.py <fresh> <resumed> \
  --steps 2 --log resume.log
```

Core and both maintained/reference plugins built with `-j4`. Additional checks:

- Two-rank `aspect-release --test 'Stage-I*'`: 90 assertions in 11 cases
  passed on each rank (`/tmp/bp3-clean-stage-i.log`).
- `ctest --test-dir build-pf-cpdi/tests --output-on-failure -R
  '^phase_field_fault_linear_exhaustion$' -j1`: passed, 26.84 s.
- Standalone Theta audit: five near-bound cases, deliberately wrong Theta
  rejection and zero interval passed; output-schedule test passed.
- Preserved analytical Airy and replay-stop tests passed. Python syntax and
  scoped whitespace checks passed. No broad suite or historical campaign run.

Qualified executable SHA256:
`517fa444414efd6237646dce97f036637845cbaa532648ec472c4c9fe87e98b8`.
Qualified maintained plugin SHA256:
`645d09b90e1ed48f775191a531b21d8c969ebdedf1e66f27c197fef8a0068407`.
Source base is `3335d3d26c298ff5aaeba77062b0a77c8d20f0b5` plus the recorded
working diff in each launch directory; there is no new commit hash.

## Remaining scope

Use `benchmarks/reconstructed_fault/bp3/LONG_RUN.md` for preparation,
checkpoint branches, graceful termination and output meanings. No long
trajectory, first-event correctness, 30-degree case, cross-rank restart or
spatial accuracy was tested here. Historical stdout goldens were not broadly
refreshed for the intentionally reduced verbosity. No further simulation is
needed for this cleanup handoff.
