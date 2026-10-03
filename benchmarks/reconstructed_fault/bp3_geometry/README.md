# BP3 runtime-cleanup section 2 review

**Qualified follow-up:** the selected [filter-derivative correction](filter_derivative/README.md) resolves all remaining comparisons (927/927 pass). This report retains its earlier evidence.

**Follow-up:** the separately selected [endpoint correction](endpoint/README.md)
removes the source-admission gap, but one of the seven field comparisons remains
above tolerance. Section 3 remains paused. The report below retains the original
pre-correction evidence.

Implemented the shared geometry description requested by section 2, starting from
accepted frozen-H commit `126049420`. **Qualification is incomplete:** 794 checks
pass and seven fail, all in the 45-degree reversed-order physical comparison.
The separate anticipated profile-roundoff correction is committed as `8df2bc5b5`.
Geometry changes remain reviewable separately. Section 3 has not started.

## Changes and scope

| Consumer | Geometry responsibility after this change |
|---|---|
| `bp3/plugin/geometry.cc` | Native prescribed-fault reader plus live Box bounds; one validated, read-only description of upper/lower intersections, length, physical tangent/normal, dip, peak, horizontal material coordinate and identity |
| `bp3_model.h`, `bp3.cc`, `particle_initialization.cc` | Initial phase/H, loading coordinates, horizontal material fractions and thrust sense use that description; survivor H and Maxwell transfer remain unchanged |
| `monitor.cc`, `bottom_constraint.cc` | Loading direction, endpoint diagnostic windows and bottom tangent use shared geometry; the monitor no longer reparses fault coordinates independently |
| `mesh.cc` | Shared geometry classifies support-intersecting cells for compact mesh statistics; existing leaf replay remains unchanged |
| `output.cc`, `work_audit.cc` | Stations and endpoint windows use physical down-dip coordinates and actual length; official station distances that fit are retained and the lower endpoint is added |

Every consuming plugin configures geometry during its own parameter parsing. The
first call distributes and parses the file; subsequent calls check and reuse the
immutable configuration. Geometry is available before field queries, constraints
and refinement, without relying on postprocessor execution order. The native
manager later parses the same source and its retained anchors are checked against
the early description. No second fault manager, core ownership change, resampling
change or input reordering is introduced.

Admission requires one stationary straight 2D polyline, at least two strictly
ordered collinear vertices, nonzero segments, constant admissible peak phase, and
interior top/bottom intersections. Curved/multiple faults, corners, tangencies,
duplicates/backtracking and incompatible peak values fail explicitly. Physical
orientation points down dip independently of input order. Thrust sense accounts
for the native tangent/normal convention on either dip side.

The material extension remains horizontal, `(y_top-y)/sin(dip)`, with the existing
weakening length and 3 km transition. Production domains must contain the complete
transition. `Postprocess / BP3 / Allow truncated transition = true` is an explicit
small-fixture exception; it does not alter the transition or physical parameters.

**Section boundary:** compiled geometry is replaced here. Removing required
`profile.txt`, `target_cells.txt` and legacy `completion.txt` dependencies belongs
to section 3. The new geometry still requires compatible existing profile/mesh/
completion inputs. The 45-degree coupled checks use the already qualified
`automatic prescribed` path and the same affordable saved mesh/profile. This
pass does not claim a complete fault.txt-only runtime package or a fresh server
input. No full-resolution production run, mesh-policy replacement, R7 or server
job was performed.

## Verification and remaining defect

Release core/plugin builds pass, including required 2D/3D template instantiations;
BP3 geometry itself explicitly admits 2D only. Existing solver tolerances,
constitutive settings, 20 m normal filter, full bottom constraints and native
4×4 / 12–24 policy are unchanged.

- The final comparator records **794 passed / 7 failed** checks and exits 1.
  `results/checks.json` preserves every result, including failures.
- The 60-degree one/two-rank comparisons with accepted frozen-H evidence retain
  identical timestep/active-set/Newton/Krylov/line-search decisions. Maximum
  component/column infinity-norm differences are below `8e-13` for bulk and
  particle fields and `3.2e-14` for fault profiles. The derived 60-degree frame
  differs by one ULP from the old compiled trigonometric constants. These are
  measured differences, not bitwise equivalence claims.
- Rejected/direct smaller steps and two-rank checkpoint/uninterrupted continuation
  match exactly for full particles/bulk/fault/work outputs and accepted summaries.
  The second retry has nonzero incoming Maxwell history and RNG use on both
  ranks. Survivor H remains exact; every newborn matches the shared initializer
  before audit capture, including exterior Hc.
- Geometry/loading checks pass for 60/45 degrees, both input orders, left dip and
  a translated 2×1 km Box. Unsupported-geometry checks also pass. Each rank emits
  a completion marker before the isolated guard stops intentionally. The small
  Box guard is geometry-only, not the section-5 resolved physical smoke.
- Regular-particle 60/45-degree coupled cases and their reversals all reach
  accepted steps 0–2 with fresh residual checks. The 60-degree reversed physical
  comparison passes. The 45-degree reversed physical comparison **fails**.
- Evolving-model H transfer remains native. Same-identity checkpoints resume;
  changed geometry and pre-identity checkpoints fail explicitly. Historical
  checkpoints require their original plugin; committed archives are not converted.

The 45-degree failure is an existing gap between normal-strip projection and
automatic source continuation at endpoint planes. At `(51000,1000)`, the native
resampled first segment gives `xi=-3.6209821701049764e-16`, so
`project_to_normal_profiles_unchecked()` rejects it. The raw endpoint frame gives
`s=0`, so `project_to_bulk_source()` also skips it (`s>=0`). Reversing input order
gives `xi=1` and admits the point. Four positive-phase quadrature points change
association on both one and two ranks. Startup particle rows are identical;
Ih at common active samples agrees within `8e-16` relative. The work-weight
maximum difference is about 6.2% of the maximum row weight. Initial fault-field
maxima differ by `6.20541e-13 m/s` in V, `420.93 Pa` in shear traction and
`761.68 Pa` in normal traction. This is not dismissed as field roundoff.

The relevant core code is unchanged by the geometry pass:
`source/reconstructed_fault/utilities.cc` (strict xi interval admission) and
`source/reconstructed_fault/manager.cc` (automatic endpoint continuation).
`results/endpoint_plane_gap.json`, `endpoint_plane_mpi_comparison.json` and
`order-45-association-differences.json` preserve the concrete reproduction.
**Proposed next bounded task:** make endpoint-plane ownership consistent between
those two operations, with a regression for these points in both input orders
and MPI counts. A scoped roundoff treatment must retain already admitted
associations and existing transverse/support/overlap checks. Do not work around
it by reordering user geometry, changing native resampling or relaxing tests.
This separate numerical correction is not implemented here.

The earlier peak failure and its separate correction are detailed in
[PROFILE_ROUNDOFF.md](PROFILE_ROUNDOFF.md). Its two-rank unit tests pass 38,967
assertions per rank; the independent table error is `6.41154e-14` under the
unchanged `2e-11` gate. No peak, physical parameter, history assertion or solver
tolerance was changed to pass the 45-degree startup.

All simulation/startup/unit runs, including failed/development attempts, total
about 308 s, excluding compilation. Maximum reported child RSS is 344476 KiB,
not aggregate MPI memory. Coupled geometry tests use the existing coarse
functional ell=4000 m fixture (1875 cells and 30000 initial particles), not the
production ell=20 m model. Long-time inflow accuracy, resolved small-model
behavior, production memory and first-event timing remain unqualified.

## Reproduction and evidence

Build with GCC 12.4 and Open MPI 5.0.6 on PATH:

```sh
cmake --build build-refactor-r6b -j2
cmake -S benchmarks/reconstructed_fault/bp3/plugin -B benchmarks/reconstructed_fault/bp3_geometry/build/bp3 -DAspect_DIR=$PWD/build-refactor-r6b
cmake --build benchmarks/reconstructed_fault/bp3_geometry/build/bp3 -j2
cmake -S benchmarks/reconstructed_fault/bp3_geometry/plugin -B benchmarks/reconstructed_fault/bp3_geometry/build/observer -DAspect_DIR=$PWD/build-refactor-r6b
cmake --build benchmarks/reconstructed_fault/bp3_geometry/build/observer -j2
```

The final immutable executable is `build-refactor-r6b/aspect-profile-bounds-qualified`.
Old runs retain `aspect-particle-lifecycle-qualified`; neither old qualified
executable nor previous plugin/output evidence was overwritten. Rebuild the
current source/plugin together; artifact hashes identify the exact tested pair.

`run_cases.py --unit 2` runs the focused unit checks. `run_cases.py --batch`
accepts `CASE:RANKS` pairs. Final fresh cases are `qualified-serial-fixed:1`,
`qualified-mpi:2`, `qualified-retry:1`, `qualified-direct:1`,
`qualified-staggered:2`, `qualified-create:2` and the four `qualified-dip-*`
60/45-degree forward/reverse cases on two ranks. The six `guard-*-qualified`
cases use the ranks recorded in `results/runs.json` and explicit early-stop
markers. One-rank `qualified-dip-45[-reverse]-serial` cases isolate the endpoint
problem from partitioning.

For restart checks, use `prepare_restart.py SOURCE TARGET` to clone
`output-qualified-create` into absent `output-qualified-resume`,
`output-qualified-retry2`, `output-qualified-direct2` and
`output-qualified-changed-resume` directories, then run those cases on two ranks.
The old-identity check clones `frozen_particle_H/output-create` into
`output-qualified-old-resume`. `qualified-evolving:2` checks native H transfer.
Expected rejections/guard exits require explicit markers; arbitrary failure is
not success. Preserve existing output evidence before reproducing; the helper
refuses to overwrite a restart target.

`compare.py` deliberately exits nonzero while the seven 45-degree ordering
comparisons fail. Compact CSVs, metrics, all statuses, exact commands, binary/
source hashes and local-file preservation are under `results/`. Large raw logs,
outputs and builds are ignored. The initial development guard lacked an all-rank
completion barrier; the qualified guard fixes that instrumentation and records
completion before intentional MPI abort. Intermediate compiler/fixture failures
are retained in the run manifest rather than counted as qualification.
