# Post-r3 homogeneous-coefficient repair

The user's full-grid 48-rank attempt failed during step-zero mechanics because
fault-point harmonic averages of equal moduli were not bitwise constant across
chemical labels. The same assertion is reproduced in a bounded two-rank fixture.
The material now uses configured G directly throughout homogeneous bulk/fault/
history coefficient evaluation; legacy averaging remains. The exact assertion,
normalization, laws, solver/publication tolerances and four parameter files stay.

Fresh local qualification is in `qualification/assessment.json` and
`verification.json`; the source report is
`doc/reconstructed_fault/particle_free/BP3_DG_HOMOGENEOUS_COEFFICIENT_FIX.md`.
Native tests: **127 cases / 165094 assertions pass**, including 1001 mixtures,
four averaging selections and three constitutive intervals. Old harmonic kappa
spans 1.2815248119788469e17 to 1.2815248119788472e17 Pa s; the repaired homogeneous
coefficient equals the configured value exactly. Earlier optimization/startup
evidence remains under `qualification/inherited-optimization/`.

Fresh mixed-label 1/2/4-rank checks pass: two-rank optimized/reference advancement,
serial-to-four-rank restart, complete one-owner rejection and once-only publication.
Accepted adaptive steps are 4e6/7.64e6 s; source/allowance arrays match bitwise,
frozen B/G/C relative differences <=1.01e-15/4.17e-15/1.71e-15. Rejection/retry
fields match exactly; MPI restart differences are reported rather than called exact.
All four unchanged parameter files validate using the rebuilt ordinary plugins.

The r4 snapshot already contains the fix; the small six-file patch is for r3
source only. Rebuild core/plugins, preserve failed output and rerun bounded
startup in a fresh output directory. The supplied server log has no completed
positive accepted step. All earlier accuracy failures and strict reference
matching=false remain; local repair checks do not qualify 48-rank full-event
performance, physics or the server compiler stack.

---

# Stationary mature DG optimization qualification

17 October 2026; reviewed base `b2dfdebe43437d6028b06d4f9f862cc80a00cc71` plus
preserved incoming work. The full snapshot/manifest identifies this revision.
See `qualification/verification.json`, `assessment.json`, `operator-checks.json`,
`parameter-validation.json`, `regression.json` and the source report
`doc/reconstructed_fault/particle_free/BP3_DG_OPTIMIZATION_REPORT.md`.
Earlier startup evidence is retained under `qualification/inherited-startup/`.

The selected homogeneous fixed-phase mature DG route enables independent exact
cached/sparse evaluation. It keeps unlimited DGQ2, mesh/10 m spacing, Gauss6/source3,
Helmholtz20, pressure scaling, native adaptive controls and all existing numerical
acceptance tolerances. Consistent endpoint phase stays off; the false optimization
selector retains the old algorithms and legacy/default backend behavior is unchanged.

- Final nonoverlapping two-rank comparisons, two positive steps of 200000 s:
  4136 cells, **30.888 → 18.305 s (1.69×)**; 11528 cells,
  **80.940 → 46.709 s (1.73×)**. Krylov counts/timestep sequences match. Peak
  native RSS increases **13.5–36.0 MiB/rank**. Initialization/terminal checkpoint
  are included. These short local strengthening cases do not measure full-grid
  weakening/event performance or simultaneous node memory peaks.
- Rebuilt native suite: **126 cases / 149074 assertions pass**. Actual 5776-node
  search kernel passes, reference .2642 s versus indexed .002393 s plus .001300 s
  setup. Manufactured filtered surface algebra differs by 2.15e-16 relative;
  native frozen operator probes independently verify the actual work measure.
- Source coefficients and allowance arrays match bitwise for all nine DG modes;
  independent native-QP reconstruction difference <=3.09e-12 Pa. Frozen B/G/C
  relative differences <=1.88e-15/1.77e-14/9.46e-16 with hanging constraints,
  physical pressure scaling, variable friction and endpoint/restricted traces.
  Full timing-pair tensor RMS differences <=2.10e-10 Pa; phase/I/mass identical.
- 1/2/4-rank ownership, serial-to-four-rank restart, complete candidate rejection,
  actual outer/nonlinear rejection/cutback and single publication pass. MPI solver
  roundoff differences are documented; they are not claimed bitwise equal.
  A short native adaptive control accepts 4e6 and 7.64e6 s timesteps with the
  unchanged first/max caps, log-state bound and growth/cutback settings.
- The non-root-only collective wall-stop and existing scheduled-output restart
  controls pass again within bounded timeouts. Both continuations match the
  uninterrupted control exactly in all compared fields and particle/bulk audits.
- All four standalone inputs validate with ordinary matching plugins and no
  local test macros. Physical/adaptive equivalence is checked independently.
  The full mesh is not constructed by parser validation.

Full-grid startup through step 5/resume through 8 is **prepared, unexecuted**;
first-event evolution remains an **unqualified experiment**. Checkpoint all_gather
memory and full diagnostic costs remain server measurements. Fixed diagnostic
particles and their independent matching-cell publication check remain. All
original spatial/endpoint temporal failures, surface-state imbalance and
**strict_reference_match_passed=false** are preserved. No server/full event was
launched. Prior package evidence follows without alteration.

---

# Revised server-package qualification

17 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved
incoming changes. `qualification/verification.json`, `regression.json`,
`costs.json`, source selection and the file manifest identify this revision.

- Completed locally: two-rank forced wall stop from **rank 1 only** at accepted
  step 2. Other stops and ordinary heavy/profile intervals were not due; both
  ranks observe stop/heavy/profile=1. Exit is successful, terminal native and
  scheduled fields are written, and checkpoint label is step 2, t=400000 s.
- The step-2 checkpoint continuation to step 5 has **zero difference** against
  the uninterrupted accepted timestep sequence in all 27 DG coefficients/cell,
  physical FE fields, phase/I/Z, rate/state/slip, full reconstructed stress and
  weak loads. Full particle-property and bulk rank audits are byte identical.
  Existing cumulative accepted-publication and per-solve callback checks pass.
- The independent scheduled-output/restart regression is retained: 0/3/5 field
  snapshots in the uninterrupted case, step 5 only after restarting step 3, and
  no spurious step-4 output. Local native tests have a 240 s process-group cap.
- Rebuilt native core: **124 cases / 118034 assertions pass**. The only core edit
  here is a `Resume snapshot` timer; existing equations/checkpoint format are
  unchanged. The inherited independent reference suite remains 55/2625.
- All **four** standalone inputs pass native parser validation using matching
  ordinary BP3 plugins, requested runtime layout and relative library paths.
  Independent parameter-map comparison confirms only resume mode, accepted-step
  bounds, wall budget and checkpoint cadence differ. Native adaptive controls,
  initial constitutive interval, physical end time and all physical settings
  are identical. This does not construct the full mesh.
- The server wrapper is built without local strengthening/wall-stop macros.
  The forced request and traces exist only in the dedicated local test build.
  Dense-export timing and collective scheduling changes do not affect history
  updates or the independent publication check/tolerances.

Measured local fixture: 4136 cells, 149538 native DoFs plus 111672 DG history
DoFs, 66176 stationary points. Five-step control: 100.55 s harness elapsed,
427.12/414.73 MiB peak native RSS on ranks 0/1; wall-stop and continuation:
56.72/47.77 s, peaks 421.72/408.27 and 426.16/421.48 MiB. Step-2 checkpoint:
14.235 MiB; final all-rank dense snapshot: 13.198 MiB. These are local resource
observations, not full-grid estimates. See `RESOURCES.md` for required server
per-rank/node and checkpoint-window measurements and timer interpretations.

Prepared, unexecuted server startup: step 5, then resume to 8; checkpoint every
accepted step, graceful 7200 s. First-event `parameters.prm`/`resume.prm` remain
an **unqualified experiment**, with native adaptive timestep controls. No server
job, full fine-grid mechanical trajectory, limiter sweep or new physics is run.
The inherited 6.598 kPa spatial change, failed endpoint 1% temporal screen,
49.704 Pa surface-state interpolation imbalance and
`strict_reference_match_passed=false` are preserved. Neither DG nor particles
is an accuracy reference. The earlier package evidence follows unchanged.

---

# Previous package verification and limits

17 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus the preserved
incoming worktree. The manifest, complete source snapshot and `source.patch`
identify this distribution; the base revision alone does not identify its source.

- ASPECT source configuration from the assembled distribution passes locally,
  including bundled World Builder and `ASPECT_WITH_VORO=ON`. Every ASPECT core,
  include, CMake and unit-test input matches the already tested native build.
  A new full ASPECT compile was not performed for this source-only packaging task.
  The inherited native suite passes 124 cases / 118034 assertions; independent
  reference suite 55/2625. Rebuild and check on the server stack.
- The new CMake adapter builds ordinary BP3 and its observer, and exposes
  `plugin/aspect-release` and both libraries at the requested relative paths.
  Both standalone fresh/resume inputs pass native parser validation. This does
  not construct the full fine mesh or prove all runtime restrictions.
- Two-rank short fixed-mesh controls: default observer reproduces **36 retained
  field files byte for byte**. Scheduled mode matches **20 corresponding field
  files byte for byte**, emits initial/terminal tensors without intermediate
  step-one/two dumps and writes replicated weak loads on root only.
- A five-step scheduled reference and step-three checkpoint continuation to step
  five have **zero difference** in all compared DG coefficients, physical FE
  fields, phase, I/Z, rate/state/slip, reconstructed stress and weak loads.
  Diagnostic scheduling restores its reference correctly: no spurious step-four
  field dump. Fresh restart-branch per-step CSV headers are present.
- All checks still run every accepted step. The independent native publication
  validator/allowance, physical laws, limiter/filter settings and production
  defaults are unchanged. Native full-geometry MPI first-event conditioning,
  temporal/spatial accuracy and costs have not been qualified by these tests.

The first output-only five-step attempt retained the fixture's old T=600000 s
end cap and therefore ended at step three. That input/log remains evidence;
the corrected reference reaches T=1000000 s and is the restart comparator.
No physical failure or prior report has been overwritten.

Only `coupled_checks.cc` changes runtime C++ in this packaging task: a default-off
schedule reuses the existing BP3 time/slip/event helpers; schedule state is saved
under a separate optional checkpoint key. The default archive format is unchanged.
Per-step CSV header detection now uses file existence for clean restart branches.
No core API, constitutive equation, solver or acceptance tolerance is changed.

The older controls qualify serial/2/4-rank mature ownership, actual rejection/
cutback and restart. The selected full-geometry surface initializer has 5776 nodes,
independent friction discrepancy <=2.23e-16 and a retained **49.704 Pa** interior
state-interpolation imbalance. The fixed-FF mechanical refinement changes local
stress by **6.598 kPa**. Paired halving changes that spatial contrast by .1935 Pa,
but the absolute endpoint 1% temporal screen remains unpassed. Neither DG nor
particles is an accuracy reference. `strict_reference_match_passed=false`.

This is a first-event **experiment**, with physical inputs unchanged and user-
selected native adaptive timestep controls. No full server run, propagation,
phase evolution, transport, moving geometry or automatic AMR was launched here.
