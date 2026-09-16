# Coarse BP3 first-event preparation: corrected smoke/restart pass; launch held

The boundary-panel correction passes its analytic one-/two-rank reproducer.
The actual cell/remote BP3 comparison still misses the unchanged allowance,
so the fixture uses corrected remote integration. The new initialization and
three real steps genuinely converge, and a filesystem restart at accepted
step 1 reproduces steps 2/3 at roundoff level, including bulk/particle history,
surface states, accumulated slip, frozen background and the next timestep.
The accepted-only event observer, station/output stream and checkpoint
configuration are implemented. No first event or server job was run.

One requested condition is **not met**: this is not numerical reproduction of
the old boundary-defective trajectory. The corrected I_h changes the derived
initial shear background and the controller's accepted times. Supplied initial
Theta/C and physical inputs are unchanged. Do not undo the correction or call
these differences roundoff. The prepared launcher fails closed pending review
of this corrected coarse baseline; no solver/physics tolerance was changed.

The current task preserves the physical/model configuration of the accepted
`performance/interface_zero_bp3` trajectory: 100 km box, 60-degree through-going
open fault, ell=400 m, level-10 crossed cells (97.65625 m), stress-change bulk
history and pressure, once-initialized fixed background tractions, horizontal
VW/VS and initial-Theta extension, official state/friction, deep Vp, Vmin=1e-20.
No Airy field, fine pilot, GMG or few-mode correction enters this fixture.

The qualified pivoted inverse and explicit B are selected. G remains the
independent original action because its marginal timed gain was not established
in a native sparse-only comparison. No reference comparisons or profiling
callbacks belong in the server environment.

## Approved boundary correction and backend decision

The remote profile state retains the found physical boundary coordinate
`boundary_low` independently of adaptive width. Halving a panel does not mark
the remaining interval complete. Accepted subpanels continue, with widths
clamped to the known remaining physical interval, until that coordinate is
reached. No tolerance, tail criterion, support or equation changes.

The analytic boundary reproducer passes at relative error 3.17e-14. Actual
BP3 comparison `first_cycle_ih_check` removes the large endpoint omission but
still finds a maximum projected nodal remote/cell relative difference
5.237661365509183e-6, above the established quadrature+tail allowance 2e-10.
The comparison intentionally fails before mechanics and its outputs are
preserved. Therefore **use corrected remote points**, not the cell backend.
The remaining smaller interior discrepancy is not resolved or hidden here.

## Persistent state and output-only interfaces

The manager already serializes geometry, properties (including the frozen
background data), committed V, Theta, C and previous I_h. The BP3 postprocessor
now serializes slip, previous Theta used by its independent history audit,
last accepted step, event state and output scheduling. On resume the plugin
reselects the manager background property before mechanics and reapplies the
deep prescribed-rate map. It never recalibrates the background or initializes
history again. Older fresh-start-only benchmark checkpoints are explicitly
incompatible. Generic caches and all linearization matrices/factors rebuild.

The solver emits accepted-only observer data: Newton updates, total Krylov
iterations including fresh-residual restarts, minimum accepted alpha and the
final active/prescribed mask. It does not emit a successful summary on failure.
Surface weak diagnostics retain total normal-traction moments and actual
constitutive sample extrema from the pre-publication evaluation. No Maxwell
update is reevaluated after history commit to create those outputs.

A post-checkpoint notification is emitted only after the complete checkpoint
and last-good marker exist. BP3 can archive event milestones from those files
without changing any physical or solver state. Every-step checkpoints retain
three rolling slots; milestones preserve before onset, onset, highest sampled
peak, latest down-crossing, and termination. This costs checkpoint I/O each
accepted step deliberately to ensure a genuine pre-onset full state, not just
a late visualization. Archive failures are reported; the original complete
checkpoint is retained.

## Accepted-state event semantics

The initial observer is interseismic. At physical max(V)>=1e-3 m/s it records
onset. After onset, five consecutive accepted states with max(V)<1e-3 terminate.
Any intervening value >=1e-3 resets the streak (including equality). Onset,
sampled peak and its down-dip coordinate/time, down-crossing and termination
are persisted. The maximum is over all Q1 vertices, including prescribed rows,
so it is the actual maximum of the represented physical rate. Free/lower
counts separately exclude deep prescribed rows. A standard independent end
time of 1500 Julian years remains a safety stop, never event success.

The event plugin observes only; it neither constrains output times nor changes
the timestep. Every accepted state appends `accepted_steps.csv` and the twelve
official station rows in `stations.csv`. Tractions are projected from actual
surface weak moments, not bulk-column averages. Total normal extrema are
unprojected constitutive extrema. Profiles are normally every 0.1 yr, then
every accepted step at max(V)>=1e-5 m/s and all event milestones. Bulk VTU is
normally every year, every accepted coseismic state and all milestones.
The minimum event snapshots are complete archived checkpoints, not merely VTU.
Bulk FE stress arrays are published old-history **perturbations**; accepted
current particle stresses reside in checkpoints and audit particle output.

## Bounded verification results

All code was built with `-j4`. Final executable SHA256:
`41e923c4eb5c2476315af5533355b8830d07e06009aa82545444481e86b70b48`.
BP3 Release plugin SHA256:
`cad4ed109f5cc6a75af3bcb44a1c5fc1a4e05419b7ff3fad24cde988bc836242`.

| Check | Result |
|---|---|
| `[ih_boundary_reproducer],Stage-I*`, one and two ranks | 91 assertions in 12 cases per rank pass; analytic boundary relative error 3.17354e-14 |
| Actual BP3 corrected remote/cell comparison | Qualification fails at 5.23766e-6 maximum projected nodal relative difference versus 2e-10; corrected remote selected |
| Endpoint profiles / endpoint vertices in that comparison | 2.67121e-7 / 3.90529e-8 maximum relative differences; large old endpoint omission is removed, not a claim that the cell comparison passes |
| Interior profiles | Maximum relative difference 1.20481e-5; no further integration redesign performed |
| Seeded random/basis, derivative, condensed action/recovery checks | One/two ranks pass, 21.73/16.06 s; new normal-traction outputs do not change the actions |
| Standalone first-event observer | Onset, equality/re-entry reset, five-state termination and retained peak checks pass |
| New server `.prm` syntax | `--validate` passes |
| Initialization + 3 real steps | Pass, 530.742 s, 4,509,424 KiB peak RSS, 502 total Krylov iterations |
| Resume from accepted step 1 through step 3 | Pass, 391.256 s, 4,493,096 KiB peak RSS, 398 total Krylov iterations |
| Full numerical restart comparison | Pass at 1e-8 unchanged nonlinear coefficient; actual differences much smaller below |
| Old-trajectory numerical equivalence | **Not passed** after mandatory boundary correction; quantified separately below |

### Genuine convergence, not just exit zero

| Accepted step | Time, s | dt, s | Final relative bulk | Final relative surface | Newton updates / Krylov | Minimum accepted alpha |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 5.06189e-15 | 4.85076e-11 | 2 / 51 | 1 |
| 1 | 2618120.21649 | 2618120.21649 | 5.16794e-13 | 8.27559e-11 | 2 / 53 | 1 |
| 2 | 5232101.20506 | 2613980.98857 | 3.00605e-13 | 2.91852e-9 | 12 / 219 | .11859803 |
| 3 | 10224804.89322 | 4992703.68817 | 1.50724e-13 | 6.66721e-14 | 9 / 179 | .85887588 |

All 29 returned directions pass the fresh residual check. All final masks
have 400 free RSF nodes and zero lower-active nodes (deep prescribed rows are
excluded); intermediate bound-contact directions remain handled by the
unchanged active-set solver. Initial max V is 1.01854248e-9 m/s. Actual normal
traction ranges are 49.85259--50.07540 MPa at initialization and
49.65634--50.20524 MPa at step 3. Deep Vp is exact, and deep accumulated slip
differs from Vp*time by only 1.735e-18 m. The step-3 independent Theta-update
relative error is 9.99201e-16. The physical end time is about 0.324 yr; the 4e6 s
Maxwell initialization interval is not counted as physical elapsed time.

Every accepted state has twelve station records and a valid coarse bulk PVTU
with existing rank pieces. The observer correctly remains interseismic. The
short wrappers enable every-step audit exports; the server file disables them
and uses its adaptive output schedule. Real-event milestone archival has not
been exercised by a seismic trajectory, which is outside this task.

### Filesystem restart, including actual working history

The completed step-1 `restart/02` directory was copied byte-for-byte to
`first_cycle_checkpoint/restart/02`; its last-good marker is 2. This reuses the
continuous run instead of repeating initialization. `first_cycle_resume.prm`
then enters the actual Simulator filesystem resume path at step 2. The explicit
checkpoint wrapper is retained for future reproduction but was not separately
run. No old/current histories were reset from output.

The comparison requires identical stable particle IDs and bulk DoF/component
indices; each velocity/pressure/composition component is tested separately.
Thus pressure magnitude cannot mask velocity or history differences. It
compares every particle property, including retained/committed stress and H,
and the actual published FE history consumed by subsequent mechanics.

Across steps 2/3, maximum relative differences are 1.90891e-14 for bulk
components, 1.43814e-14 for particle columns, and 1.09494e-15 for the compared
surface fields. Geometry and both background arrays are bitwise identical.
Times and dt are exactly identical, including the next timestep chosen from
the restored state. Maximum absolute differences include V=2.068e-25 m/s,
Theta=1.863e-9 s, C=5.821e-10 Pa, I_h=3.638e-12 m and slip=1.735e-18 m.
No stale background selector, extra history lag or slip reset was observed.
The event/output bookkeeping is serialized; the synthetic observer test covers
its nonzero state transitions, while the physical restart test remains aseismic.

### Old baseline change: do not silently redefine equivalence

Comparison uses the latest successful `performance/interface_zero_bp3` output,
not an old Airy or fine-pilot snapshot. At initialization the maximum I_h
change is 726.763 m (5.71655% of the old global maximum), initial shear
background changes by 1964.41 Pa, and V changes by 4.26311e-12 m/s (0.419594%
of the old maximum). Supplied Theta0 and retained C0 are exactly unchanged.
The background changes because its approved discrete initialization contains
C_eval,0=kappa*Vinit/I_h+beta*C0; retaining the defective old background would
not preserve that approved initialization equation.

| Quantity | Step-3 maximum absolute change | Relative to old global maximum |
|---|---:|---:|
| Physical time | 15370.097 s | .150096% |
| V | 3.35277e-13 m/s | .0335277% |
| Theta | 14554.604 s | .147639% |
| C | 4948.128 Pa | .901127% |
| Total shear q | 10068.192 Pa | .0369905% |
| Accumulated slip | 1.53701e-5 m | .150096% |

These are matched accepted-index differences, **not common-time errors**;
the unchanged controller responds to the changed initialization. Histories
were never reset. None of these differences is called numerical equivalence
under the unchanged 1e-8 comparison coefficient. Review acceptance of the
boundary-corrected coarse baseline before enabling the first-event launcher.
This is an initialization/reference correction, not a proposal to change
loading, support, constitutive laws, solver tolerances or event thresholds.

## Commands and artifacts

Run commands below describe completed bounded tests, not instructions to launch
another case. Paths are relative to the repository root.

```sh
build-pf-cpdi/aspect-release --test '[ih_boundary_reproducer],Stage-I*'
mpirun -np 2 build-pf-cpdi/aspect-release --test '[ih_boundary_reproducer],Stage-I*'
python3 benchmarks/reconstructed_fault/performance/run.py benchmarks/reconstructed_fault/bp3/first_cycle_ih_check.prm --cap 240 --env ASPECT_IH_COMPARE_CELL=1
python3 benchmarks/reconstructed_fault/performance/run.py benchmarks/reconstructed_fault/bp3/first_cycle_verified.prm --cap 900 --no-performance --env ASPECT_FAULT_EXPLICIT_B=1 --env ASPECT_FAULT_SURFACE_SOLVER=tridiagonal
python3 benchmarks/reconstructed_fault/performance/run.py benchmarks/reconstructed_fault/bp3/first_cycle_resume.prm --cap 600 --no-performance --env ASPECT_FAULT_EXPLICIT_B=1 --env ASPECT_FAULT_SURFACE_SOLVER=tridiagonal
python3 benchmarks/reconstructed_fault/bp3/check_first_cycle_restart.py benchmarks/reconstructed_fault/bp3/first_cycle_verified benchmarks/reconstructed_fault/bp3/first_cycle_checkpoint --continuous-log benchmarks/reconstructed_fault/bp3/first_cycle_verified.log --resume-log benchmarks/reconstructed_fault/bp3/first_cycle_resume.log --output benchmarks/reconstructed_fault/bp3/first_cycle_restart_comparison.json
```

The preserved first launcher attempt `first_cycle_continuous` used the invalid
environment spelling `ASPECT_FAULT_SURFACE_SOLVER=pivot` and failed backend
validation before mechanics (59.45 s). The correct supported spelling is
`tridiagonal`. This was a launcher correction, not a numerical retry or solver
retuning. The actual failed backend comparison is also preserved (45.15 s).
Every runner records status, environment, binary/input hashes, wall and RSS.

Deliverables are `bp3_first_cycle_coarse.prm`, the updated restartable `bp3.cc`
and `first_event.h`, `stampede3_first_cycle.sh`, `README_first_cycle.md`,
`first_cycle_readiness.json`, and the short test wrappers/checkers under
`benchmarks/reconstructed_fault/bp3/`. The launch script was prepared only after
filesystem restart equivalence passed; `bash -n` passes. It is fail-closed while
`server_ready=false`. No Slurm job, long event run, fine pilot, GMG work, or
additional solver/integration redesign has been launched or implemented.
