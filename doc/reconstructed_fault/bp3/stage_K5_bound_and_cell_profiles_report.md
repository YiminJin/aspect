# K5 bound contact and generic cell-profile prototype

## Decision summary

The bound bug is closed. Absolute contact values now survive evaluation,
acceptance and publication, including exact surface endpoints. The unchanged
coarse BP3 case with Vmin=1e-20 completes initialization and three real steps;
all nonlinear and fresh-linear checks pass. No equation, bound tolerance,
Armijo budget, support, history rule or acceptance criterion changed.

Cold profiling identified repeated geometric searches and eager diagnostic
formatting, not degradation evaluation, as the largest avoidable costs. Those
invariants are reused without changing results. The alternate cell-owned ray
backend passes the evolving K3 comparison: required property preparation falls
from 31.293 to 0.301 s, with no completed-value hits, and wall time from 111.18
to 80.62 s.

**Stop for review:** the long BP3 comparison exposes a legacy boundary-panel
omission, independently reproduced, plus smaller unresolved interior errors.
The new backend is opt-in; remote points remain the default. The legacy fix
is proposed below, not implemented. No fine pilot or propagation was run.

## 1. Bound-contact invariant and coarse BP3

`reconstructed_fault_trial_value` uses the existing contact fraction and emits
exactly Vmin at contact. Every other entry uses the ordinary affine trial.
`ReconstructedFaultManager::set_slip_rate_trial_values` validates the complete
absolute vector, including prescribed values, before replacing trial state.
The same values enter residual evaluation and acceptance: there is no
base-relative subtraction followed by addition. Existing incremental APIs
remain available to their other callers.

The first replay exposed another instance of the same cancellation at an exact
surface endpoint. For left=4.879984001041503e-13, right=1e-20 and xi=1,
`left+(right-left)` produced 9.999999985577813e-21. Exact endpoints now return
the corresponding stored nodal value in manager interpolation and the surface
constitutive input. Interior interpolation is unchanged. The failed replay
is preserved as `bp3/bound_corrected_smoke`; its follow-up JSON finds no
infeasible nodal candidate. This is not a change to fraction-to-boundary or
active-set mathematics.

`bp3/bound_tip_smoke` is the successful replay, with the same low-floor fixture,
physical parameters and solver settings. Wall time is 573.463 s, peak RSS
4,475,552 KiB (4.268 GiB). All 29 returned directions pass fresh residual
checks (502 total linear iterations). Backgrounds and geometry stay fixed;
deep prescribed V is exact; retained initial Theta/stress and subsequent split
updates pass. Maximum independent Theta-update relative error is 8.88e-16.

| State | Accepted physical time (s) | Final bulk relative residual | Final surface residual (Pa) | Surface relative residual | Free V/Vinit range |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 4.74384e-15 | 2.29601e-5 | 2.09885e-11 | 0.987149--1.016008 |
| 1 | 2,624,650.7189 | 5.08064e-13 | 9.76405e-5 | 9.17354e-11 | 1.000144--1.018971 |
| 2 | 5,241,669.3687 | 2.79324e-13 | 7.71601e-3 | 2.90103e-9 | 2.12959e-4--0.997805 |
| 3 | 10,240,174.9898 | 1.49787e-13 | 4.59582e-8 | 4.22086e-14 | 6.69108e-5--0.987589 |

All accepted states have 400 free and zero lower-active nodes; contact during
Newton is temporary, not an accepted artificial floor. The per-iteration CSV
and `cache_audit.json` retain Fmin, free/active counts, fraction limits, accepted
alphas and separate residuals. `perturbation_report.json` retains dimensional
bulk residuals, actual total normal traction, pressure and histories.

Existing coarse spatial limitations are **not certified away**: initial
maximum omitted fraction is 1.49449e-4; maximum sampled column normalization
error is 6.526% at the tips. Global Q3 normalization errors over states 0--3
are 1.51448e-4, 1.51822e-4, 7.54018e-5, 7.59088e-5; Q8 diagnostics are
2.07420e-4, 2.08196e-4, 1.08499e-4, 1.09192e-4. No width/normalization change
was made. Successful three-step mechanics is not a fine BP3 accuracy gate.

## 2. Reconciled cold cost and safe invariant reuse

Opt-in per-operation timings in the same coarse BP3 preparation explain the
previous opaque material/geometry scope. Seconds below exclude overlapping
parent timer scopes; rounding accounts for the sub-millisecond remainder.

| Cold preparation component | Eager reference work | Invariant reuse, same remote integrator |
|---|---:|---:|
| Another-fault/support geometry | 33.58864 | 0.30972 |
| Coordinate/diagnostic context | 32.32734 | 0.32110 |
| Phase admissibility | 0.36706 | 0.31516 |
| Degradation/material mixture | 0.80820 | 0.64122 |
| Integrand guards | 0.45169 | 0.38689 |
| FE sampling/lookup | 18.76622 | 18.06530 |
| Adaptive/request overhead | 1.20824 | 0.98021 |
| Adaptive MPI | 0.00450 | 0.00440 |
| Surface composition projection | 1.56012 | 1.34832 |
| Key copies/comparison | 0.00023 | 0.00021 |
| Profile geometry/mixtures | 0.00066 | 0.00065 |
| Final projection/reduction/publication | 0.00015 | 0.00021 |
| **Complete I_h preparation** | **89.08306** | **22.37342** |

Key MPI is below 1 microsecond. With one fault, an overlap with another fault
is impossible: bypass that repeated search, retaining the multi-fault check.
Format detailed point context only for a new raw minimum or a diagnostic
failure, rather than at every admissible quadrature evaluation. Error messages
and physical validation remain. `ASPECT_IH_BASELINE_GUARDS=1` preserves the
eager reference work for comparisons; profiling remains opt-in. This measured
3.98x cold improvement is distinct from the earlier completed-value cache.

The frozen BP3 cache hits now cost 1.351--1.416 s (surface preparation remains),
with zero integration requests. Five calls including the cold one total 27.95 s;
whole property preparation is about 28.3 s, 5% of this 573 s run. **These frozen
hits are not counted as generic/evolving acceleration.** Particle projection is
timed separately; batching is not implemented in this pass.

## 3. Cell-owned ray backend and evolving results

The implementation contract is in `stage_K5_cell_profile_addendum.md`.
`Material model/Phase field fault/I h integration backend = cell intervals`
selects the alternate path. Supported geometry is 2-D axis-aligned affine Box
cells with exact Cartesian/Q1/degree-one Q mappings and no mesh deformation.
Inclined/polyline fault profiles keep their actual segment normals. Other
configurations use the remote backend.

Locally owned cell intersections cache only geometry and Q1 DoF indices. Phase
changes resample current values without locating adaptive points. MPI sums are
by profile identity; independent coverage checks reject gaps/double ownership.
Cell boundaries and Q1 ray zeros split the quadrature, the latter resolving
the max(phi,0) kink. Four/eight-point refinement and existing tail coefficients
retain accuracy; no support clipping or analytical I_h substitution occurs.
Consistent Q1 projection is unchanged. Geometry keys reuse unchanged profiles;
mesh changes conservatively clear all traversals. Selective AMR and propagation
scaling remain unmeasured, not claimed completed.

Primary performance evidence uses the corrected periodic K3 32x128/fault32,
dt=0.375 s trajectory through 3 s: nine genuinely different phase states.
The only harness repair admits the now-explicit, unchanged geometry default
and recognizes the existing common-dt fixture/reference paths. Numerical
guards are unchanged; all nine pass. Maximum phase change is 2.31516e-5 and
I_h change 0.00561957, so completed-value hits cannot explain the result.

| Measure | Remote/eager baseline | Cell backend |
|---|---:|---:|
| Cold first I_h preparation (s) | 5.682 | 0.02425 |
| Subsequent changing-phase I_h (s/call) | 3.171--3.203 | 0.02203--0.02285 |
| Nine I_h preparations (s) | 31.195 | 0.20246 |
| All required property preparation (s) | 31.293 | 0.30094 |
| End-to-end wall (s) | 111.1786 | 80.6189 |
| Peak RSS (KiB) | 728,024 | 504,544 |
| Completed-value hits | 0 | 0 |

Property-only acceleration is 104x; conservatively including the separate
manager geometry cache is about 33x. Cell preparation is 0.37% of total, or
about 1.2% including manager/QP caches. End-to-end improvement is 27.5%.
Timings include identical opt-in instrumentation on both paths; a separate
both-backend validation run is not counted as a performance run.

First construction has 96 profiles, 12,288 intervals/candidates and 884,736
bytes of interval records. Later states rebuild zero profiles, reuse all 96,
and do zero tree queries/remote requests; each evaluates 155,520 current FE
samples over four windows. This memory cost replaces larger point-lookup
storage: measured peak RSS drops from 711 to 493 MiB.

All projected nodal I_h comparisons over nine states are within about
8.13e-11 relative (unchanged allowance 2e-8). Maximum accepted-field absolute
differences are V=1.84e-13 m/s, Theta=5.17e-9 s, C=1.65e-10 Pa,
I_h=8.79e-9 m, phi=2.92e-16 and H=3.19e-12 Pa.
Saved K2 phase/actual fault coordinates at t=0.5 s give maximum nodal I_h
difference 8.129e-11. This latter frozen-data test restores its working state
and does not purport to replay K2 mechanics or histories.

## 4. BP3 numerical comparison: explicit unresolved blocker

`bp3/cell_initialization` compares both paths on the same actual coarse BP3
phase and geometry before mechanics. The comparison fails, correctly, at
maximum nodal relative difference 0.113897. Endpoint I_h changes from about
5968 to 6647 m; the difference cannot be accepted as roundoff.

The prototype constructs 3465 profiles, 266,680 intervals (19,200,960 bytes),
testing 52,154,764 bounding-box candidates. It uses 2,721,156 FE samples, five
windows, zero remote requests. Geometry costs 0.506 s and complete cell
integration 0.896 s, versus 21.015 s for the optimized remote reference;
surface projection is 1.354 s. These timings are promising but **not a
validated BP3 speedup**. RSS before the comparison's remote work is 1,702,904
KiB and after it 3,452,348 KiB; this is not a cell-only full-trajectory peak.

A subsequent read-only boundary audit confirms the legacy state-machine
defect. Boundary search sets a final panel ending at the known in-domain
boundary. If its quadrature estimate fails, halving reduces its width but
leaves `boundary_final_panel` set. The next accepted smaller panel marks the
whole side complete, omitting the remainder. Sixteen actual profiles show
discarded lengths of 33.75--59.11 m. For example profile 1 integrates only
through zeta=37.77929 m although its found boundary is 86.56819 m; its I_h
shortfall against the cell backend is 1468.954 m. Consistent Q1 projection
spreads the endpoint error across neighboring nodes.

The independent hidden kernel reproducer integrates h=exp(20|zeta|) on
[-0.073,0.073], with 1e-12 quadrature/tail coefficients. The old path gives
0.0440514008 instead of 0.3305959528: an 86.675% error on one and two ranks.
It discards 0.05475 on each side after the accepted first 0.01825 panel.
This test intentionally remains failing under `[.ih_boundary_reproducer]`,
excluded from ordinary Catch selection, as a recoverable proposed-fix test.

**Smallest correction proposal, not implemented:** retain the discovered
boundary coordinate independently of the adaptive panel width. After accepting
a subpanel, integrate the remaining in-domain distance; cap subsequent panels
at that coordinate and terminate only when it is reached. Retain all current
quadrature/tail coefficients, lookup semantics and other panel rules. Then
require the analytic test, short boundary profiles and actual BP3 profile/nodal
comparison to pass before rerunning affected initialization.

This explains the dominant endpoint error, **not every difference**. Outside
those 16 profiles the current BP3 per-profile maximum relative discrepancy is
1.205e-5, still above the allowance and not yet assigned to remote error versus
cell quadrature error. A tighter *diagnostic reference*, not a production
tolerance change, will be needed if it persists after the boundary repair.
Corrected I_h may change initial cohesive/background projection near BP3 tips;
that consequence must be recorded rather than reset or hidden. No such fix or
new BP3 dynamics was attempted with the cell backend.

## 5. Focused tests and recoverable evidence

Paths below are under `benchmarks/reconstructed_fault/`. Each `.prm` runner
records exact executable command, source hashes, wall cap/status, elapsed time
and RSS in its adjacent `.resources.json`; full logs are retained. Core and
plugins were built in Release with `-j4`.

| Check | Result |
|---|---|
| `bp3/bound-tip-unit-one.log`, `bound-tip-unit-two.log` | 109 assertions / 17 cases pass on each rank; captured contact, endpoint, infeasible input, release, rejection/exhaustion, evaluated/accepted equality, publication/rollback |
| `performance/cell_stage_i` | Pass, 18.31 s; actual final bulk relative 5.092e-8 meets unchanged 1e-6, surface zero; positive convergence required |
| `performance/cell_rollback_one`, `_two` | Pass, 6.63/6.98 s; intentional nonlinear failure after accepted update verifies bulk/current and committed V/particle/surface histories restored; fixture failure strategy allows process exit 0 |
| `bp3/cell_actions_one` | Pass, 8.24 s; surface/bulk coupling and background derivatives |
| `bp3/cell_cache_roots_one` | Pass, 8.03 s; repeated exact hits, phase/geometry invalidation, singular failure, restart invalidator |
| `bp3/cell_cache_roots_two` | Preserved comparison failure 2.032e-8 vs 2e-8 after shifted geometry |
| `bp3/cell_cache_roots_reference_two` | Pass, 9.39 s; production tolerances unchanged, diagnostic remote coefficients 100x tighter; maximum nodal difference 5.849e-11 |
| `performance/k2_saved_ready` | Pass, 8.08 s; actual saved Q1 phase/fault data, 8.129e-11 nodal comparison; restored state |
| `performance/k3_compare` | Pass, all nine evolving states/guards, both backends evaluated independently |
| `performance/k3_remote_checked`, `k3_cell` | Both pass all accepted-state guards; speed/memory/trajectory comparison in `k3-cell-result.json` |
| `bp3/bound_tip_smoke` | Pass initialization + three real steps; 573.46 s |
| `bp3/cell_initialization`, `cell_boundary_audit` | Numerical comparison failures retained, 43.02/51.65 s, before mechanics |
| `performance/boundary-reproducer-one.log`, `-two.log` | Expected hidden test failure; confirms unfixed legacy boundary omission |

Earlier cell quadrature attempts missed a narrow positive Q1 sliver: root
splitting fixes that without weakening the comparison. The two-rank near-limit
failure is explicitly retained rather than refreshing an expected output.
The checkpoint check invokes the installed invalidator; it is **not** a new
full checkpoint write/read or adaptive-mesh test. Existing passed evidence
for the accepted value cache/frozen localization is retained. No full suite,
fine BP3, propagation or large convergence campaign was run.

The saved K2 harness's initial configurations failed three setup checks
(missing prescribed-Stokes plugin, missing particle postprocessor, incompatible
Stokes BCs in a no-Stokes test). These logs are retained. The final wrapper
only evaluates saved phase/geometry; removing irrelevant mechanical BCs from
that no-solve diagnostic does not change a production benchmark.

### Commands

From the repository root, existing evidence is never overwritten by the runner:

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
build-pf-cpdi/aspect-release --test '[phase_field_fault_cohesive],Stage-I*'
mpirun -np 2 build-pf-cpdi/aspect-release --test '[phase_field_fault_cohesive],Stage-I*'
python3 benchmarks/reconstructed_fault/performance/run.py PARAMETER.prm --cap 120
# Production-vs-reference comparisons, no tolerance relaxation:
python3 benchmarks/reconstructed_fault/performance/run.py PARAMETER.prm --cap 120 --env ASPECT_IH_COMPARE_CELL=1
# Pending-fix reproducer: currently EXPECTED TO FAIL.
ASPECT_IH_BOUNDARY_AUDIT=1 build-pf-cpdi/aspect-release --test '[.ih_boundary_reproducer]'
```

Use the exact per-case resource manifest for longer approved caps, MPI count
and saved-data paths. A fresh output wrapper is required to repeat a case.

## 6. Changes and remaining decision

This pass changes manager trial publication (`manager.h/.cc`), scalar candidate
construction (`reconstructed_fault_nonlinear.h/.cc`), solver trial use
(`solver.cc`) and exact endpoint sampling (`surface_system.cc`).
`phase_field_fault.h/.cc` owns the opt-in geometric traversal cache/backend,
fine cold timers, safe invariant reuse and reference-comparison diagnostics.
Tests cover the captured bound and proposed boundary defect; the cache
postprocessor adds the saved-state comparison. Benchmark runner/comparison
scripts, focused wrappers, the narrowly repaired K3 guard and this report
preserve future reproducibility. Authoritative Stage-I/current-design notes
and the cell-profile addendum document the approved scope.

Previously accepted BP3/cache/depth/Theta/timer/phase-revision changes and
unrelated working-tree edits are preserved, not attributed as new fixes here.
No commit was requested. The prototype is deliberately **not** qualified for
BP3; the hidden boundary test is deliberately failing, and remaining interior
comparison errors are explicit. A source snapshot/manifest accompanies this
report for recovery.

The evolving performance gate is exceeded without value hits. A second
bulk-strip design is not justified by these measurements. Review the minimal
legacy boundary-panel correction and subsequent reference-accuracy check
before promoting the backend; do not proceed to the fine pilot merely because
either the frozen cache or K3 ray implementation is fast.
