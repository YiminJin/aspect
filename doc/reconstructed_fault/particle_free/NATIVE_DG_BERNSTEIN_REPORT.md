# Native DG Bernstein phase: bounded integration report

Completed 9 November 2026 under the [selected task](CODEX_NATIVE_DG_BERNSTEIN_PHASE_FIELD.txt).
The element is integrated and locally equivalent on the selected controls.
Adopt it in the active tip fixtures for native coefficient meaning and fewer
conversions, not as a general speed optimization. Ordinary defaults stay nodal.
The [reading guide](NATIVE_DG_BERNSTEIN.md) is the compact equation-to-code path.

## Starting state and identities

Work remained in `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, revision `d481a09c1415578ec6fb148e6b9e13c80184b6c4`.
The completed efficiency implementation was **uncommitted**, not discarded.
All 48 source hashes in its closeout matched at entry. The previous source
archive, retained checkpoints and binary `8a652f3e…89115` remain the reference.
No reset, checkout, server submission or commit was performed.

Local evidence is under
[`evidence/native_bernstein`](../../../benchmarks/reconstructed_fault/particle_free/evidence/native_bernstein/).
`entry.json`, `incoming-source.tar.xz`, `closeout.json`, source archive and the
entry-relative patch distinguish inherited efficiency work from this task.
Per-run input, clock, source/core/checker hashes and commands remain in
`build-particle-free/bulk-tip/runs/<label>/command.json`; outer timeout/thread
records are `refactoring_boundary/evidence/bernstein-*.json`.

| Build | SHA256 | Use |
|---|---|---|
| Retained efficiency core | `8a652f3ec0ce1efda2608db133376bfae21551aa0205a7c96f3fed8a66989115` | Original nodal physical endpoints and timings |
| Initial native core | `e498704dc171aa885039850e3b980e70bd2a6b91a16152070d3aaad3b98b5172` | Difficult/localized tip and MPI comparison; preserved in `controls/native-bernstein-initial` |
| Final native core | `76e9c5c1f16caf64e4f98e763463936dc12ba7d4649b280febbd34f5dbd5641a` | Mature checkpoint correction, final oracle and ordinary control |
| Native tip checker | `fc5ce66ddace9cb4eed53c7c3ca34570c2ccd632177c1619f3bbfc50bfa0838e` | Benchmark import, derivative/oracle and lifecycle checks |

The final core adds the mature serialization correction, explicit unsupported-route
guard, diagnostic sample-anchor correction and indentation cleanup; no tip
operator, solver policy or tolerance changed after the difficult comparison.
Mature checker/provider were rebuilt in fresh `native-bernstein` directories
against the new parameter layout. Old DSOs were not loaded into the new core.

## Representation and removed work

`aspect::FE_DGQBernstein` derives from the installed deal.II 9.6 FE_DGQ polynomial
constructor. Local headers and the available `source/fe/fe_dgq.cc` were inspected.
It reuses L2 topology, cell numbering, polynomial FEValues, all-ones constant
mode, reference embedding and additive restriction. It overrides nodal
interpolation assumptions, generalized sample conversion, face support, clone
and name registration. Root-product polynomial evaluation preserves endpoint
positivity through FEValues without clamping. No library patch is needed.

The one native semantic phase composition stores Bernstein coefficients. Its
base element is isolated from ordinary/stress compositions. Generalized sampling
and inverse interpolation reproduce the original nodal interpolant. Current,
old, old-old and accepted obstacle copies carry the same native meaning.
Fracture assembles directly in that basis; material/geometry/diagnostic consumers
continue to receive FE-evaluated physical phase.

Removed from the native coupled path: repeated phase coordinate transforms,
fracture congruence construction, a trial nodal copy and a second ghost exchange.
The nodal map is constructor-local compatibility data, not a persistent member.
Reference basis values and immutable parent samples are cached for one attempt;
trial constitutive responses and section roots are not reused across iterates.
Mechanical u=dt*v and p/G scalings remain. Q2Bounds shares one enclosure/scaling
implementation with a small nodal adapter. No shadow phase field or solver was
added; ordinary advection face assembler classes were untouched.

BE, signed section closure, physical tau/A, SSPRK3, dense Lobatto4 mass,
Lobatto5 transport, Gauss5 fracture, Gauss6 mechanics, progress search, exact
polishing and physical acceptance gates remain unchanged. Growth keeps the
coefficient box; transport keeps the less restrictive polynomial enclosure and
JxW-mean scaling. Existing physical AMR restrictions stay.

The task described two native registration routes. Actual current source admits
DG phase only as a semantic composition; separate native DG registration was
already retired. Its guard remains, and legacy separate Q1 is unchanged.
Historical replicated cohesive/endpoint solvers retain nodal assumptions and
now explicitly reject the Bernstein selector. Integrated bulk tip and growth-off
mature material transport are the supported fault routes.

## Verification

All errors below compare the same physical polynomial/state, not unmatched
vector entries. Mechanics 1e-9, phase KKT 1e-7 and independent ledger/dissipation
gates were retained.

| Check | Result |
|---|---|
| `bernstein-points02` | **5692 assertions / 17 cases pass**, including retained point suite; degrees 0–3, interpolation, constants, clone/registry, physical values/gradients, face traces, endpoint positivity, affine/bilinear mapping, child embedding/restriction, mass congruence and limiter contracts |
| `bernstein-operators02`, MPI2, bilinear/nonconforming | SIPG action relative 1.47e-15; energy absolute 2.27e-13; unlimited/limited transport coefficients 1.55e-15; source/flux/balance 2.03e-15; integral 1.78e-15; 119 limiter events |
| Semidiscrete generator | Relative difference 1.77e-13. Recovered actual production L*u+g from the cubic single-step unlimited SSPRK3 polynomial at stationary inflow; no duplicated flux operator or private test accessor |
| `bernstein-oracle-final01` | 318 free unknowns; Krylov/direct 3.29e-8, manufactured-direction 3.29e-8, weighted symmetry 5.00e-17; existing strict checks pass |
| Difficult endpoint derivatives | Tangent action 7.60e-10; phase functional 2.37e-8; scaled Lagrangian 2.51e-9 |
| `bernstein-mpi01` | Active 1→2 rank restart from the native localized checkpoint; accepted parent immutable, rollback complete, retry deterministic, publication once |
| `bernstein-basis-rejection01` | Expected nonzero exit with explicit storage/index/family/degree/**basis** mismatch before deserialization |
| `bernstein-ordinary01` | `depth_average.txt` and `statistics` byte-identical to retained ordinary no-fault Stokes control |

The subdivision test includes a bounded polynomial whose Bernstein coefficient
exceeds one, and an actual overshoot that must be limited without changing mean.
Transfer tests concern parent-representable polynomials; they do not qualify
arbitrary irreversible coarsening or physical AMR.

### Same-parent difficult .0575 → .06 endpoint

The old checkpoint's actual pending endpoint is `.060000000000000019` with
`dt=.0025000000000000001`. `bernstein-parent-export01` restores its accepted
parent without replaying the trajectory. A serial, CellId/component/local-shape
benchmark archive saves all three solution copies, tau/A, section state,
geometry/material identity, accepted phase and exact clock. Only phase copies
are transformed on import. Material restoration validates the full parent;
independent Gauss6 phase samples differ by 1.11e-16. No anonymous checkpoint
bytes are patched and ordinary restart never converts a basis.

Fixed-parent residual/actions compare the current retained nodal path with the
current native path, isolating basis assembly from nonlinear trajectories.
Coordinates agree bitwise. Raw residual maximum difference is 4.85e-16
(relative 1.40e-13); tangent action maximum difference is 1.42e-14. Mechanical
rows agree bitwise. The action's global relative error is artificially small
because intact phase derivatives are large; absolute and per-component scales
are retained in `fixed-parent-components.json`, not hidden behind that norm.

The accepted endpoint comparison uses the **original efficiency binary's**
`efficiency-final01` step24, not a new control trajectory:

| Quantity | Maximum absolute difference |
|---|---:|
| Phase | 4.68e-16 |
| Physical stress components | 1.59e-12 Pa |
| A_xy | 1.73e-18 |
| Driving force Y | 2.84e-12 Pa |
| Section V | 1.01e-16 m/s |
| Age / cumulative slip | 1.78e-15 s / 2.71e-19 m |
| Section resistance, including tiny rates | 3.60e-11 Pa |
| Resistance for absolute V>1e-8 m/s | 6.82e-13 Pa |
| Bulk / cohesive / fracture storage | 0 / 6.94e-18 / 1.78e-14 J/m |

`fine-comparison.json` retains scaled maxima, L2 errors and every work/energy
component. Work differences are about 1e-17 J/m or below. Final mechanics is
2.19e-14, KKT 7.61e-8 and ledger 2.93e-16. The independent section law agrees to
1.14e-13 Pa. No new material discrepancy justified a tighter solve. The earlier
35 Pa sensitivity is not used as a tolerance here.

Polynomial reconstruction of the independent Gauss6 endpoint exports gives the
same diagnostic active set: 1004 lower, 148 free, zero upper coefficients.
The reporting threshold is 1e-12; it does not change solver contact tests.

### Localized growth, moving profile and layered coupling

`bernstein-localized01` runs the existing equilibrated control through .08 at
.005. Integrated damage increment is 8.42234623794e-5, so this tests actual growth.
Against `efficiency-localized-coarse01`, maximum phase difference is 2.72e-15,
stress 4.26e-12 Pa, Y 2.03e-13 Pa, V 7.81e-17 m/s and resistance 2.27e-13 Pa.
The physical seed and material/loading data are identical. Restart/retry final
phase differs from the uninterrupted native run by 3.72e-15 and stress by
1.49e-12 Pa. Exact rollback assertions and distributed physical comparison are
separate evidence.

The short MPI2 translated compact profile has 576 DoFs and two .05 steps.
Native/nodal final L2 errors both equal .03188142691466; final mass differs by
5.55e-17, centroid by 1.11e-16 and width by about 1e-16. Native flux-balance
error is 3.03e-17, with 148 limiter events (nodal146). Roundoff at zero-amplitude
cells can change limiter event counts without a physical profile difference;
this is not an identical-branch-count claim. Minimum sampled phase is zero.
The coarse profile's appreciable shape error is retained, not cured by basis.

The layered mature comparison uses the identical level3, translation2,
background3.7, dt=.0125 fixture and rebuilt plugins. Initial phase differs by
1.67e-16; endpoint phase by 1.11e-16, physical stress by 8.74e-16, shear/normal
traction by 1.04e-16/1.46e-16, slip rate by 1.11e-16 and age by 3.55e-15.
Geometry agrees exactly; normalization I differs by 6.22e-15. Storage differs
6.94e-18, localization measure 2.22e-16 and flux 2.71e-20. The independent
background recurrence/balance check is below 8.89e-16. Transport's physical
balance error is -9.35e-18, with 133 limiter events. The ordinary joint transaction
publishes history and geometry once.

### Discovered defect and correction

`bernstein-layered01` completed its endpoint but crashed while writing mature
schema9: its serializer indexed ordinary support points, absent for Bernstein.
It now writes/verifies native coefficients with **generalized interpolation
anchors**; outer phase-space metadata checks basis identity first. This preserves
old nodal layout and does not reinterpret anchors as physical coefficient values.
The same diagnostic assumption was corrected in the history trace. Adjacent
historical nodal-only solvers are guarded rather than silently enabled.

`bernstein-layered03` passes with the correction and writes its checkpoint.
A two-rank restore (`bernstein-layered-restart01`) validates phase/tensor/surface
history and completes one more accepted step to .025, with publication counts
2/2 and no retry. The ordinary restart loop executes that pending step even with
the copied .0125 end time; it is recorded as a short extra step, not a restore-only
check. This does not expand the mature qualification beyond the bounded window.
`bernstein-layered02` was an MPI socket sandbox startup failure, not a numerical
failure. Both failed records are preserved. Initial build link failures required
including deal.II's FE-factory template definitions; their logs remain.

## Cost and decision

Single measurements, not a statistical speed claim; nested timers are not added:

| Scope | Retained nodal | Native Bernstein |
|---|---:|---:|
| Difficult step, excluding setup | 101.356 s | 109.935 s |
| Difficult trial evaluation total / calls | 13.301 s / 1096 | 12.129 s / 1100 |
| Difficult tangent total / calls | 3.670 s / 44 | 3.716 s / 45 (includes fixed-parent oracle) |
| Localized physical-step total through .08 | 12.488 s | 10.610 s |
| Standalone two-step transport kernel | .01087 s | .00650 s |
| Layered phase transport timer | .0204 s | .0210 s |
| Layered complete wrapper | 12.388 s | 12.485 s |

The native difficult trial evaluation averages about 11.0 ms versus 12.1 ms.
However Krylov work varies with floating-point assembly: difficult inner Stokes
iterations increase 100725→111397, despite practically identical physical endpoint.
The full difficult step is 8.5% slower. Localized step time is 15.0% lower. DoF
count and transport operator storage (428560 bytes/rank0) are unchanged. Warm
DoF setup is about .002 s; the first initialization-only check recorded a 9.3 s
setup outlier and is retained. No preconditioner/tolerance changes were made to
improve timings. The justified benefit is clearer native ownership and removal
of unnecessary work, with mixed end-to-end cost.

New wrapped numerical checks total **243.19 s (4.05 min)**, including failed
startup and expected basis rejection, below the two-hour budget. Separate build
log intervals total about 702 s (11.7 min; rounded filesystem timestamps, overlapping
intervals included). One early points build inadvertently overlapped the tail of
an initial core build, each launched with -j4; the aggregate four-worker limit
was therefore not assured during that brief build overlap. Subsequent heavy
jobs were sequential; numerical runs used at most two MPI ranks and one numerical
thread/rank. All jobs finished. Nine inherited protected binary hashes and the
retained efficiency pair were checked; no checkpoints or outputs were deleted.
Inherited losses remain 39/45 pilot field frames, early temporary logs and the
absent old `bulk_continuation.zip`.

## Remaining scientific work

Implemented and locally verified: native coefficient integration, unchanged
operators/bounds, accepted-parent conversion for the benchmark, active restart
and rollback, and short moving mature coupling. This is not production or general
propagation qualification. BE work error, near-zero-rate sensitivity, spatial
accuracy and lack of resolved front advance remain. Dynamic irreversible AMR,
moving cohesive growth, turning and branching remain unsupported.

The next justified scientific task is the separately selected endpoint time-update
study: point and small coupled comparisons against this retained BE baseline,
judged by state accuracy and consistent work accounting. Compact-loading
mechanical screening can accompany that task before any longer growth campaign.
Neither subsequent task was started here.
