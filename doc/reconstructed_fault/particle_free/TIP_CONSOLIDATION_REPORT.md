# Numerical decisions and active-path consolidation — 9 November 2026

Entry is `e8b6dd0912640433676a69089d2b2ad17701bcbf` on
`pf-rsf-particle-free`, in the existing worktree. The completed midpoint study
and manual documentation changes were present and preserved. Source/executable
identity is recorded independently; no reset, checkout, server job or production
checkpoint conversion was performed.

## Frozen decisions

- Native DGQ2 Bernstein is the supported phase representation for the active
  research model. Ordinary ASPECT FE_Q/FE_DGQ infrastructure is unchanged.
- BE remains the reference/default time law. The same experimental midpoint
  implementation remains selectable; it is **not adopted** as a replacement.
- The compact-loading experiment is closed: boundary damage preceded resolved
  axial advance, and opposite-sign fault deformation nearly cancelled section
  slip. No additional loading variants or growth campaigns were run here.
- Successful controls, failed common-parent paths, the compact pilot and all
  retained checkpoints remain evidence with their original qualification limits.

The prior [time-update report](TIP_TIME_UPDATE_AND_GROWTH_REPORT.md) is historical
scientific evidence, not rewritten to claim a repaired midpoint solve.
[`entry.json`](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_consolidation/entry.json)
records the preserved source and final-study core/checker identities. The latter
were copied into `build-particle-free/bulk-tip/controls/time-update-final/` before
building. Final-study core is `ad50ceb83d726d81c26f4392306394d7d0cecdba4d74fc920841d0bd9b34eb79`.

## One bounded midpoint repair audit: closed without a local correction

`tip-consolidation-diagnosis01` imports the exact retained accepted .0575 parent,
uses dt=.0025, and attempts only the .06 endpoint. Diagnostic prints expose the
largest phase natural-map rows and candidate line-search values. The same
80-iteration limit and mechanics/KKT tolerances 1e-9/1e-7 are retained. The run
lasts 98.00 s and reproduces the original failure **exactly**: mechanics
2.50583e-15, phase KKT 6.34003e-7. No endpoint is accepted.

The obstructing rows are not a pressure or section-root failure. At iteration79:

| Phase row | Coefficient | Accepted lower bound | Gradient | Natural-map magnitude |
|---|---:|---:|---:|---:|
| 1055, 863 | about 3.31138e-11 | about 8.36022e-12 | about -1.10167e-9 | 6.34564e-7 |
| 288, 96 | about 3.31138e-11 | about 8.36022e-12 | about -1.10150e-9 | 6.34466e-7 |
| 125, 353 | about 5.33191e-5 | about 3.29886e-5 | about +9.80709e-10 | 5.64888e-7 |

Their diagonal phase measure is .001736111111111111. The four near-intact rows
are above the lower bound, with a negative residual; they cannot be declared
inactive merely because their coefficients are tiny. The upper bound is about
.5 in midpoint stage coordinates. Section rate defects remain about 1e-15.

At iteration79 the exact direction has slope -7.7364e-14, below the functional
summation scale 1.01235e-13. The full step increases scaled mechanics/KKT merit
from 6.34564 to 347487.8; half gives 352.68, quarter 87.21, and 1/128 gives
6.37280. Only 1/256 gives a decrease, to 6.34003. Thus the residual safeguard
prevents accepting a huge phase defect through an unresolved energy decrease.
Late iterations repeatedly take this tiny step. Earlier positive phase searches
also encounter sharp near-intact releases and very small accepted fractions.

No incorrect bound, stale parent, missed MPI row, pressure admission or justified
local algebraic correction was demonstrated. The current audit identifies a
strongly nonlinear near-intact globalization limitation; it does **not** prove
that every possible direction or solver is inadequate. Removing the merit gate,
clipping small phase, increasing the iteration budget, or relaxing KKT would not
be a justified repair. Resolving it requires a separately designed treatment of
these nonlinear transitions. The bounded repair attempt is therefore closed,
with no numerical change to BE or midpoint and no further prerequisite sweep.
Temporary instrumentation is archived locally and removed from active source.

## Removed implementations and retained dependencies

| Area | Caller audit and change |
|---|---|
| Separate Q1/CPDI phase owner | Removed standalone phase-variable registration, particle interpolation helper, CPDI phase residual/Jacobian, Q1 fracture integration, old phase Newton/box solve and wider particle-domain graph from `source/simulator/phase_field.cc`. Native semantic composition and DG face graph remain. |
| Q1 profile motion | Removed `profile_transport.cc` and simulator fallback dispatch, including bounded MacCormack; removed old Q1/file-seed interpolation from `initialize_fe_shared_phase`. Mature phase goes through ordinary native DG SSPRK3/BP only. |
| Nodal tip coordinates | Removed `BulkTipCoupling::transform_phase` and fracture row/column congruence. The accepted obstacle and coupled unknown are native Bernstein coefficients. Polynomial conversion remains in independent representation tests and explicit parent import. |
| Replicated phase solvers | Removed `bulk_cohesive_phase.cc` and `reconstructed_fault_endpoint_phase.cc`, their simulator entry points and `benchmark direct` selection. Their physics is not silently redirected into the active bulk-tip model. |
| Historical endpoint mechanics | Removed endpoint-only direct surface Schur dispatch, reduced phase-tangent adapter and cached exact velocity inverse. The common mature coupled Stokes solver retains its ordinary preconditioner and physical residuals. The independent bulk inverse test remains under `unit_tests/support/endpoint_bulk_reference.h`. |
| Stress advection | Removed the production serial matrix/SSPRK path and boolean dispatch. The owned-row operator and ghost refresh serve all communicator sizes. Projection remains a separate operation. The sparse serial implementation is now solely `benchmarks/.../support/serial_tensor_advection.*`, checked against production on one rank. |
| Historical test entry points | Old `phase_field_*` fixtures and the PF1–PF6/replicated plugin umbrella are no longer active registrations. Their source, inputs, outputs and old binaries are retained. Current tip, native spatial and mature checks have maintained separate CMake projects. |

The material parser admits only integrated tip or growth-off transported mature
configurations with mesh data and native Bernstein phase. Q1/nodal/CPDI and
retired endpoint settings fail explicitly. Generic ASPECT particle, FE_Q and
FE_DGQ facilities remain; tensor stress still uses ordinary DGQ2. Full physical
cohesive history and section friction are unchanged. Material/history helpers
shared with mature reconstruction or checkpoint interpretation are retained;
this is not a deletion of all historical constitutive algebra.

The historical `Material profile transport` string remains an **identity-only
compatibility input**, with its existing support admission. Some valid native
DG checkpoints contain `bounded MacCormack` because that parameter preceded
native phase transport; their DG trajectory already used SSPRK3. Keeping the
identity does not execute the removed remapper or admit Q1 checkpoints. Physical
law, basis and BE/midpoint method guards remain independent. There is no automatic
conversion or hidden restart change.

## Retained regression set

The final core SHA-256 is
`5d8f9ab1adc94662f7b96a9b63267657a18bc0674dfb51693e763f5877c18e3e`.
The core and matching tip, mature, background and independent-reference plugins
were built with at most four workers. All numerical runs used one thread per
rank, at most two MPI ranks, and ran sequentially. Nine bounded wrappers,
including the failed midpoint audit and one failed input setup, total **166.95 s**.
No long campaign was repeated.

| Check | Measured result |
|---|---|
| Full executable unit suite | **189,151 assertions in 169 cases pass**, including the retained direct bulk-inverse test. |
| BE localized control, dt=.005 to .01 | Point fields, section state and accepted ledger reproduce `tip-time-be-final01` exactly. |
| Successful midpoint control, dt=.005 to .09 | Point fields, section state and accepted ledger reproduce `tip-time-local01` exactly. Coupled tangent and functional derivatives pass at step8. |
| Midpoint old-binary checkpoint → new core, 1→2 ranks | Accepted .075 parent advances to .08. Two rollback checks preserve fields and all histories. Against the serial endpoint, max shear-stress difference 1.36e-12 Pa, phase 2.94e-15, age 1.78e-15 and slip 4.34e-19. |
| Independent serial versus owned-row tensor operator | Positive translation, negative translation/shear and divergent flow: projection difference ≤6.22e-15, SSPRK3 step difference ≤8.44e-15, material-constant defect ≤1.11e-15 (gate 2e-12). |
| Layered mature motion, dt=.0125, two intervals | First accepted bulk/surface fields reproduce `time-mature-smoke` exactly; storage counts match. RSS is excluded from numerical comparisons. |
| Mature checkpoint, 1→2 ranks during transport | Accepted .0125 parent advances to .025. Maximum bulk-field difference 5.33e-15 (velocity gradient), surface normalization 3.55e-15, age 1.78e-15, rate 1.11e-16. Geometry is identical. |
| Ordinary non-fault Stokes control | Statistics and depth average are byte-identical to the retained Bernstein-era ordinary control. |

The initial mature restart input accidentally changed the diagnostic library
path while relocating the run directory. It failed before loading the model;
the input and log remain preserved. A fresh input with the correct library path
passes. This was a benchmark setup defect, not a checkpoint or solver failure.
The first core build also exposed a missing retained `initialize` declaration;
it was restored before the successful build and all reported tests.

Exact commands, wall times, executable/plugin hashes and preservation checks are
in the local [closeout manifest](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_consolidation/closeout.json).
[Tip endpoint comparisons](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_consolidation/tip-comparisons.json)
and [mature comparisons](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_consolidation/mature-comparisons.json)
record per-field errors. The run labels are `tip-consolidation-units01`, `be01`,
`midpoint01`, `restart01`, `mature01`, `mature-restart01/02`, and `ordinary01`,
each with the same `tip-consolidation-` prefix. The separate diagnostic run is
`tip-consolidation-diagnosis01`.

These are unchanged-behavior and lifecycle checks, not a new physical
qualification. They do not resolve the difficult midpoint window, BE work
accuracy, finite-end stress representation limits or propagation validity.
Protected final-study binaries retain their original hashes. Generated outputs,
logs, archives, binaries and checkpoints remain local and outside the source
commit.

## Reading path and next experiment

[ACTIVE_RESEARCH_MODEL.md](ACTIVE_RESEARCH_MODEL.md) is the single entry point
through supported configuration, bulk-tip equations/operator/solver/history,
native spatial assembly, mature transport and checkpoint ownership.
[TIP_TREATMENT.md](TIP_TREATMENT.md) and its linked pages have been reconciled
with the removed files and active Bernstein-only interfaces. Historical reports
remain evidence rather than current API instructions.

The next justified scientific experiment is a short **fixed-phase mechanical
screen with uniform shear plus a compact perturbation**, preserving compatible
side loading and examining signed section cancellation before attempting growth.
That is a proposed new experiment, not launched here. A broader midpoint
convergence redesign is not an indefinite gate on independent useful work.
Long propagation, moving cohesive growth, turning/branching, and server campaigns
remain deferred. Cleanup does not improve the retained BE physical-work accuracy.

Inherited losses remain 39/45 pilot frames, early temporary logs and the missing
old `bulk_continuation.zip`. No retained output or checkpoint was deleted.
