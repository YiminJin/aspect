# Stress-history consolidation — 4 November 2026

The selected behavior-preserving consolidation is complete. The tested direct
physical-stress implementation and final publication fix were first committed
as **`e4b74c556`**, before restructuring. Its parent and the unchanged local
origin-tracking reference are `69e8a10a93a08718d31af961197899e320657c04`.
No fetch or push was performed; the live remote was not independently queried.
See [starting baseline](STRESS_HISTORY_CONSOLIDATION_BASELINE.md),
[declared plan](STRESS_HISTORY_CONSOLIDATION_PLAN.md), and
[evidence index](../../../benchmarks/reconstructed_fault/particle_free/evidence/stress_history_consolidation/README.md).

## Implemented boundaries and caller audit

| Responsibility | Current location and decision |
|---|---|
| Physical stress access | `dg_stress_access.cc` owns raw tensor sampling, common physical observation and stationary mechanical sample caches. Direct returns tau; separated reconstructs T−R−B with its surface memory. Mechanics and traction share the observer. No second mechanical solver. |
| Initial/source projection | `dg_history.cc` retains initialization, source projection and publication roundoff allowance. `dg_material_projection.cc` performs the existing material mass projection with current staged phase and accepted velocity gradient; it never publishes. |
| Transport and rotation | `dg_transport.cc` / `dg_objective_transport.cc` retain smooth advection. `dg_material.cc` / `dg_objective.cc` retain existing transport/finite-rotation ordering and the incoming-gradient spin lag. The projection callback supplies the representation-specific field. |
| Transaction and reaction | Existing material preparation, coupled reaction, acceptance/rollback and `dg_accepted.cc` checkpoint ownership remain. Mature direct/separated tensors are distinct from cohesive formed-side physical elastic histories and retained lifting/native memories. No schema conversion or shared cohesive/mature owner. |
| Serial limiter | `particle_free/support/serial_dg_limiter.{h,cc}` now owns the limiter, candidates, statistics and limited SSPRK3 experiment. Only limiter/stabilization checker DSOs compile it. Core limiter declarations, stage hooks and statistics are removed. |
| Checker snapshots/reference setup | Moved into the existing `tests/phase_field_fault_test_access.h`, without additional production friends or accessors. Actual callers in PF3/PF5/PF6 and BP3 native checkers were migrated and their includes made explicit. The unused production wrappers are deleted. |
| Standalone spatial relaxation | `transported_objective/native/spatial_relaxation.h` attaches the same prescribed relaxation blocks to a scratch checker operator. Removed the core helper/attachment flag and benchmark-only per-source shape allocation; checker FEValues reconstructs those shapes using the same quadrature. Coupled material reaction remains in production. |
| Affine residual diagnostics | `simulator/solver/reconstructed_fault_affine_audit.h` owns the optional audit's assembly, shadow matrix, comparisons, logging and scratch restoration. Newton construction, Armijo search, physical acceptance and final publication remain in `reconstructed_fault_stokes.cc`. |

Serial smooth transport and its serial matrix are **retained**: PF5's prescribed
accepted-step control and active transport checks still call them. The
`divergence_correction` diagnostic is also live. These are not dead helpers.
The benchmark limited-stage integrator remains a scratch experiment, not a
runtime selector. The internal checker ABI changed, so checker DSOs must be
rebuilt with the matching core; an old DSO is not compatible merely because
its embedded revision string matches.

No default, physical transport equation, bound/irreversibility rule, endpoint
law, pressure convention, spin lag, Gauss rule, convergence threshold or
checkpoint meaning changed. The baseline's final projected-reaction gate still
passes its converged private slip guess to publication. No tolerance relaxation,
new damping, profile limiter selection or endpoint-aware tensor space was added.

## Bounded verification

Core and matching native/checker builds passed with at most four workers. The
full existing suite passed **183170 assertions in 152 cases** (CTest 7.79 s).
The first checker build exposed missing explicit test-access includes; those
were fixed and the failed log is retained. Independent compilation without
unity/PCH passed for both extracted material translation units, the audit
header and both affected BP3 native checkers.

Fresh pre/post native comparisons use 16² cells, two MPI ranks, dt=.05 s,
end=.1 s, fixed .1 s initialization, compact Q1 phase, bounded MacCormack,
accepted tensor trace, full-normal support and zero imposed shear. Constant
eta=10 and layered eta=10(1+.5 tanh(4x)) are each tested with direct and separated
history. Exact inputs and command records are retained. The post layered-direct
run extends to .2 s to supply the uninterrupted restart reference.

| Matching accepted states | Bulk export max absolute difference | Surface export max absolute difference | Physical tensor L2 difference | Work-traction max difference |
|---|---:|---:|---:|---:|
| Constant direct | 6.98e−17 | 1.07e−14 | 1.81e−17 | 8.34e−18 |
| Constant separated | 3.26e−19 | 5.33e−15 | 2.47e−19 | 2.72e−19 |
| Layered direct | 8.89e−16 | 7.11e−15 | 1.52e−16 | 6.74e−18 |
| Layered separated | 1.58e−15 | 7.11e−15 | 2.55e−16 | 8.60e−18 |
| Entry layered checkpoint, resumed 2→4 ranks | 1.78e−15 | 1.07e−14 | 6.27e−16 | 5.31e−17 |
| Forced rejection vs retained rejection/cutback controls | 1.58e−15 | 1.25e−14 | 7.03e−16 | 6.22e−17 |
| Default separated/Q1 case | 4.73e−15 | 1.78e−14 | 2.07e−15 | 1.88e−16 |

Bulk/surface maxima combine exported quantities with different units and are
regression screens, not a physical norm. Tensor and work-traction differences
are separately reconstructed by the existing independent reducer. The declared
absolute screens (5e−13 bulk, 5e−12 surface) pass unchanged. Some records are
bitwise identical; overall equivalence is at roundoff, not bitwise. Geometry and
history publication counts match, as do equal-duration mechanical iteration
sequences. The constant-direct run exercises the extracted affine audit and
source/rotation checks. Continued restart uses only a fresh same-law local
checkpoint, never a production checkpoint or conversion.

Existing profile/donor-bound, surface age/memory, storage/work/flux reductions
are retained in `equivalence.json`. The direct/separated stress difference at
.1 s remains 3.32925% constant and 3.30067% layered; work-traction differences
remain 2.17163% and 2.16216%. These coarse-grid physical differences are not
expected to vanish under source reorganization. This check neither resolves
finite-end approximation nor requalifies coupled energy.

The moved serial limiter and stabilization fixtures pass their existing native
assertions on bounded 4² fixtures. Across 62 limiter CSVs, the largest scaled
historical difference is 1.41e−12 in the large-amplitude cancellation fixture;
all other limiter files are within 4.20e−15. Across 30 stabilization CSVs the
maximum is 2.63e−15. These are measured floating-point differences, not a claim
of historical bitwise identity. No checker assertion changed.

The standalone layered operator uses the existing 512-cell rectangular fixture,
80 synthetic substeps of .025 s, both directions and two MPI ranks, with zero
outer physical steps. Its CSV is **byte-identical** to `layered-native-r3`:
relative L2 errors 6.07615e−4 (negative) and 3.00127e−4 (positive). This verifies
the migrated relaxation assembly and removal of its core shape payload. The
BP3 initialization support library was rebuilt and preserved; no BP3 event was
replayed. Post representative wall times were 18.91/12.24/24.22/21.31 s for
dc/sc/dl/sl, including extra audit work and different dl duration. These single
runs do not support a speedup claim. Default control cost was 21.91→16.95 s;
its 20 linear-solve iteration counts are identical.

## Preservation and remaining work

Final hashes verify all **39 protected files**: 33 earlier controls, five entry
core/checker copies and the additional BP3 support-library copy. Source/input,
actual binary hashes, commands, original failed evidence and existing outputs/
checkpoints remain. The core was compiled against baseline revision text;
`final-provenance.json`, not the printed revision, identifies the tested files.
No manual work or unrelated server package was removed. Inherited loss of
39/45 native-pilot frames and temporary logs remains explicitly unrepaired;
production BP3 and cohesive-interface evidence remain separate.

Implemented: dependency/interface cleanup, shared physical access, projection
and audit boundaries. Locally verified: existing units, short constant/layered
native equivalence, default, rejection and cross-rank restart, migrated checker
operators. Physical comparison: prior finite-end/direct/separated limitations
are preserved, not solved. Production readiness: unchanged and unestablished.
Cohesive energy/tip failures, moving AMR, birth and general shape work remain
open. The scientific work stays paused. If subsequently selected, the justified
next experiment remains a **frozen endpoint-aware tensor-space comparison with
matching work integration**, before a coupled implementation or server run.
