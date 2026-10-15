# Bounded P2 cleanup recovery — 2026-10-13

The requested structural cleanup was already committed when this session
recovered. Verification found one missing export for the retained-entry test
interface. Its explicit 2D/3D instantiation is restored without changing a
numerical body. The corrected retained growth replay passes with byte-identical
accepted fields. The selected cleanup and bounded verification are complete,
with three baseline initialization-fixture failures recorded below. BP3 testing
has not begun.

## Reference and recovery

Reviewed reference: `41272d26366b1953f09f1da82c386af5bbea8ec2`. Recovery checkout:
`/home/ein/repository/aspect-fault-propagation`, branch `fault-propagation`, HEAD
`f9aa961999b4ed2c8f62527d85649bfafc3387b1`. No staged changes; incoming unstaged
changes were the three status/plan documents. All 2301 incoming untracked files
and both pinned reference binaries remain unchanged. No reset, checkout switch,
other-worktree build, commit or push was performed during recovery.

Completed commits are `7136d0156` (FE movement), `d0b2bd004` (diagnostics),
`35077cc3c` (normalization, lazy projection, zero initialization and recurrence),
`08eadf296` (ASPECT formatting), and `f9aa96199` (explicit manager include).
Recovery adds one instantiation line in `fe_strain.cc` and this review/status
documentation. Structural movement and formatting remain separate in history.

Release build: GCC 12.4, deal.II 9.6.2, Trilinos 14.2, p4est 2.8.7, existing
`build-propagation-integration`. The pinned executable/observer SHA256 prefixes
are `e78821a5b12b` / `f5b7567699a8`; initial cleanup checks used
`1256219dc5a7` / `c38a3522308b`. Final export correction uses
`0ac2e871813e` / the same observer. Full identities, inputs, commands and logs
are retained in the evidence linked below. The native growth control retains
16×512 Q2/Q1, six normal points, 64 surface segments, dt=1/1024 s, original
FGMRES/reduced Newton, material, loading, accepted anchors and KKT target.

## Responsibilities and preserved behavior

| Files | Responsibility / change |
|---|---|
| `phase_field_fault/fe_strain.cc`, `fe_strain_internal.h` | Accepted FE state, stress samples, immutable-anchor recurrence, trial/commit/publication and mechanical quadrature; shared implementation data only. Seed and retained-entry interfaces stay here. |
| `phase_field_fault/fe_normalization.cc` | Fixed normal-ray geometry, analytic integrals, consistent projection and its dense/action/adjoint derivatives. Values are recomputed at the explicit phase; fixed geometry/projection is reused. |
| `phase_field_fault/fe_phase_energy.cc` | Bulk/cohesive phase energy, gradients, tangent actions and endpoint friction potential. |
| `phase_field_fault/normalization.cc`, `history.cc` | Generic normalization dispatch/non-FE path and history orchestration. Explicit zero traction uses native validation/publication; seeded nonzero initialization still projects particles. Physical candidates and projection diagnostics have distinct names. |
| `manager_particle_projection.cc`, manager declarations | Association/domain quadrature and collective support checks precede generation-dependent matrix/factor preparation. Actual projections share cached quadrature and retain singularity checks. |
| `surface_system_diagnostics.cc`, surface declarations/internal support | Combined CSV writing, particle/stencil sampling and diagnostic line clipping. Numerical constitutive evaluation and assembly remain in `surface_system_particle.cc` / `surface_system_bulk_work.cc`. |
| `manager.cc`, existing test/observer access | Explicit quadrature dependency and focused normalization/cache checks; no new public fixture framework. |
| `current_design.md`, `specification.tex`, `AGENTS.md`, cleanup/status/plan docs | Selected implementation boundaries and preserved FE/particle distinctions. No governing-equation change. |

The sole material plugin and class ownership are retained. Initial RSF state,
chemical composition, nonzero cohesive histories, reverse interpolation and
particle/fault geometry services remain live. Normalization-only calls bypass
bulk integration, gradients, dense derivative allocation and energy reductions:
the scalar integral branch and empty derivative requests establish this in
source; timer checks independently establish zero phase-energy calls.

The common recurrence still uses the exact Maxwell coefficients and stable
`expm1`; it never changes the incoming anchor. The independent q I_h versus Z
recurrence and derived particle-publication checks remain. No serialized name,
layout, default, physical parameter, quadrature, normalization mass solve,
acceptance tolerance or scientific algorithm is changed.

FE history still applies `consistent_deviator` to the raw symmetric gradient;
legacy particle history still consumes the full symmetric gradient. This is a
constitutive distinction, separate from representation, integration measure,
generic particle transfer and normalization/coupling. Direct mechanics remains
serial, experimental and default off; unsuccessful quasi-Newton remains inactive.

## Verification

| Check | Actual result |
|---|---|
| Release/observer build and independent compilation | Retained split builds pass; all 11 touched implementation/test units compiled without unity/PCH. Recovery release build passes in 27.57 s; corrected FE unit independently compiles for 2D/3D in 8.56 s. |
| Export coverage | All 22 FE method definitions have explicit instantiations; all 154 ASPECT symbols required by the observer are exported. |
| Final compatibility | 21,112 assertions in 55 cases pass. |
| Serial and two-rank normalization/operators | 15 states per rank count: value-only normalization equals the FE energy calculation exactly, zero phase-energy calls. Retained energy/gradient/adjoint/tangent CSVs are byte-identical at matching rank count. Maximum cross-rank energy difference 3.55e-15 J/m, below the original 4.55e-13 allowance. Serial 127.02 s retained from the interrupted session; fresh two-rank suite 157.46 s. |
| Legacy history, restart, temperature and cache CTests | Six pass: Stage-J, restart create/resume, temperature serial/MPI, lazy cache. Cache checks association-only access, factor reuse/invalidation, constant reproduction and one-owner missing-value rejection followed by successful reuse. Stage-J preserves all state after one-owner history failure. |
| Mature explicit-zero initialization | Existing boundary-completion test passes, 19.34 s; native normalization/support and initialization markers retained. Unloaded FE initialization also reports direct zero publication. |
| Extracted history CSVs | Two-rank unchanged Stage-J history-audit control passes on checkpoint and candidate; both rank CSVs are byte-identical. |
| Retained complete P2 growth | Corrected replay passes, 761.55 s including setup/diagnostics. All phase/surface/stress CSVs and entry metrics are byte-identical to the retained serial reference: KKT 2.88e-13 ≤ 2.68e-11 J/m, zero healing, phase increment .00157378, minimum I_h 111.13338 m, publication once; 135 tangent actions. First missing-export failure remains preserved. |
| Preservation/style | All 2301 incoming untracked files and pinned reference binaries unchanged. Repository astyle 2.04 applied only to the corrected source; `git diff --check` passes. |

Three older I_h CTests fail after exhausting their unchanged ten-iteration
initialization budget, before reconstructed geometry exists. The pinned serial
reference reproduces the same residual 1.293e-3 versus 1e-5 and missing-geometry
error. These are recorded failures, not passing qualification. The existing lazy
cache fixture uses its already established 50-iteration initialization budget
and passes the inherited I_h/nonzero-initialization checks. No test expectation
or tolerance was changed.

An attempted normal/stress-sample diagnostic on Stage-J is rejected by the
existing true-pressure requirement because that fixture uses adiabatic pressure.
The compatible history-only audit passes without changing physics. The bulk
diagnostic line/particle samplers independently compile; no fresh BP3 observer
campaign was run. Native four-rank complete-step recovery remains retained
evidence; this cleanup reruns the changed cache/history collective failure paths
on two ranks. Restricted FE checkpoint conversion remains unsupported; the new
restart checks concern the legacy particle path.

Timings are observed run costs, with some concurrent checks, not a paired
normalization performance study. No normalization-only speedup is claimed.

## Helper review and next step

Each new operation with one production caller has a substantial responsibility:
`prepare_fe_normalization_geometry` constructs complete rays/projection;
`ray_integral_value` evaluates the analytic integral and stable series without
derivative jets; `add_fe_normalization_energy_pullback` applies the complete
gradient/Hessian chain rule; `add_fe_normalization_action_pullback` applies its
matrix-free first/second adjoint terms; `sample_normal_diagnostic_particles`
interpolates and inventories owned/ghost stencil samples;
`sample_normal_diagnostic_lines` clips and evaluates complete line samples;
`write_particle_diagnostics` owns three substantial CSV outputs. The shared
`update_fe_strain_trial` has both trial and commit callers. No trivial forwarding
helper, new manager or general cache/evaluation framework was introduced.

History-audit switches retain extra collective FE sampling and constitutive
evaluations. Normal diagnostic callbacks retain extra interpolation/material
evaluation and derived stress-cache effects. Their lifetimes and assembly-point
conditions are preserved; they are not assumed to be passive file formatting.

Remaining restrictions include horizontal/unit-tangential FE geometry,
homogeneous coefficients, fixed mesh and no history transport/birth. Broader P2
physical/temporal/stability qualification is unchanged. No selected cleanup work
remains; stop for review. Proposed next task: a bounded stationary BP3 port of
accepted FE history/normalization to the actual dipping geometry, with a frozen
profile comparison against the retained particle backend and the constitutive
distinction explicit. No BP3 implementation or testing is selected here.

Evidence: `benchmarks/reconstructed_fault/propagation/p2_cleanup_evidence/`
contains recovery/compilation/export/preservation/comparison records;
`native_mpi_evidence/p2-cleanup-recovery-*` contains native inputs/run identities;
`benchmarks/reconstructed_fault/refactoring_boundary/evidence/p2-cleanup-recovery-*`
contains bounded commands, outcomes and logs. Native outputs stay under this
worktree's `build-propagation-integration/runs/`.
