# ASPECT architecture refactor — 4 November 2026

The selected change removes the rejected mature timestep driver and restores
ordinary simulator/solver ownership. It preserves the direct-stress implementation
and final converged-slip publication fix. The maintained algorithm is
[MATURE_MOVING_FAULT.md](MATURE_MOVING_FAULT.md); the before/after ownership sketch
is in [the plan](ASPECT_ARCHITECTURE_PLAN.md).

## Actual baseline and preservation

Root: `/home/ein/repository/aspect-pf-rsf-particle-free`; branch:
`pf-rsf-particle-free`; HEAD: `a22214d57f3ce809b3b20c9883b853c96e04033f`.
HEAD alone was **not** the incoming tested source: the previous refactor was dirty.
The entry source hashes were checked against its completed provenance. Incoming
source archive/patch, six copied runtime binaries and the preceding 45 protected
hashes are recorded in
[`evidence/aspect_architecture`](../../../benchmarks/reconstructed_fault/particle_free/evidence/aspect_architecture/).
No reset, checkout, fetch, push or production restart occurred. Historical/manual
outputs and cleanup losses remain intact.

## Implemented ownership boundaries

| Owner | Implementation and interface |
|---|---|
| Simulator | Ordinary `solve_timestep` retains the common endpoint snapshot, predictor, four explicit material preparation operations, original scheme dispatch and existing pending publication callback. Normal timestep control decides acceptance. |
| Coupled solver | `solve_reconstructed_fault_stokes` owns the 24-iteration projection-consistency loop and calls the same native inner solver. Its final call retains the converged private slip guess. |
| Material | `material_history.cc` owns history lifetime/preparation/rollback and material admission; `stress_evolution.cc` supplies transport inflow, separated-memory transformation and spin field; `stress_reaction.cc` supplies the endpoint law/localization and correction response. `evaluate` runs no global solve or physical advancement. |
| History discretization | Independent `HistoryDiscretization::TensorHistory` owns numerical coefficients/FE space; `TensorAdvection` owns serial/distributed advection assembly and SSPRK3. Explicit projection receives fields/evaluators, cells, velocity and quadrature. These components do not depend on `PhaseFieldFault` or interpret friction. |
| Geometry | Manager `stage_material_motion(dt,velocity,gradient)` maps coordinates and normal supports. It delegates validated clone/staging to existing infrastructure and carries generic properties unchanged. `material_geometry.cc` supplies native FE samples, Q1 profile and normalization; it assigns material meanings. |
| Mechanics/observation | `physical_stress.cc` supplies common physical stress. Direct returns tau before separated reconstruction. Existing assemblers, work traction and acceptance diagnostics retain their physical response. |
| Persistence | `dg_accepted.cc` / `dg_transfer.cc` retain checkpoint packet ordering, law/material identity and accepted transfer semantics. Independent tensor types replace nested containers without converting archives. Cohesive formed-side physical histories remain distinct. |

Removed completely: `mature_material_step.cc`, `solve_mature_material_timestep`,
`solve_mature_projected_reaction`, `make_mature_step_rollback`, `MatureRollback`,
`solve_timestep_equations`, their declarations/instantiations and dispatch.
No replacement executor/class/pipeline/signal was introduced. The prior
projection-input wrapper is removed; the numerical projection takes actual fields.
The useful accepted-input view and rollback/candidate state are now material-owned
in `material_history_internal.h`, independent of any special driver. Useful
physical-law algebra moved from geometry utilities to material rheology.

Active coefficients are named `coefficients`; the internal history selector and
raw evaluator are `uses_dg_tensor_history` and `evaluate_history_tensor`.
`DG trial stress` remains the legacy input spelling at the parser boundary.
Representation tags identify physical tau versus auxiliary T; direct Z remains
diagnostic. The separated default and every physical/default parameter remain.
Connected checker/support callers were updated, including the stationary runner's
header freshness check. CMake's existing recursive source discovery handles the
removed/new files without another registration framework.

Transport → geometry/profile → rotation → reaction ordering, accepted-gradient
spin lag, source/coefficient sampling, mass projection, age/friction conventions,
inner/outer tolerances and terminal residual/KKT gates are retained. Accepted,
transported, rotated and reaction candidate histories remain distinct. The normal
endpoint callback joins bulk/surface/geometry publication once; rejected attempts
and exceptions restore the same parent and solver bookkeeping.

Comparable ASPECT patterns were inspected in registered viscoelastic material
models/rheology, `SimulatorAccess`, assembler registration, simulator schemes,
signals and manager clone/swap. The [official extension guide](https://aspect-documentation.readthedocs.io/en/latest/user/extending/index.html)
was consulted. No unrelated signal was used to hide ordering.

## Verification

Core, four native checker DSOs and BP3 support build successfully. The existing
unit suite passes **183170 assertions in 152 cases** (7.80 s). Eight affected
translation units compile independently without unity/PCH. Final source hashes
match the frozen manifest; the build-consistency check is clean. All **51**
protected files match their recorded hashes. Formatting was limited to the new
numerical/material components.

All seven sequential moving controls pass. They reuse identical incoming
arguments; parsed material parameters confirm direct/separated selection,
material-Lie rotation, profile and inflow choices. The declared 5e-13 bulk /
5e-12 surface screens pass. Accepted clocks, geometry/history publication counts,
linear iterations, projection-outer indices and line-search rejection counts
match. Floating alpha values are compared numerically, not as exact log text.

| Moving control | Bulk max | Surface max | Physical tensor L2 | Work traction max | Seconds |
|---|---:|---:|---:|---:|---:|
| Constant direct | 8.88e-16 | 5.42e-20 | 8e-17 | 4.8e-18 | 20.70 |
| Constant separated | 1.08e-16 | 5.42e-20 | 1.76e-17 | 1.08e-19 | 11.94 |
| Layered direct | 0 | 0 | 0 | 0 | 23.87 |
| Layered separated | 0 | 0 | 0 | 0 | 18.25 |
| Default Q1/zero inflow | 3.38e-15 | 1.73e-18 | 2.13e-15 | 2.36e-16 | 15.40 |
| Rejection/retry | 1.33e-15 | 8.13e-20 | 6.19e-16 | 3.94e-17 | 22.31 |
| Restart on four ranks | 0 | 0 | 0 | 0 | 8.94 |

Bulk/surface maxima combine export columns of different units and serve as
regression screens. Tensor and work traction are independently reconstructed from
the exported fields. Single wall times are observations, not speedup claims.
The explicit profile/gradient projection check is **1.96081e-15 < 3e-12**.

The four-rank restart continues the preserved local two-rank layered checkpoint
and matches this refactor's uninterrupted run within **2.47e-15 bulk / 1.07e-14
surface**. Completed-step rejection matches the retained accepted cutback schedule
within **1.58e-15 / 1.25e-14**. Exception and recoverable-no-candidate branches
retain the common restoration policy by source review; individual throw sites
were not separately fault-injected.

The additional stationary fault case passes one physical step on two ranks.
Accepted-step data, objective summary, both surface profiles and both ranks'
final tensor coefficients are **byte-identical** to the incoming-binary baseline.
The ordinary non-fault single-advection/single-Stokes case completes initialization
and two physical steps; all reported statistics, including solver iterations,
are **byte-identical**. These checks exercise ordinary dispatch, not a production
BP3 event. See `dispatch-equivalence.json`.

The incoming non-fault and stationary baselines passed. The first stationary
launch failed before model initialization because the sandbox blocked MPI sockets;
the authorized two-rank retry passed. The first build exposed a missing explicit
instantiation for the new geometry overload; the instantiation was added. Original
failure logs are retained. Matching checkers were rebuilt after interface changes;
no numerical criterion was relaxed.

Nine affected serial checker targets rebuild successfully. The existing limiter
and stabilization support fixtures pass on one rank (3.87 s each); all their CSV
numeric values are identical to the retained controls, excluding timer columns.
These exercise the extracted serial numerical kernels, not a newly enabled
limiter or production configuration. See `serial-comparison.json`.

Evidence includes exact inputs/commands, final source/binary hashes, failed and
successful build/launch logs, seven-case equivalence, lifecycle crosschecks and
configuration audits. Eleven bounded native controls were run sequentially;
no further verification campaign is needed for this structural task.

## Qualification boundary

This is structural implementation and bounded behavioral verification. It does
not improve or requalify finite-end approximation, energy/flux accuracy, moving
AMR/birth, cohesive tip/energy failures or production readiness. Defaults and
checkpoint meanings remain. Scientific campaigns are paused; the next justified
scientific experiment remains a frozen endpoint-aware stress-space comparison
with matching work integration, **only on a new user selection**.
