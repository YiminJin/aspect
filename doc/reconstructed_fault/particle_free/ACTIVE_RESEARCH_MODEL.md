# Active research model: code reading path

This is the entry point after the November 2026 consolidation. The supported
fault phase field is the semantic `phase_field` composition with **native DGQ2
Bernstein coefficients**. Ordinary ASPECT velocity, pressure, temperature and
other compositions retain their own FE_Q/FE_DGQ choices. The two physical routes
below share infrastructure, not constitutive histories or qualification.

| Route | Physical history | Time update and ownership | Current qualification |
|---|---|---|---|
| Fixed-direction bulk tip | Full plane-strain physical tau, cohesive A, phase, one age and signed slip per section | BE reference; shared experimental midpoint; ordinary simulator acceptance | Bounded local growth, derivatives, power and recovery; no resolved propagation or instability |
| Growth-off mature moving fault | Direct tau, or separated tensor histories with physical-stress reconstruction; moving curve/profile and surface age/slip | SSPRK3 tensor/phase transport, lagged accepted spin, retained endpoint reaction, joint material transaction | Retained constant/layered moving controls with their documented accuracy limits |

Start at [tip treatment](TIP_TREATMENT.md) for the bulk closure and
[mature moving fault](MATURE_MOVING_FAULT.md) for the distinct mature law.
[Consolidation evidence](TIP_CONSOLIDATION_REPORT.md) records the supported
settings, regression endpoints and retired-path boundaries. The compact-loading
experiment is closed. The [specimen assessment](PROPAGATION_SPECIMEN_ASSESSMENT.md)
rejects the current small specimen for the next physical growth test; its
numerical-control evidence remains valid. The subsequently selected
[synthetic specimen](SYNTHETIC_PROPAGATION_BENCHMARK.md) uses a new explicit
material/geometry calibration. Its fixed-phase preflight is followed by
[two freely evolving BE histories](SYNTHETIC_FREE_EVOLUTION.md), both starting
from the equilibrated t=0 seed. They show central broadening and weak sub-cell
extension with small timestep sensitivity, but numerical work loss still exceeds
added fracture energy. The subsequent [midpoint assessment](MATERIAL_MODEL_CLEANUP_AND_OVERNIGHT_RESULTS.md)
completes three coarse histories with much smaller physical-work errors and a
passing final temporal screen. Fine-mesh accepted-prefix growth differs strongly;
its scalar-solver failure is repaired through one interval, not through the full
.06 s trajectory. BE remains default and midpoint selectable. Complete that
spatial comparison before longer growth or instability calibration. Shared
age/RSF ownership and scalar residual correction are documented in the material
and [time-integration](tip_treatment/time_integration.md) guides.

## One bulk-tip step

1. [`Parameters::parse_parameters`](../../../source/simulator/parameters.cc)
   admits the single DGQ2 Bernstein phase composition.
   [`PhaseFieldFault::parse_parameters`](../../../source/material_model/phase_field_fault.cc)
   admits integrated bulk-tip or growth-off transported mature physics. Old
   Q1/CPDI, nodal phase and replicated endpoint settings are rejected.
2. [`Simulator::solve_single_advection_iterated_newton_stokes`](../../../source/simulator/solver_schemes.cc)
   constructs one `BulkTipCoupling` from the accepted parent. It owns the
   nonlinear search, unchanged mechanics/KKT/ledger checks and pending acceptance.
3. [`BulkTipCoupling::evaluate`](../../../source/simulator/solver/bulk_tip_assembly.cc)
   assembles physical mechanics, phase VI and work using native Bernstein
   coefficients directly. [`Tip::update/derivative`](../../../source/material_model/phase_field_fault/bulk_tip.cc)
   supplies the rational local tau/A elimination;
   [`FaultFriction::section_endpoint`](../../../source/material_model/rheology/fault_friction.cc)
   condenses signed section friction and its age-dependent derivative.
4. [`compute_bulk_tip_search_direction`](../../../source/simulator/solver/bulk_tip_linear.cc)
   solves the release-capable linearized box VI with the existing ASPECT Stokes
   inverse. The small direct oracle is in
   [`bulk_tip/checks.cc`](../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/checks.cc),
   not a production alternative solver.
5. [`PhaseFieldFault::prepare_bulk_tip_candidate`](../../../source/material_model/phase_field_fault/bulk_cohesive_history.cc)
   constructs endpoint tau/A, phase, age and signed slip once from the accepted
   parent. Ordinary timestep acceptance publishes it; rejection leaves the parent
   unchanged. Stage observations are not additional physical histories.

The equations, all factors of two and work identity are derived in
[time integration](tip_treatment/time_integration.md). BE is unchanged; midpoint
uses half-duration algebra for its stage, endpoint extrapolation, the equivalent
stage bound `(1+b_n)/2`, and exact constant-stage-rate aging. Neither method
changes the cohesive energy or section work adjoint. No tolerance was relaxed
in the bounded repair audit.

## Spatial operations and moving-state transaction

[`FE_DGQBernstein`](../../../include/aspect/fe_dgq_bernstein.h) supplies the scalar
phase representation. The existing native phase assembly and SSPRK3/BP path
lives in [`native_phase.cc`](../../../source/phase_field/native_phase.cc),
[`dg_transport.cc`](../../../source/phase_field/dg_transport.cc) and
[`dg_sipg.cc`](../../../source/phase_field/dg_sipg.cc).
[`assemble_dg_fracture`](../../../source/phase_field/fracture.cc) is the owned-row
SIPG fracture operator used by the tip residual, tangent and seed obstacle solve.
The phase handler retains geometric/degradation laws, native setup and this
operator; it no longer carries a second CPDI/Q1 solver.

For mature motion, [`Simulator::solve_timestep`](../../../source/simulator/core.cc)
prepares accepted material inputs, transports tensors and the native phase,
stages geometry/profile, freezes the rotated history, solves mechanics and
completes the joint acceptance transaction. This remains the ordinary simulator
pipeline, not a dedicated mature timestep driver. Material ownership stays in
[`material_history.cc`](../../../source/material_model/phase_field_fault/material_history.cc)
and its existing geometry/stress helpers.

[`TensorAdvection::assemble/advance`](../../../source/material_model/history/tensor_advection_distributed.cc)
uses the same owned-row cell/neighbor operator and ghost refresh on **every rank
count**, including one. [`TensorAdvection::project`](../../../source/material_model/history/tensor_advection.cc)
projects an explicit evaluator. The material equation's divergence correction,
SSPRK3 stages, CFL guard, inflow callbacks, spin lag and physical endpoint forcing
are unchanged. The independent sparse serial assembly is solely a
[benchmark reference](../../../benchmarks/reconstructed_fault/particle_free/support/serial_tensor_advection.cc).

## Checkpoints and archived controls

Physical-law, phase-basis and numerical-method identities retain their meanings.
There is no automatic old-checkpoint conversion. Bernstein BE/midpoint restarts
must match their method; explicit common-parent benchmark import remains separate.
Mature DG checkpoint identity retains the historical `Material profile transport`
string and its support admission, even though both old Q1 names now identify
metadata only. They do not dispatch a remapper. The actual phase algorithm is
native SSPRK3/BP. This preserves existing DG checkpoint interpretation without
claiming that old Q1 checkpoints are supported.

Historical reports/inputs and protected executables remain available. Old
`phase_field_*` tests and the PF1–PF6/replicated plugin umbrella are no longer
registered against this implementation; maintained independent tests and current
coupled fixtures have their own CMake projects. The cleanup inventory records
what was retired and what remains for history interpretation. Original data
losses—39/45 pilot frames, early temporary logs and unavailable old
`bulk_continuation.zip`—are unchanged.
