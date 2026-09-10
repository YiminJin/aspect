# Stage K: K0 inventory and proposed K1 benchmark

**Bounded transfer correction and timeline verified:** DGQ1-only Maxwell fields
hit existing vertex-indexing and mixed-base batching incompatibilities. The
approved alternative retains continuous Q2 and replaces shared-node INSERT by
incident-cell MPI ADD/count in the generic particle composition transfer.
One-/two-rank and traversal differences are at roundoff (history <=2.274e-13 Pa;
weak load <=8.029e-14); constant and independent weak-integration controls pass.
One unchanged-physics coarse replay through 0/.5/1 s took 38.01 s and verifies
actual first-assembly inputs consume the preceding commit, with supplied stress
retained at zero. Bulk VTU stress compositions show old published inputs;
particle VTU stress shows terminal commits. No extra computational lag exists
in this measured timeline. Three nonlinear acceptances, 17 fresh linear checks
and 52 Stage-I unit assertions pass. Initial projections/profile/I_h are
unchanged; at 1 s weak-traction RMS changes by 6.366e-5 Pa. No convergence
campaign or K2.3 was run. Gate K2 and the provisional reference remain unchanged.
See `stage_K2_stress_transfer_timeline.md` for scope, tests, limitations and
recoverable evidence. Earlier unimplemented/proposed statements below are
historical records, superseded by this approved bounded correction.

**K2.2 campaign accepted complete; separate transfer audit requires review:**
the reference remains provisional and Gate K2 unmet. A bounded frozen-data
8x32-cell test passes the forward-production, constant and independent FE
weak-integration controls on one/two ranks, but reversing shared-node transfer
writes changes the actual constrained history by 4.431 Pa RMS and its weak
load by 80.7%. Reverse-order one/two-rank loads differ by .583%; normal-order
MPI loads agree to roundoff for this partition. The physical constraint lift
is measured separately from published history. The proposed smallest fix is
an explicit MPI ADD/count incident-cell average at shared continuous DoFs;
it is not implemented and needs transfer-policy review. No connection to
the K2.2 temporal plateau is established. Runs took 1.386/3.822 s; no further
timestep, cancellation analysis, production change, or K2.3 run was made.
See `stage_K2_history_transfer_audit.md` and `nonuniform/history-transfer/`.

**K2.2 approved sequence completed; temporal reference needs review:** the
accepted Cartesian/lookup performance baseline is frozen and archived. Only
the missing .25/.125-s cases were completed (1750.999/2908.272 s, about
5.52 GiB each), with opt-in timings. All 155 overlapping exports match saved
states byte-for-byte. The three timestep trajectories pass 31 final nonlinear
acceptances, 136 fresh linear checks, unchanged profile/initialization and
both separate 1e-4 allowances. Spatial evidence is reused, not rerun.
Total errors contract, but at 2 s actual weak-traction anomaly differences
barely decrease (.001056798 -> .001035198 Pa); raw-stress anomaly differences
decrease slowly (.04943618 -> .04429940 Pa). Signed-component cancellation is
measured, but an asymptotically resolved nonuniform temporal reference is not
established. No production/criterion change or additional run was made.
All runs have finished; stop for review before further expensive cases or
K2.3. The separate bulk-history transfer concern remains separate. See
`stage_K2_2_temporal_completion.md` and
`nonuniform/domain-convergence-completion/` for the complete assessment,
saved-data diagnosis, unchanged baseline snapshot, numerical checks and costs.
Older entries below are historical execution records.

**Accepted performance baseline closed; approved K2.2 completion resumed:**
the exact lookup reuse and Cartesian early rejection are frozen. Recoverable
source archive, working-tree patch, executable, plugin and per-file manifest
are in `nonuniform/performance/accepted-cartesian-baseline/`. The tested
executable SHA is e2451b97ad5a76ba431967c4938bccb5b94cf910ff1182f9bd830e7666f1b453.
No further I_h integration/cache development belongs to this task.
The t=1-s temporal anomaly reversal is traced in saved weak terms to
cancellation; slip's endpoint/time-load accounting closes to 1e-19 m, and
cohesive accounting leaves a separately reported small projection remainder.
This is an interpretation of the reversal, not a temporal convergence pass.
Only the approved .25/.125-s cases are being completed, under their existing
9000/14400-s caps, in `nonuniform/domain-convergence-completion/`. No executable
checkpoint exists, so replayed prefixes must match the saved accepted data.
Both runs use distinct physical cores, opt-in timings and recorded peak RSS;
spatial and dt=.5 results are reused. See `temporal-cancellation.json` and the
completion README for the decision and pre-run resource bounds. K2.3 remains
out of scope, and production/criterion changes still require review.

**Production Cartesian lookup verified; K2 resume point:** exact
MappingCartesian support is now active in the real undeformed-box I_h path.
The bounded K2 t=0/.5-s replay genuinely converges and reproduces all 19 saved
exports byte-for-byte; profile preparation is 3.194/1.636 s cold/warm, with
2,448 excluded requests and 1,295 warm batch hits. The same-Cartesian nine-profile
comparison improves cold time 17.217 -> 3.715 s with unchanged integrals and
found/missing sequence; one-/two-rank checks pass. See
`nonuniform/performance/cartesian-review.md` and its machine-readable artifacts.
Resume from the completed spatial sequence and preserved partial temporal
states, not fresh full-fine runs. The previously reported mean-removed temporal
trend remains unresolved; no Gate K2 claim follows from this performance check.
Use `ASPECT_FAULT_PERFORMANCE=1` for the next necessary authorized run and
obtain runtime approval before exceeding the current bounded-run budget.
Historical stopped/in-progress entries below remain the execution record.

**User-requested stop and review, 2026-09-10:** both temporal processes are
stopped (SIGTERM, confirmed absent); do not restart automatically. The
domain-rule 32/64/128 spatial sequence is complete through 2 s, with
contracting errors and both 1e-4 allowances satisfied. The .25 s run has
accepted outputs through 1.75 s and the .125 s run through 1 s. All 17
completed temporal states pass the final solver criteria and 79 fresh
linear checks. Initial temporal datasets are byte-identical. Partial total
errors contract, but mean-removed q/C/slip errors do not contract uniformly:
at 1 s q anomaly RMS differences are 2.81087e-4 then 6.53755e-4 Pa across
successive timestep pairs. The cause is not diagnosed; no production
correction is inferred. Later temporal normalization and raw bulk-stress
comparisons are unmeasured. K2.2 remains incomplete, K2.3 unstarted, and
the bulk-history transfer concern separate. See the updated
`stage_K2_2_domain_convergence.md` and `nonuniform/domain-convergence/`
for the review, preserved outputs, costs and partial checks. Earlier
"in progress" entries below describe the execution history, not active jobs.

**K2.2 baseline accepted; new-rule convergence resumed, 2026-09-10:** the user
accepts the global-accumulator correction and demonstrated endpoint improvement
as the tested baseline. Solver/quadrature development is closed absent a new
reproducible failure. Evidence under `nonuniform/global-accumulator/` is
preserved; source snapshots and the executable/plugin hashes are recorded in
`nonuniform/domain-convergence/`. No new-rule 64/128 checkpoint exists, so
complete through-2-s runs are required, with shared-prefix checks against
the accepted runs. The new 32/64/128 sequence now completes through 2 s using
the domain rule throughout; all 15 nonlinear acceptance and 75 fresh linear
checks pass, as do both support/normalization allowances. Shared prefixes for
64 and 128 are byte-identical to the accepted through-1-s data. At 2 s the
actual weak-traction anomaly RMS contracts from .00112168 to .000165367 Pa;
raw bulk-stress anomaly RMS contracts from .0686696 to .0341007 Pa. Initial
projection offsets remain explicitly reported. This supports the planned
fine-grid .5/.25/.125 time sequence: .5 is reused and the two finer time runs
are in progress, with resource estimates recorded before launch.
Both separate 1e-4 allowances and all
physical/solver settings are retained. Stop before K2.3; the bulk-history
transfer concern stays separate. See `stage_K2_2_domain_convergence.md` for
the current record and resource estimates.

**Approved separate-global accumulation, 2026-09-10 (bounded verification
complete):** frozen Maxwell/profile weak loads now remain independent of
unknown-dependent bulk/BV terms through constraints and MPI, with completed
global vectors combined afterward. No equation, surface rule, history,
support, I_h, tolerance or iteration budget changed. The unchanged positive
Stage-I fixture genuinely converges; represented-increment and nonzero-dV
audits pass on one/two ranks. Geometry/Stage-I units (815 assertions per rank),
the 12-case focused coupling/load/lifecycle batch and a fresh sequential
two-rank restart pair pass. The prepared short K1 and K2-64 replays reach 1 s
with final convergence and measured support/normalization checks passing.
K1's coarse initial raw-stress maximum retains its known pointwise-resolution
limitation, numerically matching the saved same-mesh baseline; later raw
stress and all mean/history checks pass. K2-128 also completes through 1 s
(1651.82 s, 4.29 GiB), passing final criteria and measured allowances. The
mean-removed left/right endpoint traction mismatches at 1 s fall from
3.67339e-3 / 3.66451e-3 Pa to 5.71007e-5 / 6.99507e-5 Pa in magnitude.
Whole-fault traction-anomaly RMS falls from 4.92231e-4 to 2.48890e-4 Pa,
but the new initial central discrepancy is larger and raw bulk stress
differences remain. Initial phase/particle histories are unchanged; changed
surface projections are recorded, not reset. Gate K2 is not declared passed;
the larger campaign and separate bulk-history transfer concern remain held
for review. See `stage_K2_global_accumulator_review.md`
and `nonuniform/global-accumulator/`. This supersedes the pending-correction
hold below; the older entries remain as the recoverable decision history.

**Bounded residual-consistency audit, 2026-09-09:** the polyline quadrature and
its passing evidence are preserved. The unchanged failing Stage-I fixture
identifies mixed residual/load accumulation as the cause: at the audited
linearization the velocity affine mismatch is 5.74370e-10, whereas represented
increment effects are 2.14881e-20. Newton-only assembly agrees with A to
4.27230e-21 and the frozen load does not change. Cell-only separation remains
inconsistent at 1.36437e-10; independent global accumulation gives 4.27230e-21
consistency. All eight surface nodes are active and dV=0. No stopping rule,
physics, support, initialization or quadrature is changed. Positive tests now
explicitly require the final convergence criteria; intentional failure tests
remain separate. The proposed residual-accumulation correction requires review
before implementation and before short K1/K2-64/K2-128 trajectory replays.
See `stage_K2_residual_consistency_audit.md` and the new `nonuniform/residual-audit/`
artifacts; the preceding nonlinear hold below remains in effect.
The completed audit replay reproduces all twelve residual/rejection records
exactly and now fails the explicit positive convergence guard (exit 1).
Wall time is 1042.02 s, peak RSS 633696 KiB. Strengthened positive temperature
checks and rollback pass on one/two ranks; linear exhaustion and all 815
quadrature/Stage-I unit assertions per rank pass. No golden output is refreshed.

**Polyline extension: nonlinear verification hold, 2026-09-09:** the approved bounded map
is defined in `stage_K2_polyline_quadrature_addendum.md`, written before its
production implementation. Actual reconstructed polylines are partitioned
without flattening, normal clipping or changes to parent admission/P0 inputs.
The installed Voro++ cut/copy probe favors the existing 2-D polygon
representation. Final one-/two-rank geometry and nonlinear-safeguard tests
pass (815 assertions in 18 cases per rank), including the captured formerly
blocked geometry. All seven selected surface/condensed fixtures pass. The
concurrent integration batch is 14/19: three wall-time timeouts, a stale
rate-state golden output and a stale-reference restart comparison. An isolated
dynamic-pressure Stage-I replay then genuinely exhausts its 12 nonlinear
iterations: last reported bulk residual 3.47631e-10 versus target 5.88894e-13,
with fresh linear residual 3.17288e-17. Its configured continue-on-failure
strategy returns zero; this is not convergence. The fresh isolated restart
create/resume pair passes (340.34/119.93 s). Short K1 and 64/128 K2 replays are held; no solver, support or
stopping-rule workaround is made. See `stage_K2_polyline_quadrature_review.md`.
Evidence is under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/polyline-quadrature/`.
The previous straight-only stop below is historical, not the current geometry
scope. This uncommitted revision is not yet a verified trajectory baseline.

**Domain quadrature implementation stop, 2026-09-09:** the approved integrated
rule is implemented for straight 2-D faults and passes production-domain
manufactured tests on one/two ranks (149 assertions per rank). Five coupling
fixtures fail before derivative checks because their reconstructed geometry
is not straight; a bounded replay measures a 1.47e-7 m chord deviation versus
1.71e-14 m roundoff. The first implementation's geometry scope is too narrow.
See `stage_K2_domain_quadrature_addendum.md` and
`stage_K2_domain_quadrature_progress.md`. No fixture straightening/tolerance
relaxation or trajectory replay was performed. A polyline-domain partition
needs geometric review; K1 recheck and 64/128 through-1-s comparisons remain
pending. This partial, uncommitted revision is not a verified baseline.

**Endpoint-moment regression, 2026-09-09:** constant reproduction passes but
smooth nonconstant transverse fields expose displacement-dependent endpoint
weak-moment errors. Periodic areas alone fail the odd-field control; full Q1
domain moments on unchanged wall-clipped domains recover the reference mass
and decrease traction error with refinement. Ten focused analysis tests and
an independent Voro++ area cross-check pass. See
`stage_K2_endpoint_moment_regression.md` for the proposed surface-quadrature
revision (approval required), scope limits and separate history-transfer item.
No production correction or mechanical replay was made in this step.
The 64/128 replays through 1 s await correction approval and focused checks;
temporal refinement and true normal stress remain held. K2.2 is incomplete.

**Bounded endpoint diagnosis, 2026-09-09:** saved 0.5/1 s data and one
field-identical 64x256 replay identify moving wall-clipped particle-domain
volumes/coordinates as the dominant endpoint traction signature. Their signed
projection contribution accounts for about 96% of the 64--128 endpoint
difference at 1 s. Correct old particle history is verified; the bulk FE
transfer and its physical periodic constraint are separately measured.
See `stage_K2_endpoint_diagnosis.md`. No production correction or Jacobian
change was made. Endpoint-moment/seam semantics need review; the temporal
sequence and true-normal-stress branch remain held. K2.2 is not complete.

**K2.2 review stop, 2026-09-09:** the 32/64/128 spatial sequence completed
through 2 s. Support and actual slip-normalization checks pass under the
separately approved family allowance below. The late-time nonuniform actual
particle/Q1 traction shows an endpoint-localized plateau: adjacent RMS
differences at 2 s are 1.00656e-3 and 9.71051e-4 Pa. Temporal runs are held
under the user's explicit review gate; K2.2 and Gate K2 are not complete.
See `stage_K2_2_spatial_review.md` for initialization/raw-stress errors,
actual weak-balance checks, resources, artifacts and remaining uncertainty.
No production correction or true-normal-stress run was made.

**K2.2 follow-up:** the user accepted K2.1 as a feasibility pilot and explicitly
revised this fixed-profile, prescribed-pressure K2.1/K2.2 family's omitted-
fraction allowance from 1e-6 to provisional 1e-4. The independent 1e-4 actual
slip-normalization requirement is unchanged. The spatial refinement is
complete and awaiting review before temporal runs; see
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/refinement/README.md`.
Production solver/physics remain at the committed K1 baseline. The diagnostic-
only initial replay is field-identical to the pilot, and actual particle/Q1
balance checks pass. No true-normal-stress test is authorized by this revision.
Earlier containment status below is historical.

**Prior status before K2.2 authorization, retained for history:** Gate K1 passes with the reviewed K1-only
1e-4 containment allowance. `stage_K1_verification.md` records the completed
five-case spatial/time cross, raw stress and trajectory errors, MPI/restart
and focused failure regressions. Production correctness baseline is frozen.
The bounded K2.1 prescribed-pressure pilot completes through 2 s; see
`stage_K2_1_pilot.md` for its measured perturbation, coupled response and
ParaView outputs. Its normalization/lifecycle checks pass, but its omitted
fraction 5.914e-5 exceeds K2's original 1e-6 containment target. K2 inherits
no containment relaxation. K2.2 and true normal stress remain unstarted;
the refinement sequence is proposed only. The paragraphs below are historical.

Latest follow-up: `stage_K1_residual_floor_review.md` records the bounded
ASPECT infrastructure audit and the represented-increment diagnosis. Stable
cellwise velocity-gradient evaluation reduces removable assembly cancellation;
the remaining iterate-representation floor is handled by the user-authorized
mixed absolute/relative bulk criterion, with an independently computed
matrix/state precision scale fixed for convergence and merit. The separate
surface criterion, original pressure-compatibility cap, fresh linear checks,
configured tolerances/budgets and history-publication semantics are retained.
The corrected coarse half-second one-rank trajectory now completes through
6 s. One-/two-rank unit tests pass (52 assertions / 10 cases per rank), and
the completed coarse reference comparison passes its assumption gates.
Final-source MPI, restart, focused regressions and the existing five-case
spatial/time cross are still being completed; this is not yet Gate K1.

The accepted physical-pressure normalization and fresh-residual / pressure-
complement corrections remain in place. Their historical diagnoses and
verification are in `stage_K1_linearization_diagnosis.md` and
`stage_K1_linear_correction_review.md`.

Gate K1 remains unmet and K2 remains unstarted. Only K1 has the reviewed
provisional 1e-4 containment allowance; support, full I_h and the 1e-4 actual
slip-normalization requirement remain unchanged. The entries below are
historical records, not the current completion status.

Current bounded follow-up: `stage_K1_support_resolution.md` records the width
scan, separate retained-fraction diagnostic and authorized corrected mechanical
pilot. Initial mechanics is now accepted and compared; the first real timestep
fails its line search. A K1-only containment error-budget revision is proposed
for review, not applied. **K1 remains unmet; K2 remains unstarted.**

Pre-pilot gate review: `stage_K1_gate_review.md` audits the corrected initialization
against the existing Gate K1. Geometry, CPDI consistency, independent I_h and
initial cohesive projection pass, but strip containment still fails its 1e-6
target. Corrected accepted mechanics and spatial/temporal K1 convergence are
unmeasured. **Gate K1 is not satisfied; K2.1 has not started.** The records below
remain historical; they do not override the latest measured status.

Current follow-up: `stage_K1_pilot_report.md` records the approved initial
previous-profile correction and the bounded, incomplete K1 pilot. The K0
audit and proposal below remain historical, not a current execution status.

Follow-up: `stage_K1_prerequisite_report.md` records the authorized boundary
constraint correction, checked-in fixture refresh and a newly exposed
initial-profile history/root gate. The initial root screen below is historical:
it assumed previous phi equals the converged initial phi, unlike the source's
zero previous FE phase field. Do not use that screen as a verified pilot root.

Review packet, 2026-09-07. K0 has been audited and its focused baseline run.
K1 is proposed only: no K1 input, plugin, reference program, or benchmark run
has been implemented. K2 and later families have not started.

## Authority and revision

Production revision: `f4032b1824ef4892020af8c58ef981aca03dc0ec`
(`Reconstructed fault: add nonlinear history feedback`). There were no tracked
source changes at the beginning of K0. This progress document is the only
repository addition made for K0; older untracked documents remain untouched.

Authorities inspected:

- `doc/reconstructed_fault/current_design.md`, especially sections 13–14,
  18–25: initialization, associations, normalization, constitutive state,
  coupled mechanics, and history publication.
- `doc/reconstructed_fault/specification.tex`, especially the surface/bulk
  coupling, pressure scaling, Stage-I nonlinear solve, and Stage-J feedback
  subsections.
- `doc/reconstructed_fault/pf_rsf.tex`: the current local theory note;
  equations `eq:equilibrium`, `eq:tau coh`, `eq:H`, `eq:H+`, `eq:H small-step`,
  and `eq:g`. Its small-step approximation is not substituted for finite-step
  history evolution.
- `doc/reconstructed_fault/review_and_cleanup/stage3/stage3_J_recovery.md`:
  the committed final recovery/evidence record. Earlier F–J plans do not
  override this record or the corrected specification.
- `doc/reconstructed_fault/benchmarking/stage_K_benchmarking_instructions.md`
  and the repository's `refactoring.md` responsibility/validation rules.

## K0 source map and mathematical contract

All symbols below are in namespace `aspect`; paths are repository-relative.

| Responsibility | File and fully qualified symbol | Invariant or implemented equation |
| --- | --- | --- |
| Friction | `source/material_model/rheology/fault_friction.cc`, `aspect::MaterialModel::Rheology::FaultFriction<dim>::friction_coefficient` | Stateful `rate state` and stateless `rate dependent` are distinct dispatch paths. K1 selects regularized rate state. Material fractions average law parameters before nonlinear evaluation. |
| State and timestep | Same file, `aspect::MaterialModel::Rheology::FaultFriction<dim>::update_state`, `compute_time_step` | Exact constant-accepted-V aging update; mechanics uses old Theta. One global Dc. Rate strengthening has no operator-split RSF restriction; otherwise dt is bounded by CFL*a*Dc/(b*V). |
| Maxwell law | `source/material_model/phase_field_fault.cc`, `aspect::MaterialModel::PhaseFieldFault<dim>::compute_maxwell_coefficients`, `compute_maxwell_stress`, `compute_creep_viscosity` | beta=exp(-dt*G/eta), kappa=-eta*expm1(-dt*G/eta); stress=2*kappa*(sym grad u-upsilon*S)+beta*old stress. No rotation. Kappa is the Stokes viscosity. |
| Initial particle stress | `source/particle/property/maxwell_stress.cc`, `aspect::Particle::Property::MaxwellStress<dim>::initialize_one_particle_property` | User-supplied stress components enter through explicit compositional-field/property mappings. They remain frozen during Newton. |
| Initial H | `source/particle/property/crack_driving_force.cc`, `aspect::Particle::Property::CrackDrivingForce<dim>::initialize_one_particle_property`; `source/reconstructed_fault/manager.cc`, `aspect::ReconstructedFaultManager<dim>::initialize_crack_driving_force` | Baseline H=c^2/(2G). Prescribed normal profiles replace it where phi exceeds activation; elsewhere baseline H remains. This cutoff must appear in an independent initializer check. |
| Initial profile formula | `source/simulator/phase_field.cc`, `aspect::PhaseField::PhaseFieldProfile`, `aspect::PhaseFieldHandler<dim>::stationary_crack_driving_force` | Stationary profile tabulation initializes H, not the final FE phase field. H=Gc*alpha(phi_hat)/(c0*ell*h(phi_hat)*g(phi)^2) in the contributing region. |
| Phase equation | Same file, `aspect::PhaseFieldHandler<dim>::assemble_phase_field_system`, `evolve_phase_field` | CPDI history-field weak equation; no separate nodal irreversibility active set is implemented here. Nonlinear exhaustion throws; singular-branch trials are rejected collectively. The relative tolerance is anchored to each call's initial residual. |
| Geometry | `source/reconstructed_fault/manager.cc`, `aspect::ReconstructedFaultManager<dim>::reconstruct_initial_faults`, `reconstruct_initial_fault` | Reconstruction follows the converged initial Q1 phase solve. Ordered open polylines, normal-offset ridge fit, replicated geometry. Initial association widths come from the full prescribed stationary-profile support. |
| Associations and projection | `source/reconstructed_fault/utilities.cc`, `aspect::ReconstructedFaultUtilities::project_to_normal_profiles`; `source/reconstructed_fault/manager.cc`, `aspect::ReconstructedFaultManager<dim>::project_particle_scalar`, `rebuild_particle_projection_cache` | Finite-segment xi in [0,1], no end caps; particle-domain-volume weighted consistent Q1 projection. Only owned particles contribute, followed by small MPI reductions. |
| I_h | `source/material_model/phase_field_fault.cc`, `aspect::MaterialModel::PhaseFieldFault<dim>::compute_normalization_integrals`, `integrate_normalization_profiles`, `project_normalization_integrals` | Distributed adaptive normal profiles and consistent Q1 projection. One projected surface composition mixture for the complete +/-normal profile. Degradation uses max(raw phi,0); excessive negative undershoot below -1e-4 fails. The 1e-4 guard is empirical error detection, not a convergence parameter. g=0 is explicitly singular; no upper clipping. |
| Surface material/temperature | Same file, `aspect::MaterialModel::PhaseFieldFault<dim>::compute_fault_surface_temperatures`, `evaluate_reconstructed_fault_localization` | Sample frozen FE temperature at fault vertices, then Q1 interpolate. Projected chemical fields supply the surface mixture. Bulk coefficients instead use local bulk temperature/composition. Applicability requires L_mat much greater than ell. |
| Initial cohesive/state data | Same file, `aspect::MaterialModel::PhaseFieldFault<dim>::initialize_cohesive_state_from_initial_fields`, `prepare_reconstructed_fault_mechanical_solve` | Project g(phi_eff,p,f_Gamma)*sqrt(2*G_bar(f_Gamma)*H_p) to initialize C. Save I_h,0. Project supplied positive Theta once. Initialize V at its lower bound, then solve it. |
| Constitutive point response | Same file, `aspect::MaterialModel::PhaseFieldFault<dim>::evaluate_reconstructed_fault_point`, `evaluate_reconstructed_fault_bulk_point`, `compute_cohesive_response` | C=(kappa_Gamma*V+beta_Gamma*I_old*C_old)/I; upsilon=h*V/I+beta_Gamma*C_old/kappa_Gamma*(h*I_old/I-h_old). The shared localization implementation serves both particle and QP paths. |
| Surface system | `source/reconstructed_fault/surface_system.cc`, `aspect::ReconstructedFaultSurfaceSystem<dim>::assemble_surface_system`, `linearize_surface_system`, `surface_residual_rms` | R_Gamma is the owned-particle Q1 weak residual; K_V=-dR_Gamma/dV. MPI-reduced replicated blocks use UMFPACK. Convergence uses sqrt(r_F^T M_FF^-1 r_F / measure_F), not an unweighted nodal norm. |
| Bulk B and surface G | `source/simulator/assemblers/reconstructed_fault_stokes.cc`, `aspect::Assemblers::ReconstructedFaultStokes<dim>::linearize_B`, `apply_B`, `evaluate_slip_dependent_bulk_residual`; `source/reconstructed_fault/surface_system.cc`, `aspect::ReconstructedFaultSurfaceSystem<dim>::apply_G` | Bulk QP geometry is cached with exact quadrature identity. 2*kappa_b*chi*S is frozen per linearization. Residual includes both -B*V and the frozen history contribution. G uses physical bulk directions and the selected normal-pressure mode. |
| Condensation and scaling | `source/simulator/solver/reconstructed_fault_condensed_system.cc`, `aspect::StokesSolver::ReconstructedFaultCondensedSystem<dim>::linearize`, `Linearization::make_physical_bulk_direction` | Canonical simulator-owned components; one linearization lifetime. Solver pressure p_hat=p_physical/s_p, converted once by delta p=s_p*delta p_hat. Perturbation constraints are homogeneous. Semantic surface solves permit free-set restriction. |
| Nonlinear lifecycle | `source/simulator/solver.cc`, `aspect::Simulator<dim>::solve_reconstructed_fault_stokes`; `source/simulator/solver/reconstructed_fault_nonlinear.cc`, `aspect::internal::update_reconstructed_fault_active_set`, `reconstructed_fault_armijo_line_search` | FGMRES for generally nonsymmetric condensation. Stabilize active set, then fraction-to-boundary and Armijo; rejected candidates never commit. Rebuild the active set at later Newton iterations. |
| History publication | `source/material_model/phase_field_fault.cc`, `aspect::MaterialModel::PhaseFieldFault<dim>::commit_reconstructed_fault_mechanical_history`, `compute_crack_driving_force_candidate`; `source/reconstructed_fault/manager.cc`, `aspect::ReconstructedFaultManager<dim>::commit_slip_rate_nonlinear_solve` | All candidate calculations, allocations, MPI validation and commit checks precede terminal writes. Failure restores bulk and V and preserves histories. Timestep zero skips physical particle/surface history evolution. |
| Advection and cache lifecycle | `source/particle/manager.cc`, `aspect::Particle::Manager<dim>::advance_timestep`; `source/reconstructed_fault/manager.cc`, `aspect::ReconstructedFaultManager<dim>::particle_projection_cache_is_valid`, `invalidate_stokes_qp_projection_cache` | Advect particles through the normal ASPECT scheme; exchange ghosts before rebuilding Voronoi/CPDI domains. Particle IDs, locations, volumes and geometry versions invalidate associations. QP caches persist until mesh/geometry changes. |
| Restart | `include/aspect/reconstructed_fault/manager.h`, `aspect::ReconstructedFaultManager<dim>::save`, `load`; `source/reconstructed_fault/manager.cc`, `rebuild_after_deserialization` | Serialize geometry, widths, generic committed fields and committed V. Rebuild transient caches; recompute current I_h and surface temperature. Standard ASPECT checkpointing carries bulk/old fields and particles. |
| Timestep ordering/controller | `source/simulator/solver_schemes.cc`, `aspect::Simulator<dim>::solve_single_advection_iterated_newton_stokes`; `source/time_stepping/reconstructed_fault.cc`, `aspect::TimeStepping::ReconstructedFault<dim>::execute` | H_old -> phase -> thermal/composition/advection -> coupled mechanics -> history publication. Controller is opt-in and uses committed V. No new post-solve cutback/retry. |

For the selected law, the exact equations are

\[
\mu=a\operatorname{asinh}\left[\frac{V}{2V_{\rm ref}}
 \exp\left(\frac{\mu_0+b\log(\Theta V_{\rm ref}/D_c)}{a}\right)\right],
\qquad
\Theta_k=\Theta_{k-1}e^{-x}-\frac{D_c}{V_k}\operatorname{expm1}(-x),
\quad x=V_k\Delta t_k/D_c.
\]

The unregularized option replaces asinh by the logarithmic law. K1 does not
use that option or the stateless weakening law. The regularized derivative
uses (a/V)*Z/sqrt(1+Z^2); production uses its large-Z limit at Z>=1e6.

For real steps, raw accepted cohesive samples determine a consistent Q1 C
projection. The projected/interpolated C is used for H. Maxwell stress uses
the accepted local constitutive upsilon, not a strain reconstructed from the
projected C. With S=(s tensor n+n tensor s)/2, S:S=1/2 and tau:S=tau_xy in
K1. There is no engineering-shear factor of two to insert into V.

The finite-step H candidate is dt*(a_H-b_H)*(a_H+b_H)/(2*kappa_Gamma), with
a_H=C_k/g_k and b_H=beta_Gamma*h_old*C_old/(1-g_k). Commit max(H_old,candidate),
without separately clipping negative candidates. At g_k=1,h_old=0 use
dt*C_k^2/(2*kappa_Gamma); g_k=1,h_old>0 is rejected as healing.

Normal stress is either physical p-tau:N or prescribed p_ad. In the latter
mode neither dynamic pressure nor deviatoric normal traction contributes,
and G delta x=2*kappa_b*S:delta epsilon_dot. Constant adiabatic pressure can
be obtained with the existing computed profile, zero gravity and positive
Surface pressure. Full solution pressures are physical; Stokes directions
are scaled. A bulk pressure gauge cannot be chosen arbitrarily when true
normal stress enters friction. In K1's prescribed-pressure mode it is a
gauge: use volume normalization initially and compare gauge-subtracted bulk
pressure. The coupled terminal publication does not explicitly call the
ordinary normalize_pressure() path, so do not assume it reimposes a particular
mean pressure at each accepted step.

The approved timestep-zero contract is:

| Field | Initial solve / commit |
| --- | --- |
| V_0 and bulk u,p | Solve with the positive numerical Initial time step; publish accepted kinematics. |
| Theta_0 | Retain the supplied, projected value; no aging update. |
| H_0 | Retain prescribed initialized history; no finite-step H update. |
| C_0 | Retain the initialization-specific particle-to-Q1 projection, not the trial cohesive response from the initial mechanics. |
| tau_0 | Retain user-initialized particle stress, not the trial Maxwell response from the initial mechanics. |
| I_h,0 | Store as the previous-I_h snapshot for the first real step. |

Previous phi is evaluated from old_solution at the particle's current spatial
position. It is not a separately advected phi property. This distinction is
benign for the ideal x-independent K1 solution with u_y=0, but transverse
motion and profile drift must be measured. `Evolve phase field=false` freezes
H publication, not particle advection or the phase solve; C, Theta, stress and
V still evolve at real steps. Existing repeated phase solves can refine the
same frozen-H state further because their relative stopping scales restart.

Current coupled support is 2-D, separate velocity/pressure blocks, assembled
block AMG, the single-Advection/iterated-Newton-Stokes scheme, and no melt.
Geometry/topology is held fixed here. With a>b the selected RSF controller
returns the largest finite double; convection and the requested maximum dt
still constrain the accepted sequence.

## K0 measured baseline

Build: Debug executable `build-pf-cpdi/aspect`, Voro enabled, deal.II 9.6.2,
Trilinos 14.2.0, p4est 2.8.7, Open MPI compiler at
`/opt/openmpi/5.0.6/bin/mpic++`.

SHA-256 of the executable is
`432e0df9058430aae9060dfb79d96e188625c2edfc01a7c8a16a12736084501d`.
The Stage-J plugin hash is
`b07273c35996143c3d81bdac77290eda24443d499346d57224fd29ecff32015c`;
the dynamic and adiabatic condensed plugin hashes are respectively
`c2510ab3be850700c3c542fbd93ef7afc7f05cb4dedb12b21b6c6c1abb5bea02` and
`36368eaa33ae07a442b28017162f2865735dd69c6f11d23c088cadd7adcf97d1`.

Build and focused test commands:

```sh
cmake --build build-pf-cpdi --target aspect -j4
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_stage_j$' -j1
build-pf-cpdi/aspect --test '[phase_field_domain],*cohesive*,MaxwellStress*,Stage-I*'
build-pf-cpdi/aspect --test '[phase_field_fault_ih_accuracy]'
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_(rate_state|condensed_adiabatic|condensed_dynamic)$' -j1
```

| Check | Result on the current revision |
| --- | --- |
| Build | PASS, -j4; production source unchanged. |
| Stage-J evolving lifecycle, MPI 2 | PASS, CTest 320.99 s; application wall time 311.8 s. Initialization and two real steps converge in 20/15/12 phase Newton iterations to 9.395e-6 / 4.747e-6 / 4.879e-7 against 1e-5. Initial linear solves each take five iterations against 2e-7. |
| Stage-J observables | H grows 6.230% on step one, exact per-particle plateau on step two; phase increments 7.336 and 10.35; Theta increment/tolerance 4.545e5 and 9.182e3. One-owner failure preserves all state. |
| Constitutive/domain/nonlinear unit selection, MPI 1 | PASS, 65 assertions / 15 cases. |
| Existing adaptive and Q1 I_h accuracy, MPI 1 | PASS, 59,117 assertions / 2 cases. |
| Existing rate-state integration test, MPI 1 | PASS, 27.46 s. |
| Unmodified condensed adiabatic/dynamic CTests | FAIL, 53.06 / 57.73 s. Both exhaust their inherited 10-step phase budget at 1.293e-3 > 1e-5, before reconstruction. Their default continue-after-failure policy then reaches a postprocessor requiring absent geometry. These are not passing coupling baselines. |
| Temporary dynamic-pressure condensed fixture, MPI 1 | PASS, application wall time 125.4 s. With budget 50, phase converges in 20 iterations to 9.395e-6; existing surface/bulk/condensed assertions report verified. |
| Temporary adiabatic-pressure condensed fixture, MPI 2 | PASS, application wall time 68.23 s. Phase converges in 20 iterations to 9.395e-6 with five linear iterations per solve; existing surface/bulk/condensed assertions report verified. |

K0 diagnostic inputs and outputs are preserved in
`/tmp/aspect-stage-k0-NJV4th`. `stage-j.log` and `stage-j.screen-output`
preserve the lifecycle output. `legacy-coupling-ctest.log` preserves the failed
original CTests. Temporary `condensed-{dynamic,adiabatic}.prm` include the
generated existing fixtures, change only output directory, phase iteration
budget to 50, and failure policy to abort. Accuracy and test assertions are
unchanged. The direct commands, run from `build-pf-cpdi/tests`, are:

```sh
../aspect /tmp/aspect-stage-k0-NJV4th/condensed-dynamic.prm
mpirun -np 2 ../aspect /tmp/aspect-stage-k0-NJV4th/condensed-adiabatic.prm
```

Both direct runs exited successfully. These direct assertion runs do not
claim that the old golden-output CTests pass. No golden
output or production parameter default was changed. The last recorded Stage-J
restart, transverse-temperature, rollback and CPDI-regeneration results are
reused from the final recovery report: they exercised the same production
implementation committed in this revision. Their limitations, particularly
the restart fingerprint rather than a full entrywise comparison, remain.
No complete test suite or K1 experiment was run. MPI unit commands required
socket access outside the sandbox; this did not change numerical settings.

## K0 findings requiring review before K1

1. **Old regression inputs are stale.** The corrected exhaustion reporting
   exposes their insufficient phase budget. Both temporary successful
   runs identify this as an initialization-budget issue, rather than a
   demonstrated B/G/K_V defect. Updating these checked-in fixtures and their
   numerical output belongs in an explicit focused cleanup, not an unexplained
   K1 tolerance change.
2. **Nonzero velocity boundary values need a solver check/fix.** Source audit
   finds that `solve_reconstructed_fault_stokes()` copies `solution` to
   `working_x`, computes current constraints, and then updates `working_x`
   exclusively with homogeneous bulk directions. It does not first distribute
   the physical inhomogeneous velocity constraints to that copy. Initial field
   setup copies temperature/composition, then pressure, but does not publish a
   velocity boundary lift; later ramps change boundary constraints without
   changing the copied boundary values. Thus the usual homogeneous-perturbation
   invariant has no corresponding admissible base-state construction in this
   path. The zero-velocity Stage-J test does not verify it. This is a concrete
   source-level gap for K1's imposed shear, not yet a new runtime reproducer.
   Residual assembly also uses current constraints for its local/global scatter;
   a fix must check the residual/lift convention together, rather than merely
   writing boundary values after convergence. No production fix is made in K0.
3. **A frozen-H switch is insufficient for an exactly fixed FE profile.**
   The proposed K1-only prescribed-phase constraint fixture below makes the
   restriction explicit through existing ASPECT infrastructure. It must not be
   presented as the semantics of `Evolve phase field=false`.
4. **Periodic FE support is not proof of periodic particle domains.** Box
   periodic constraints and RK particle wrapping exist. The Voronoi builder
   uses nonperiodic local containers and unshifted neighboring cells. K1 must
   check boundary support/volume and along-fault uniformity; it must not claim
   periodic Voronoi topology has been verified. Finite fault endpoints remain
   separate endpoints, even with periodic bulk boundaries.

No constitutive equation conflict requiring a new model was found. Some
parameter help text (especially Initial time step's old return-mapping wording)
predates the corrected lifecycle; the current specification and actual commit
behavior above determine the benchmark initialization.

## K1 concrete proposal: finite-width homogeneous shear

This proposal is conditional on resolving the nonzero-boundary base-state gap
above. It uses a short x-periodic strip with square cells, so reducing the
tangential extent saves unknowns without extreme cell aspect ratios.

| Quantity / parameter | Proposed value | Units / meaning |
| --- | --- | --- |
| Domain | (0,0.25) x (-0.5,0.5) | m; L_s=0.25, W=1 |
| Prescribed fault | (0,0) -> (0.25,0), phi_hat=0.6 at both ends | Ordered open line; n=e_y, s=e_x |
| Geometry evolution | None after initial reconstruction | No endpoint joining or propagation |
| Phase model | AT1, alpha(phi)=phi, c0=8/3 | No AT2 |
| ell | 0.15625 | m |
| Cohesion c | 1,000 | Pa |
| G_b=G_Gamma | 1e6 | Pa; one homogeneous background material |
| eta_b=eta_Gamma | 1e8 | Pa s; lambda_b=lambda_Gamma=100 s |
| Viscosity cutoffs | 1e6 to 1e10 | Pa s; avoid default 1e17 lower cutoff |
| Critical energy release rate | 80/3 = 26.6666666667 | J/m^2 |
| Degradation curvature p | 1 | Dimensionless |
| Derived degradation m | 128 | m=Gc*2G/(c0*ell*c^2), not a new tunable parameter |
| Temperature | 293 everywhere; reference 293, thermal viscosity exponent 0 | K; static thermal field, no heating |
| Chemical composition | No chemical fields, background fraction 1 | Stress/state fields are not chemical fractions |
| Activation / upper admissibility | 0.1 / 0.99 | Existing normal activation and fault admissibility semantics |
| Normal stress in friction | sigma_*=1,000; adiabatic mode true | Pa; zero gravity, initialized computed adiabatic profile, Surface pressure=1,000 |
| Bulk pressure gauge | Initial volume normalization | Compare p minus its volume mean; log the raw mean too |
| Friction | `rate state`, regularized true | Production parameter spelling contains spaces |
| mu_0, a, b | 0.6, 0.025, 0.013 | Dimensionless; fixed-state slope positive, rate strengthening |
| V_ref / V_min | 1e-5 / 1e-12 | m/s |
| D_c | 0.001 | m |
| Theta_0 | 200 | s; deliberately not steady |
| eta^d | 1e5 | Pa s/m |
| Initial particle stress | tau_xx=tau_yy=0, tau_xy=1,500 | Pa, supplied through explicit Maxwell mappings |
| Numerical Initial time step | 2 | s; fixed across spatial/time refinement; not physical aging |
| Tangential velocity difference | U(t)=1e-4*[1+0.2*min(t/(4 s),1)] | m/s, sampled at ASPECT's current accepted endpoint time |
| Real steps / pilot duration | dt cap 2; t_end=6 | s; nominal initialization plus three real steps |
| Time selection | Explicit `convection time step, reconstructed fault time step`; CFL 0.5 | Record actual accepted dt; a>b makes the RSF limit inactive here |
| Pilot grid | 16 x 64 square cells, delta x_n=delta x_s=1/64 | 1,024 cells; box repetitions 1 x 4, global refinement 4 |
| Particles | Reference-cell 3 x 3 per cell, RK2, no addition/removal | 9,216 particles; Voronoi/CPDI enabled |
| Particle spacing | 1/192 | m |
| FE | Q2 velocity, Q1 pressure, Q1 phase; standard continuous auxiliary fields | Existing assembled block-AMG path |
| Fault spacing / ridge | 1/32 m / 1 | Eight segments, nine vertices initially |
| Phase evolution setting | `Evolve phase field=false` plus explicit benchmark-only frozen-phase constraints after initialization | H frozen; C, Theta, stress and V evolve normally |
| Algebraic accuracy | Phase linear 2e-7, phase nonlinear 1e-8; coupled nonlinear 1e-8, Stokes linear 1e-9 | Existing parameters, tighter than Stage-J accuracy |
| Iteration budgets / failure | Phase 50, coupled 30; abort on nonlinear failure | No success on exhaustion; revisit a budget only with convergence evidence |
| I_h quadrature / tail tolerance | 1e-10 / 1e-10 | Existing parameters |

Stress/state mappings will explicitly map `tau_xx`, `tau_yy`, `tau_xy` to
`maxwell stress[0..2]` and `theta_initial` to `phase field fault state[0]`.
The latter uses the established initial-composition particle property path.
All four fields use particles; their types are stress, stress, stress, generic.
There are no independently adjustable surface elastic or viscous constants.

An independent parameter-screening quadrature of the ideal stationary AT1
profile gives half support 0.308821594 m, I_h about 108.652564 m, and initial
H full width at half maximum w_H0=0.078753769 m. The pilot therefore has
ell/delta x_n=10, about 5.04 cells across w_H0, and about 15.12 particle spacings
across that width. Ideal H at the core is 29,568.05 Pa and at phi=0.1 is
16.8962 Pa; the production background H is 0.5 Pa. These are screening values,
not claimed outputs: activation cutoff, CPDI and the initial FE solve change
the realized profile. The calibrated m=128 avoids the extreme m=480000 of
the Stage-J smoke fixture while leaving a contained, measurable profile.

### Boundary and fixed-profile contract

Top and bottom prescribe (u_x,u_y)=(+U_k/2,0) and (-U_k/2,0). Left/right use
the existing Box X-periodic FE constraints. Apply periodicity to bulk fields,
not a new cyclic fault data structure. The homogeneous constant solution
satisfies the separate natural endpoint rows of the open surface system;
endpoint half support is accounted for by the consistent projection.

Check reconstructed vertex displacement from y=0, segment angle, endpoint
position, and all particle/QP association coordinates, including the two end
strips. Check constant surface fields at both endpoints independently. The
short pilot's estimated maximum displacement is less than 0.00036 m, below
the initial first-particle distance to a lateral boundary even on the proposed
finest grid; this pilot does not establish correctness after particle wrapping.
Volume/CPDI and endpoint-uniformity checks still apply. If periodic support is
unsuitable, stop: the explicit compatible alternative is lateral Cauchy data
t_right=(-p_*,q_k), t_left=(p_*,-q_k), or full lateral reference velocity from
K1-d, with their pressure/particle-inflow implications reviewed first. Zero
lateral traction is not an equivalent boundary condition.

Proposed fixed-profile fixture: let the initial production phase solve and
reconstruction finish normally. Save its Q1 values. For k>0, use the existing
`SimulatorSignals<dim>::post_constraints_creation` hook (or equivalent
PrescribedSolution plugin) to constrain every independent phase DoF to those
saved values. Preserve pre-existing periodic relations. This defines a
prescribed-phase benchmark: the phase residual has no free test functions,
so the ordinary phase call can return without advancing it. Keep the initial
unconstrained residual/accuracy check separately. Check exact phase-vector
preservation and consistent old/current phase sampling. On restart the frozen
field is recovered from the ordinary checkpointed phase vectors.

This fixture belongs solely in the benchmark plugin, uses existing constraint
infrastructure, and changes no production switch or public interface. It
does not overwrite particle stress, surface C/Theta, V, or H after each solve.
Reconstructed geometry stays fixed. H retains its user-initialized particle
values through the supported false setting, while normal advection continues.

### Initialization and reference independence

The production initializer first assigns H_c=c^2/(2G). For the prescribed
normal profile phi_*(y), it assigns

\[
H_0(y)=\begin{cases}
\dfrac{G_c\alpha(\widehat\phi)}{c_0\ell\,h(\widehat\phi)g(\phi_*(y))^2},
&\phi_*(y)>0.1,\\
H_c,&\text{otherwise}.
\end{cases}
\]

The independent initialization check reconstructs phi_* by quadrature of
\(d\zeta/d\phi=-\ell\sqrt{h(\widehat\phi)/
[h(\widehat\phi)\alpha(\phi)-\alpha(\widehat\phi)h(\phi)]}\), with the
cosine-squared endpoint substitution, then checks actual particle H against
the piecewise rule. It evaluates the CPDI weak residual from exported weights,
gradients, volumes and initial FE values with independently written AT1/g
formulas. This checks the supplied initialization data and discrete residual;
it is not the fully independent 1-D phase solver deferred to K3.

Independently assemble the small Q1 mass projection of
g(phi_eff,p)*sqrt(2G*H_p) from exported particle data and associations to check
C_0. Report its transverse residual, not only its mean. Perfect uniformity
in the normal direction is not presumed. Uniformity of the projected C along
the fault, and of the accepted bulk shear stress in space, is required for
the scalar reduction.

Export initial phi_h along several normal lines as FE-cell polynomial data,
not just a few output pixels. Reconstruct Q1 phi_h independently and integrate
h(max(phi_h,0)) with g=(1-phi)^2/[(1-phi)^2+m*phi*(1+p*phi)]. Split at FE-cell
boundaries and zero crossings. Compare successively tighter adaptive
quadratures; do not import production I_h as the reference. Independently
integrate over the actual association strip as well as the whole domain:

\[
\epsilon_{\rm strip}=
\frac{\int_{\Omega\setminus\text{strip}} h(\phi_h)\,dy}{I_{h,\rm ref}}.
\]

Require this omitted contribution to be negligible at the declared target.
Also report the outermost normal windows; compare the realized profile with
the compact-support prescribed profile without identifying them. A significant
omitted strip contribution invalidates K1-a. Do not silently renormalize V,
expand production widths, or change ell to hide it.

For the conditional time-update check, record the actual initial projected
C_0 and Theta_0 once. The committed initial Maxwell stress is the prescribed
1,500 Pa. Check their spatial ranges before using scalar initial values.
Retain them throughout independent reference integration; never reset from
ASPECT at later steps. Record separately the trial/evaluated initial
q_0^eval,C_0^eval and the retained committed q_0,C_0. They need not be equal
under the approved timestep-zero contract. Compare the initial mechanical
solve with the reference evaluated using the numerical 2 s interval, but do
not advance reference committed histories through that interval.

For each real accepted timestep use independently computed beta and kappa
and solve the strictly decreasing scalar function

\[
F_k(V)=\beta_bq_{k-1}+\frac{\kappa_b}{W}(U_k-V)
-\frac{\kappa_\Gamma}{I_h}V-\beta_\Gamma C_{k-1}
-\sigma_*\mu(V,\Theta_{k-1})-\eta^dV=0.
\]

Then update q, C with the two corresponding Maxwell relations, and Theta
with the exact aging formula above. Use SciPy's existing Brent root solver
with a growing positive upper bracket; check F(V_min)>0 and an interior
root, not a clipped root. For the nominal ideal-profile screen, the initial
root is about 3.16e-4 m/s; the first real-step state increment is about -94 s.
Those estimates are not expected outputs for the actual FE profile. They
show that this proposal can resolve a nonzero state update with generous
separation from V_min. Every actual root must still be checked.

Recover the velocity using

\[
u_x(y)=-U_k/2+\frac{q_k-\beta_bq_{k-1}}{\kappa_b}(y+W/2)
+V_k\int_{-W/2}^{y}h(\phi_h(z))/I_h\,dz.
\]

Here U is the imposed total velocity difference, not V. Compare both the
accepted FE velocity and newly committed particle shear stress; a successful
surface solve alone cannot validate the bulk stress publication. Also
compare raw/projected C in this along-fault homogeneous case, where their
projection difference should approach zero. Accumulated slip uses the declared
accepted-rate rule D_k=D_(k-1)+dt_k*V_k.

The reference will be a small benchmark-only Python program using the local
SciPy installation (1.18.0 was available for the K0 screening calculation).
It will not call production constitutive code. It will reject nonconverged
runs, missing columns, nonpositive roots and uncontained normalization.
Actual accepted times, dt and endpoint-sampled loading drive the direct
discrete comparison. For temporal accuracy, use a separately refined scalar
time integration at common physical times, holding initial data fixed; this
distinguishes matching an update formula from reaching a time-continuous limit.

### Error targets fixed before the pilot

Report dimensional errors and min/mean/max over the fault. Use
abs(error)<=absolute allowance+relative allowance*abs(reference), with fixed
scales V_scale=1e-4 m/s, Theta_scale=200 s, traction_scale=1,500 Pa,
velocity_scale=1e-4 m/s, and slip_scale=6e-4 m. No machine-minimum denominators.

| Check | Target |
| --- | --- |
| Initial unconstrained phase residual | Requested relative 1e-8; report absolute residual and initial scale too. Never infer convergence from exit code alone. |
| Coupled residual | Both blocks below 1e-8 relative; retain dimensional bulk norm and mass-consistent surface RMS. |
| Reconstructed line | max abs(y)/ell <=1e-6; max abs(segment angle)<=1e-6 rad; endpoints checked at the same scale. |
| Reference quadrature/root accuracy | I_h relative self-change <=1e-10 and absolute <=1e-9 m; scalar root bracket width <=1e-14 m/s and abs(F)<=1e-7 Pa. |
| Production I_h vs independent same-FE integral | Relative <=1e-6. |
| Unrepresented tail/strip fraction | <=1e-6; report domain truncation evidence separately. |
| Normalization identity | abs(integral upsilon - V)/max(abs(V),1e-4 m/s)<=1e-4. |
| Pilot V, q, C, velocity profile, cumulative slip | 2% relative plus 1e-5 of each fixed dimensional scale; a pilot passes only as a smoke/feasibility result. |
| Resolved conditional comparison | 0.2% relative plus 1e-5 of each scale; weighted velocity L2 and pointwise maximum both reported. |
| Theta update at each vertex | <=1e-10 of Theta_scale for the independently evaluated update using that vertex's accepted V; independent accumulated scalar-history error uses the resolved comparison criterion. Each expected increment must exceed 100 local update tolerances. |
| Along-fault symmetry | Range/scale <=1e-4 for surface histories and shear stress; report endpoint rows separately. |
| u_y and divergence | norm(u_y)/velocity_scale<=1e-4; W*norm(div u)/velocity_scale<=1e-4, with volume-weighted norms. |
| Fixed fields / lifecycle | No H history writes, no phase or geometry drift; compare per-particle H by ID. C/Theta/stress/V do change and survive restart. |

Also report max abs(log10(V_ASPECT/V_reference)) with positive interior
states, the raw nodal/bulk pressure gauge, all iteration counts, and the
minimum raw phi. An error plateau requires a diagnosis, not relaxed targets.

### Bounded execution and estimated cost

Proposed new layout after approval:
`benchmarks/reconstructed_fault/uniform_shear/` for the input, small diagnostic/
prescribed-phase plugin, README and output schema;
`benchmarks/reconstructed_fault/reference/` for the scalar calculation.
No expensive convergence campaign is added to the default integration suite.

The pilot command sequence will be explicit and run from the repository root:

```sh
cmake --build build-pf-cpdi --target aspect -j4
cmake -S benchmarks/reconstructed_fault/uniform_shear -B build-k1-uniform-shear -DAspect_DIR="$PWD/build-pf-cpdi"
cmake --build build-k1-uniform-shear -j4
/usr/bin/time -v build-pf-cpdi/aspect benchmarks/reconstructed_fault/uniform_shear/pilot.prm
python3 benchmarks/reconstructed_fault/reference/uniform_shear.py --run output-k1-pilot
```

These paths/commands describe proposed files, not files already implemented.
The future input will specify the compiled plugin and output directory
explicitly. Preserve the executable/input/plugin hashes and local diff with
each result, and save machine-readable histories plus sparse profile data.

Measured anchors are the current 4,096-cell/36,864-particle Stage-J run at
321 s on two ranks and the 125 s one-rank coupling diagnostic (78 s of which
is its unusually extensive postprocessor). The proposed 1,024-cell/
9,216-particle pilot eliminates later phase Newton solves but has three real
mechanical updates. Estimate **2–6 minutes on one local rank**, excluding
first compilation; reserve a 10-minute pilot limit. Budget approximately
0.5–2 GiB and under 20 MiB of selected output; memory/output estimates have
not yet been measured. Record actual wall time and peak RSS before accepting
the convergence matrix.

The provisional matrix is spatial 16x64, 32x128, 64x256 (square cells,
ell fixed), with 3x3 particles per cell and fault spacings 1/32, 1/64, 1/128 m.
This gives approximately 5/10/20 cells across the ideal H FWHM. Use the same
physical initial data and dt=0.5 s for spatial comparison if affordable.
For time comparison use dt=2,1,0.5 s on the first spatially resolved level,
at t=2,4,6 s. Do not run a full Cartesian product. Include one selected
two-rank comparison and a checkpoint at a real step followed by resumed vs
uninterrupted evolution. If a plateau occurs, vary particle or surface
resolution separately. The finest level has 16,384 cells and 147,456 particles;
its campaign runtime is not inferred to be a few minutes from the pilot.

K0 stops at review with the concrete K1 proposal and the boundary-condition
prerequisite exposed. No approval to implement K1, later families, or a server
job is assumed from this record.
