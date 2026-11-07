# DIRECT PHYSICAL-STRESS HISTORY: IMPLEMENTATION AND COMPARISON HANDOFF
4 November 2026

Revised against BOUNDED_PROFILE_TRANSPORT_REPORT.md and
POSITIVE_TRANSLATION_REFINEMENT_REPORT.md. This is the next task for the
existing coding thread and worktree, using the selected 6.1-sol/xhigh setting.

## Selected task
Pause the previously recommended accuracy/energy continuation and implement an
opt-in direct physical-stress history for the existing mature moving-fault
model. Complete a small native comparison, including coupled mechanics; do not
stop at an algebra review or another standalone transport demonstration.

The bounded-profile implementation and the two positive-refinement runs are
finished. Park the broader accuracy campaign now. Incorporate the limited
accepted-state boundary audit below into this task; it does not require a
separate read-only assignment or another approval before implementation.
Proceed through the authorized code changes and comparison without stopping
at intermediate milestones to ask whether to continue.

The scientific question is whether storing the physical stress directly gives
a simpler and competitive implementation at the accuracy required by the
coupled problem. A useful outcome may be adoption, retention as a reference,
or rejection on measured accuracy/cost grounds. Faster execution is not assumed.

This task changes the numerical stress representation. It does not select the
new cohesive/phase-energy assumptions in sections_2_1_to_2_3_direct_stress.tex.
In particular, retain the native exponential Maxwell reaction and the existing
RSF age lag. Do not copy the manuscript's proposed backward-Euler update or
its endpoint-state friction convention into this comparison.

### 1. Resume from the actual worktree

Work in the existing aspect-pf-rsf-particle-free worktree. Read AGENTS.md and
verify branch, revision, dirty work, current task status and binary identity.
The remote branch reviewed for this handoff was
69e8a10a93a08718d31af961197899e320657c04. This is a reference, not a reset target;
the coding chat may contain newer local work. Preserve it and park the preceding
task at a recoverable boundary. Do not kill unrelated jobs or alter production
checkpoints to start this experiment.

Read the relevant current sections and evidence indexes of:

```text
  doc/reconstructed_fault/current_design.md
  doc/reconstructed_fault/specification.tex
  doc/reconstructed_fault/particle_free/STATUS.md
  doc/reconstructed_fault/particle_free/IMPLEMENTATION_MAPS.md
  doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_SPEC.md
  doc/reconstructed_fault/particle_free/BOUNDED_PROFILE_TRANSPORT_REPORT.md
  doc/reconstructed_fault/particle_free/POSITIVE_TRANSLATION_REFINEMENT_REPORT.md
  doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md
  doc/reconstructed_fault/particle_free/STAGE_C_COHESIVE_DIAGNOSIS.md
  doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md
```

The current source already has a complete restricted mature material transaction,
spatial-viscosity reaction, optional bounded MacCormack profile transport,
accepted-trace tensor inflow, and full-normal support checks. Earlier statements
that the entire material transaction is missing are obsolete. Reuse this work.

The latest positive-refinement report records clean tracked source at
69e8a10a93a08718d31af961197899e320657c04. Some retained executables still print
27178813c; their actual hashes identify them. Verify any newer local changes.
Preserve these completed findings as the starting comparison:

- Bounded MacCormack improves resolved profile/localization accuracy, but is
  not a conservative remap. Retain its measured mass drift and donor bounds.
- At 64x64 and label spacing .05 m, dt=.025/.0125 s gives endpoint energy
  remainders of 8.7411%/4.4151% of active work. The roughly first-order decrease
  does not establish a defect in stress representation or select a new scheme.
- Raw face power is -3.11469e-6/-4.19177e-6 W/m in those two runs, while weak
  reaction power is near zero. Neither a converged raw traction nor all weak
  residual components have been independently established by that power check.
- Surface-grid changes alter sampled initial age/loading and initialization;
  the approximately .67% slip-rate difference is largely present initially.
  Do not confound a surface-grid change with direct-versus-separated evolution.

Keep the current separated implementation available as the comparison control.
Protect the core/checker binaries needed to reproduce it before rebuilding.
Use at most four local workers and run native simulations sequentially. This
task authorizes local implementation and the bounded tests below, not a server
campaign, long event replay or migration of an existing production trajectory.

Perform one focused baseline audit on the existing accepted states before
interpreting the paired results. Independently evaluate the mechanical and
continuity weak residuals in the actual admissible test space, constrained
reaction components, and net force components; zero power for one imposed
translation is not a complete equilibrium test. Compare raw per-face stress
resultants and interior DG jumps using the same stress/pressure conventions.
Use retained exports or a non-advancing checker, not a new refinement campaign.
If weak equilibrium is verified, preserve the unresolved raw-trace discrepancy
and proceed; fully repairing strong DG traction reconstruction is not a
prerequisite to this representation experiment. If an actual common residual,
boundary-assembly or observer bug is found, fix it narrowly and recheck the
affected control before drawing an A/B conclusion. Continue independent
direct-history work while isolating that defect.

### 2. Preserve the physical model

Initially admit only the existing mature material scope: 2D/full-3D plane strain,
constant G, positive constant or prescribed spatial eta, fixed affine Cartesian
bulk mesh, one interior fault with the same material labels, and growth disabled.
Retain the selected profile/inflow/support methods, geometry transaction,
normalization, friction/pressure/loading conventions and physical bounds.

The governing physical branch stress is

```text
  J_v tau + (G/eta_b) tau = 2G (D - chi V_tilde S_tilde),
  D = dev_3 sym(grad v),
  J_v tau = partial_t tau + v.grad(tau) - W tau + tau W,
  W = skw(grad v).
```

Use the total regularized FE spin. Store xx, yy, xy and recover zz=-xx-yy;
use |tau|_3^2 = xx^2 + yy^2 + (xx+yy)^2 + 2xy^2. Pressure does not rotate.
No auxiliary localization/geometry commutator or residual tensor is required
in this direct equation. The Jaumann stress-spin terms -W tau + tau W remain
inside J_v and must still be integrated. "No commutator" must never be
interpreted as restoring a non-objective component rate.
Current geometry/profile/association still determines chi, V_tilde and S_tilde
at the endpoint. Removing the commutator does not remove that kinematics.

For this branch, the existing representation is tau=T-R-B, B=2 chi Z S.
Its Z is auxiliary, with no cohesive storage or cohesive resistance. The direct
path must evaluate physical stress from its own tensor history, never by
subtracting B again. Keep Z as a comparison diagnostic if existing surface
storage requires it; it must not feed the direct stress, traction or tangent.
Age, cumulative slip, geometry and phase/profile histories are still needed.

Here "physical stress" means the physical tensor of the existing evolved
constitutive branch. BP3's imposed shear drive and confinement do not define a
unique full bulk prestress. Do not invent or begin rotating/relaxing an inferred
prestress, or change how prescribed background loads enter mechanical work.

Do not modify the older cohesive energy formulation or the newer sharp-jump
cohesive pilot in this task. Their elastic memories, native-gradient/lifting
remainders and physical phase/shape variations are distinct. Direct storage
does not by itself justify deleting those terms. Do not enable growth, birth,
resampling, moving AMR or deformed cohesive evolution as a side effect.

### 3. Define the direct discrete update before changing the solver

Use one independent DGQ2 physical tensor per cell. Let P_h be the actual common
cell projection, with its mass matrix and quadrature, and A_h the existing DG
SSPRK3 material-advection operation. For each attempt, from the immutable
accepted parent and its accepted velocity/gradient:

```text
  tau_adv = A_h[tau_n],
  Q = exp(dt W_minus),
  tau_bar = P_h[Q tau_adv Q^T],
  beta_b = exp(-G dt/eta_b),
  kappa_b = eta_b (1-beta_b),
  tau_loc = beta_b tau_bar
             + 2 kappa_b (D(v_new)-chi_new V_tilde_new S_tilde_new),
  tau_candidate = P_h[tau_loc].
```

Use the existing stable expm1 coefficient evaluation and elastic limit.
Match the current accepted-gradient sampling and substep ordering; this task
does not introduce a different spin trajectory or objective integrator.
Variable eta is evaluated with the existing endpoint spatial convention;
project the complete products. Do not commute P_h with variable beta/kappa
or replace bulk eta by a projected surface viscosity.

This is a proposed direct-DG discretization of the SAME continuum stress law.
It is not generally pointwise identical to the separated discretization: B is
not a DGQ2 polynomial. Source projection and loss of analytic localization are
part of the experiment and must be measured.

The projected candidate must be the stress used by the converged native bulk
and fault residuals, by traction/work evaluation, by publication and by the
accepted observer. Do not solve with tau_loc, then silently store P_h tau_loc
after acceptance. That would change the stress without re-establishing balance.

The existing local Maxwell matrix may be reused inside a converged correction
outer iteration. Define the scratch correction Delta_tau=P_h tau_loc-tau_loc;
freeze it only during each inner native solve, update it from the solved v,V,
and repeat before the original final residual/KKT/publication gate. The outer
fixed point, not the frozen local matrix alone, represents the projected law.
Keep the work-conjugate transfer and its quadrature consistent. Alternatively
assemble the exact projected tangent if that is clearly simpler in the code.

Important current-code trap: the existing correction outer loop is entered
only for material_variable_viscosity. Direct DG generally needs projection
consistency even with constant eta, because chi V S is nonpolynomial. Introduce
a narrow representation-aware dispatch; do not pretend that the material is
heterogeneous just to activate the loop. Retain the established convergence
criteria and verify the terminal projected residual independently.

For accepted-trace inflow, freeze the incoming trace of accepted tau_h during
the advection substep. The separated comparator freezes T and U=R+P_h B;
their difference supplies the corresponding projected physical trace. Identify
this common finite-domain boundary approximation explicitly. It is not exact
infinite-domain continuation. Do not compare zero inflow with continued-tensor
inflow and attribute their different boundary-value problems to representation.

Retain the existing incoming-age use in the friction solve and its once-only
post-acceptance aging update. Use the same accepted velocity for tensor,
curve, support-width and phase/obstacle transport. Fault normals come from
geometry, not Q. V is relative slip, not material label velocity.

### 4. Implementation locations and ownership

Add one opt-in history-representation selection, valid initially only with
the supported mature material branch. Keep existing defaults and law labels.
Use an explicit representation tag or separate narrow owner so a vector called
trial_stress cannot silently alternate between T and tau without dispatch.
Reuse the existing tensor space and transport rather than copying the solver.

Primary files to inspect and update as needed:

```text
  include/aspect/material_model/phase_field_fault/dg_history_internal.h
  include/aspect/material_model/phase_field_fault/fe_strain_internal.h
  source/material_model/phase_field_fault.cc
  source/material_model/phase_field_fault/dg_history.cc
  source/material_model/phase_field_fault/dg_material.cc
  source/material_model/phase_field_fault/dg_material_reaction.cc
  source/material_model/phase_field_fault/fe_strain.cc
  source/material_model/phase_field_fault/dg_accepted.cc
  source/simulator/solver/reconstructed_fault_stokes.cc
```

Reuse dg_objective_transport.cc, dg_material_profile.cc and the manager's
existing staged geometry/history transaction. Audit the reconstructed-fault
assembler, surface residual, work adjoint and observers for assumptions that
history means T-R-B. Change only the necessary dispatch and stress access.

Every nonlinear trial and retry uses the same accepted tensor parent and actual
attempt dt. Publication happens once. Rejection restores tensor, solution,
geometry, profile/obstacle, maps and surface state together. No second rotation
on observation or restart. The new checkpoint identity must distinguish direct
physical stress from separated history and retain integrator/inflow/profile/
support identity. Start fresh fixtures; do not convert production checkpoints.
Rebuild matching checker libraries after history-layout changes.

For the enabled direct scope, no accepted T/R pair or analytic B correction may
remain necessary to reconstruct tau. Temporary projection scratch is allowed;
it is not a second independent physical history. Leave other representations'
native FE/lifting state untouched. Do not broaden to non-affine meshes by
dropping a native source remainder that the admitted affine case does not need.

### 5. Small, decisive test sequence

**A. Independent update and representation checks**

  - Verify uniform rotation plus relaxation against a componentwise analytic
    tensor solution; verify a forced constant-coefficient recurrence. Use
    independent full-3D norms and a representable field for roundoff checks.
  - Reuse the existing moving/layered prescribed-field reference for a compact
    direct-history check with specified inflow, nonzero spin and local eta.
    Exercise the native new path, not only the old standalone Python solver.
  - Check one source projection independently at higher quadrature. Compare
    candidate stress, its native residual/traction and the published observer.
    Audit a nonzero slip perturbation as well as a velocity perturbation.
    Distinguish the frozen-inner Jacobian from the derivative of the full
    projected residual, including the projection correction's variation.
    Independent finite-difference actions should expose that distinction;
    a converged outer correction is valid without an exact full Newton matrix.
    A frozen-inner tangent check alone does not validate the final projected law.
  - At one common frozen configuration, decompose the tensor difference as

    ```text
      tau_direct - tau_separated
        = (tau_direct - P_h tau_separated)
          + (P_h tau_separated - tau_separated).
    ```

    This separates disagreement within the DG space from the separated field's
    unresolved analytic localization. Use common geometry, profile, material
    coefficients and prescribed velocity/slip data for this diagnostic; do not
    attribute differences between evolved trajectories solely to projection.
    Report native-quadrature and higher-quadrature norms distinctly. This is
    an offline/checker comparison, not a production checkpoint conversion.
  - Verify that changing diagnostic auxiliary Z at fixed physical tau and
    other physical state does not change direct-mode mechanics.

Reuse trustworthy existing transport tests; do not rerun the entire historical
Stage A-C matrix. Specify tolerances and normalization before inspecting new
results. Use retained arithmetic bounds for identities and explicit h/dt errors
for PDE accuracy. Do not demand roundoff pointwise agreement between different
stress spaces, and do not relax an identity tolerance to conceal a mismatch.

**B. Native coupled comparison**

Use the corrected full-support bounded-profile fixtures, not the earlier
clipped-support or sheared/zero-inflow cases. The starting selection is:

```text
  profile transport = bounded MacCormack;
  tensor inflow = accepted trace;
  association support = full normal cross-section;
  growth disabled; identical material labels, loading, friction and tolerances.
```

Run separated/direct pairs on the existing 32x32, dt=0.05 s, final t=0.2 s
fixture with initial label spacing 0.1 m:

  1. constant viscosity, positive translation;
  2. constant viscosity, negative translation;
  3. layered viscosity, positive translation.

Retain the common 0.1 s initialization convention. Locate exact current inputs
and commands in the existing evidence; do not reconstruct their material and
boundary parameters from this prose. If newer equivalent fixtures are already
available, use them and document the substitution.

Use the same physical initial data, sampled surface state and initialization
convention in each pair. Project initial stress consistently into the admitted
spaces and report any initial representation error separately. First check a
representative separated run in the modified build against the protected
baseline. Reuse matching retained controls where source and input equivalence
is established; do not rerun completed trajectories merely to regenerate a
table. Never compare differing label grids or initial loading as if only the
stress representation changed.

Add one paired refinement of case 1 to 64x64 and dt=0.025 s, keeping the same
label grid. This tests the representation difference against refinement while
avoiding a new surface-discretization variable. It does not isolate a temporal
order or establish surface-continuum accuracy. Use a separate single h-only or
dt-only check only if an observed discrepancy needs that diagnosis. The existing
fine separated run is a numerical comparator, not ground truth; compare both
representations with independent references and their refinement behavior.

The completed positive-refinement runs use label spacing 0.05 m. They are useful
for the baseline audit and time-error context; they do not silently replace
the 0.1 m-label comparator above. If adopting their 0.05 m grid instead, match
that grid and its initialization in both representations and relevant mesh
levels. Do not add another surface-spacing or time-refinement campaign by
default. Exclude the retained old reference's spurious 2.78e-17 s terminal
interval from physical step/order analysis, while retaining its recorded cost;
new fixtures should end on the intended accepted time sequence.

If a nonrepresentable direct stress causes an accuracy loss, report it. Do not
silently retain analytic B in the direct path, change chi/width, add filtering,
or refine only one run and then claim equal-mesh equivalence.

**C. Lifecycle and regression**

Force one completed-trial rejection and cutback in direct mode. Compare with
the same accepted time sequence. Restart one layered direct run from two to
four ranks and continue through at least two steps with cell crossing. Compare
the complete accepted state, not just the load event. Check direct/separated
checkpoint mismatch rejection and the existing default regression. Run the
required unit suite once after the final source change. No moving-AMR or
production-restart qualification is selected by these tests.

### 6. Measurements and decision

For each paired case report:

  - full tensor L2 and component/traction errors, including fault-band and
    deeper-interior windows; show point traction AND work-conjugate traction;
  - velocity, pressure, V, cumulative slip, age, phi, I and geometry differences;
  - projected residual, inner/full derivative distinction, outer convergence and
    accepted-vs-observed stress agreement;
  - tensor coefficients/bytes, total history/checkpoint size, peak-memory scope,
    transport/rotation/projection time, nonlinear/linear iterations, solve time
    and total wall time, using equal ranks and output settings.

At the same mesh, pure plane-strain DGQ2 stores 27 tensor coefficients/cell,
versus 54 for complete T+R. Count ghosts, trial copies, surface state and retained
caches separately. This is not a claim of halved total memory or doubled speed.
Use a few repeated timings of one pair only if a speed conclusion needs them;
otherwise label short timings indicative and report operation/storage counts.

Preserve the pressure-aware work accounting: branch storage |tau|_3^2/(4G),
bulk work, slip work, Maxwell loss, friction/radiation, imposed background work,
material-energy flux and weak boundary reaction work. Report endpoint and
trapezoidal diagnostic remainders separately, with signed and absolute sums
and an active-work scale. Do not substitute raw unequilibrated DG face traction
for weak reaction power or normalize only by large cancelling gross work.
Measure transport/rotation/source projection changes independently. For the
source projection, compare the quadrature energies of tau_loc and P_h tau_loc;
the orthogonal-projection loss is numerical, not extra physical dissipation.
Claim nonnegative projection loss only in the inner product for which P_h is
orthogonal; a higher-quadrature diagnostic need not have the same sign.

Do not require this representation change to eliminate the existing endpoint
energy remainder or satisfy a newly invented 1% energy gate. The retained
4.4151% result is context for the current time discretization, not a tolerance
for identities or equilibrium. Report any change in that error and its source.
The direct representation does not itself repair profile mass conservation,
finite-domain inflow, raw DG traction reconstruction or time-integration work.

The earlier Stage B prescribed-field screen reached a 1e-3 tensor target with
direct N16 (6912 tensor coefficients) and separated N8 (3456 plus surface
scalars). This warns of a localization-resolution tradeoff; it does not decide
native coupled cost on a mechanics mesh that is already fixed by other fields.
Show equal-mesh comparisons first, then discuss equal-accuracy cost only where
the measured references/refinement support it.

Conclude with one recommendation:

- adopt direct history for the tested mature scope; or
- retain it as an independent native reference pending a named accuracy issue; or
- retain separated history because its measured accuracy/cost is better.

Separate continuum equivalence, finite-step/spatial differences and runtime.
Where coupled accuracy and cost are comparable, prefer the simpler physical
history if it actually removes T/R/B ownership and correction bookkeeping.
If localization accuracy is inadequate, retain the result and identify that
limitation; do not expand this task into stress enrichment or a new solver.
A pass does not close the separate cohesive energy/tip or moving-shape problem.

### 7. Deliverables and stopping point

Deliver working opt-in code, reproducible small fixtures/reduction, and one
DIRECT_TRUE_STRESS_REPORT.md with the equations actually implemented, source/
binary/input identities, measured comparisons, limits and adoption decision.
Update STATUS.md and the formulation/owner map, including the paused task and
how to resume it. Follow the worktree's commit/push policy. Preserve failures
and existing evidence; no unrelated cleanup or broad solver refactor.

Proceed through implementation and this bounded native comparison. A conflict
in one unsupported formulation does not block independent mature-path work.
Stop at the comparison report and recommendation; do not automatically resume
the former accuracy campaign or start production calculations.
