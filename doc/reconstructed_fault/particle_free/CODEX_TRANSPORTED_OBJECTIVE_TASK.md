# Complete transported objective particle-free histories and moving-fault qualification
Prepared for Codex 6.1-sol with xhigh reasoning effort — 2 November 2026

## MISSION AND AUTHORIZATION

Work in the existing ASPECT research worktree. Implement and locally qualify the common numerical capabilities required by our three-example paper before preparing expensive server experiments. This is an implementation task, not a request for another proposal or a particle-only demonstration.

The paper plan is:
1. BP3-derived mature-fault calculation, with its deviations from official BP3 explicitly assessed.
2. Controlled viscoelastic-crust comparison, including a fault transported across the Eulerian mesh.
3. Unstable RSF coupled to damage and fault extension.

Proceed through the stages below without requesting approval at each milestone. You may modify source, add necessary history state, extend existing tests, update repository documentation, and run bounded local verification. This instruction expands the earlier review/spin-only scope to transported particle-free implementation and bounded cohesive/growth qualification. Preserve unrelated repository rules. Do not submit server jobs, launch long scientific trajectories, alter protected production checkpoints, push changes remotely, or introduce bulk inertia.

Use the selected 6.1-sol/xhigh configuration throughout; this document is task guidance, not a mechanism for changing the model setting. Work as one agent unless the user or applicable repository instructions explicitly authorize delegation. Use deliberate analysis for the mathematical/state decisions, then implement and measure. Do not spend the task repeatedly reconsidering settled choices or expanding peripheral test matrices.

## KNOWN STARTING POINT — VERIFY AGAINST THE CURRENT WORKTREE

Read applicable AGENTS.md files and locate RESEARCH_HANDOFF.md, CLEANUP_REPORT.md, BP3_REVIEW.md, OBJECTIVE_STRESS_REPORT.md, IMPLEMENTATION_MAPS.md, particle_free_history.tex, the current review plan, and their relevant evidence indexes. Follow actual source paths; names here are discovery targets, not assertions that every file is in the repository root. Read the equations and source behind the portions you change. Missing remote outputs do not block local implementation.

The supplied reports describe entry revision 27178813cebd471b93a7a9308f41215b71b4a20b on pf-rsf-particle-free plus subsequent changes. Record current HEAD and existing changes; do not reset or check out the report's revision automatically.

Reported facts to preserve unless newer evidence supersedes them:
- The legacy BP3 run reaches accepted step 4237. Its latest supplied spatial profile is step 4182, with maximum slip rate about 1.37553 m/s, still accelerating. Peak, arrest, and a completed first event are not established.
- The observer repair establishes internal agreement, not external benchmark fidelity, spatial convergence, or a complete energy balance.
- The selected spin convention is total-spin Jaumann, W = skew(grad v). The current opt-in implementation is a lagged-spin, first-order Lie split, qualified with fixed stationary particles and constant material coefficients. DG, transport, AMR, and cohesive combinations remain disabled.
- Production mature histories currently reconstruct tau = T - B, B = 2 chi Z S. The report identifies missing residual history under general rotation. The newer cohesive path additionally contains native FE remainder and lifting memory and uses a different time integrator.
- The protected propagation checkpoint was reported as four-rank step 748, t = 374.58003334330067 s. Its RSF episode was incomplete and fracture/spatial screens had failed. Verify whether an authorized newer checkpoint exists, but do not advance or convert either trajectory here.

Protect unique source changes, current controls, complete restart dependencies, server packages, and evidence of unresolved failures. Reuse existing builds where practical. At most four build workers or four local MPI ranks, sequential simulation runs, unless stricter repository limits apply. Do not create another full worktree/build copy without a demonstrated need. Keep compact numerical evidence and selected PNG/GIF output; avoid large diagnostic sequences and unrelated cleanup.

## INTENDED SCOPE AND SETTLED CONVENTIONS

Implement the capabilities needed by this paper: existing two-dimensional quasi-static mechanics, particle-free histories, objective stress evolution, spatially heterogeneous viscosity, moving reconstructed geometry and histories, and compatibility with phase evolution and fault extension. Keep constant shear modulus initially; do not expand into arbitrary temperature-dependent or nonlinear material laws, 3-D, or a new finite-strain elastic theory. Support the mesh adaptation/transfer operations actually needed by the planned production configurations.

Retain the selected total-spin Jaumann law and legacy controls. Do not subtract a slip spin, switch objective rates, alter friction, inflate regularization widths, add damping, or loosen guards to make tests pass. Preserve the declared lagged-spin integrator unless the derived coupling requires a justified change; label and validate any integrator change explicitly. Do not claim finite-strain hyperelasticity or width-independent physical behavior from objective-rate tests.

Do not replace physical-state completeness with more quadrature. Do not enable a rejected configuration merely by deleting its admission guard. Particle calculations may remain useful controls, but the required production path is particle-free.

### STAGE A — CLOSE THE EQUATIONS AND STATE OWNERSHIP

Create a concise implementation plan and acceptance matrix, then derive the changes before coding them. Update existing maps rather than duplicating contradictory specifications. For every bulk/surface state, identify its physical meaning, storage, transport velocity, reaction law, geometric dependence, trial/accepted ownership, transfer rule, and checkpoint representation.

Resolve these concrete items:
1. Write the physical branch equation, including material transport, total-spin rotation, relaxation, and localized slip forcing. Preserve full 3-D deviatoric tensors in plane strain, the pressure convention, and the actual gradient-index convention.
2. Distinguish bulk material velocity, physical fault motion, and tangential motion used to redistribute fault nodes. For a material no-opening interface, establish its common normal velocity. Derive surface-history transport for the chosen parametrization. Relative slip V is not the node transport velocity. A tangential remeshing choice must not silently advect or reset RSF age and slip.
3. Derive the evolution of normals, tangents, localization/profile coefficients, projection/extension maps, and integration weights. Geometric normal evolution under general deformation is not simply stress rotation by W. Separate prescribed-motion verification from motion driven by the solution.
4. Define what freezes in each control: stationary geometry, disabled fracture growth, and stationary Eulerian history are different restrictions. A translating mature fault carries its existing damage/profile. State the appropriate material/geometric irreversibility condition so vacated cells do not retain a damage trail through an Eulerian maximum operation.
5. Derive all missing product, transport, and coefficient terms for separated stress histories. The fixed-profile commutator in the report is a special case, not the generic residual equation. Account for differences between surface and bulk transport and between projected and local relaxation coefficients in layered material.
6. Specify heterogeneous eta evaluation and time integration along transported histories. Constant G is allowed; a history traversing changing eta cannot silently use the old homogeneous update without a declared, convergent coefficient treatment.
7. Resolve prestress and loading conventions. Two scalar background fault tractions do not define a complete bulk prestress tensor. Distinguish evolved physical stress from explicitly imposed background loading, and include the latter's work where appropriate. Do not invent a full prestress tensor from those scalars.
8. Connect mature and cohesive physical equations and their time-discrete energy/work measures. Identify the mature limit without discarding auxiliary slip memory just because cohesive resistance vanishes. Different storage layouts/integrators may remain if their physical meaning and compatibility are demonstrated.

Deliver an executable design decision, with the relevant equations and source owners. Choose routine implementation details autonomously. If an unresolved constitutive assumption would change the scientific model, isolate it, provide the smallest decisive counterexample, and continue independent work; do not silently choose new physics.

### STAGE B — RESOLVE THE HISTORY-REPRESENTATION BLOCKER FIRST

Implement the bounded comparison proposed by OBJECTIVE_STRESS_REPORT.md before extending the full native solver.

Use an oblique fault with a resolved normal profile, nonzero localized slip forcing, and spatially varying spin. Compare an independently evolved direct physical tensor with the separated representation tau = T - B - tau_res. Under the report's fixed-geometry, no-advection assumptions, recover its residual source [W,B]; verify the sign and factors by substitution into the physical equation. Then test transport/profile variation and heterogeneous eta using the Stage A equations.

The reference must not be the production reconstruction evaluated twice. Use independently coded component equations, an analytical/manufactured solution, or a demonstrably converged direct evolution. If direct stress is ultimately chosen as the production representation, retain an independent reference for it.

Measure full tensor error including zz, reconstructed normal/shear traction, stored energy, and projection error. Separate time, normal-resolution, and quadrature studies. Use at least three resolutions for each required convergence claim, with nontrivial stress and forcing. Preserve the existing pressure/work adjoint in coupled comparisons.

Select and implement the smallest defensible history space. You may add residual DG degrees of freedom or another justified representation; do not assume the exact storage choice in advance. If standard DG cannot resolve the residual at intended width, compare a targeted enrichment/refinement or direct physical-stress alternative before committing to native integration. Record accuracy and cost. No particle substitution and no relabeling an omitted tensor as cohesion.

### STAGE C — INTEGRATE THE COMPLETE PARTICLE-FREE UPDATE

Integrate the qualified representation into residuals, loads, tangents, observers, transport, and publication. For explicitly lagged spin, retain a clearly defined accepted-state anchor. If geometry or coefficients depend on the current iterate, implement the corresponding derivative or a declared, converged outer iteration. Demonstrate the tangent for the residual actually solved.

Implement moving geometry and history transfer through cell crossings, surface resampling, changed quadrature, and boundary intersections. Preserve the work/traction adjoint and localization normalization. Handle geometry caches as state-dependent data, not valid forever after construction. Exercise solution-driven normal motion/deformation in addition to prescribed translation before claiming general moving-fault support.

Extend the accepted-state transaction to all new histories, fault vertices, profile/phase state, accepted-gradient anchors, maps, and related bookkeeping. Rejected trials must restore the accepted state; retries use their actual dt. Diagnostics must not advance state. Restart must neither rotate nor transport a history a second time. Add explicit compatibility metadata and reject unsupported old-to-new conversions.

Audit and implement the same physical conventions in the cohesive path, including its native FE remainder and analytic lifting memory. Reuse common operators where appropriate without forcing incompatible histories into one superficial abstraction. Exercise phase evolution and fault extension with the updated histories. Existing unresolved fracture/spatial failures must remain visible and be addressed when they prevent qualification of the planned configurations.

Use a fixed mesh first to isolate transport. Then qualify the existing refinement/coarsening and distributed transfer needed by the production plan. A refined corridor can support the early translation fixture; it is not evidence that moving AMR is qualified. Do not broaden this into adaptive anisotropic-quadrature research unless the measured failure makes it necessary.

### STAGE D — BOUNDED INTEGRATED VERIFICATION

Before each test, record the physical scales, expected result/convergence, error norm, tolerance rationale, and step/time/output budget. Use dimensional and relative errors with a stated nonzero reference scale. Exact identities and approximate spatial transfer need different criteria. Do not relax tolerances after a failure to obtain a pass. Revise a defective test only with a documented independent reason.

Use compact fixtures and analytical/manufactured references for slow physical processes. The required tests are:

1. Legacy compatibility and constitutive checks: retain relevant zero-spin, rotation/relaxation, forcing, observer-covariance and simple-shear checks. Reuse existing valid evidence unless affected by changes. Preserve the known hypoelastic limitations.
2. Nonuniform translation: move an oblique curve, localized profile, and nonuniform histories across several cell boundaries in both horizontal directions. Measure position, profile width/normalization, accumulated diffusion/overshoot, stress, traction, and surface-history error. A translated uniform field alone is insufficient.
3. Translation with deformation/rotation: exercise transport and spin together, with changing geometry as required by the manufactured motion. Separate timestep and spatial error.
4. Heterogeneous relaxation: verify layered eta and a history crossing a viscosity transition, including storage and viscous dissipation. Use an independent time-dependent-coefficient reference where necessary.
5. Coupled RSF: a short stable case may isolate implementation errors, but also run a bounded velocity-weakening case that demonstrably exercises acceleration and the coupled state/timestep controls. Compare against a refined or independent reference. It is not necessary to complete a decades-long nucleation history or a full BP3 event locally.
6. Cohesive/growth compatibility: use the smallest meaningful evolving-phase/tip fixture to exercise the new state and mapping terms. Track growth, physical work, fracture work, and unresolved spatial/temporal errors. Translation-only success does not qualify tip growth.
7. Lifecycle and transfer: force rejection after a completed moving-geometry trial; compare with an equivalent accepted-step control. Compare uninterrupted versus restarted evolution while geometry crosses cells, include repartition, and exercise one meaningful refinement/coarsening sequence when AMR is required. Use exact equality only where justified and numerical tolerances for projection/partition changes.
8. Energy and width: evaluate the physical balance from independent mechanical work, storage, viscous/frictional/radiation terms, and fracture work when active. Include prescribed loading work and transport fluxes for open/fixed control volumes. Define flux signs and compute the remainder; do not manufacture a closing term. Report errors relative to each scientifically important contribution, especially fracture work. Separate mesh refinement at fixed width from width sensitivity, and report integrated spin, stress/G, and deformation measures in the actual operating regime.

A homogeneous rotation pass, a tiny observer discrepancy, or a closing constitutive substep ledger does not replace these integrated tests. Retain failures and diagnose them. Re-run only tests affected by new changes or unresolved concerns, then the relevant repository-required suite after integration. Avoid repeatedly rebuilding/rerunning the whole matrix after documentation-only edits.

### STAGE E — DEFINE THE EXPERIMENT CONFIGURATIONS AND FREEZE THE HANDOFF

Prepare configurations and analysis scripts, not production trajectories:

__Example 1__: retain the legacy BP3 control and its exact provenance. Record the incompressible nu -> 0.5 versus official nu = 0.25 discrepancy, finite boundaries, diffuse width, filtering, and initialization differences. Keep strict_reference_match_passed=false unless independently justified. Assess a matching external reference or a separately scoped compressibility extension; do not add a major compressible formulation silently or block independent transport work on unavailable participant data.

__Example 2__:
A. Uniform velocity-weakening RSF, a=0.010 and b=0.015, effectively elastic bulk, stationary geometry, growth disabled.
B. Same friction and other controlled settings, layered lower-crust viscosity, stationary geometry, growth disabled.
C. The translated counterpart of B, with the existing curve/profile/histories moving across the fixed mesh, growth disabled.

For the translation comparison, define a constant horizontal c and the target relations
  F_c(t) = F_0(t) + c t,
  q_c(x,t) = q_0(x-c t,t),
  v_c(x,t) = v_0(x-c t,t) + c.
Translate all applicable data consistently. Uniform translation adds no strain or spin and must not change relative slip. Use periodic/manufactured compatible boundaries for the exact local comparison. For a finite BP3-like domain, explicitly assess transformed inflow/outflow and boundary effects; retaining incompatible side conditions invalidates an exact-equivalence claim. Measure errors in moving coordinates, with translation distance expressed in cell widths and fault widths. Both directions are local checks; one can be selected for the initial server case.

Use vertical depth for rheological layering and down-dip distance for official BP3 friction coordinates. Distinguish them explicitly. Retain constant G initially. Specify the viscosity transition, behavior below 40 km, background loading, and duration relative to eta/G. Preserve the proposed smooth upper-to-lower-crust transition and label any unchosen numerical viscosity as a proposed default, not an approved physical result. Do not aim parameters at forcing nucleation to a predetermined depth. Retain the uniform-friction elastic control to isolate viscosity.

__Example 3__: prepare the intended unstable-RSF/growth configuration only after its required shared operators and evolving-phase path pass the local qualification. Preserve the old continuation package separately. Do not describe an unfinished acceleration or tip-damage increase as a completed rupture with persistent fault extension.

Prepare a pinned source/configuration package and a bounded server qualification script for build/runtime, MPI, restart, and output checks. Do not execute or submit it. Record source revision and complete dirty delta if present, core/plugin hashes, build environment, input hashes, schema compatibility, and all run commands. Do not convert old checkpoints into the new law. Server outputs should distinguish accepted steps from attempted solves and include event maxima, peak/decay indicators, station/profiles, energy terms, and complete checkpoint manifests at a controlled output cadence.

Local scaling estimates are preliminary; do not claim measured large-rank performance. Mark each package ready or blocked by named acceptance criteria. One qualified release may include explicitly selected legacy and new laws; do not silently change a law mid-trajectory. Later changes require reassessment of affected results.

## DOCUMENTATION, CONTINUITY, AND COMPLETION

Keep authoritative documentation outside disposable build output. Extend the existing AGENTS/research supplement narrowly with state-ownership, admissible-configuration, restart, and verification rules; preserve unrelated rules. Avoid generic instructions that prevent this authorized implementation.

Maintain these compact deliverables, adapting paths to the repository:
- TRANSPORTED_OBJECTIVE_SPEC.md: final equations, selected representation, interface motion/transport, prestress and energy conventions, assumptions, and source map.
- TRANSPORTED_OBJECTIVE_REPORT.md: acceptance matrix with measured values, convergence plots/tables, commands/provenance, costs, failures, and per-configuration server readiness.
- Updated IMPLEMENTATION_MAPS.md and research handoff: one accepted timestep from physical equations through state storage, mechanics/RSF/phase/geometry, rejection/commit, and energy output. Include a plain-language explanation for the researcher.
- Reproducible compact fixtures/reducers and candidate server package with an exact source/configuration manifest.

Maintain a short task-state record after meaningful milestones: current code identity, decisions made, tests passed/failed, evidence locations, and the next concrete action. After a context transition, resume from that record rather than repeating completed audits. Provide concise progress updates reporting findings and next actions, not raw reasoning or huge logs.

Do not stop merely because Stage A or the first fixture is complete; continue through authorized implementation and verification. A failed gate blocks its dependent production-readiness claim, not independent work. If a genuine mathematical or representation blocker remains after bounded, evidence-driven diagnosis, preserve a minimal reproducer, state the exact missing decision and its consequences, and finish all unaffected work. Never label partial particle-only or fixed-geometry success as completion of this task.

At handoff, state clearly:
1. Which formerly unsupported combinations now work and which remain disabled.
2. What changed physically versus only numerically, with the final equations.
3. Which measured criteria support each claim and which still fail.
4. Whether each planned server configuration is ready, with the precise next action if blocked.
5. That no server job or long scientific trajectory was launched.

## REFERENCE POINTERS

Treat repository source and current supplied reports as authoritative for implementation facts. External references inform conventions and benchmark scope, not proof of this code:
- Official BP3-QD/FD specification, 1 October 2021: https://strike.scec.org/cvws/seas/download/SEAS_BP3.pdf
- Schrank et al. (2017), A comparative study of Maxwell viscoelasticity at large strains and rotations: https://doi.org/10.1093/gji/ggx297

Use predeclared, physically meaningful tests to determine readiness. The goal is a reproducible, qualified implementation for the stated paper scope, followed by deliberate server experiments.
