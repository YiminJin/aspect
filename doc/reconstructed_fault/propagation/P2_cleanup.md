Perform a bounded cleanup of the reconstructed-fault implementation following the P2 closeout. The purpose is to make the accepted FE-history machinery clear and reusable before extending it to BP3.

Complete the authorized cleanup and verification without pausing for approval between routine steps. Stop after the cleanup report; BP3 implementation and new scientific experiments are subsequent tasks.

1. Establish and preserve the reference.

The reviewed remote checkpoint is fault-propagation commit 41272d26366b1953f09f1da82c386af5bbea8ec2. Inspect the actual current checkout and preserve any subsequent edits. Record the reference revision, dirty state, executable, and relevant retained evidence.

Read the applicable AGENTS instructions, CONTRIBUTING.md, doc/reconstructed_fault/refactoring.md, and relevant sections of current_design.md and specification.tex. This instruction selects the cleanup below; historical task blocks do not authorize other experiments.

Preserve the accepted P2 reference, legacy particle-history behavior, numerical defaults, existing physical parameters, and checkpoint compatibility. Keep structural movement, removal of redundant work, and formatting distinguishable in commits.

2. Follow ASPECT style and control abstraction.

The implementation must balance efficiency, readability, and conciseness. Follow ASPECT’s existing coding conventions and repository formatting configuration.

In particular:

- Keep the mathematical sequence visible in the main routines.
- Expand compressed multi-statement lines and deeply compressed loops.
- Use descriptive names, conventional control flow, const correctness, and comments explaining mathematical meaning, ownership, or non-obvious decisions.
- Keep established mathematical symbols where their definitions are clear.
- Avoid comments that simply narrate the next statement.
- Format only touched files with the repository-supported tooling. Do not run an unrelated repository-wide formatting pass.

Do not create a helper merely to shorten a function or give a few statements a name. Extract a helper when it either removes meaningful repeated computation or represents a substantial, coherent numerical/lifecycle operation with clear inputs, outputs, and state effects.

A single-use helper is acceptable for a substantial operation such as constructing normalization geometry or publishing validated history. It is not justified for trivial vector setup, one assignment, a short conditional, or a forwarding call.

Prefer a few substantial routines over chains of tiny wrappers. Small local lambdas are acceptable when their purpose and capture scope are immediately clear. Avoid nested lambda frameworks.

Do not introduce new managers, abstract backends, factories, policy classes, generic cache frameworks, or plugin families. Reuse the existing PhaseFieldFault and ReconstructedFaultManager ownership.

Do not merge similar-looking loops unless their quadrature, material mixture, support, ownership, and lifecycle semantics are actually identical.

3. Use this specific FE implementation layout.

Paths below are relative to the repository root.

| File | Responsibility |
|---|---|
| source/material_model/phase_field_fault/fe_strain.cc | Accepted FE history, stress evaluation, shared FE-history recurrence, preparation/validation, trial and commit support, publication, and existing mechanical quadrature |
| source/material_model/phase_field_fault/fe_normalization.cc — new | FE normal-profile geometry, analytic normal integrals, consistent normalization projection, and normalization derivatives/actions/adjoints used by the phase functional |
| source/material_model/phase_field_fault/fe_phase_energy.cc — new | FE bulk/cohesive phase energy, phase gradients, Hessian/tangent actions, and P2-specific endpoint potential construction |
| include/aspect/material_model/phase_field_fault/fe_strain_internal.h — new | Only implementation data definitions genuinely shared by those translation units, including the existing private FE state/update structures where necessary |

Retain the existing material class declaration in include/aspect/material_model/phase_field_fault.h and the sole plugin registration in source/material_model/phase_field_fault.cc.

Keep generic normalization dispatch and the established non-FE normalization path in normalization.cc. Keep common constitutive formulas in constitutive.cc and accepted-history orchestration in history.cc.

Do not add separate files for each helper, derivative order, diagnostic quantity, or fixture.

Keep the small existing seed-file initialization and retained-entry test-access methods in clearly identified sections of the relevant files unless existing benchmark/test interfaces can host them without expanding the public API. Do not create another fixture framework.

Retain current restrictions, including the horizontal/unit-domain assumptions, during structural extraction. Generalizing them to dipping BP3 geometry is not part of this cleanup. Label those assumptions accurately.

Move complete definitions before restructuring their bodies where practical. Handle includes, template instantiations, and ordinary independent compilation correctly; do not rely on unity-build ordering.

4. Make normalization available without evaluating phase energy.

Currently compute_normalization_integrals() calls evaluate_fe_phase_energy(..., false) in the initialized FE path just to retrieve normalization. The false argument disables the dense Hessian, but the routine still computes derivatives, bulk energy, gradients, and reductions.

Provide a direct normalization-only path through fe_normalization.cc. Reuse the same integral and projection implementation for the energy/derivative paths.

A normalization-only request must avoid:
- Bulk stress/energy integration.
- Phase-gradient and Hessian assembly.
- Dense derivative matrices and derivative-only work.
- Energy/gradient reductions and diagnostic output.

Preserve the definition of I_h, integration rules, projection weights, admissibility checks, and consistent derivatives/adjoints. Keep required mechanical-state checks in their owning mechanical/energy operations.

Use a small private interface with clear requests and results. Do not introduce a general evaluation framework or many narrowly forwarding methods.

Retain valid existing caching. Any added reuse must have explicit dependencies and invalidation for the relevant phase, material, geometry, mesh, and projection data. Geometry validity alone does not establish normalization-value validity.

Do not combine this task with replacing the normalization mass solver, changing quadrature, mass lumping, or redesigning the derivative algorithm.

5. Separate association data from particle-projection solver preparation.

Keep this work within manager_particle_projection.cc and the existing manager declaration.

Currently rebuild_particle_projection_cache() constructs particle/fault associations and quadrature while also assembling and factoring the projection matrices. Consequently, requesting associations can prepare a projection solver.

Separate these responsibilities:
- Particle/fault associations, domain quadrature, and their validity.
- Projection matrices/factors, prepared when a projection actually needs them.

Reuse the association quadrature; do not independently reconstruct it for each projected quantity. Use minimal cache state, with explicit dependencies on the association generation.

Preserve support/admission rules, domain coverage checks, MPI participation, and collective error propagation. Do not turn an unsupported association or singular projection into a silently accepted state. Keep geometry-only validity distinct from projection-factor validity.

Do not add a new cache class or split this code into additional files. Preserve public interfaces where practical; make only focused internal changes.

Keep these live capabilities:
- Initial RSF-state projection.
- Surface chemical-composition projection.
- Nonzero cohesive initialization and legacy cohesive evolution.
- Fault-property interpolation at particle associations.
- Required particle/fault geometry services.

Do not delete manager_particle_projection.cc or the legacy surface assembly.

6. Bypass projection for explicitly zero cohesive initialization.

In initialize_cohesive_state_from_initial_fields(), the unloaded and mature-frictional cases currently construct zero particle values and project them to obtain zero nodal traction.

Initialize the fault nodal cohesive values directly for these explicitly zero cases, using the existing validation and publication operations.

Preserve normalization preparation, initialization markers, support/admissibility requirements, and later-time/restart behavior. Ensure the shortcut does not accidentally bypass checks previously reached while preparing projection data.

Keep the existing particle projection for seeded nonzero initialization. Do not change that initialization’s physical meaning.

7. Clarify FE-history ownership and consolidate genuine duplication.

Extract the common FE displacement-history recurrence currently repeated in make_fe_endpoint_trial() and prepare_fe_strain_commit(). Keep the P2-specific friction-potential calculation with fe_phase_energy.cc.

Preserve:
- Immutable incoming accepted history throughout nonlinear trials.
- Noncommitting trial construction.
- Separate validation and terminal publication.
- Exact Maxwell coefficients and stable expm1 evaluation.
- Collective rejection before any rank publishes.

Retain the independent consistency check between the surface traction/normalization representation and the Maxwell Z recurrence. Do not replace an independent verification with two calls to the same computation and claim equivalent coverage.

In the FE path, make it clear that particle stress is sampled from the authoritative FE state. Retain the existing recurrence/publication equivalence check during this cleanup; removing it from routine execution can be considered after BP3 qualification.

Rename misleading internal data where useful. In particular, a field named cohesive_projection should not imply that projection occurred when it contains directly computed surface candidates. Keep diagnostic projection results distinguishable from the physical candidate values.

Avoid renaming serialized properties or changing property layouts.

8. Extract substantial surface diagnostics without expanding the architecture.

Review the disposable history audit and CSV output in surface_system_particle.cc, and the diagnostic sampling/line-clipping work in surface_system_bulk_work.cc.

Prefer existing observer and benchmark interfaces where they provide the required values at the correct lifecycle point.

If substantial core-side diagnostic implementation needs a separate translation unit, use only:
source/reconstructed_fault/surface_system_diagnostics.cc

Use existing surface-system declarations/internal support for the necessary private interfaces. Do not add another public diagnostic framework, a new manager, or a collection of per-output files.

Keep actual constitutive evaluation, quadrature, residual/Jacobian assembly, and acceptance in their existing numerical owners. Diagnostics should consume the values used by those operations wherever possible.

Preserve diagnostics needed for the BP3 ringing comparison. Preserve switch semantics, MPI participation, and observer lifetimes. Do not assume every environment-controlled block is observational: identify extra evaluations, cache preparation, or state effects before moving it.

9. Preserve scientific distinctions and limit experimental cleanup.

The FE and particle paths currently differ in strain treatment: FE history applies consistent_deviator, while the legacy particle history update uses the full symmetric gradient.

Document this distinction for the subsequent BP3 comparison. Do not silently unify it, remove stress trace, or label a constitutive change as refactoring.

Also preserve the distinction between:
- Bulk-history representation.
- Particle-volume versus bulk-conjugate mechanical integration.
- Generic particle-property transfer.
- FE normalization projection and bulk–fault coupling.

Keep reduced Newton/FGMRES as the P2 reference and direct mechanics experimental, serial, and default off.

Leave the unsuccessful quasi-Newton path inactive for this pass. Its later removal is a separate, small deletion task; do not reorganize the entire endpoint solver just to isolate it now.

Do not generalize RSF, prestress, transport, restart, mesh adaptation, or fault geometry in this cleanup. Do not delete the working particle-history backend before the replacement covers the required BP3 behavior.

10. Verify the changed responsibilities with bounded existing checks.

Use the established build and test commands. Do not create a new test framework or repeat completed campaigns unnecessarily.

Verification should cover:
- Independent compilation and correct template instantiations after file movement.
- Existing relevant reconstructed-fault/history/projection tests.
- New normalization-only results against the preserved FE calculation, plus existing derivative/adjoint checks.
- Association access without unnecessary factor preparation; projection results and invalidation after subsequent projection requests.
- Explicit zero initialization and unchanged nonzero initialization.
- A retained accepted P2 growth entry with unchanged settings.
- A small existing MPI control for changed collective/cache behavior; include failure/retry coverage if that lifecycle path changes.

For pure movement, preserve deterministic results where applicable. For removed redundant computation, use the existing numerical tolerances and reference fields. Never relax acceptance or replace expected results to accommodate a difference.

Verify that the normalization-only path actually avoids derivative and energy work. Report measured timings only if collected; do not infer a numerical speedup from line-count reduction.

11. Finish with a compact review.

Update the existing status document and provide:
- Changed files and their responsibilities.
- Preserved behavior and deliberately removed redundant work.
- Verification results, clearly distinguishing new checks from retained evidence.
- Any unresolved restriction or scientific issue.
- One brief justification for each new helper with only one production caller.
- The next bounded BP3 implementation step.

Use the specified file layout. If one item requires broader architecture or a scientific change, complete the independent cleanup items and report that specific limitation rather than expanding the task.

Keep the result readable as scientific C++: a small number of coherent operations, explicit state ownership, and no unnecessary abstraction. Stop after this cleanup.
