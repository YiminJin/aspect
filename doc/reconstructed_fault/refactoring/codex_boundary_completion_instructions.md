Codex instructions for prescribed fault boundary completion

Continue on pf-rsf-refactor from the completed R2b geometry-preparation extraction. Read the current AGENTS.md, refactoring guidelines, plan, and latest review in this worktree. Those files and the current implementation are authoritative; do not use line numbers from earlier discussions.

The agreed objective is to detect boundary contacts separately for every prescribed fault and apply paired completion at supported contacts. This extends the BP3 treatment to other supported fault placements and boundary faces. Geometry alone does not determine the exterior phase field: automatic activation also requires a defined continuation and compatible mechanical coupling.

1. Update the scope and keep the changes reviewable.

Record this decision in the existing refactoring guidance and plan. Separate two changes: first, extract the existing completion implementation without numerical changes; second, add per-contact detection and the supported automatic treatment. Identify the second change as a behavior extension, not a pure refactor. Implement both within the scope below, with a short progress report after the first.

Keep the particle-domain/CPDI implementation and phase-field evolution implementation unchanged. Do not proceed to the normalization value-reuse extraction or unrelated solver cleanup. Preserve unrelated user edits.

2. Define the supported first implementation.

Support 2D prescribed faults meeting a single planar, nonperiodic boundary face transversely. Allow either endpoint, both endpoints, multiple independent faults, and different boundary faces. A fault may curve elsewhere, but its terminal neighborhood must be straight over the entire region used by this completion, with unambiguous projection.

Initially retain the existing mature/frozen-field restrictions. Use the prescribed phase-field profile and material data that already make the BP3 exterior continuation well defined. An initial fault that subsequently evolves is not automatically covered: its initial profile cannot silently substitute for the current exterior field.

Detect but do not silently approximate tangential contacts, corners, ambiguous projections, overlapping completion regions, unsupported exterior material data, curved terminal neighborhoods, or 3D contacts. Preserve the existing handling of periodic boundaries. If supporting an additional case requires a new physical assumption, report that case and the missing input separately.

“Arbitrary prescribed settings” here means supported placements, orientations, and fault identities, not unrestricted geometry or phase-field evolution.

2a. Record phase-field boundary compatibility as a prerequisite.

An oblique prescribed stationary profile generally does not satisfy homogeneous Neumann phase-field data. Exterior normalization and interior source completion cannot repair distortion introduced when solving for the physical phase field. Audit how the current BP3 and proposed automatic cases obtain phi, including initialization, constraints, and test plugins. A fully prescribed compatible frozen profile can proceed without a new phase-field boundary implementation. Freezing a field after an incompatible initialization does not establish compatibility.

Document a separate follow-up task for an evolution-compatible phase-field boundary treatment. Do not implement that task inside R2b or change modules 1 and 2 now. Complete the extraction and contact detection, and enable the new automatic path only for cases with a verified compatible prescribed field. For an H-driven solve with an incompatible or unqualified boundary treatment, report that dependency and keep the new case unsupported until it is resolved. Preserve legacy behavior and identify any existing limitation.

Distinguish a permanently prescribed boundary profile from a fault prescribed only as an initial condition. A fixed Dirichlet trace, phi_D(x) = Phi(r(x); core_phi, length_scale, constitutive_parameters), fixes the phase field on the selected boundary patch, not throughout the interior. It may suit a permanently prescribed/frozen profile, but must not be imposed automatically on an evolving fault merely because its initial H is nonzero there.

For an evolving fault, first specify the intended exterior physics. Homogeneous natural data permit boundary adaptation. Preserving an oblique continuation while allowing boundary damage to change requires an appropriate boundary/exterior model. One candidate is inhomogeneous natural data c grad(phi) dot m_out = q_phi. For a locally straight stationary profile depending only on transverse distance, its compatible initial value is q_phi = c Phi_prime(r) (n dot m_out), where m_out is the outward boundary normal and c is the isotropic gradient coefficient in the actual weak form. This prescribes the normal derivative rather than pinning phi. It is a candidate for the later design, not authorization to implement a new boundary law now.

A flux computed once from the initial profile does not generally remain compatible with an evolving profile. Specify how the exterior continuation and boundary data evolve, and keep that choice consistent with normalization completion. Do not impose both Dirichlet and Neumann data on the same patch. Initial-only Dirichlet data followed by homogeneous Neumann data can reintroduce the original profile distortion; do not treat this switch as an automatic solution.

Reuse the same profile definition when constructing initial H, any profile-based boundary data, and the exterior continuation. Do not infer a universal pointwise phi(H) law: the phase-field equation contains spatial derivatives. A known stationary-profile relation may provide a consistent construction, but its assumptions must be explicit.

H > 0 at the boundary is not equivalent to a fault-centerline crossing; a diffuse interior fault or other driving force can also give nonzero boundary H. Keep contact detection geometric. Determine the boundary-condition footprint from the intended profile, not a binary H test or just the intersection point. Do not prescribe phi = 0 over an entire boundary merely because it has one contact.

Before implementing the follow-up, specify its initialization/evolution lifecycle and compatibility with irreversibility. Dirichlet data belong in the existing constraint infrastructure where suitable; distinguish physical values from homogeneous Newton increments. Natural data require the corresponding boundary contribution to the weak form. Cover refinement and restart for the chosen treatment. Keep profile evaluation separate from the generic phase-field boundary mechanism.

Qualify that later task with a small oblique-fault initialization comparison between homogeneous natural and profile-consistent boundary data, then a short changed-driving-force case showing that the intended boundary phase field can evolve. Check the profile and normalization near the boundary. Requalify the affected completed case afterward; this is a separate numerical change, not part of the move-only BP3 regression.

3. Extract the existing BP3 path first.

Keep the completion-file loading, validation, integral addition, and diagnostics in a private PhaseFieldFault helper under source/material_model/phase_field_fault/. Retain current algorithms, restrictions, cache behavior, MPI operations, and restart behavior.

Document the existing interior source-continuation path alongside this helper. Do not relocate mechanical assembly into the material model merely to put both contributions in one file.

Use the existing short BP3 qualification to establish that this extraction preserves behavior before adding automatic detection.

4. Detect contacts geometrically, per fault and per contact.

Inspect all prescribed fault polylines against the actual domain boundary. Use boundary geometry and identifiers; remove assumptions that the relevant fault is fault zero or that its contact is on the bottom boundary.

Check segment intersections as well as endpoint distances, so a crossing cannot be missed when input vertices lie outside the domain. Reuse existing clipping/topology handling where available. If an intersection cannot be represented by the current fault topology, report it as unsupported rather than inventing a new endpoint and history mapping in this task.

Deduplicate contacts at shared vertices. Distinguish an interior endpoint from a boundary crossing; proximity alone must not create a continuation. Use a documented, scale-aware geometric tolerance.

Return a small geometry record containing fault/contact identity, endpoint or segment identity, boundary ID, intersection position, inward fault tangent, fault normal, inward boundary normal, and support/admissibility information. Keep constitutive coefficients, completion-file parsing, and h(phi) evaluation out of this geometry record.

Build these records when prescribed geometry is prepared, and rebuild them through the existing relevant geometry/mesh/restart invalidation paths. Avoid repeating global boundary searches during quadrature or every nonlinear iteration. Keep MPI decisions consistent and diagnostics rank controlled.

5. Use one local description for the two completion regions.

At a supported contact let e be the intersection, t the unit fault tangent pointing into the domain, n the fault normal, and m the inward boundary normal. Write

    x = e + s t + r n
    a = m dot t > 0
    b = m dot n

The local physical domain is a s + b r >= 0. The exterior normalization region belongs to the real fault side s >= 0 but lies outside that half-plane. The interior continuation region lies in the physical domain with s < 0.

For symmetric transverse extent |r| <= R, the geometric influence length is R |b| / a. Use this to check that the straight terminal neighborhood and unambiguous projection cover the required region. Use the actual profile extents where they are asymmetric.

The two regions are mapped into each other by (s,r) -> (-s,-r), but equal geometry does not imply equal integrands or mechanical weights. Do not implement completion by blindly doubling an interior integral.

At an exactly perpendicular crossing b = 0, both regions have zero measure and the additional contributions are zero. For nearly perpendicular contacts, evaluate the small actual correction or establish a documented error tolerance; do not silently rotate the prescribed fault.

6. Complete normalization using the existing discrete profile convention.

At each affected real-fault profile, retain the current in-domain integral. Integrate only the missing exterior part and form I_total = I_in + I_out before the existing projection to fault nodes. Leave the projection mass matrix and fault quadrature weights unchanged. Ensure each correction is applied once.

Evaluate the same localization integrand, degradation law, and material coefficients as the normal normalization path. For the supported prescribed frozen field, reuse the prescribed profile evaluator and BP3 ghost-Q1 continuation convention. Preserve the mesh orientation, spacing, interpolation, and integration conventions needed to reproduce BP3; do not substitute a continuum analytical normalization merely because it is easier.

Keep exterior field evaluation outside generic reconstructed-fault geometry code. Do not reflect current interior values unless that field symmetry has been explicitly established. If the prescribed profile or required exterior coefficients are unavailable, identify the unsupported contact rather than guessing them.

Keep legacy completion files as a compatibility path where needed. New supported geometry should not require a hand-authored CSV for each crossing. Legacy and automatic completion must be mutually exclusive at a contact to prevent double counting.

7. Keep the interior mechanical continuation consistent.

In the physical interior wedge associated with the virtual extension, evaluate phase field, pressure, stress, and particle history at the actual physical point. Use the established endpoint continuation for the surface quantities: endpoint slip rate, normalization, and applicable surface state. Do not create physical exterior cells, exterior particles, or new exterior slip unknowns.

Use one common association and endpoint shape-function rule for all affected bulk and surface residuals and Jacobian/coupling terms. For a free endpoint, retain derivatives with respect to its slip unknown. Preserve the established prescribed-endpoint treatment.

Do not add a source only to the bulk residual while omitting its surface work or coupling derivatives. Do not assume G = B transpose. Preserve the selected integration backend and its weighting conventions. An integration extent used for I_h must not become a new mechanical cutoff that truncates nonzero physical phase-field support.

Handle multi-fault ownership through the existing association policy. Do not count one physical contribution for two contacts; detect unsupported overlap rather than choosing a nearest continuation without justification.

8. Preserve decoupling and compatibility.

Reconstructed-fault code owns boundary geometry, contact records, and geometric association. PhaseFieldFault owns the constitutive continuation and normalization correction. Existing coupling/assembly code owns mechanical contributions. Share the contact description; do not build a generic boundary-completion plugin framework.

Reuse existing selectors where suitable. If automatic completion needs a new selector, make it an explicit parameter value and preserve existing input-file behavior by default. Selecting automatic mode should classify every prescribed fault without per-fault manual flags. In that mode, an unsupported contact requiring completion must produce an actionable error before assembly, not silently run with an incomplete correction.

Distinguish: interior/no contact; supported perpendicular/no correction; supported paired completion; unsupported contact. An interior fault whose diffuse support touches the boundary is not a crossing and must not acquire a fictitious extension.

9. Run a small, targeted qualification.

Reuse the existing BP3 fixture for the extraction check and compare automatic completion against the legacy BP3 result, including both normalization and mechanical contributions. Require exact agreement for the move-only change where the existing harness supports it; justify any tolerance for the new path.

Use small geometry/integration cases to cover an interior fault, a perpendicular crossing, oblique contacts on different faces, both endpoints, reversed polyline ordering, and multiple independent faults. Verify contact identities, no double counting, zero perpendicular correction, and an actionable rejection for a corner or tangential contact.

Include a focused free-endpoint coupling check or reuse an existing equivalent check; use a directional finite-difference residual/Jacobian check if coverage is missing. Reuse the short existing restart and MPI harness where needed to confirm contact reconstruction and that corrections do not multiply with rank count.

Keep runs suitable for a laptop. Do not launch a long earthquake-cycle test or rerun unrelated qualification. If compilation or runtime testing is unavailable, state exactly what remains unverified.

10. Report for human review.

Give a short report with: what changed; what is now supported; what remains unsupported; focused checks and outcomes; any numerical differences from BP3; and the next decision, if one is actually needed. Include changed file names and a compact per-contact summary for the test fixtures. State how phase-field boundary compatibility was established and which cases await the separate boundary-condition task. Keep detailed logs separate.

Finish this boundary-completion task before proposing the next refactoring extraction.
