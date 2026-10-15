Implement and verify a bounded transfer of native FE internal-strain history from P2 to stationary, frozen-phase-field BP3.

The deliverable is a verified local BP3 implementation, small comparison tests, and a prepared fresh 150 × 50 km BP3 configuration. This task does not include a long first-event run or cluster submission.

**1. Establish the actual baseline and preserve both worktrees**

Work in the current cleaned `fault-propagation` worktree. Read its applicable repository instructions, the P2 cleanup report, current design/specification, and P2 closeout.

The BP3 reference worktree is `../aspect-pf-rsf-refactor/`. The user’s local data path is `benchmarks/bp3`; the remote maintained source path is `benchmarks/reconstructed_fault/bp3`. Locate and distinguish the actual local inputs, plugin sources, outputs, and checkpoints instead of assuming these paths are interchangeable.

Inspect local changes and recent commits in both worktrees. The local cleanup may be newer than the remote branch. Preserve the recovery correction and all unrelated edits.

Use the other worktree as a read-only source of inputs and evidence. Build ASPECT and the BP3 plugin against the current worktree in separate build/output directories. Never load an old plugin binary into the new executable.

Inspect the maintained BP3 `production/README.md`, `production/bp3_fresh.prm`, `plugin/README.md`, and relevant recent local reports. The historical `bp3_150x50_first_event.prm` is a restart input, not a fresh template.

Bring over only the required maintained sources and inputs. Do not merge the entire refactor branch, copy all historical experiments, or overwrite the existing BP3 data.

**2. Define the first supported configuration**

Implement the following explicitly supported case:

- Two-dimensional, straight dipping fault, including the actual BP3 60° geometry.
- Frozen phase field and reconstructed geometry; mature frictional fault.
- Fixed mesh after initialization, including graded meshes and hanging-node constraints.
- Homogeneous bulk shear modulus and viscosity.
- Existing spatially varying BP3 friction parameters, RSF state evolution, radiation damping, and mechanical normal-traction feedback.
- Fresh initialization with zero bulk Maxwell stress change and the existing background fault tractions.
- Native FE history as the authoritative mechanical history.

Keep `Evolve phase field = false` and the P2 consistent-endpoint phase solve disabled. Do not run phase-energy minimization, phase Hessians, saddle searches, or the experimental P2 direct solver for this task. Retain the established iterative BP3 mechanics.

Start with a stationary-history diagnostic: no material-history transport, mesh motion, or mesh adaptation during stepping. Make this restriction explicit in the inputs and report. For matched comparisons, apply the same stationary treatment to both backends.

Do not merely remove particle-position or birth assertions. Moving particles, insertion/deletion, and FE history transport require a separate ownership argument. Either retain explicit rejection in this first implementation or support a narrowly defined derived-sampling lifecycle with its own verification. Do not claim that fixed Eulerian FE history reproduces advected particle history.

Keep legacy restart behavior intact. Do not convert an old particle checkpoint into FE history. FE restart and transport may remain explicitly unsupported in this bounded task; clearly distinguish the resulting fresh-run candidate from a production-ready long-run model.

**3. Derive and implement the mature-fault history correctly**

Before editing the recurrence, write a short derivation showing that the FE representation reproduces the existing frozen-profile mature-fault Maxwell update.

Check the representation against

\[
\tau^n=\beta_n\tau^{n-1}
 +2\eta_{\mathrm{ve},n}
 \left[D_3(u^n)-\chi V^n S\right],
\]

where the tensor convention, slip sign, localization \(\chi\), and background/perturbation split must match the actual implementation.

For fixed geometry/profile and homogeneous coefficients, assess the existing decomposition

\[
\tau=2G D_3(d_R)-2\chi ZS,
\]

with recurrences

\[
d_R^n=\beta_n d_R^{n-1}
      +\frac{\eta_{\mathrm{ve},n}}{G}u^n,
\qquad
Z^n=\beta_n Z^{n-1}+\eta_{\mathrm{ve},n}V^n.
\]

Here \(Z\) represents the accumulated slip contribution to bulk stress. **In mature mode it must not be inferred from \(I_h\tau^{coh}\), because mature cohesive traction remains zero.** Preserve the existing cohesive-mode identity and its independent P2 verification.

Retain the exact Maxwell coefficients and stable `expm1` evaluation. Preserve existing RSF aging, state-lagging, initialization, and timestep conventions; this is not a change to the RSF integration scheme.

Keep the background normal/shear tractions in their existing surface representation. Do not inject them into the FE bulk history or count them twice. The maintained BP3 initialization uses zero bulk perturbation stress.

Every nonlinear trial must read the same incoming accepted histories. Validate the complete candidate before publication, commit once per accepted physical step, and restore all histories on failure or cutback.

**4. Extend geometry and normalization without importing P2 assumptions**

Replace horizontal/unit-domain indexing with the existing reconstructed-fault geometry and bulk-source association interfaces:

- Use actual segment coordinates, normals, tangents, shear sense, and physical lengths.
- Handle the actual graded mesh and constraints.
- Preserve endpoint source continuation and automatic boundary completion.
- Use the same associations and localization in assembly, stress evaluation, history update, and diagnostics.

For frozen BP3, prefer reusing the established consistent \(I_h\) calculation and its boundary completion. The FE history consumer must use the same accepted nodal normalization as BP3 mechanics. Do not overwrite it with P2’s full-normal-domain calculation.

P2’s analytic normal-ray formula assumes a linear phase restriction along each ray interval. An oblique ray through a bilinear Q1 cell generally has a quadratic restriction. Do not generalize that formula by rotating endpoints. Use the existing general cell-interval integration, or demonstrate correctness of a necessary extension against independent quadrature.

No normalization floor, lumping change, phase-profile adjustment, or omitted endpoint contribution is permitted to make the tests pass.

Do not construct a dense fault-wide inverse for BP3. Reuse the existing consistent projection solve and valid caches. Keep memory proportional to the supported discretization.

Check the mechanical quadrature separately: P2’s horizontal tangential-panel construction is not automatically valid on a dipping, graded mesh. Use a consistent rule across residuals, tangents, coupling operators, and checks, then perform one bounded quadrature comparison.

**5. Keep the implementation compact and within existing modules**

Use these responsibilities:

| Existing file/module | Intended responsibility |
|---|---|
| `source/material_model/phase_field_fault/fe_strain.cc` | FE state, mature slip memory, geometry-aware stress evaluation, recurrence, validation and publication |
| `include/aspect/material_model/phase_field_fault/fe_strain_internal.h` | Necessary internal FE-state data |
| `source/material_model/phase_field_fault/history.cc` | Initialization, accepted-state lifecycle, derived particle samples and rollback integration |
| `source/material_model/phase_field_fault/normalization.cc` | Dispatch to the appropriate existing normalization and completion path |
| `source/material_model/phase_field_fault/constitutive.cc` | Existing shared constitutive response; narrowly necessary integration changes |
| `source/material_model/phase_field_fault.cc` and its public header | Existing option validation and declarations |
| Maintained BP3 `plugin/` | Benchmark loading, initialization, diagnostics and input handling |

Keep `fe_phase_energy.cc` focused on the retained evolving-phase P2 implementation. Modify `fe_normalization.cc` only where genuinely shared functionality requires it.

Prefer no new core implementation files. Do not add a backend framework, new manager, duplicate constitutive law, or collection of small helper headers.

Follow ASPECT coding style. Balance efficiency, readability, and conciseness. A helper should encapsulate a substantial numerical/lifecycle operation or remove meaningful repeated work; avoid trivial wrappers and single-use helpers that only hide a few readable statements. Format only touched files using the repository convention.

Preserve the generic particle projection services still used for composition, initialization, and other callers.

**6. Resolve the constitutive difference before interpreting comparisons**

The cleanup report records that native FE history uses `consistent_deviator`, while the legacy particle history consumes the full symmetric gradient.

Therefore, a production FE-versus-particle comparison changes more than history representation.

Use a divergence-free manufactured field for an initial matched check. Also evaluate a controlled nonzero-divergence case to quantify the tensor-convention difference independently. A test-only matched constitutive reference is acceptable; do not silently change the production legacy backend or add a general-purpose switch solely for this comparison.

Distinguish errors from history representation, tensor convention, normalization, quadrature, and transport assumptions in the report.

**7. Run a small sequence of local checks**

Reuse existing test harnesses and the local dipping-fault fixtures where possible. Do not create an exhaustive test matrix.

A. **History and geometry checks**

- Exercise a dimensional, translated 60° fault with nonuniform surface slip.
- Verify the mature FE recurrence against direct pointwise Maxwell evaluation at bulk quadrature points and particle sample points.
- Include relaxation, changing timestep, and a timestep comparable to the Maxwell time.
- Check slip orientation and endpoint continuation.
- Verify \(I_h\) on an oblique profile, including boundary completion and a graded-mesh case.
- Confirm finite positive normalization, frozen phase, zero mature cohesive traction, and nonzero slip memory when slip occurs.

B. **Small coupled dipping-fault BP3**

Adapt the existing approximately 2 × 1 km local dipping-fault setup, or an equivalent bounded fixture. Keep the production length scale and sufficient local resolution. Initially use uniform velocity-strengthening friction to avoid mixing nucleation with implementation errors.

Run uniform creep, then a small smooth state perturbation, for approximately ten accepted steps. Use the same physical setup, prescribed timestep sequence, quadrature, filter, and stationary-history assumption for the FE and retained particle controls.

Retain the full bottom velocity condition unless newer local evidence establishes that the intended baseline changed. Do not automatically select the experimental fault-parallel bottom condition.

C. **One temporal and one spatial/quadrature check**

For the selected small FE case, halve the timestep over the same physical interval. Add one targeted mesh or quadrature refinement to distinguish integration/discretization effects from history errors.

Also exercise the actual BP3 weakening-to-strengthening transition in a bounded fixture or targeted material/initialization check; a uniform-strengthening test alone does not verify the spatially varying model.

D. **MPI and rejected-step lifecycle**

Repeat the selected small case on one and two ranks. Exercise one controlled failure/cutback followed by retry, verifying that FE history, slip memory, RSF state, cumulative slip, and derived samples are published exactly once.

Use existing numerical tolerances and physically scaled comparisons. Do not relax tolerances or classify a pre-existing failure as a pass.

Keep the initial simulation campaign within approximately one hour of local runtime, excluding compilation. Set explicit step limits and wall limits before launch. Extend only a narrowly necessary check; do not launch the full production mesh by accident.

**8. Rebuild the BP3 candidate and retain the filter initially**

Prepare a fresh native-FE 150 × 50 km BP3 input from the maintained local setup, with the actual 60° fault, frozen profile, spatial friction transition, background tractions, loading, and intended bottom boundary condition.

Record the resolved physical and numerical settings. Reconcile local inputs with the maintained template: the remote template labels some timestep/cutback choices as provisional. Do not silently adopt an old 100 s first-step cap, increase it, or alter the logarithmic-state limiter as part of the history comparison.

Keep the 20 m Helmholtz filter for the initial paired tests. Only after those checks pass, run one short otherwise identical unfiltered FE control. No simultaneous boundary-condition, timestep-policy, or friction tuning.

Prepare the full configuration, but execute only a bounded local version with an explicitly documented mesh reduction. A coarse startup check is an integration test, not a resolved BP3 accuracy result.

Do not resume the old oscillatory particle checkpoint. A late-time frozen-state diagnostic, if useful, must be labeled separately from a continuous native-FE trajectory.

**9. Report numerical behavior and stop at the agreed boundary**

Record compact profiles and summaries of:

- Slip rate, cumulative slip, state, and \(\Omega=V\theta/D_c\).
- Raw and filtered normal traction, with pressure and deviatoric contributions separated.
- Shear traction, nonlinear residuals/iterations, cutbacks, and runtime.
- Oscillation amplitude and spatial wavelength near endpoints, mesh transitions, and the friction transition.
- FE/pointwise recurrence errors, MPI differences, and publication/rollback results.

Do not claim that short smooth local runs prove the late first-event ringing is eliminated, or that the normal filter is unnecessary.

Deliver the implementation, reproducible small-test commands, prepared fresh BP3 input, and one concise report listing exact revisions, changed responsibilities, actual results, and remaining restrictions. Keep generated outputs out of the source tree’s tracked files.

Update the existing task status with a short recovery summary. Finish the authorized implementation and local checks autonomously; stop for a genuine unresolved mathematical conflict rather than choosing a new physical law.

Do not begin propagation, a long first-event campaign, server execution, or broad additional cleanup.