# Codex instructions: R5 manager and surface-system organization

## Active task

R4 is accepted and the user reports that the frozen AMG/GMG fixture repair is complete. Start from the locally accepted post-R4 source and repaired-fixture evidence. Confirm their actual revisions and manifests; do not assume that the published remote branch contains these changes.

Update the existing R5 guidance and implement **R5a1: manager slip-rate lifecycle organization only**. Complete its source changes, build, and focused verification as one task, without approval pauses between individual functions or tests. Report and stop before R5a2 or surface-system implementation.

The later passes below define the intended organization and review points. They are not blanket authorization to reorganize both classes at once. Routine implementation choices within the selected pass are yours to resolve.

## 1. Goal, boundaries, and working baseline

Make responsibilities, state transitions, and cache lifetimes explicit while preserving the current algorithms. File separation is useful when it follows these responsibilities; reducing line count alone is not the goal.

Read AGENTS.md, the relevant current_design.md/specification.tex sections, refactoring.md, the staged roadmap, and the accepted R4/fixture-repair records. Inspect current headers, definitions, callers, serialization, and affected tests before editing. Preserve the scientific-test worktree and unrelated local changes.

Record the branch, HEAD/local patch, accepted reference executable/plugins, inputs, and repaired-fixture evidence. Preserve reference artifacts and build the candidate separately. Reuse qualified R4 evidence where the selected diff does not reach those paths. Do not repeat the full R4 verification campaign to establish a baseline.

The responsibility boundaries remain:

| Component | Ownership and role |
|---|---|
| M3 manager/infrastructure | Fault geometry/connectivity, reconstruction, generic properties and persistence, associations/projection geometry, distinguished slip-rate storage and lifecycle |
| M4 material | Constitutive laws, normalization policy, physical history meaning and updates |
| M5 surface system/coupling/solver | Surface and bulk weak forms, MPI assembly, K_V/B/G, factorizations, restricted solves, nonlinear acceptance and publication orchestration |

The surface system belongs to M5 even though its source is under reconstructed_fault/. Do not make it constitutively generic by inventing a material interface during R5. Keep the existing concrete material interface and canonical simulator-owned instances.

M1 particle domains/CPDI and M2 phase-field implementation stay frozen. Particle creation/removal, advection, interpolation choices, domain construction, integration measures, propagation, and boundary-condition algorithms are not part of R5. In particular, do not replace the current projection-domain construction with another particle-domain API as incidental cleanup.

Update doc/reconstructed_fault/refactoring.md and doc/reconstructed_fault/refactoring/refactoring_plan.md in place. Keep AGENTS.md changes minimal and preserve historical reports. Record the accepted R4 disposition without claiming to have rerun its evidence.

## 2. R5a1 — Manager slip-rate lifecycle, selected now

Begin with a compact responsibility inventory of the manager, sufficient to distinguish geometry/reconstruction, property registry/persistence, slip-rate lifecycle, particle projection, Stokes-QP associations, and boundary-contact/source-continuation support. Identify existing extracted implementations before proposing new files. This inventory is preparation for implementation, not a separate approval gate.

For the slip-rate group, record a short table of each state member's owner, readers, writers, initialization, reset, and checkpoint treatment. Include:

- timestep-committed, current-Newton, and trial slip rates;
- initialization and active-solve/trial flags;
- prescribed nodal rates and their reattachment after restart;
- sizing triggered by geometry creation and deserialization.

Move the complete slip-rate method definitions and exclusive helpers into a focused implementation file, for example source/reconstructed_fault/manager_slip_rate.cc. Retain ReconstructedFaultManager as owner and keep its public API and data layout unchanged. The current method group includes:

- initialization/readiness, current/committed access, and slip-rate interpolation;
- prescribed-rate configuration and mask access;
- begin/validate/commit/rollback of the nonlinear solve;
- begin/set/accept/rollback of a trial, including the absolute-value setter.

Resolve these names against the current source. Preserve bodies, arithmetic, validation order, comments that describe behavior, and exception specifications. Leave generic property registration, geometry creation, archive save/load, and rebuild_after_deserialization() in their existing locations in this pass. Document their cross-file initialization responsibilities.

Do not add a separate slip-rate manager, state-machine framework, generic transaction object, or new storage copy. Do not group all manager members into a new struct during the mechanical move.

Preserve these specific contracts:

1. Beginning a solve initializes the working Newton state from committed values and applies prescribed values at the existing point.
2. Accepting a trial updates the Newton iterate; only the converged-solve commit publishes timestep slip rates.
3. Trial rollback and whole-solve rollback restore their respective states and flags exactly.
4. Absolute trial values retain exact values at bound contact; do not reconstruct them through subtract/add arithmetic.
5. M3 retains its existing generic admissibility checks. Constitutive lower-bound policy remains with its current M4/M5 callers.
6. Commit validation and the existing no-allocation/noexcept commit contract remain separate and unchanged.
7. Restart restores the existing persistent data and reconstructs transient working storage. Retain the R3 fix sizing prescribed_slip_rates to the number of reconstructed faults. Actual prescribed conditions remain caller-supplied; do not serialize them as new history.

Adjust includes, source discovery, and explicit template instantiations as needed. Keep each translation unit independently compilable and avoid duplicate whole-class/member instantiations. Do not change existing unity groups or unrelated build settings merely to accommodate the move; use the established focused independent-compilation approach when appropriate.

A successful R5a1 result makes the lifecycle easy to locate and explain while leaving ownership unchanged. Describe this honestly as organization of an existing responsibility, not complete removal of manager coupling.

## 3. R5a2 — Manager projection and cache organization, later selection

Use the inventory to select a coherent projection group. Keep particle-to-fault projection distinct from Stokes quadrature-point associations: their consumers, measures, inputs, and invalidation conditions are different.

A suitable layout may separate manager_particle_projection.cc and manager_stokes_projection.cc, retaining member definitions and the existing manager owner. Names are illustrative. Do not create empty future files or relocate the already extracted boundary-contact implementation for naming uniformity.

For the selected cache, document:

| Required item | Meaning |
|---|---|
| Contents | Geometry/associations, factors, sampled values, and diagnostics, distinguishing each |
| Validity inputs | Exact existing versions, mesh/particle/DoF assumptions, and configuration dependencies |
| Rebuild and invalidation | Actual events/callers and their order |
| Ownership and lifetime | Rank-local versus replicated data; returned references and expiry |
| MPI behavior | Local contributions, collectives, empty-rank behavior, and failure propagation |

First move a complete coherent implementation group. Any later extraction of large operations should follow the existing sequence of association construction, local accumulation, reduction, solve, and publication without moving validations or collectives across those boundaries.

Preserve projection weights and measures, Q1 mass matrices, parent sampling, property component mapping, endpoint/source-continuation rules, geometric tolerances, and all validity inputs. Keep geometric cache validity separate from current material/phase values. Do not strengthen or relax cache reuse without a separately reviewed algorithm change.

Keep source-continuation geometry distinct from surface quadrature admission. An association valid for a bulk source is not automatically valid for the surface weak measure. Preserve the established legacy and automatic boundary-completion behavior.

Property registry and serialization remain generic. Do not hard-code Theta, cohesive traction, a friction law, or material history updates into manager internals. Geometry/reconstruction and registry/restart reorganization are additional candidates only if the inventory demonstrates a clear benefit; they are not automatically required to finish R5.

## 4. R5b1 — Surface-system implementation boundaries, later selection

Before moving code, map both existing assembly paths, their dispatcher, consumers, and intermediate types:

- particle/domain-based surface assembly;
- bulk-work/Stokes-quadrature assembly;
- residual-only evaluation;
- linearization construction/publication;
- full and restricted surface solves;
- G application, including the existing explicit/reference alternatives;
- optional normal-filter and diagnostic operations.

Record what each assembly path integrates and what data it samples. Preserve both paths and their selection rules. Similar loops do not justify merging distinct quadrature measures or sampling conventions.

A reasonable first surface pass is to separate the substantial backend assembly implementations into focused source files while retaining dispatch and linearization lifecycle in surface_system.cc. Select the exact move set from the inventory; do not simultaneously redesign the assembly pipeline.

Keep SurfaceAssembly and SurfaceLinearization private. A narrow source-private header may hold complete definitions needed by multiple translation units, following the existing internal-header convention. Do not expose implementation records publicly, duplicate their definitions, or introduce broad accessors. Preserve construction/destruction requirements for private incomplete types and template instantiations.

Do not merge scratch assembly results with published linearization state merely because their members look similar. Retain existing FaultSurfaceDirect, normal-filter, sparse-coupling, and restricted-solve helpers instead of creating competing implementations or wrapper layers.

## 5. R5b2 — Clarify surface assembly and linearization operations, later selection

After the implementation boundaries are established, extract a substantial operation only where its inputs, outputs, state changes, and lifetime can be stated clearly. Possible boundaries include local weak-form accumulation, collective reduction/validation, or construction of linearization factors and coupling lookup data. Select one coherent extraction for a task.

Do not force every phase into a helper or introduce callback policies to make the two assembly paths look identical. Use existing scratch records where suitable, without turning them into containers for unrelated Simulator state.

Preserve the following:

- Locally owned contributions, replicated surface results, quadrature/parent order, reductions, and collective failure checks.
- Consistent residual/Jacobian coefficients and signs, physical pressure units, normal/shear conventions, and pressure-mode behavior.
- Full K_V factorization and the principal free-block inverse used by restricted solves; exact zero active entries and current mask semantics.
- The same data used by K_V and G, existing remote-point multiplicity handling, and explicit/reference G agreement.
- The actual invalidation/publication sequence. In the reviewed implementation, previous linearization data is invalidated and its generation advances before constructing the new candidate. Preserve what happens if construction fails; do not silently retain an old semantic inverse as a supposed exception-safety improvement.
- Generation checks and the lifetime of restricted inverses, borrowed linearization views, matrices, remote-point caches, and filter objects.
- Normal-filter raw/projected/Helmholtz distinctions, including zero-length behavior and derivative consistency.
- Diagnostic sampling and observer timing. Residual-only evaluation does not publish material history, but may update existing transient caches/diagnostics; do not describe it as side-effect-free.
- Existing timer behavior on rank-local failure paths; do not introduce implicit collectives through reorganized instrumentation.

Keep material-history commits and nonlinear acceptance with their existing M4/M5 owners. Do not move B assembly into the surface system or assume B is the transpose of G. Diagnostic extraction, switch removal, and parameter cleanup remain R6 work.

## 6. Verification proportional to the selected change

Use existing focused tests and qualified comparison rules. Do not repeat all solver/backend tests after every file move.

| Selected group | Required focused evidence |
|---|---|
| R5a1 slip-rate lifecycle | Build/link and independent 2D/3D symbol checks; existing initialization, prescribed-rate, trial/accept/rollback, absolute-bound and commit checks; restart regression covering restored rates before any prescribed-rate setter and later reattachment; a short coupled comparison with rollback, using the existing small one/two-rank coverage |
| Manager particle projection | Existing projection accuracy/support and component-mapping checks; cache hit/rebuild/invalidation checks; relevant geometry/restart and one/two-rank comparisons; one short affected coupled fixture |
| Manager Stokes-QP associations | Existing quadrature-identity/order, source/endpoint and cache checks; affected bulk-work/coupling comparisons on one/two ranks |
| Surface backend movement/extraction | Existing residual and Jacobian/G-action comparisons for each touched backend; relevant normal-filter and endpoint identities; restricted-solve/stale-generation checks where touched; a short coupled trajectory and failure/rollback comparison |

Select the smallest existing cases covering these contracts. Use preserved baseline checkpoints for restart compatibility where relevant; do not change archive layout or expected data. Add a small test only for a concrete missing contract, such as a moved symbol or an uncovered transient reset.

For unchanged arithmetic, compare deterministic fields, histories, solver decisions, and cache counters exactly at matching ranks/stacks, excluding time/path metadata. Investigate differences without loosening tolerances. Where cross-rank comparisons already use qualified tolerances, retain them.

The repaired frozen AMG/GMG fixture need not be rerun for slip-rate definition movement alone. Reuse it when a later selected change affects the frozen operator, linearization/inverse lifetimes, or observer contract it exercises. Ordinary melt/BFBT/direct/GMG tests are unnecessary unless shared solver code is actually touched.

Keep known scientific limitations and historical Stage-J failures separate. A newly exposed defect is a separate correction task, not permission to redesign a numerical operation inside R5.

## 7. Review and completion

Keep mechanical movement, structural extraction, documentation, and any separately selected bug fix distinguishable in diffs/commits. Follow the existing local commit policy. No push, merge, or scientific-worktree modification is requested.

Update the rolling review with a short responsibility/state/cache table for the touched group. The main user-facing report should be about 250 words plus a compact PASS/FAIL/BLOCKED/NOT RUN table:

1. What changed and the boundary it clarifies.
2. Whether any API, owner, lifetime, or behavior changed.
3. Reference/candidate identification and focused verification.
4. Remaining limitations.
5. One proposed next bounded task.

Do not make the user approve every helper or test invocation. Review occurs after a coherent pass. If a proposed extraction requires a new owner, public interface, numerical change, or substantial build redesign, identify that scope change instead of silently proceeding.

R5 is complete when the selected manager and surface operations have clear owners and documented lifetimes, the affected invariants are verified, and the remaining code is understandable without further speculative extraction. There is no mandatory file-count or line-count target.

For this instruction, finish **guideline updates and R5a1 only**, then report for review.

