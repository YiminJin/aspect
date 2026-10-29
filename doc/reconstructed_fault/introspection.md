We are returning to evolving-phase development while the BP3 first-event test runs on the server. Initial BP3 results are encouraging. After many diagnostic and optimization cycles, the worktree needs a coherent development direction and targeted consolidation.

The long-term objective is a reliable, efficient particle-free fault model that supports coupled phase evolution and eventually fault propagation. The immediate objective is to establish one clearly documented, maintainable evolving-phase path in the current branch, verify that it still works, and identify the next bounded scientific task.

Please complete the consolidation and verification described below. Do not stop after producing an inventory or plan. Resolve routine implementation choices independently; report a material scientific conflict before changing the affected equations or acceptance rules.

## 1. Establish the actual current baseline.

Verify the worktree, branch, revision, local modifications, and available builds. The remote revision reviewed for this instruction was `612b75d`, but use the actual current state and preserve subsequent work.

Protect the running BP3 configuration, executable/plugin identities, inputs, outputs, and checkpoints. Use separate build/output locations for evolving-phase work. Identify uncommitted changes before modifying overlapping files.

Read the applicable `AGENTS.md`, `current_design.md`, `specification.tex`, and `refactoring.md`. Then read the relevant current particle-free status and PF6 reports, using `propagation/P2_CLOSEOUT.md` as the older reference. Consult additional historical reports only to resolve a concrete question.

Some instruction paths are stale—for example, the particle-free derivation is at `doc/reconstructed_fault/particle_free_history.tex`, and the previously referenced `particle_free/PLAN.md` was absent from the reviewed revision. Resolve references against the actual tree. Correct proven path errors without changing scientific authority.

## 2. Reconstruct the scientific development line.

Keep these stages distinct:

- The older P2 native-FE/CPDI evolving-phase experiments.
- The newer PF6a independent-DG-history, ordinary-FE-fracture, consistent-endpoint evolving-phase implementation.
- The stationary mature-fault DG route currently used by BP3.

For each, identify its purpose, accepted-state ownership, relevant source entry points, maintained test, established evidence, and remaining limitation. Make this a compact map rather than a catalogue of every historical experiment.

Use the current PF6a/domain-free implementation as the starting point for renewed evolving-phase work. The original propagation launch instructions and early failed reports are historical context. Later successful checks supersede only the specific limitations they addressed.

Distinguish phase evolution along fixed fault geometry from actual geometric extension and physical tip propagation.

## 3. Consolidate the active implementation around clear responsibilities.

Inspect the code used by the selected evolving-phase path and its shared BP3 dependencies. Identify concrete problems such as duplicated calculations, obsolete temporary branches, unclear state ownership, stale comments, scattered mode checks, and helpers that mix constitutive calculations with acceptance or diagnostics.

Make small, behavior-preserving improvements where they directly clarify the coming work. Follow the existing material, geometry, coupled-solver, and simulator-lifecycle boundaries. Prefer existing ASPECT/deal.II infrastructure and straightforward scientific C++.

Do not launch a repository-wide refactor. Do not introduce a new framework, broad inheritance hierarchy, or another implementation of an existing operator. Avoid adding a new configuration switch for every temporary diagnostic.

Remove obsolete scaffolding only after checking callers, tests, build registration, and reproduction dependencies. Preserve unique evidence and useful independent references. Historical experiments do not all need to remain part of the active execution path.

Keep structural changes separate from numerical fixes so their effects can be assessed independently. If a numerical defect appears, establish its cause and handle it explicitly rather than hiding it inside cleanup.

## 4. Preserve the established scientific contracts.

For the selected cohesive evolving-phase path, retain independent DGQ2 bulk history, native FE strain information, surface cohesive history, consistent normalization and its derivatives/adjoint, and the selected endpoint energy and coupled solve.

The previous accepted physical phase supplies the irreversible lower obstacle. Every trial starts from the same accepted bulk and surface histories. Trial iterates and predictors are not accepted history. Publication occurs once after the actual timestep acceptance decision, with rejection/cutback restoring the accepted state.

Keep the full plane-strain tensor convention and all required DG modes. Retain unlimited DG as the baseline for these stationary tests; reopen limiter work only if a demonstrated problem requires it.

Keep cohesive elastic-slip history and mature-fault auxiliary slip memory semantically distinct. The successful frozen-phase BP3 route does not automatically qualify evolving phase.

Audit the eligibility and invalidation of recent stationary optimizations where code is shared. Phase-dependent normalization, localization, derivatives, and coupling operators must update consistently during phase trials. Reuse cached data only when its dependencies justify reuse.

Preserve existing physical inputs and acceptance criteria during consolidation. A genuine design conflict should be stated explicitly, with its consequences and a concrete proposed resolution.

## 5. Verify the maintained evolving-phase path with bounded work.

Select one existing short cohesive/domain-free evolving-phase fixture with measurable phase change and clear reference evidence. Establish its current behavior before relevant edits, then repeat it after consolidation.

Check the quantities needed to establish preserved behavior: physical phase and irreversibility, authoritative history and reconstructed stress, normalization, coupled residual/KKT and energy acceptance, and once-only publication.

Use existing rejection/restart or MPI controls when the edited ownership or lifecycle code makes them relevant. Reuse unchanged-source evidence elsewhere. Do not rerun every historical suite, undertake a convergence matrix, or launch a long trajectory merely to complete a checklist.

Report actual differences on meaningful physical scales. Distinguish functional correctness, equivalence to the entry baseline, and scientific accuracy.

Preserve the unresolved PF6a mirror-branch discrepancy and spatial/temporal limitations. Do not align fields, impose a branch preference, alter tolerances, or replace references just to obtain agreement.

## 6. Leave one concise current handoff.

Update `doc/reconstructed_fault/particle_free/STATUS.md` with a short current view: selected implementation, source/build identity, what was verified, remaining blockers, and the next action. Link detailed existing reports.

Reduce duplicated or contradictory active guidance. Preserve historical evidence through clear references; retain any uniquely recorded information before shortening a status document. Add a brief pointer from the older propagation entry point to the current continuation.

Avoid another collection of reports for individual patches. Use one compact consolidation record, preferably within the existing status/audit structure, containing the necessary changes, commands, results, and limitations.

## 7. Define the next scientific task after consolidation.

The current source explicitly guards against coupled phase evolution on hanging-node meshes. Existing frozen local-remap checks do not qualify that capability.

Assess the next bounded task as fixed-geometry phase evolution on a small locally refined mesh: independent-master constrained variations, consistent gradient/Hessian reduction, physical obstacle treatment, and acceptance/rejection/restart behavior. Do not simply remove the guard or apply independent nodal bounds to constrained DoFs.

Explain whether this is the next necessary capability for the intended experiment, or whether a more immediate demonstrated blocker should come first. Keep the mirror-branch reference question separate from constrained-solver correctness.

Automatic AMR, lossy obstacle transfer, transport, surface birth, moving geometry, and physical tip propagation belong to later explicitly scoped tasks.

Completion means that the current evolving-phase route is understandable, its directly relevant patchwork has been reduced, a short native evolving-phase control has been verified, the BP3 baseline is preserved, and the repository contains one clear next-task proposal.

At the end, summarize what became simpler, what was tested, what remains scientifically unresolved, and the precise next implementation step. Stop at that boundary rather than automatically beginning the next research campaign.
