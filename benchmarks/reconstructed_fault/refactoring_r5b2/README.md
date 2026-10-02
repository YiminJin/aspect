# R5b2: prepare a complete surface linearization candidate

Reference: accepted R5b1 `dfb7ad9f2`, branch `pf-rsf-refactor`.
Immutable reference: `build-refactor-r5b1/aspect-r5b1-qualified`, SHA256
`fd8c03ab1f1f4e363e2b69cb69d8517a35c648c5a682308004aee4453ed4910c`.
R5b1 source/artifact/local manifests were checked before its commit.
The scientific worktree and local temporary review files remain untouched.

## Selected boundary, recorded before source editing

Extract one private `prepare_surface_linearization(const SurfaceAssembly &)`
member in surface_system.cc. Its return value is an owned, complete but
unpublished SurfaceLinearization candidate. Move the contiguous candidate
construction, copy, fault-factorization, RPE and optional sparse-G block without
changing expressions or order. Do not extract either backend or another phase.

| Operation / lifetime | Owner, inputs, outputs and effects |
|---|---|
| Caller invalidation | Canonical surface owner resets old unique_ptr, increments generation, clears diagnostic before assembly. Same total timer and UMFPACK admission guard. Failed preparation leaves old inverse unavailable. |
| Assembly and observer | Caller dispatches existing backend with physical bulk state and nodal V, then invokes existing diagnostic observer before preparation. Both backend files/records untouched. Residual-only cache effects unchanged. |
| Candidate preparation | Input const SurfaceAssembly contains reduced residual/K/mass, rank-local ordered coupling samples and shared immutable normal-filter factors. Output unique_ptr owns copies, full per-fault factors, RPE and optional sparse G. No publication, generation update, history commit or acceptance. |
| Dependencies and MPI | Existing owner supplies grid cache, FE/mapping/DoF/triangulation/introspection, timer and output; environment selects sparse G only without filters. Same factor loop then collective RPE reinit and optional routing/matrix construction. Point order, multiplicity division, missing-point behavior, physical pressure/signs unchanged. |
| Publication | Caller publishes diagnostic then moves candidate pointer, returning borrowed residual. No old-state retention or new atomicity guarantee. |
| Views and filters | Full inverse/G live in published candidate; restricted factors borrow owner/generation and expire on supersession. Shared immutable filter factors survive residual-trial cache replacement. No manager/particle state moves. |

The helper and private declaration stay under existing UMFPACK compilation
admission; no unsupported-build behavior is added. No public API/data layout,
owner, validation, MPI, cache, physical policy or timer-scope change is intended.
The explicitly stopped lookup timer moves with its body; total timer remains
in caller. Only completed temporaries end in the helper before publication.

## Planned focused evidence

Release build and independent compilation, unique 2D/3D helper symbols; exact
extracted block and retained caller/source checks. Reuse recorded qualified
R5b1 one/two-rank reference outputs; run matched candidate residual/K/G, pressure
modes, explicit/reference sparse G, filter derivative/raw preservation, full/
restricted/stale solves, short coupled pressure/rollback and automatic BP3.
Keep original singular fixture unchanged and recorded; reuse its separate
current-diagnostic probe to verify failed preparation invalidates old state.
Rerun the repaired frozen AMG/GMG fixture on both binaries, four ranks, because
this task touches candidate construction and the inverse/G lifetimes it uses.
Require actual backends, fresh residuals, complete pass before intentional stop,
exact state/directions/actions, and no step-two publication. No numerical fixes.

## Reproduction and harness notes

`evidence/*.json` records commands, environments, exit codes and durations;
matching `.log` files retain the full output. Configure/build commands use the
accepted GCC/OpenMPI/deal.II/Voro Release stack and
`-fno-finite-math-only -ffp-contract=off`. The build and output directories are
separate from every qualified reference. `compile_independent.py` compiles the
changed TU without PCH/unity; the other two backend TUs are unchanged and their
independent R5b1 evidence is reused.

Build `plugin` against the R5b2 build package. A reference plugin build against
R5b1 needs only `reconstructed_fault_frozen_gmg` and `bp3_frozen_reference`.
`prepare_inputs.py` supplies unchanged existing fixtures with local paths.
Run `run_checks.py reference frozen`, then `run_checks.py candidate` with MPI
socket access. Optional case names restrict candidate runs. The small reference
cases reuse R5b1 candidate logs/outputs; only the repaired frozen case runs both
binaries afresh. `compare.py` compares the small matrix and `compare_frozen.py`
requires both complete frozen runs, backend/state/pass markers and matched data.
Exit 1 alone does not qualify an intentional-stop case. The original singular
fixture's known failure is explicitly separate from its successful supplemental
invalidation probe; no original test is edited.

`verify_extraction.py` checks the exact preparation block, reconstructs the old
caller/source, checks the single private declaration and all unchanged source,
tests, CMake, accepted artifacts and temporary local files. `verify_symbols.py`
checks actual member definitions, excluding nested lambda call operators.
Its initial parser counted those nested operators twice; initial symbol reports
are retained and show the correct member definitions already present. The
parser correction required no compilation or simulation rerun. The reference
MPI launch denied local sockets in the sandbox; the original log is preserved
as `reference-frozen-sandbox.*`, before the successful authorized launch.

`qualify.py` runs the comparisons/checks, verifies 647 protected reference-evidence
entries, freezes the candidate executable, and records source/artifact manifests.
Timing/path metadata are excluded from numerical comparisons; the existing
comparison rules retain exact field/trace/work values, with timer-count multisets
for asynchronously printed rank-local summaries. No tolerance is relaxed.

## Completed verification and review baseline

Candidate: uncommitted R5b2 over `dfb7ad9f2`; frozen executable
`build-refactor-r5b2/aspect-r5b2-qualified`, SHA256
`edf23c823e86fe579a231110f2e2167dd929fa531996501516759d8faa432cd2`.

Release build/link and independent compilation pass. Two unique helper and two
caller definitions cover 2D/3D. Six source/protection checks confirm the pre-edit
boundary/lifetime table above after extraction. No CMake, record, backend, other
production or existing test change; the only header addition is private.

All **248 matched checks pass**: 148 small one/two-rank checks against retained
R5b1 outputs and 100 fresh four-rank frozen checks. Units pass 636 assertions
in three cases per rank. Exact small comparisons include 166 BP3 CSVs, DataArrays
in 16 VTUs, four filter action CSVs and four coupled history fingerprints. The
original singular fixture still fails its stale text guard on both matched rank
counts; the supplemental probe reaches its full marker after checking generation
advance and unavailable old inverse. Neither original fixture nor diagnostics
were changed.

The frozen fixture compares 59 field files and 20 rank-local binary payloads.
Both binaries verify actual AMG/GMG use, unchanged state/RHS/operator actions,
complete pass before intentional exit 1, and accepted steps exactly [0,1]. Both
backends use 17 iterations; fresh residuals are `0.00044068965535591382` (AMG) and
`0.00048708263397949716` (GMG), below target `0.0012258892187472356`. All directions,
fields, histories, solver decisions and work counts match exactly across binaries.
No numerical tolerance or physical input was changed; elapsed times may differ.

Qualification records 31 final build/runtime outcomes, 861 source/header/unit entries and
77 candidate/reference/prepared-input/harness artifacts. Accepted source/artifacts,
647 protected evidence entries and local temporary files remain unchanged. Python
syntax and git diff checks pass. No demonstrated numerical or MPI defect.

Not run: Debug/3D runtime, unsupported-UMFPACK configuration, restart or long
scientific campaign, dedicated native-QP/line diagnostic observer test. Those
observer sites are byte-preserved; historical scientific/restart and broader
manager cache-lifecycle limitations remain. R5b2 is ready for review, uncommitted.
No further operation extraction, manager cache organization or R6 begun.
Recommended next bounded task: a separate correction of the original singular
fixture's diagnostic expectation, preserving its lifecycle assertions.

## Accepted post-R5 baseline

The user accepted R5b2 and selected R6a. The commit containing this entry records
the accepted R5b2 source, guidance and harness. All 248 comparisons and source/
artifact/protection checks were rechecked before committing, without rerunning
simulations. The immutable executable and hash above are the R6a reference.
Local temporary files and the user-supplied R6 instruction document remain
outside this R5b2 commit. The original singular-fixture text issue remains separate.
