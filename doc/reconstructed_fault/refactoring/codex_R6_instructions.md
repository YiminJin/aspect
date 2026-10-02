# Codex instructions: R6 diagnostics and configuration organization

## Selected task and review boundary

R5 is complete according to the user. Use the locally accepted post-R5 source, qualified artifacts, and recorded fixture repairs as the reference. Confirm the actual revisions and preserve unrelated local work; do not reconstruct the baseline from an older remote snapshot.

Update the existing R6 guidance and implement **R6a: refreshed feature-switch inventory plus one coherent material-history diagnostic extraction**. Complete implementation, compilation, and focused verification as one task. No intermediate approval is needed for individual functions or tests.

R6b and R6c below define subsequent work. Stop after R6a with a concise report; do not automatically reorganize all output, migrate parameters, remove experiments, or begin R7.

Read AGENTS.md, the accepted R6 plan, current_design.md/specification.tex where relevant, refactoring.md, and the post-R5 review. Preserve the separate scientific-test worktree. M1/CPDI and M2/phase-field implementation remain frozen.

## 1. Purpose and responsibility boundaries

Make material and solver algorithms readable without embedded CSV formatting, while retaining the diagnostic evidence needed for the current fault-mechanics research. Keep numerical controls and benchmark inputs recognizable and owned by the appropriate component.

This is a behavior-preserving organization task. Do not change equations, physical/numerical defaults, sampling or output frequency, convergence decisions, checkpoint formats, error policy, or MPI ownership.

Use the existing three categories, based on actual behavior rather than names:

| Category | Treatment in R6 |
|---|---|
| Observation and verification | Extract presentation/recording where useful; preserve the original measurements, checks, side effects, and capture points |
| Numerical implementation choice | Document owner, dependencies and current selection semantics; preserve implementation and defaults |
| Benchmark initialization/loading | Identify the existing setup/output boundary and possible existing extension points; relocate only in a separately selected pass |

Verification is not always passive observation. A residual audit may assemble again, modify scratch state, invoke collectives, or deliberately stop execution. Record those behaviors separately from its output formatting.

Do not classify a switch as harmless because its name contains DIAGNOSTIC. Completion input, prescribed loading, filtering, reference evaluation and cache controls may affect the calculation or its execution path. In particular, retain the existing meaning of boundary-completion inputs and numerical normal-stress filtering.

Keep M3 generic infrastructure, M4 material physics/history, and M5 mechanics ownership unchanged. Diagnostic code may consume the exact values supplied by these owners; it must not become another owner of geometry, material history, or nonlinear state.

## 2. Refresh the R0 inventory

Locate the R0 switch inventory and update it against current production code, feature headers, relevant test/benchmark plugins, and their callers. Include environment variables, feature-related parameters, setters, and observer registrations where they provide alternative control paths. Do not audit unrelated ASPECT configuration.

Group entries by owning component. Record:

- Exact selector/API name, owner, and current readers.
- Where and when it is evaluated: startup, timestep, nonlinear solve, or individual call.
- Current default and parsing semantics: presence versus value, empty string or "0", numeric parsing, invalid input, and precedence between existing controls.
- Primary category and any additional numerical, verification, output, or failure effects.
- MPI participation and any existing requirement for consistent settings across ranks.
- Output files/streams, naming, units, schema, and known plotting scripts, tests, or research consumers.
- Current disposition: retain, selected extraction, or future compatibility decision.

Distinguish a live reader from a historical name that appears only in scripts or reports. Document stale usage; do not revive an obsolete production switch or delete archived evidence.

Useful entries to check, if still present, include ASPECT_FAULT_PERFORMANCE, ASPECT_FAULT_LINEAR_PERFORMANCE, ASPECT_FAULT_NONLINEAR_DIAGNOSTIC, ASPECT_FAULT_EXPLICIT_G, ASPECT_FAULT_INTERFACE_MODES, ASPECT_DISABLE_IH_VALUE_CACHE, and the material-history trace switches selected below. Resolve all locations against the post-R5 tree.

Keep this inventory concise. Update the standing guidance and R6 roadmap in place, and store the detailed table in the existing review/evidence structure. Do not copy the full table into AGENTS.md.

## 3. R6a implementation — Material-history traces

The first extraction concerns the existing history diagnostics controlled by:

- ASPECT_STRESS_CYCLE_TRACE;
- ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC.

These currently describe the inputs and candidate stresses used during material-history preparation. Confirm their post-R5 locations and consumers before editing. If they have already been extracted, do not redo that work; report the current boundary and select an equally bounded formatting-only family only if it fits this instruction.

Move their file setup, trace-cell selection handling, CSV headers, and row formatting into narrowly scoped M4 implementation support under source/material_model/phase_field_fault/. For example, history_diagnostics.cc with a source-private header may be appropriate. File names are illustrative; follow existing conventions and avoid introducing a new public material interface.

Prefer a few named functions or one small per-call recorder if it simplifies both streams. Do not introduce a generic logging framework, a global diagnostic registry, or one configuration class shared across all modules.

Keep capture calls at their existing execution points. In particular:

1. Record the old particle stress, sampled gradients/coordinates, computed coefficients, and candidate stress actually used by the update. Do not reconstruct them later from updated particles, FE output, or a repeated material evaluation.
2. Preserve the distinction between candidate data and committed histories. A row produced before validation/publication must not be moved to a post-timestep callback merely to simplify output.
3. Preserve the existing selected-cell filter and continued-source admission conditions. Trace-cell files are diagnostic input; they must not alter particle selection for mechanics.
4. Keep recorder/stream lifetime compatible with the current history-candidate lifetime, including stream closure during exceptions. Do not retain borrowed references after their source data expires.
5. Preserve lazy execution. Disabled diagnostics must not trigger new sampling, cache preparation, allocation of a full trace, material evaluation, or MPI communication. Keep any diagnostic-only argument computations under the appropriate existing guards.
6. Leave history sampling, candidate formation, collective validation, publication, and rollback control flow with their existing owners.

Preserve filenames, rank/timestep suffixes, overwrite/append behavior, header/column order, row selection/order, numeric precision, units, and missing-input behavior. A historical column label is not permission to rename it during this pass; record misleading labels for a separate compatibility decision.

Preserve file-error behavior exactly. If the current path silently produces no rows when a stream cannot open, do not make it abort; if it currently throws, do not suppress the error. Do not move I/O across existing try/catch or collective boundaries in a way that changes failure propagation.

Retain current switch-read timing and parsing. Centralizing all getenv calls at startup is not a neutral refactor if the current code or tests read them later.

The extraction must not add persistent diagnostic state to the Simulator/manager/material object or checkpoints. Keep existing public headers unchanged where practical; any unavoidable private declaration change must be narrow and justified.

## 4. R6b — Coupled-solver and surface diagnostics, later selection

Use the refreshed inventory to select one coherent diagnostic family per task. Candidates include nonlinear bound/line-search reporting, surface stress-sample/weak-moment output, and linear performance summaries. Reuse existing recorders rather than wrapping them in another framework.

Separate captured values and formatting from the numerical operation that produced them. Keep these at their original sites:

- residual/Jacobian evaluations and any audit-only reevaluation;
- cache preparation and exact intermediate values being measured;
- pressure-complement compatibility checks and fresh linear residual checks;
- active-set and line-search decisions;
- MPI reductions and collective failure propagation;
- observer invocation and deliberate test-stop points.

A formatting helper must not call back into assembly to obtain a quantity that is already available. A stored record must identify the state it describes: trial, accepted Newton iterate, converged candidate, or committed timestep.

Retain the existing normal-stress, particle/FE stress, decomposition, and oscillation diagnostics used in current research. Do not remove columns, reduce output frequency, coalesce snapshots, or change cumulative-slip output as part of extraction.

Preserve stream state as well as emitted diagnostic text: precision/scientific flags applied to a shared output stream can affect later output. Do not silently change that behavior when introducing local string streams.

Retain instrumentation scopes, nested timer behavior, work counters, and rank participation. In particular, do not introduce MPI synchronization through a recorder destructor or timer on a rank-local failure path. Measured wall times may change, but the operation boundaries and deterministic counts must remain comparable.

Do not defer a synchronous observer to a later callback. Its operator/preconditioner references may only be valid at the existing call site. Preserve the repaired frozen AMG/GMG probe contract.

## 5. R6c — Benchmark setup boundaries, later selection

Identify which benchmark choices already live in benchmark/test plugins and which remain embedded in production implementation. Separate benchmark-specific initialization/loading from reusable mechanics.

Where an existing callback supplies the same state at the same time, a selected pass may relocate setup or output to a benchmark plugin. First document the old/new callback timing, available data, initialization/restart reattachment, and error behavior.

Do not add a new signal, public accessor, or generic callback layer solely to shorten a file. If the existing extension points cannot preserve the behavior, retain the integration point and record the required future design decision.

Keep generally applicable mechanisms in production, including established boundary-contact geometry, supported completion algorithms, prescribed-rate interfaces, and solver/history contracts. A mechanism's use in BP3 does not by itself make it benchmark-only.

Do not copy constitutive formulas into a benchmark plugin, move convergence/publication there, or make M3 depend on a benchmark.

Environment-variable removal, migration to ParameterHandler, changed defaults, deletion of experiments, altered diagnostics/output frequency, and numerical fixes require a separately selected task with an explicit compatibility decision. R6 classification does not authorize those changes.

## 6. Verification of the selected extraction

Record the accepted post-R5 source/artifact/input baseline and preserve it. Build the candidate and any required plugins separately. Use existing short fixtures with nonempty diagnostic output; a header-only file is insufficient to establish that row capture was exercised.

For each selected observational family, use the following comparison structure:

| Comparison | Purpose |
|---|---|
| Baseline off vs candidate off | No change to ordinary numerical behavior |
| Baseline on vs candidate on | Same physical/history outputs, solver decisions, diagnostic rows and failure behavior |
| Off vs on within each version | Confirm the diagnostic has the claimed numerical neutrality |

For R6a, cover both history streams, using small existing configurations that exercise selected trace cells and continued-source rows. Reuse a fixture that covers both if available. Include the existing small one/two-rank coverage where it checks rank-local files and history/error propagation; do not create a full switch-combination matrix.

Compare deterministic physical/history fields, solver decisions, headers, data rows, and available work counters exactly at matching ranks/stacks. Exclude actual time measurements and path metadata where appropriate; do not discard scientifically meaningful columns. Capture output before subsequent runs can overwrite it.

Exercise relevant existing failure/rollback cases and the selected missing trace-input/file-open behavior in isolated test output directories. Do not change source failure policy to simplify the test. Add a small test only for a concrete coverage gap; no new production getters/counters merely for testing.

For diagnostic-only checks that deliberately abort or alter scratch/cache work, require matched baseline/candidate behavior with the same selector instead of incorrectly demanding a completed off/on trajectory or identical cache-work counts. Distinguish physical-state neutrality from extra diagnostic computation. Any baseline on/off physical difference is a separate finding, not something to normalize away.

Compile new files independently, verify required 2D/3D symbols, and avoid duplicate definitions or reliance on unity/PCH include order. This does not request 3D fault simulations, a compiler matrix, ordinary melt/BFBT qualification, or a production earthquake run.

Rerun the repaired frozen probe only if the selected extraction touches its observer, relevant operator lifetime, or associated output contract. R6a history CSV extraction alone does not require the full R4 campaign.

Do not weaken tolerances, change expected scientific results, or fix unrelated numerical defects to obtain passing tests. Document blocked checks and pre-existing failures honestly.

## 7. Report, review, and R6 completion

Keep documentation classification, source extraction, necessary build/test scaffolding, and any separately approved behavior change distinguishable. Do not mix assertion cleanup, broad renaming, or formatting-only churn into the extraction. Follow the existing local commit policy; no push, merge, or scientific-worktree change is requested.

Provide a main review of roughly 250 words plus a small PASS/FAIL/BLOCKED/NOT RUN table:

1. Which diagnostic family moved and where its capture points remain.
2. What dependencies and state lifetime the helper/recorder has.
3. Baseline/candidate comparison results, including nonempty diagnostic output.
4. Any compatibility or coverage limitations.
5. One proposed next bounded extraction.

Keep long switch tables, commands and logs in supporting evidence. Mark only the completed pass as complete.

The R6 exit conditions are: every live feature switch has a recorded owner and purpose; selected numerical controls/defaults remain available; substantial targeted formatting is outside the physics/solver operations; and the diagnostic evidence used by existing tests and research is preserved. No new logging framework or blanket plugin migration is required.

For this instruction, implement **guideline updates, the refreshed inventory, and R6a material-history traces only**, then stop for review.

