# Reconstructed-fault code-quality and refactoring guidelines

## Purpose and authority

Make the reconstructed-fault implementation easier for ASPECT developers to
read, review, debug, and extend, while preparing coherent upstream contributions.
Prefer straightforward scientific C++ whose structure follows the mathematical
algorithm. Minimize unnecessary dependencies and modifications to existing
ASPECT code; do not optimize for minimum line count.

This document supplies standing code-quality guidance. It does not select an
active task or authorize a whole refactoring campaign.

- `AGENTS.md` defines the repository workflow.
- `doc/reconstructed_fault/current_design.md` and
  `doc/reconstructed_fault/specification.tex` define the authoritative scientific,
  architectural, ownership, and MPI contracts.
- `doc/reconstructed_fault/refactoring/refactoring_plan.md` defines the staged roadmap,
  verification strategy, and review points. It supersedes the former
  `doc/reconstructed_fault/refactoring_plan.md` roadmap (removed by the user);
  historical inventories remain reference material, not active instructions.
- The current source tree defines actual APIs and reusable infrastructure.
- The user's current instruction selects the stage/pass to perform. Status
  reports provide evidence and context, not an instruction to resume unrelated
  scientific experiments.

Read the relevant contracts, callers, and tests before editing. Report material
specification/source conflicts instead of silently changing the implementation
or rewriting the specification to match it. If a referenced plan is missing,
report that fact; do not invent authorization for a stage.

## 1. Scope and reference behavior

Typical areas include:

- `include/aspect/reconstructed_fault/` and `source/reconstructed_fault/`;
- `include/aspect/material_model/phase_field_fault.h` and
  `source/material_model/phase_field_fault.cc`, including subsequently extracted
  implementation files;
- `source/material_model/rheology/fault_friction.cc` when directly relevant;
- `source/simulator/solver.cc` and reconstructed-fault files under
  `include/aspect/simulator/solver/` and `source/simulator/solver/`;
- reconstructed-fault Stokes assemblers and necessary simulator lifecycle,
  checkpoint, access, and signal integration;
- associated tests, timestep control, postprocessors, and benchmark support.

Inspect only the parts required by the selected task and its dependencies.
Resolve names against the current tree; this list is not a mandate to edit all
of these areas.

Use an explicitly recorded source revision and local test baseline as the
behavioral reference. Existing convergence failures and scientific questions
may remain unresolved. Matching the baseline demonstrates refactoring
equivalence, not scientific correctness.

Do not fix a suspected numerical bug within a structural pass. Report it and
handle any selected fix separately. A discovered bug is not automatic permission
to change the reference algorithm, parameters, or expected test results.

## 2. Preserve responsibility boundaries

### Accepted module scope (R6 closed)

R6b is accepted as `00ad5ce1c`, with its immutable qualified executable and
one/two-rank evidence. Follow [R6 §5](refactoring/codex_R6_instructions.md) and
[the R6c boundary assessment](../../benchmarks/reconstructed_fault/refactoring_r6c/README.md).
Keep the [rolling switch inventory](../../benchmarks/reconstructed_fault/refactoring_r6a/switch_inventory.md)
current; classification does not authorize migration/removal or changed defaults.

The selected legacy-completion boundary is retained. Benchmark plugins already
own supported input attachment, loading, initial fields and output scheduling.
Production owns profile geometry/integration, consistent projection, cache/MPI
lifecycle and current legacy admission. A setup callback cannot reproduce the
cache-miss application point or access its private profile/integral data. Passing
the legacy environment path through the explicit setter would change restart
rejection, empty-path handling, read/failure timing and invalidation behavior.
No relocation or new public hook is justified under the current contract.

For future benchmark moves, first compare actual callback arguments/order,
available state, fresh versus restart reattachment, error propagation and MPI
participation. Retain integration if these differ; document the smallest
separate compatibility decision. Do not move generic geometry, prescribed-rate
interfaces, supported completion, constitutive computation or acceptance into a
benchmark merely because BP3 uses them. Preserve all scientific algorithms,
parameters, switch semantics, ownership, checkpoint and observer lifetimes.

R6c is accepted and R6 is closed. This R6c result changes documentation only. Verify source/artifact/local-edit
preservation and relevant existing environment-guard coverage; reuse numerical
qualification of the identical source. No new runtime campaign is implied by
classification. No refactoring pass is currently selected. R7 and legacy compatibility
migration require separate selection.

The separately selected post-R6 BP3 lifecycle correction is a numerical-state
ownership fix, not a new refactoring stage. Native particle backups/checkpoints
must include every reached population/placement RNG; gather per-rank checkpoint
state in an all-rank lifecycle hook, never inside serialization. Benchmark audits
follow particle backup/restore/management, retain exact survivor checks and baseline
initialized births before mechanics. Keep acceptance in the simulator and physical
history computation in the material. Runtime-named native limiter policy is shared
between local and production candidates. See the
[section-1 lifecycle report](../../benchmarks/reconstructed_fault/particle_lifecycle/README.md).
This does not reopen particle-domain/CPDI algorithms, geometry consolidation or R7.

| Module | Owns | Refactoring scope |
|---|---|---|
| M1 — Particle domains / CPDI | Domain construction, integration weights, shape-function information and domain lifecycle | Current implementation frozen |
| M2 — Phase-field method | Generic equations, discretization, solution and material capability interface | Current implementation frozen |
| M3 — Reconstructed-fault infrastructure | Geometry/connectivity, reconstruction, bulk–fault associations, generic projections, property storage/persistence and distinguished V storage/lifecycle | Active after R1, only through selected passes |
| M4 — Fault material physics | Maxwell/cohesive/friction responses, localization/current I_h, material parameters and constitutive history semantics/updates | Active after R1, only through selected passes |
| M5 — Coupled mechanical system | Surface/bulk assembly, MPI reduction, K_V/B/G, restricted solves, condensation/preconditioning, nonlinear iteration, acceptance/rollback | Active after R1, only through selected passes |

These are responsibility boundaries, not new libraries or plugin hierarchies.
Keep existing names and locations during R1. The surface system belongs to M5
although its files are under `reconstructed_fault/`. M3 stores Theta generically;
M4 computes its update; M5 determines when convergence permits publication.
Preserve manager-owned committed/current/trial V, history candidates, collective
validation and terminal writes.

M3 geometry/property storage must not interpret constitutive laws. Reconstruction
and projection adapters may use M1/M2 without coupling every geometry operation
to them. M4 owns the physical meaning of I_h and localization; R2a relocates its
implementation privately within M4. Generic sampling/projection services remain
reusable; do not move material normalization into M2/M3 to shorten a file.
R4 concerns M5 and preserves concrete material responses and canonical
simulator-owned surface/coupling objects; no abstract constitutive base is added.

M1/M2 should eventually be usable without concrete reconstructed-fault physics.
Record current reverse dependencies, including the PhaseFieldFault mature-mode
check in PhaseFieldHandler::evolve_phase_field(), without changing them in R1.
Narrow boundary-interface changes need separate selection; broad prerequisite
cleanup and upstream PR preparation are deferred. Chronology is not a module
boundary: the first reconstruction commit also touched phase/material code.


| Component | Responsibility |
|---|---|
| `ReconstructedFault` | Lightweight application-owned geometry and vertex-major generic runtime properties |
| `ReconstructedFaultManager` | Geometry lifecycle, generic projection infrastructure, distinguished slip-rate lifecycle, property registry, and persistence |
| `PhaseFieldFault` | Common Maxwell/cohesive/localization mechanics, current normalization, material parameters, and constitutive history semantics |
| `Rheology::FaultFriction` | Existing replaceable friction-law calculations |
| Surface system and Stokes coupling | Weak-form assembly, MPI reduction, factors, and B/G coupling actions |
| Condensed system and nonlinear driver | Coupled algebra, linearization lifetime, active-set/line-search orchestration, and solve acceptance |

The manager stores and checkpoints generic constitutive properties without
interpreting their physical meaning. Do not hard-code Theta, cohesive traction,
or a fixed friction-law state into the geometry container. Preserve the
application-owned fault representation and replicated-fault/distributed-bulk
MPI design.

Keep material semantic ownership when moving implementation between files.
Normalization integration belongs to the material capability; moving it into
the generic manager merely to shorten the material source would violate that
boundary. Keep finite-element coupling algebra out of the material interface.

Retain canonical simulator-owned surface and coupling objects. Do not create
duplicate stateful instances during a solve. Preserve the current concrete
constitutive interface unless a separately selected architecture change
justifies generalization.

### M4 implementation layout

Keep the material plugin entry and its sole registration in
`source/material_model/phase_field_fault.cc`. Implementation support belongs in
`source/material_model/phase_field_fault/`; R2a relocates normalization to
`normalization.cc` in that directory. This supersedes the earlier flat
`phase_field_fault_normalization.cc` proposal. It introduces neither a plugin
family nor a generic material-model interface. Later selected M4 moves should
use this same directory convention; do not create future-stage files now.

M4 retains the normalization integrand, material-mixture policy, admissibility
and singularity rules, current-I_h validity decisions, and current/previous-I_h
semantics. R2a moves complete definitions and exclusive file-local helpers,
retaining the existing class, header, data and visibility. It introduces no
helper class. The selected R2b pass extracts only cell-profile geometry
preparation into a private PhaseFieldFault operation in `normalization.cc`.
It retains material ownership, existing cache criteria, collective ordering,
and diagnostics; current phase sampling, material evaluation and quadrature
remain in the integration operation. No generic integration framework or
cross-module extraction was selected in proposal 1. The subsequently selected
[boundary-completion instructions](refactoring/codex_boundary_completion_instructions.md)
authorize proposal 2 in two distinguishable steps: extract the legacy completion
without numerical changes and verify BP3, then implement geometric per-contact
detection and supported automatic paired completion as a behavior extension.
M3 owns contact geometry/association, M4 exterior constitutive continuation,
and M5 mechanical terms. Automatic activation requires verified compatible
prescribed frozen phase data and complete mechanical coupling. M1/M2 and the
normalization value-reuse extraction remained out of scope for proposal 2.
Evolution-compatible phase-field boundary conditions remain a separate design/implementation task.

The separately selected R2b proposal 3 extracts one private PhaseFieldFault
operation that captures existing normalization cache inputs, determines
composition independence and performs the unchanged collective reuse decision.
Keep invalidation, automatic-completion qualification, composition projection,
cache counters and final publication in the caller, in their original order.
Every rank must agree to reuse; no comparison, dependency, equality rule or MPI
operation may change. Report any discovered missing dependency separately;
this extraction does not authorize a correctness fix. Use the completed
proposal-2 implementation as reference and compare existing hit/miss/MPI cases
and short legacy/automatic trajectories and work counters exactly (excluding
time). Stop after this pass for review.

### M5 solver architecture and staged extraction

Retain the user-facing `single Advection, iterated Newton Stokes` scheme and
its existing dispatch to `Simulator::solve_reconstructed_fault_stokes()`.
Keep that dedicated driver; do not merge it into ordinary
`do_one_defect_correction_Stokes_step()` or import its Picard, stabilization,
residual or publication semantics.

The intended responsibilities are three levels:

- Driver: prepare frozen solve data, maintain the outer iteration and private
  accepted bulk state, test convergence, publish and restore on failure.
- Iteration operations: residual evaluation, linearization, bounds/active sets,
  increment recovery and joint bulk/V line search.
- Linear solve: solve the supplied condensed operator using existing Stokes
  preconditioner components.

R4a moves complete definitions without restructuring their bodies. Shared
Stokes operator/Schur definitions may move mechanically to a private internal
header; preserve one implementation and existing interfaces. Compile both
translation units independently, including 2D/3D instantiations. This does not
authorize consolidation of duplicated preconditioner setup.

R4b selects one substantial private extraction per reviewable subpass, starting
with the condensed solve/preconditioner setup, then trial residual evaluation.
Record inputs, outputs, mutations, collectives and lifetime before each move.
A private `do_one_reconstructed_fault_stokes_step()` is optional only if it fits
existing control flow cleanly; focused operations plus a readable driver are
acceptable. Do not force ordinary `DefectCorrectionResiduals` or `use_picard`
onto the fault algorithm. Scratch must be solve-local, with no duplicate
canonical surface/coupling owner, persistent state or all-Simulator context.

For coupled residual evaluation, non-committing means temporary manager trial V
is rolled back and histories are not published. It does not mean a pure query:
assembly flags, current linearization point and assembled RHS retain their
existing side effects. Keep whole-solve restoration in the driver and audit
matrix/RHS restoration in its callers. Preserve trial opening, exact absolute V,
bulk assembly, velocity/pressure norm reductions, surface evaluation and guarded
trial rollback in their existing order, including exception paths.

R4c may consolidate only a demonstrated common operation, such as existing
preconditioner construction. Retain `C = A - B K_V^-1 G`, the condensed RHS
`-R_bulk + B K_V^-1 R_Gamma`, and recovery
`dV = K_V^-1(R_Gamma + G dx)`, with the restricted free-block inverse and
projected residuals. Never substitute A for C or lag coupling to reuse
`solve_stokes()`. Preserve ordinary cheap/expensive and fault total iteration
budgets, fresh-residual restarts, pressure checks, AMG/GMG dispatch, signals and
failure behavior. GMG remains a velocity preconditioner for the assembled
coupled fine operator. No generic integration framework, new persistent owner,
plugin family or abstract constitutive interface is introduced.

All passes retain preparation once per solve, non-committing trial evaluations,
manager committed/current/trial V, final publication/rollback, canonical
linearization/constraint lifetimes, physical versus homogeneous constraints,
pressure scaling and temporary assembly controls. Preserve fixed residual
scales, precision bounds, active-set rebuilding, exact bound-contact V and joint
Armijo acceptance. Algebraic pressure-complement projection and permitted
physical pressure normalization remain distinct. Preserve MPI ordering,
observers, defaults and checkpoint compatibility; extraction does not establish
stronger atomicity. M1/M2 remain frozen and M3/M4 ownership is unchanged.

## 3. Reduce the core-facing interface

### Post-R6 header and build integration

Core declarations and internal template support belong under `include/aspect/`,
mirroring their implementation directories. Use canonical `<aspect/...>` includes
in core, tests and benchmark consumers; do not depend on source-directory headers
or retain forwarding copies. Installed internal headers remain implementation
details: their placement does not change private nested types, namespaces,
ownership, or the supported public interface. Keep definitions in the split
translation units and make each header independently includable without PCH.

The selected post-R6 cleanup audits unity and PCH exclusions separately from
header relocation. Keep any necessary exclusion local and explain the concrete
instantiation/helper collision it prevents. Do not globally disable either build
facility or change scientific code to make a build pass. Verify normal unity/PCH
and independent compilation, installed-header consumers, and matched short
coupled/MPI/restart behavior. The pinned BP3 server package remains historical
and must not be silently regenerated by this cleanup.

For each modification to existing ASPECT code, explain why it is needed and
which feature owns it. Prefer narrow integration points for initialization,
solver dispatch, accepted-state publication, and restart. Move substantial
feature-specific algorithms into feature-specific implementation files.

Use existing plugin, assembler, or signal interfaces where they provide the
required information at the correct lifecycle point. Do not add public getters,
callbacks, or abstract classes simply to move code out of one file. Some core
integration is necessary for the coupled bulk-fault solve.

Review public-interface changes with their callers and tests. Do not duplicate
physical parameters already supplied by phase-field, material, particle, or
solver interfaces. Changes to numerical defaults or supported configurations
are separate behavior changes.

Keep scientific experiments and generally useful independent improvements
distinguishable from the eventual upstream contribution. Preserve research
outputs and checkpoints; excluding them from a PR does not authorize deleting
them.

## 4. Make top-level routines follow the algorithm

Split functions at meaningful numerical or lifecycle boundaries. Avoid deeply
nested lambdas, generic metaprogramming, and many tiny helpers that obscure
control flow. Retain small local lambdas when their purpose is immediately clear.

Useful conceptual structures are:

- **Reconstruction:** determine support; reconstruct/resample each prescribed
  fault; validate and publish geometry; invalidate dependent caches.
- **Particle projection cache:** establish associations; assemble local Q1
  systems; reduce fault-sized data; validate/factorize; record cache versions.
- **Normalization:** prepare surface material information; establish whether
  exact reuse is valid; construct and integrate owned profiles; project and
  publish the resulting field.
- **Initial cohesive state:** distinguish complete persistent state from fresh
  initialization; prepare current normalization; evaluate/project the initial
  quantity; validate and publish.
- **Mechanical history:** sample the accepted solution; compute and project
  candidates in the required order; validate collectively; publish histories.
- **Nonlinear solve:** prepare frozen data; evaluate residuals; linearize and
  determine the active set; solve/recover increments; evaluate trials; accept
  or reject; publish convergence or restore after failure.

These are readability targets, not permission to reorder the implemented
mathematics. Preserve the existing candidate/projection dependencies and
collective ordering.

Move complete methods before restructuring them where practical, and keep
those changes distinguishable. When splitting template implementations, handle
explicit instantiation and registration correctly. Do not introduce new public
classes merely to reduce file length. Group private state only when its members
share a meaningful responsibility or lifetime.

## 5. Protect numerical, lifecycle, and cache contracts

Preserve the existing reconstruction mathematics, projection weights,
quadrature/support, normalization definition, endpoint/completion treatment,
and material-mixture rules. Preserve stable numerical expressions and operation
order during pure extraction, including the stable Maxwell coefficient
evaluation.

Residual and Jacobian evaluations must remain non-committing. Preserve:

- committed/current/trial slip-rate semantics and exact accepted absolute
  values at lower-bound contact;
- the distinction between accepted Newton trials and committed timestep state;
- timestep-zero initialization and later-time history evolution;
- particle stress remaining unchanged throughout nonlinear trials;
- collective candidate validation before terminal persistent writes;
- restoration after failed solves or rejected timesteps as specified;
- the canonical A/B/G/K_V linearization lifetime, free-block inverse, signs,
  constraints, pressure scaling, and existing residual/acceptance criteria.

Keep geometry-only caches separate from cached field values. A valid geometric
lookup does not establish validity of cached I_h or constitutive values.
Preserve invalidation on relevant geometry, mesh, particle, field, or parameter
changes and the existing collective agreement on reuse.

Keep property names, component order, and serialization compatible during
structural work. Restart restores committed state and rebuilds transient data
according to the existing contract.

Do not introduce new interpolation, filtering, constitutive regularization,
propagation, line-search, timestep, or preconditioner algorithms in a structural
pass. Do not change failure handling through new exception-catching or
destructor behavior without reviewing its numerical and MPI consequences.

## 6. Use assertions according to their role

Classify checks before removing or changing them. Assertion cleanup is a
separate pass from algorithm extraction; its value must be demonstrated in the
current code rather than assumed from an older report.

Keep `AssertThrow` for invalid user input, unsupported configurations,
corrupt/incompatible checkpoints, externally supplied API data, and physical or
numerical failures that require safe termination in Release builds. Retain
singularity, admissibility, and solver-consistency checks required by the
algorithm. Do not impose positive-definiteness on a surface solve that supports
indefinite or nonsymmetric blocks.

Use `Assert`, `AssertDimension`, and `AssertIndexRange` for internal programming
invariants whose enforcement is appropriately Debug-only. Do not automatically
downgrade every lifecycle or size check: some protect public operations or
prevent invalid persistent writes.

Remove repeated checks only after demonstrating that an upstream guarantee
remains valid at the downstream use. Consider mutation, arithmetic range,
interpolation/projection weights, and rounding. Finite inputs alone do not
guarantee finite outputs; overflow, cancellation, and ill-conditioning can still
occur. Projection can also violate bounds that held for the input samples.

Preserve collective error propagation. Removing a check or changing where an
exception is thrown must not leave other ranks entering a collective that the
failing rank skips. Avoid repeated MPI boilerplate only when a small shared
operation has genuinely identical participation and error semantics.

## 7. Keep geometry/property and test interfaces small

Encapsulate uninitialized-property sentinel handling in the geometry/property
abstraction. Material code should ask whether a value is initialized rather
than inspect an IEEE-754 bit representation. Preserve restart and diagnostic
semantics.

Keep test-access implementations in testing-only code where feasible. Retain
only the narrow friend/declarations needed by the production interface; do not
expose implementation details publicly just for tests.

These are continuing constraints. Inspect current source before treating them
as unfinished tasks; both areas have already received cleanup in the reviewed
development history.

Extract shared code only when the result is easier to understand than the
duplication. Similar formulas or loops may implement different quadrature,
mixtures, ownership, or support. Verify their numerical meaning before merging
them. Prefer named structs over opaque tuples when names explain the data.

## 8. Separate diagnostics from numerical choices

Inventory environment switches and options by purpose:

- observational timing, tracing, and formatting;
- numerical implementation selection, reference paths, or preconditioning;
- benchmark initialization, loading, and specialized setup.

Preserve names, defaults, and behavior during initial extraction. Removing a
numerical option or migrating it to a parameter is a separately selected task.
Do not assume all environment switches are disposable debug code. A diagnostic
may run extra residual/material evaluations, prepare caches, invoke collectives,
mutate scratch state or intentionally stop. Record those effects separately from
formatting; only require off/on physical neutrality when the behavior claims it.
For presence tests, empty and "0" remain enabled; preserve repeated reads rather
than centralizing configuration at startup. Numerical filtering, completion
inputs, loading and cache controls stay numerical/setup choices.

Extract file handling and formatting where useful. Pass values actually used
by the computation to the diagnostic recorder; avoid recomputing them in a way
that loses the original history, sampling location, or trial state. Preserve
necessary research diagnostics until their replacements provide equivalent
evidence. Check whether diagnostics are consumed by tests before changing text.

## 9. Style and documentation

Follow ASPECT's repository formatting tools and documented conventions. Keep
formatting-only edits distinguishable from structural changes and avoid
reformatting unrelated files. Expand compressed multi-statement lines and
conditionals that obstruct review.

Use descriptive names for cross-module interfaces. Mathematical local names
such as `phi`, `H`, `I_h`, `xi`, and `V` are appropriate when their meaning is
clear. Avoid both opaque abbreviations and unnecessarily long repeated prefixes.

Comments should explain numerical reasons, state transitions, collective
participation, or non-obvious invariants. Remove comments that only restate the
next statement. Give complicated numerical algorithms a short introductory
explanation rather than commenting every line.

Keep mathematical contracts in the authoritative documents, stage details in
the refactoring plan, and progress/evidence in the status and review reports.
Do not duplicate the entire plan here or change scientific documentation merely
to rationalize an unintended implementation change.

## 10. Verification and review

For the selected pass:

1. Confirm branch/worktree, local edits, baseline revision, scope, relevant
   callers, and existing tests. Preserve unrelated work.
2. Establish the needed reference evidence before changing the affected path.
   Distinguish newly executed results from historical reports and missing local
   fixtures. Do not describe unresolved behavior as already qualified.
3. Implement the agreed change in a coherent, independently reviewable diff.
   Ordinary choices within that scope do not require repeated confirmation.
4. Build and run the stage-relevant tests and short comparisons. Prefer existing
   checks; add a test only for a concrete uncovered contract.
5. Report source revisions, changed responsibilities/interfaces, actual
   pass/fail/skip results, numerical differences, and remaining limitations.

Use the agreed local resource budget. A full production run is not required for
every extraction. Apply restart/rollback, MPI, pressure-mode, material-mode, and
ordinary non-fault checks where the touched responsibilities require them.

Compare baseline and candidate under the same stack and rank count before
cross-rank comparisons. Preserve exact structural facts and inspect numerical
differences using justified per-quantity scales. Do not require bitwise equality
across different MPI decompositions or loosen tolerances to accommodate a
changed result. Investigate altered solver decisions even when final fields
are close.

Record pre-existing failures separately. If a failure prevents verification of
the affected path, explain what evidence is missing and propose a separate fix
or suitable independent check. Do not silently repair the baseline or update
expected outputs to declare success.

Review the final diff for changes to core integration, MPI collectives,
projection and quadrature, normalization caches, trial/committed state,
publication order, configuration, property layout, and serialization.

At the end of the selected pass, provide the review packet and one proposed next
task. Follow the user's agreed stage boundary rather than executing the whole
roadmap. If a pass has not yet been selected or its boundaries are unresolved,
produce the concrete proposal first; an already selected implementation pass
does not require restarting the planning process.

## BP3 runtime-cleanup section 2 review boundary

The separately selected section 2 replaces compiled BP3 geometry with one
validated read-only description from the native prescribed-fault reader and Box
bounds. Each consumer configures it during parameter parsing; no postprocessor
execution order supplies geometry. Preserve native anchors/resampling and the
horizontal material extension. Table generation/removal remains section 3.

The anticipated stationary-profile peak roundoff fix is separate (`8df2bc5b5`):
preserve the interpolation expression and its mathematical endpoint bounds;
never lower prescribed peak or loosen H/solver assertions. The new 45-degree
reversed-order comparison exposes an additional endpoint-plane association gap
in unchanged core projection/automatic-continuation code. Record its failed
qualification separately; do not canonicalize input ordering, change resampling,
or broaden a numerical fix inside this geometry pass. See
[the section-2 report](../../benchmarks/reconstructed_fault/bp3_geometry/README.md).


The user subsequently selected the endpoint-plane correction as a separate
numerical task. Use a coordinate/terminal-tangent floating-point error bound,
retain every already valid association, and keep support/overlap checks and
native resampling unchanged. Its implementation and evidence are recorded in
[the endpoint review](../../benchmarks/reconstructed_fault/bp3_geometry/endpoint/README.md).
Full QP admission agreement and the original 60-degree/lifecycle successes are
verified, but one of the seven original field comparisons still fails (also
reproduced on one rank). Do not treat the endpoint correction as fully qualified,
change comparison tolerances, or proceed to section 3. Further timestep-zero
order-sensitivity investigation is a separately selected bounded task.


The user selected the subsequent filter-derivative correction. At shared-vertex
planes use the mean of the two one-sided stiffness integrands; at endpoint
planes use the interior/zero-exterior mean. Average gradient outer products,
not gradients, to retain the tridiagonal operator. Use only a local geometric
floating-point bound; do not reassign valid source associations or canonicalize
native input. The [qualified correction](../../benchmarks/reconstructed_fault/bp3_geometry/filter_derivative/README.md)
passes all 927 existing qualification checks, including the seven prior failures.
Keep this numerical correction distinct from the section-2 geometry work and
endpoint-admission correction. Section 3 remains a separately selected task.


## BP3 runtime-cleanup section 3 review boundary

Accepted section-2/filter work is committed as `bc2b5eb64`, `82a1daaa8`, and
`48f23492d`. Section 3 replaces only the maintained plugin's three table inputs;
retain historical data and version-pinned binaries. Use the live material/profile
factory, preserve discrete Ih, and keep automatic completion in core. Prepare
geometry-only meshes through native initial-global tagging before particles
exist; initial adaptive passes and later AMR are unsupported in this BP3 path.
Do not bypass history audits to accommodate repeated startup initialization.

The [section-3 report](../../benchmarks/reconstructed_fault/bp3_runtime/README.md)
records passing uniform-mesh/lifecycle comparisons and a baseline-reproduced
graded-boundary admission failure. Do not claim resolved-mesh readiness, change
the retained grading rule, or narrow core support without a separately selected
correction. Section 4 production packaging and longer qualification remain
unselected.


## BP3 runtime-cleanup section 4 review boundary

Section 3 is committed as `e2ff248fb`. Section 4 selects maintained packaging,
resolved configuration/identity and observational output cleanup only. Preserve
all historical inputs/evidence and the single plugin source package. Explicitly
label unconfirmed production settings; parser validation is not a server gate.
Keep output schedules and physical/RNG checkpoint ownership unchanged. Diagnostic
population counts refresh at native backup and must not be mistaken for new
physical history or gross insertion/removal event counters. A changed output
selection must leave accepted physical results and checkpoint recovery intact.

The existing graded-boundary completion failure is outside this selection;
retain its admission, mesh policy and evidence. Do not proceed to Section 5 or
server execution without the separately selected prerequisite correction.


## BP3 runtime-cleanup section 5 local comparison boundary

Section 4 is accepted as `1678949c2`. The user's subsequent selection authorizes
Section 5 and the bounded velocity/mesh/timestep extension, superseding the
preceding stop-before-5 workflow boundary. It does not select a core completion
correction. The two diagnostic policies share a conservative fine boundary
buffer satisfying current completion admission; the maintained unbuffered
production policy and its documented failure remain unchanged. Recover B's
exterior slope/cap from `tests/bp3_length_scale_mesh.cc`; do not call an invented
coarsening policy historical. Compile the isolated functional fixture from the
maintained sources, keeping its rate/mesh selectors absent from the default build.

Follow [the predeclared local plan](../../benchmarks/reconstructed_fault/bp3_local_tests/PLAN.md).
Retain all positivity/history/solver gates, native ownership and checkpoint
format. Use the existing bounded pending-restart-step hook, never rewrite old dt
or accepted history. Exact-position transport shadow components are separate
from stored Maxwell history and use native LLS/continuous-Q2 publication.
Unexpected ID/lifecycle behavior is evidence to report separately, not authority
to change production insertion, audit or communication semantics. No production
refinement replacement, server run, Section 6 deployment or R7 is selected.


## Selected post-Section-5 birth identity and production buffer correction

The user selected two bounded correctness/qualification tasks after the local
comparison: identify births despite retired-ID reuse, and retain/check the
boundary strip on the actual 60-degree production mesh. This supersedes the
previous stop-before-production-policy-change boundary only for that buffer.
Do not change native ID allocation, particle properties, RNG, MPI ownership,
core completion admission, physical settings or solver gates. Preserve existing
post-management audit publication and native retry/checkpoint handling. Record
newborn events after initialized insertion; ID thresholds are not lifetime
identities. Additional audit bytes use existing transfers, with no new collective.

Keep the incoming Section-5 changes distinguishable (saved tracked patch plus
preserved original package/evidence). Preserve accepted artifacts; rebuild plugins
against the corrected core. Record full production boundary/material qualification
separately from a mechanics solve and resource-limited attempts. The historical
unbuffered 45-degree failure does not block qualified 60-degree completion. Stop
for review before server execution, another cleanup section, or R7.
