# ASPECT: The Advanced Solver for Planetary Evolution, Convection, and Tectonics.

A parallel, extensible finite element code.

## Overview
- ASPECT is a C++ CMake project
- unit tests are in /unit_tests/ and are implemented using the Catch2 framework; they are run via ``aspect --test`` (also wired as a ctest target)
- tests are in /tests/, can be run using ``ctest``. By default only a small subset runs; enable the full suite with ``make setup_tests``. Also see /doc/sphinx/user/extending/testing/running-tests.md
- use ``./aspect --validate parameter-file.prm`` to parse a parameter file and exit, reporting any syntax or parameter errors without running the simulation
- instructions for code formatting, use ``./contrib/utilities/indent`` at /CONTRIBUTING.md

## Layout
- ``source/`` and ``include/aspect/`` — paired implementation and headers
- ``tests/`` — integration / regression tests
- ``unit_tests/`` — unit tests
- ``cookbooks/`` — example programs / tutorials
- ``benchmarks/`` — similar, but for benchmarks
- ``doc/sphinx/`` — user documentation
- ``contrib/`` — vendored third-party code; do not casually edit

## Plugins
New physics is usually added as a plugin under matching directories in ``source/`` and ``include/aspect/``. See ``doc/sphinx/user/extending/write-a-plugin.md`` and ``doc/sphinx/user/extending/idea-of-plugins.md``.


<!-- phase-field-fault-refactor:begin -->
## Phase-field fault development on ASPECT 3.1.0

These project instructions apply to the research transplant. Read the [migration plan](refactoring/phase-field-fault/PLAN.md) and the [working record](refactoring/phase-field-fault/STATUS.md) on arrival. Use the [reference and verification guide](refactoring/phase-field-fault/VERIFICATION.md) for the current increment. A new session should be able to resume from these files and the actual Git diff.

### What counts as a good change

1. Preserve the selected equations, discretization, and accepted history. Compare against a reproducible small case and state what it establishes.
2. Make the change readable to an ASPECT maintainer. Ownership, public interfaces, comments and consistency with neighboring code are acceptance criteria on every increment.
3. Measure resource use where the change affects it. Account for candidate copies and MPI replication.
4. Carry an experimental option only when a paper example or a discriminating test needs it.
5. Name public operations so a caller can tell what entity they change and what data they use. Define project-specific physical and geometric terms in the owning header before using them in method contracts.

The implementation belongs in this worktree. The research donor is `../aspect-pf-rsf-particle-free/`, relative to this root. Treat its source, dirty changes, executables, build trees and checkpoints as evidence. Direct new builds, outputs, patches and replays to separate destinations. Linked worktrees can share Git refs. Preserve the donor and any incoming work. Installed code, regression tests and normal runs must work without that sibling.

### How to make a change

- Start at the ordinary Simulator call site and identify the owner of each persistent quantity. Let the material model evaluate a local response. Let the Simulator coordinate solves, retries and acceptance. Give geometry, FE operators and physical history the responsibilities described in the plan.
- Read the nearest release implementation and its callers before adding an interface. Reuse compatible variable declarations, assemblers, FE data, constraints, solver and output infrastructure. For a custom operation, name the actual upstream interface and the mathematical or lifetime mismatch. Keep common-core changes small and independently reviewable.
- Keep the accepted parent immutable throughout an attempt. Re-evaluation, Jacobian assembly, line search and output may execute more than once. Rebuild trials from the same parent and publish the converged fields together once the step is accepted. A retry with a new duration starts again from that parent.
- Write code in the style of its neighbors. Read the checkout’s contribution rules, formatter configuration and PR template. Use Doxygen for public interfaces and non-obvious persistent state. Explain units, stress frame, stage factors, measures and collective calls at the point where a wrong interpretation changes the result. Keep related residual and derivative calculations easy to read together.
- Prefer a small public interface with visible data flow. A short top-level call that hides an entire second timestep driver fails review. An optional handler earns its place by owning a coherent resource or operation. Keep experimental selectors, environment hooks and run-specific names out of production interfaces.
- Before adding or retaining a file, class, or forwarding header, identify the responsibility it owns and its production callers. Keep a small operation with the existing owner when the separate file only forwards, duplicates a formula, or makes a reader follow an extra include. Keep a separate unit when it exposes an independently used contract, registers a plugin, isolates a real dependency, or contains a coherent operator. Review these choices across the final branch diff, including installed callers. File length alone does not decide placement.
- Preserve the ordinary ASPECT path when the feature is off. Keep fixes to the scientific method distinguishable from code movement and shared API changes. Keep worktree planning material separate from patches prepared for upstream review.
- For a change visible to an ASPECT user, begin with the smallest realistic .prm example and name the nearest existing ASPECT example to follow. Show what the scientist edits to change one physical assumption. Keep numerical implementation choices internal wherever the physics determines them. Put ordered spatial data in files with documented units and identity. Before handoff, read the example as a new user and trace each setting to the calculation it actually controls.

### Editing established ASPECT core

- Read the release implementation, its callers, and the closest existing extension point before changing shared simulator, parameter, assembly, or solver code. Prefer a small opt-in addition or dispatch call. Keep the numerical implementation with its owner and preserve existing defaults and the meaning of old parameter files.
- Retain existing upstream lines, comments, interfaces, and control flow wherever practical. Revise or remove them only for a concrete correctness or integration need. Avoid incidental renaming, reordering, formatting, and cleanup. A targeted edit or shared extraction is appropriate when an additive workaround would duplicate a solver or introduce a second owner of state.
- Review the final diff as if the author of each affected line were reviewing the patch. For each substantive core change, identify the release file/symbol, why the existing extension point was insufficient, the existing callers affected, and the regression that exercises their previous behavior. Distinguish changes to upstream ASPECT from refactoring code introduced on this branch. Keep an independently useful core change separately reviewable when practical.

### Phase-field fault architecture

- Phase-field laws, reference profiles and field operators must be usable without a reconstructed fault. Fault reconstruction consumes the bulk phase. Trace ownership, section membership, normal integrals and slip transfer belong on the fault side. The ASPECT storage mechanism for a field does not decide who owns its physics.
- Document a constitutive choice where it is selected. Qualify crack geometry, energetic degradation and their coupled calibration separately. At handoff name the supported combinations and a test that would fail for an unsupported substitution. An observer comparing two exports must be paired with an independent check of the operation the paper example relies on.
- The fault owns its Maxwell update, stress history, stress contribution to Stokes and physical stress diagnostics. Fault cases run with ASPECT's original elasticity disabled. Use its generic Stokes assembly, material output base interfaces, FE infrastructure and compositional storage where their contracts match. All fault-specific physics remains independent of the original elasticity rheology, its stress conventions and its two output types.
- Give each public production type a durable physical or numerical responsibility. Match declaration and definition paths under `include/aspect/` and `source/`. At handoff, name its normal caller and the nearest release analogue. Keep global FE/MPI operations and accepted-state publication out of local constitutive evaluation.
- Trace each reused field or output to its consumer. Check units, stress frame, time level and weak form on an input where a wrong match changes the answer. Resolve coupled convergence before deciding to accept. Review the release-to-worktree diff and answer “Which production class or core edit exists only because of this test specimen?” Give the caller and the physical reason for each disputed edit.
- Start fresh-run controls from initial conditions in fresh directories. The prototype stationary checkpoints are retired. Exercise the qualified mature interior and native growing restart formats as documented in the verification guide. In-memory retry always rebuilds from the same accepted parent. Preserve the donor as read-only scientific evidence.

### Working agreement

The midpoint audit is closed and its discrete update is frozen. Read the active assignment at the top of the working record. Finish that increment and present the code for the author's architecture review. Make routine implementation decisions locally. Record a missing scientific reference, ambiguous model choice, or failed comparison against the exact path it affects. Proceed with independent work. Preserve failed evidence and explain a difference before changing a reference value.

At handoff, update the working record with the exact commands, results and next action. Show the normal call path, the equation and state owner, and the smallest representative public header and implementation a human should read. Keep the active design in one guide as code lands. Use the project's scoped tests and the checkout’s formatter. Default to four or fewer total local workers on the author's laptop and record the rank/thread layout for comparisons. Publishing changes or submitting cluster jobs requires a separate request.

Include a reproducible figure from the computed fields that answers the experiment’s main question, with units, a caption, and copyable paths to the figure and plotting command.

When referring me to a local file, print its full path relative to the worktree root in a fenced text code block, one path per line. Use the actual path in the relevant worktree so I can distinguish the refactor from the donor. Put line numbers and explanations outside the block so I can copy the path directly into my editor. Omit clickable Markdown links for local files. Apply this rule to both progress updates and final replies.

### Architecture guidance

When changing a production interface or introducing a scientific concept, choose its owner from the mathematics, the ASPECT call path, and the accepted project roadmap. The first benchmark that exercises a concept is evidence about how to test it; it does not determine who owns it.

Before settling the interface, try one concrete change already named in the roadmap. Which files would a maintainer have to edit? Which modules would acquire a new dependency? If changing a phase law would require editing fault geometry, or removing the fault would remove a generally useful phase operation, revise the ownership in the current change.

Implement the behavior assigned to this increment. A small function or concrete type in the right module is sufficient until multiple behaviors justify a common interface. If the task suggests a design that conflicts with these principles, point out the conflict and use the simpler durable design when it preserves the assigned physics and scope.
<!-- phase-field-fault-refactor:end -->
