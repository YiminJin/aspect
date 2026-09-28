We have reached a code-quality/refactoring stage for the reconstructed-fault implementation.

Please review the current implementations of:

- `include/aspect/reconstructed_fault.h`
- `source/reconstructed_fault.cc`
- `include/aspect/material_model/phase_field_fault.h`
- `source/material_model/phase_field_fault.cc`

The algorithms are currently working and their mathematical/numerical behavior should be treated as the reference behavior. This task is primarily a **structural and readability refactor**, not a redesign of the fault model.

## Primary goal

Make the code substantially easier for a human ASPECT developer to read, review, debug, and extend.

Do not optimize for minimum line count. Do not introduce clever generic abstractions merely to reduce duplication. Prefer straightforward scientific C++ in which the source-code structure follows the mathematical algorithm.

Before editing anything, inspect the code and produce a concrete refactoring plan.

## Architectural constraints

Preserve these existing design decisions:

1. `ReconstructedFault` is a lightweight application-owned fault geometry with vertex-major generic runtime properties.
2. Do not hard-code constitutive quantities such as theta, cohesive traction, etc. into `ReconstructedFault`.
3. The reconstructed fault remains replicated across MPI ranks.
4. Particle-to-fault projection remains the existing Q1 weighted least-squares projection using particle-domain volume.
5. Slip rate remains a distinguished manager-owned nonlinear variable with committed/current/trial state and explicit begin/accept/rollback/commit semantics.
6. Preserve the existing reconstruction mathematics, normalization-integral definition, MPI ownership strategy, restart behavior, and numerical results unless an actual bug is found.
7. Do not redesign the phase-field/RSF formulation in this task.

## 1. Reduce unnecessary defensive checks

The current implementation performs excessive finiteness validation. Do not check `std::isfinite()` after every computation.

Use the following policy.

### Keep `AssertThrow` for:
- user parameters;
- parsed prescribed-fault input;
- corrupted/incompatible checkpoint data;
- public API inputs that can legitimately be invalid;
- genuine physical/numerical admissibility conditions whose violation can occur during a simulation and for which continuing would be unsafe;
- unsupported configurations.

### Use debug `Assert` / `AssertDimension` / `AssertIndexRange` for:
- internal programming invariants;
- relationships between container sizes;
- state-machine invariants;
- conditions that should already be guaranteed by an upstream internal routine.

### Remove checks when:
- the same quantity was already validated at the appropriate boundary;
- finiteness follows directly from already validated finite inputs and a denominator whose admissibility was already checked;
- a result is checked again immediately after a helper that already guarantees that result;
- an interpolated value is rechecked even though the stored values were already validated when they were committed.

Do not remove mathematically meaningful safeguards such as positivity/non-singularity tests required by a linear solve.

In particular, do not blindly replace every `AssertThrow` with `Assert`; classify each check by its role.

## 2. Make high-level functions follow the algorithm

Several functions are currently too large. Split them only at **conceptual algorithm boundaries**, not into many tiny helpers.

For example, aim for structures conceptually like:

### Initial fault reconstruction

`reconstruct_initial_faults()`

should read approximately as:

1. determine reconstruction support/radii;
2. reconstruct each prescribed fault;
3. store diagnostics and commit geometry;
4. invalidate dependent caches.

Move the detailed reconstruction of a single fault into a clearly named helper if that makes this flow visible.

### Particle projection cache

`rebuild_particle_projection_cache()`

should visibly contain:

1. compute/cache particle-to-fault coordinates;
2. assemble local Q1 mass/projection systems;
3. MPI-reduce the small fault systems;
4. validate support and factorize them;
5. record cache versions.

### Initial cohesive state

`initialize_cohesive_state_from_initial_fields()`

should visibly contain:

1. determine whether persistent history already exists;
2. compute current `I_h`;
3. evaluate the initial cohesive quantity on relevant particles;
4. project it to fault vertices;
5. commit the persistent state.

### Normalization integral

`compute_normalization_integrals()`

should visibly contain:

1. prepare fault-surface material information;
2. build the owned normal profiles;
3. integrate `I_h` along those profiles;
4. project quadrature/profile values consistently to fault vertices.

The adaptive one-dimensional profile integration is inherently complicated. Do not hide that complexity using generic metaprogramming or deeply nested lambdas. Isolate it as one coherent numerical algorithm with meaningful state names and comments explaining the algorithm.

## 3. Improve responsibility boundaries without a broad redesign

`ReconstructedFaultManager` currently contains several responsibilities.

Organize the implementation clearly into:

- prescribed-fault/reconstruction lifecycle;
- generic property registry;
- particle/fault projection infrastructure;
- distinguished slip-rate nonlinear state;
- checkpoint/cache maintenance.

Do not create several new public classes merely for architectural purity.

Private helper structs are acceptable when they group state that always changes together, for example projection-cache state or slip-rate nonlinear state, but introduce them only when they make the code easier to understand.

Do not change the public API without a clear reason.

## 4. Preserve the simple `ReconstructedFault` abstraction

`ReconstructedFault` itself should remain simple.

Do not turn it into a constitutive-model object.

Keep the vertex-major generic property layout.

If code outside `ReconstructedFault` needs to know whether a generic property still contains the container's uninitialized sentinel, encapsulate that knowledge inside the reconstructed-fault/property abstraction instead of making material models inspect the IEEE-754 bit pattern of `numbers::signaling_nan<double>()`.

In particular, remove the current situation in which `phase_field_fault.cc` has to know the binary representation of the generic-property sentinel.

Do this without changing restart semantics or silently replacing signaling-NaN debugging with unchecked values.

## 5. Simplify the phase-field material-model header

The production header should primarily describe the material model, not contain a large test implementation.

Keep only the minimum friend/test declaration required by the tests.

If possible without compromising template compilation, move the implementation of `PhaseFieldFaultTestAccess` to the relevant test code or another appropriate testing-only location.

Do not expose implementation details publicly merely to make them testable.

## 6. Reduce accidental complexity

Look specifically for:

- duplicated particle-projection assembly/solve logic;
- repeated MPI error-propagation boilerplate;
- repeated manual lookup of property positions;
- four-element tuples whose members would be clearer as a named struct;
- large nested lambdas that obscure the numerical algorithm;
- repeated computation or validation of quantities already guaranteed upstream;
- very long variable names where a shorter mathematical/local name is unambiguous.

However, do not introduce a generic framework to eliminate two small pieces of duplicated code. Extract shared code only when the resulting abstraction is simpler than the duplication.

Prefer names corresponding to the mathematics where appropriate: `phi`, `H`, `I_h`, `xi`, `segment`, `vertex`, `normal`, etc., once their meaning is clear from scope.

## 7. Comments

Comments should explain:

- the mathematical/numerical reason for a step;
- MPI ownership/collective requirements that are not obvious;
- state transitions;
- non-obvious invariants.

Remove comments that merely restate the next line of C++.

For long numerical routines, add a short introductory comment describing the algorithm before the implementation rather than commenting every individual statement.

## 8. Do not mix refactoring with new physics

Do not implement additional RSF physics, crack propagation, new properties, new projection schemes, or optimizations during this task.

If you notice a possible numerical or architectural bug, report it separately before changing the behavior.

## 9. Testing/refactoring procedure

Before changing code:

1. identify the relevant existing tests;
2. establish the current test baseline;
3. present the proposed refactoring boundaries.

Then implement the cleanup in small coherent passes.

After each pass, build and run the relevant tests.

At the end, review the diff specifically for accidental behavior changes in:

- MPI collectives and ownership;
- fault reconstruction;
- projection weights;
- normalization integration;
- slip-rate current/trial/committed semantics;
- serialization/restart;
- property layout.

The final result should make it possible for a developer to understand the major algorithmic steps by reading the top-level functions without having to follow every implementation detail.

First produce the review/refactoring plan only. Do not edit files until the plan has been reviewed.