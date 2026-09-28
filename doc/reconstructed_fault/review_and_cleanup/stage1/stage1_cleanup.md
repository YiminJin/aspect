# Stage 1 cleanup before Stage 2

Do not start Stage 2 yet. Make one small cleanup commit for Stage 1 based on the following reviewed decisions. Do not redesign unrelated reconstructed-fault infrastructure.

## 1. Correct the slip-rate lifecycle

The current two-level saved/current lifecycle is insufficient. Distinguish three states:

$$ V_k \quad\text{timestep-committed state}, $$ 
$$ V^{(n)} \quad\text{current accepted Newton iterate}, $$ 
$$ V^{\rm trial} = V^{(n)}+\alpha\delta V \quad\text{line-search candidate}. $$

Required semantics:

- At the beginning of a mechanical nonlinear solve, initialize the current Newton iterate from the timestep-committed value.
- A line-search trial is always constructed from the current Newton iterate; rejected line-search candidates must never accumulate.
- Accepting a line-search step updates the __current Newton iterate only__.
- Committing the timestep is a separate operation performed only after successful nonlinear convergence.
- Checkpoint/restart and ordinary persistent output use the timestep-committed value, not an intermediate Newton iterate or active line-search candidate.
- Rollback of a rejected line-search candidate restores the current Newton iterate.
- A failed/rejected timestep must leave the timestep-committed value unchanged.

Rename internal data and methods if necessary so that `committed`, `current`, and `trial` cannot be confused. Avoid adding a generic lifecycle framework; this should remain a small manager-local implementation.

## 2. Clarify positivity ownership

ReconstructedFaultManager owns \(V\) as a kinematic slip-rate magnitude. Therefore it should enforce the generic invariant

$$ V\ge 0 $$

together with finiteness and correct layout.

It should __not__ know about \(V_{\min}\).

The stronger numerical/constitutive bound

$$ V\ge V_{\min} $$

belongs later to the material model and the coupled nonlinear line search/bound treatment.

Do not introduce V_min into reconstructed-fault geometry/manager code.

## 3. Keep the current explicit interpolation interface

Keep the existing interface based on
```
fault_index, segment_index, xi
```
for Q1 slip-rate interpolation.

Do not introduce a speculative `FaultLocation`, `SlipRateState`, or similar abstraction until the association/cache implementation gives such a type a concrete use.

## 4. Remove speculative public accessors

Review the new Stage-1 public API and remove public accessors that have no concrete current caller.

In particular, do not retain aggregate getters merely because a future solver might use them.

General rule:

    Add a public accessor only when the current implementation has a concrete caller. Prefer the narrowest interface required by that caller.

Serialization/checkpoint code should not require a public getter merely to access internal persistent state.

Do not remove an accessor that is already genuinely needed by the Stage-1 postprocessor, tests, checkpoint implementation, or another current caller.

## 5. Generic reconstructed-fault properties contain committed material state only

Establish the following ownership rule now for future stages:

- the reconstructed-fault generic property pool is persistent storage for __committed physical/material state__;
- temporary Newton quantities, trial cohesive traction, friction coefficients, derivatives, residual coefficients, etc. must not be stored there;
- those temporary constitutive quantities will later be owned as working data by MaterialModel::PhaseFieldRSF.

This is especially important for future

$$ \Theta,\quad T^{\rm coh},\quad I_h. $$

Do not add generic-property trial/rollback machinery in Stage 1.

## 6. Mutable geometry issue: document now, fix before propagation

`get_fault()` currently permits mutable access and `ReconstructedFault::append_*()` can therefore bypass manager invariants and make

$$ \#V\ne\#\text{vertices}. $$

Do __not__ redesign geometry access in this cleanup unless it can be done without breaking existing reconstructed-fault functionality.

Instead:

document this as a known temporary limitation;
propagation/topology-changing operations must eventually go through ReconstructedFaultManager;
this must be resolved before Stage 10 / fault propagation is implemented.

Fixed-geometry Stage 2 is allowed to proceed with the current geometry API.

7. Checkpoint compatibility

Do not implement backward migration for checkpoints created before Stage 1.

Document that old reconstructed-fault checkpoints are not guaranteed to be compatible with the new serialized manager state.

If an inexpensive clear diagnostic can be provided for incompatible archives, that is welcome, but do not build a version-conversion system at this stage.

## 8. Full Simulator restart test

Do not block Stage 2 on a full filesystem checkpoint/restart integration test.

Keep it as a mandatory later verification item before reconstructed-fault restart support is considered complete.

The existing manager archive round-trip test should remain.

## 9. Coding-style rules for this and later stages

Please follow these conventions throughout the reconstructed-fault implementation:

### Public interfaces

- Do not create public getters/setters/accessors speculatively.
- Add an interface when there is a concrete caller.
- Prefer narrow semantic operations over exposing entire internal containers.

### Helper functions

- Extract a helper when it represents a meaningful operation, is reused, or removes substantial nested/duplicated logic.
- Do not create trivial one-use helper functions merely to decompose code mechanically.
- Prefer keeping short, obvious logic near its call site.

### Assertions

Use Assert() / AssertIndexRange() for programmer errors and internal invariants, for example:
```
AssertIndexRange(fault_index, faults.size());
Assert(values.size() == vertices.size(), ExcInternalError());
```
Use `AssertThrow()` only for conditions that must remain protected in optimized builds, such as:

- invalid user/runtime configuration;
- malformed persistent data;
- unsupported fault overlap;
- physically inadmissible runtime states that cannot safely continue;
- singular constitutive quantities;
- unsupported solver modes.

Avoid repeated `AssertThrow()` checks in particle/QP hot loops when the same condition was already validated during construction of a cache or data structure. Validate once at the boundary, then use debug assertions for downstream internal invariants.

The motivation is primarily code clarity and correct error semantics; avoiding unnecessary release-mode checks in hot paths is an additional benefit.

## 10. Keep the cleanup small

This cleanup should not introduce:

- `Theta`;
- cohesive stress;
- `I_h`;
- RSF constitutive code;
- Stokes coupling;
- new Simulator lifecycle callbacks;
- a separate RSF handler;
- Stage-2 integration code.

Preserve the existing successful Stage-1 tests. The current review reports that the full debug unit suite, focused two-rank reconstructed-fault tests, release build, and git diff --check all pass; keep those checks passing after cleanup.

## Required output before Stage 2

After making the cleanup commit, provide only:

1. the commit hash;
2. files changed;
3. the final committed/current/trial \(V\) lifecycle in a short diagram;
4. public interfaces added/removed/renamed;
5. any remaining uncertainty;
6. test results.

Do not begin Stage 2 until this cleanup is reviewed.
