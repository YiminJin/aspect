# Mature moving-fault stress: developer guide

This guide follows one **direct physical-stress** step from its equation to its
implementation. The supported example is a two-dimensional, fixed Cartesian
mesh with full three-dimensional plane-strain stress, constant shear modulus,
constant or prescribed spatial viscosity, material fault motion and no growth.
Direct storage is opt-in; separated storage remains the default.

The method has four stages: transport accepted state; stage geometry/profile;
rotate stress using accepted velocity gradients; solve the coupled endpoint
mechanics with a projected-stress correction. The ordinary timestep controller
then accepts the joint candidate or restores its parent. There is no separate
mature timestep driver.

## Two reading routes

**Understand the algorithm:** [model and variables](mature_moving_fault/model.md)
→ [discretization](mature_moving_fault/discretization.md)
→ [geometry and localization](mature_moving_fault/geometry.md)
→ [mechanical iterations](mature_moving_fault/mechanics.md)
→ [one complete step and ownership](mature_moving_fault/lifecycle.md).

**Find code to modify:** start with the [runnable examples and code map](mature_moving_fault/configuration.md),
then the relevant topic above. [Retained alternatives](mature_moving_fault/alternatives.md)
explain what a direct-mode reader can initially skip.

## Revision and evidence

Current implementation: the consolidation on reviewed baseline
`d7e4adbe99126f8c9faf9f35c657e287f2c29cea`, 4 November 2026.
[Consolidation report](STRESS_CONSOLIDATION_DOCUMENTATION_REPORT.md) records exact
source/binary identities and bounded verification. The scientific contracts are
[design](../current_design.md), [specification](../specification.tex) and
[transported extension](TRANSPORTED_OBJECTIVE_SPEC.md). Historical reports remain
evidence at their own revisions; these linked pages describe current code.

Local equivalence does not establish production readiness. The
[finite-end comparison](FROZEN_FINITE_END_REPORT.md) leaves tensor approximation
accuracy open; coupled energy/flux accuracy, moving AMR/birth and independent
cohesive tip/energy failures remain unresolved. Production BP3 and cohesive
physical histories are distinct. No server campaign is part of this guide.
