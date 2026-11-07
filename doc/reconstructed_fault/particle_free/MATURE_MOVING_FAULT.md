# Mature moving-fault stress: developer guide

This guide follows one **direct physical-stress** step from its equation to its
implementation. The supported example is a two-dimensional, fixed Cartesian
mesh with full three-dimensional plane-strain stress, constant shear modulus,
constant or prescribed spatial viscosity, material fault motion and no growth.
Direct storage is opt-in; separated storage remains the default. The maintained
examples explicitly select `Constitutive spin = bulk slip subtracted`; the
parameter default remains `total material`. See the [correction report](BULK_SPIN_CORRECTION_REPORT.md)
for the selected constitutive change and bounded evidence.

The method has four stages: transport accepted state; stage geometry/profile;
rotate stress using coherent accepted bulk spin (or the retained total-spin
control); solve the coupled endpoint
mechanics with a projected-stress correction. The ordinary timestep controller
then accepts the joint candidate or restores its parent. There is no separate
mature timestep driver.

Phase is independently selectable: historical continuous Q1 remains the default;
the new native DGQ2 path uses material upwind transport with SSPRK3 and whole-cell
bound scaling. Stress remains ordinary DGQ2. See the [phase operator contract](DGQ2_PHASE_FIELD_DESIGN.md)
and [current qualification report](DGQ2_PHASE_FIELD_REPORT.md). The separate fixed-fault
cohesive endpoint can use DGQ2 SIPG constrained growth; mature moving growth remains
disabled. No post-growth transport limiter is applied.

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

The subsequent [frozen split-Q2 experiment](FROZEN_SPLIT_Q2_REPORT.md) supports
an endpoint-aware coupled prototype at modest history cost. It is benchmark-only;
the native algorithm described by this guide still uses uncut DGQ2.

The [bounded sheared moving-fault comparison](SHEARED_MOVING_FAULT_REPORT.md)
measures about .094 rad material spin with unchanged ordinary DGQ2/SSPRK3.
Near-fault temporal convergence does not qualify the whole domain: the retained
boundary/corner treatment gives growing global stress differences and non-monotone
energy remainders. See that report before using the sheared fixture as a reference.

The subsequent [corner isolation](CORNER_ISOLATION_REPORT.md) verifies exact
prescribed inflow and removes severe coupled corner amplification with compatible
top traction. The latter changes physical loading and slip; it is a benchmark
control, not a replacement production boundary model. Held-trace inflow retains
an independently measured boundary bias.

The [frozen non-affine replay](CORNER_REPLAY_REPORT.md) subsequently preserves
an exact manufactured tensor to roundoff under the original saved corner
velocity. Exact exterior lag converges first order, while numerical-parent
inflow retains about 16.5% corner error. This smooth prescribed control does not
reproduce the severe coupled amplification; mixed-boundary mechanics and its
evolving feedback remain open. The guide's physical algorithm is unchanged.

The explicit exterior option is `Material tensor inflow = prescribed exterior
stress`, with a nonempty `Material exterior stress identity`. It currently
requires direct physical stress. `set_material_exterior_stress` attaches a
provider of the pure-transport exterior from the accepted parent, receiving
parent time, RK stage time and attempted dt. It feeds every native incoming face;
rotation and endpoint reaction remain downstream and run once. Zero and accepted
trace retain their existing meanings. Checkpoint identity distinguishes the
exterior law; its provider must be reattached on resume. The homogeneous benchmark provider
supports its declared uniform timestep schedule only; the layered provider below
uses an accepted-time recurrence. The shared homogeneous
background supplies parent stress to transport and endpoint traction at p=0,
with zero stored initial history despite the separate .1-s initial mechanics
interval. See [comparison](EXTERIOR_STRESS_REPORT.md) and
[protocol](EXTERIOR_STRESS_PROTOCOL.md); these select no production boundary law.

The benchmark-only [layered exterior](LAYERED_BULK_CAMPAIGN_REPORT.md) supplies
equilibrated depth-dependent boundary data through the same physical-stress
callback. Its accepted stress and shear gradient follow actual accepted intervals;
ordinary start-timestep preparation, accepted postprocessing and plugin save/load
handle candidates, rejection and restart. Interior velocity, native stress
rotation and the coupled endpoint solver remain unchanged. The layered pressure
reference is p=tau_yy with zero normal top traction, distinct from the earlier
homogeneous p=0 benchmark. See the [protocol](LAYERED_BULK_CAMPAIGN_PROTOCOL.md)
and [pilot mapping](LAYERED_BP3_PILOT_MAPPING.md) before transferring these
synthetic boundary data to a physical lower-crust problem.

The revised [corrected-spin layered campaign](LAYERED_BULK_CAMPAIGN_REPORT.md)
connects these exterior histories to the selected bulk spin. Its [protocol](LAYERED_BULK_CAMPAIGN_PROTOCOL.md)
freezes one finite-slip loading, three translation controls and same-law
refinements. The benchmark loading parameter affects only fresh initialization;
resume uses saved material-label tractions, and the runner verifies matching
physical fixture arguments. Historical total-spin runs are not reused as
corrected-law convergence evidence.

The completed time/mesh/law comparisons show a persistent finite-slip effect;
the adaptive reference, rank-changing restart and provider rejection checks pass.
The existing age predictor runs before mature publication and is not an endpoint
change bound; see [lifecycle](mature_moving_fault/lifecycle.md). The principal
advected-profile limitation is its missing integrated material balance, described
in [geometry](mature_moving_fault/geometry.md). These local results do not qualify
an advected BP3-like production trajectory. The campaign report owns measured
sensitivities and cost; this guide owns the current algorithm and state meanings.
