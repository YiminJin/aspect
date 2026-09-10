# Domain-integrated surface quadrature: implementation addendum

2026-09-09. Approved discrete-formulation revision following
`stage_K2_endpoint_moment_regression.md`. The previous implementation obeyed
the previous point-volume specification. This addendum defines the revised
discrete inputs before production edits.

## Measure and evaluation locations

Admission is unchanged: the existing parent-particle center association selects
one fault or excludes the particle. Every admitted particle contributes its
**entire existing Voronoi domain D_p**, including portions outside the normal
admission width. Neither normal clipping nor additional particle admission is
performed. The geometric kernel remains one; domain construction and full I_h
are unchanged. Locally owned parents alone contribute their domains, even if
these extend into remotely owned bulk cells.

Bulk FE velocity gradient, physical pressure, temperature, current/previous
phase field retain their existing **parent-center evaluation**. They are P0
inputs over D_p. Bulk chemical fractions and old Maxwell stress retain their
parent-particle values, likewise P0 over D_p. G samples bulk perturbations at
the same parent centers and differentiates precisely this residual. This
revision does not introduce a new bulk FE quadrature or history interpolator.
Adiabatic pressure retains its parent-center evaluation as well.

At each domain integration coordinate s(x), evaluate the surface Q1 fields:
trial V, committed Theta, projected chemical fields and surface temperature,
current/previous I_h and cohesive history. Recompute the pointwise constitutive
response there with the frozen parent bulk/history inputs. R_Gamma, K_V and G
use these same coordinates/weights. No integrated mass is combined with a
single parent-coordinate nonlinear response.

Generic particle properties and caller-derived particle scalars remain P0
domain data. Their projection objective becomes integral_Dp (z_p-z_Gamma)^2.
Initial q_p retains the established parent-local phi/H and surface mixture at
its parent projection; only its projection measure changes. Particle history
updates remain parent-local, at the existing commit time. Changed initial
surface projections are recorded, not reset to previous-run values.

## Geometry, node crossings and quadrature

The straight-only scope below records the initial implementation. The approved
extension in `stage_K2_polyline_quadrature_addendum.md` supersedes that limit
and defines the corner/tip map and selected-segment frames. The parent-P0
evaluation and full-measure rules above remain unchanged.

The initial supported domain integration geometry is a straight, ordered 2-D
fault with any inclination, nonuniform collinear segment lengths, and finite
independent endpoint DoFs. Multiple nonoverlapping straight faults are allowed;
each domain retains its parent's fault assignment. Curved polylines, 3-D,
overlap and branching require a separately reviewed closest-coordinate
partition and are explicitly unsupported by this first integrated rule.

Use orthogonal arclength on the assigned straight fault. Split each convex
polygon at every surface-node plane it crosses and at projected polygon
vertices. On each resulting arclength interval the domain's transverse width
is linear. Integrate that width times surface functions with Gauss quadrature.
This reduces the full domain integral to one dimension because all remaining
bulk/parent inputs are P0. Constant endpoint continuation handles any admitted
domain portion beyond a true tip: it retains full measure without admitting
outside centers, extrapolating V below its bound, or joining endpoint DoFs.

Three-point Gauss on each linear-width interval integrates Q1 first/second
moments exactly (up to geometry roundoff). The same positive quadrature is
used for the nonlinear response and residual-square diagnostics. Exact mass
moments do not prove nonlinear accuracy: compare orders 3, 5 and 7 on smooth
manufactured nonlinear responses independently of derivative tests. If order
3 is insufficient, report the accuracy evidence before running trajectories;
do not change nonlinear solver tolerances to compensate.

## Ownership, cache and verification

Expose the already constructed polygon through a read-only particle-domain
accessor; do not reconstruct different domains in the fault manager. Cache
parent-to-integration-point offsets with segment/xi/weight, invalidating with
domain geometry as well as existing particle/fault cache changes. MPI sums
replicate the same small surface systems. Geometric coverage must satisfy
sum_q w_pq = m_p without renormalizing an uncovered tail away.

Use the integrated mass for generic projections and free-surface strong RMS.
Use the chosen nonlinear quadrature for R_Gamma, K_V, G and weak traction/
balance diagnostics, retaining fixed merit scales, homogeneous bulk directions
and history/rollback semantics. Verify finite differences of this residual,
not the old point-volume residual.

Verification order: manufactured moments and node/tip coverage; nonlinear
quadrature and derivatives; one-/two-rank ownership/coupled actions; affected
lifecycle/rollback tests; short homogeneous K1 reference; unchanged K2 64/128
cases through 1 s. Estimate resources from the smaller corrected run before
the larger replay. Keep the bulk-history shared-node transfer issue separate.
The broader temporal campaign and true-normal-stress branch remain held.
