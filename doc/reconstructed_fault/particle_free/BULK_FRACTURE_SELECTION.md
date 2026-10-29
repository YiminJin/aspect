# Bounded fracture-scale specimen — selected before native execution

21 October 2026, f1f9f29cf plus preserved local work. One fresh 2-D trajectory;
no spatial gate, law change, parameter search or additional native trajectory.
The released-side 96x128 Cartesian MPI path, Gauss8/source3, unlimited DGQ2
plus native FE complement, fixed mesh, once-only ownership and all native
source/mechanics/KKT/ledger tolerances are retained. BP3 is protected.

The 1-D calculation uses the implemented bulk law, not the old surface law.
It independently solves constant-traction equilibrium, physical Q1 obstacles
and local stick/slip, then checks every update against the actual C++ return.
The separate equilibrium profile family checks the full excess cohesive-work
area and the AT1 meaning of Gc. Viscous and residual-friction work and the
pre-existing seed fracture energy are accounted separately.

## Selected physical inputs and expected regime

G=1e6 Pa, eta=1e8 Pa s and residual R=1000 Pa remain. Select Gc=1 J/m2,
strength drop=2000 Pa, ell=.046875 m; hence Kf=8 Pa, m=4 and virgin peak=3000 Pa.
Lc=G Gc/drop^2=.25 m; slip scale=Gc/drop=.0005 m. The diffuse profile keeps
peak .6 and the same native analytic prescription, recalculated from these
coefficients. Its support diameter is .1852930 m, resolved by 23.72 normal
cells; ell spans 6 normal and 3 tangential cells. No accepted field/history is
prolonged or imported. The straight seeded interval is length 1 m, from the
left computational boundary to the physical endpoint (1,.5); the rounded cap
extends .0926465 m beyond that endpoint. Domain remains 1.5x1 m, so the new
ligament is .5 m. This is a finite-geometry specimen, not a domain study.

The initial ell=.0625 choice gave m=3, exactly the existing g''(0)>0 guard's
excluded boundary. The startup stopped before initial state or publication.
Its input/selection/strip data and log are preserved under
build-particle-free/bulk-fracture/startup-m3 and runs/startup-m3-rejected.
For p=1, g''(0)=2m(m-3); selecting m=4 analytically gives ell=.046875 without
changing Gc or strengths. The guard remains intact. This correction precedes
the sole physical trajectory; no continuation or parameter sweep is selected.

For incompressible plane strain E'=4G, the dimensional sharp-crack estimate is
G_tip approximately F^2 pi a (tau-R)^2/(4G), with unknown specimen factor F.
For F=1 and a=1 m the candidate excess threshold is 1128.4 Pa, below the
2000 Pa virgin strength drop. The retained a=.5, Gc=26.67 case would require
8240.5 Pa, above its 1000 Pa virgin drop. This estimate is an ordering check,
not a solved energy-release rate: partial damage, cap curvature, friction,
finite boundaries and actual stress redistribution matter. No stress target
or spatial nucleation prohibition is imposed. The new seed/scale choice favors
testing seeded activation before homogeneous nucleation; it does not guarantee
centerline extension. Cap thickening, disconnected damage, solver stop or no
extension remain valid measured outcomes.

Same top/bottom affine rate .018/s on (.05,.175], then hold to .185 s, lateral
ux natural and uy fixed. Dt=.00125 s, 148 positive steps, four ranks. Saved
activation is bracketed by adjacent accepted states; no speed is qualified.
Estimated cost from the same-size released run: roughly 40–80 minutes, bounded
by 7200 s wall timeout; original source/residual/phase checks stop earlier on
failure. About 12,288 cells, 125k native mixed DoFs, 331,776 DG coefficients,
110,592 stationary diagnostic particles; actual counts, wall/RSS and disk are
reported. Raw solver data/checkpoints stay in build-particle-free, separately
from compact PNG/GIF/CSV/report evidence.

## Measures and interpretation declared in advance

Check fresh seed representation and unloaded obstacle/KKT, all native numerical
gates, monotonic physical phase, once-only publications, stress/return identities
and independent energy/reaction accounting. Retain old spatial 35.86% growth
discrepancy, energy screens and unmeasured new-regime temporal uncertainty.
No new spatial/temporal accuracy claim follows from this single specimen.

Report whole-domain phase, positive-Gauss-weight physical stress/Y/rate means,
physical Q1 contours at .001,.01,.05,.1,.3,.5 and regions of initial zero phase.
Track components containing the seeded band separately from initially detached
components. Report centerline/flank changes and activation brackets using phase
increments >1e-12 (numerical onset) and >1e-5 (visible onset). Sample every
accepted state; old and new Dt=.00125 s. Compare mechanical and fracture weak
forces through the same mass-weighted physical kernel, with its radius stated;
do not evaluate a constitutive law on interpolated local histories.

Report initial fracture storage, bulk/cohesive/fracture changes, friction,
bulk/cohesive viscosity, endpoint/trapezoidal work and separate BE/remainder
terms. Keep exact algebraic ledger closure distinct from physical work accuracy.
Judge the observed mechanism and stop after the one bounded trajectory.
