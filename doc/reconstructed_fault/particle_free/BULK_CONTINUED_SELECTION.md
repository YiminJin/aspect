# One exploratory continued-loading trajectory

21 October 2026; f1f9f29cf40c plus preserved local work. Fixed 96x128 wide
specimen, Delta x=.015625 m, Delta y=.0078125 m, Delta t=.00125 s. Reuse the
native executable, unchanged physical seed, homogeneous law, source complement,
unlimited DGQ2, Gauss8/source3, physical obstacle and all original checks.
The sole physical input change is the selected loading schedule: unloaded
through .05 s, affine shear rate .018/s through .175 s, then fixed displacement
through .185 s. Loading through .15 s is identical to the spatial experiment.

The retained rolling checkpoints contain accepted steps 123,127,128, all after
the original loading stop. None is a compatible branch point: start fresh,
without transferring, resetting or independently relaxing any physical history.
Verify fresh initialization and the trajectory through .15 against the retained
96x128 run. Native checkpoints retain their usual owner/clock semantics.

Track Q1 seed-connected superlevel sets at phi=.001,.01,.05,.1,.3,.5, with
initial and current contours on identical physical axes. Report centerline
front and upper/lower flank extents, shifts from each initial contour, cap
thickening, and distinct disconnected components. Resolve diagnostic topology
on a physical subcell raster and check sensitive positions with a finer raster;
do not demand identical first-growing nodes. The original surface endpoint is
(.5,.5) m; its pre-existing diffuse cap is not newly propagated surface.
Initially intact means outside the original continuum compact support, with
native initially-zero regions also reported separately. Include first outward
crossings on the centerline and both 45-degree rays; restrict cap widening
to x>=.5 rather than confusing left-boundary widening with cap response.
Contour thresholds
and the 1e-12 nodal change threshold are diagnostic, not acceptance changes.

Save all accepted phase and original-QP q, exact local rates from .12 s,
and complete DG/native-FE/local/phase owners at every step from .15 s. Preserve
native rolling checkpoints from .11875 s onward and full fields at .15,.175,.185. Mark intervals
with max nodal per-step damage increment >=.001 or integrated-growth increment
more than twice its preceding nonzero value as candidates for subsequent
timestep verification. This is a flag, not a numerical acceptance or speed
criterion. No extra trajectory or timestep sweep is selected here.

Keep endpoint work, trapezoidal reaction-displacement work, physical storage
and dissipation, per-step and cumulative exact BE numerical remainders, and
source/mechanical/KKT/publication evidence. Animation uses constant scales and
the full domain, phase contours, total damage increment and native local slip
strain rate (1/s), with reaction-strain and energy histories. Distinguish this
bulk rate from a surface slip velocity. Stop at .185 s or an existing solver/
admissibility failure, with a 14400 s external safety timeout. Prior failed
spatial accuracy and endpoint energy screens remain limitations; growth speed
is unqualified. No constitutive, tolerance, mesh, BP3 or production change.
