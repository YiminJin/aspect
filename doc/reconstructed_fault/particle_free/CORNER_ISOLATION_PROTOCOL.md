# Corner isolation protocol — 4 November 2026

Entry: current worktree `pf-rsf-particle-free`, HEAD `cf0652547`; preceding
frozen/shear research and manual work remain. Verify six current binaries, 63
protected artifacts and all four retained sheared trajectories before edits.
Use a new benchmark DSO/build directory; do not overwrite any existing library.
Native source, defaults, physics, ordinary DGQ2/SSPRK3 and acceptance stay fixed.

## Prescribed-velocity control

Use the actual public distributed `TensorAdvection` and `TensorField` operators
on the same [-2,2]² grid with v=(.4+.5y,0). A scratch numerical tensor owns no
accepted material history. Initial stress is (xx,yy,xy)=f*(1,-.3,.45),
f=1+.15x+.1y+.08xy, continued explicitly outside the box. This smooth nonzero
field and its affine pullback lie in Cartesian Q2. No limiter/cut space is used.
Compare exact exterior data with accepted-interior trace frozen at the beginning
of each physical interval. Keep native SSPRK3 target dt_sub*r_adv<=.1.

For pure advection the exact solution is the initial tensor at
(x-(.4+.5y)t,y). For the split Maxwell case, use G=eta=10, no localized slip,
W_xy=.25, D_xy=.25. Transport, native finite rotation, then native exponential
endpoint reaction/projector are applied in their retained order. The independent
continuous characteristic reference has mean m=e^-t*m0 and complex deviator
z=(xx-yy)/2+i*xy satisfying z'=(-1-.5i)z+5i. Also retain an independent exact
discrete recurrence reference to separate Lie forcing error from inflow error.
Exact inflow during advection is the characteristic continuation of the **start
of that split interval**, not the already-reacted continuous endpoint value.

Run to .4 s at 32² with dt=.1/.05/.025, both boundary treatments and both laws;
at 64² run only dt=.025 with the same four combinations. These are nonpublishing
scratch evolutions inside two initialization-only native executions. Record full,
corner and interior tensor errors, stress amplification, storage, supplied versus
interior boundary energy flux, upwind dissipation and semidiscrete energy identity.
Exact polynomial/discrete reference tolerance: relative L2 <1e-9; energy identity
relative tolerance 1e-10. Held-trace differences are measurements, not pass gates.

## Coupled mechanical-corner control

If the prescribed transport separates the mechanisms, run two short32² coupled
controls at .05/.025 s to .4 s, keeping the original accepted-trace inflow and
all material/fault inputs. Change only the previously zero top traction to the
uniform-background traction of the **same discrete shear update** at p=0.
This retains a natural mechanical pressure reference and avoids introducing an
all-Dirichlet pressure-gauge ambiguity. For r=exp(-dt)*exp(-.5i*dt), F=5*(1-exp(-dt)),
the zero-initial-stress background is z_n=iF*(1-r^n)/(1-r); top traction is
(Im z_n,-Re z_n). At initialization use the fixture's separate .1 s interval,
giving traction (5*(1-exp(-.1)),0). The fault perturbation remains coupled and
is not forced to vanish. This is a deliberately changed boundary control, not
the same physical problem or an automatic replacement boundary condition.
The two discrete traction histories approach the same continuous background,
but differ at finite dt. Their temporal pair therefore includes the known
boundary-data discretization, rather than holding an exact traction function
identical at all intermediate times.

Compare against the already completed mixed-boundary32² .05/.025 cases. Report
initialization differences, corner/global/near-fault stress, pressure-inclusive
tractions, slip, geometry, flux/work and cost. Do not claim a continuum reference
from a smaller corner discrepancy. No new grid campaign, production restart,
server jobs, isolated rotation suite or general refactor. At most four MPI ranks
and four build workers; all native executions sequential. Keep failed evidence.
