# Fixed-mesh bulk MPI and one boundary-loading control

21 October 2026; entry f1f9f29cf40c plus preserved local work, pinned in
build-particle-free/bulk-parallel/entry.json. No BP3/default change.

First recover original Gauss8 physical tau_xy, Y and raw nodal phase forces
from saved continued-loading states (.15 through remote activation). Use actual
DG/native-FE source complement and local histories, never auxiliary T as stress.
Separate point driving from weak nodal growth/resistance. Original endpoint is
(.5,.5) m; compact seed radius .30882159 m. Keep previous failed spatial target
and unmeasured temporal uncertainty in the extended-loading interval.

Qualify the unchanged bulk residual/tangent/constitutive update on fresh 24x32
fixed grids at Delta t=.0025 s through .16 s, with identical timestep sequences
on 1/2/4 ranks. Compare physical CellId-local Q2 coefficients, native FE memory,
local histories, physical phase and reconstructed stress. Reporting tolerance:
1e-10 relative full tensor/field/storage norms (stress denominator at least
1000 Pa), alongside the unchanged native mechanics 1e-9, KKT 1e-7 and independent
source 1e-12 checks. Same-partition rejection/retry and restart should be bitwise.
Exercise growing states, a repeated outer rejection/retry before .14 acceptance,
and restart after .15. Measure wall time without overlapping timing runs;
profile assembly, collective reduction, root direct factor/solve, history and
output separately. Retain serial UMFPACK; no solver-tolerance change. Distributed
cell histories and assembly with a replicated coupled search and root direct
solve are a bounded implementation, not a scalable distributed factorization.
Use sparse raw fracture entries to avoid O(n_phase^2) replicated storage.

Only after those gates pass, one fresh 96x128 mixed-BC run to .185 s at
Delta t=.00125 s, Gauss8/source3. Same seed, material, length scale, orientation,
loading (.05,.175] at .018/s, hold to .185, unlimited DGQ2 plus native complement,
physical obstacle and all acceptance rules. Top/bottom ux=+/-.009 m/s while
loading, uy=0. Lateral uy=0, ux is free with zero natural horizontal total
traction. This removes the mechanical pressure nullspace: do not pin pressure
or add a hydrostatic offset. Friction retains the specified independent p_N=
10000 Pa. No new normal traction is imposed. Report that distinction explicitly.
Sum reaction work only on prescribed components; zero-increment vertical
restraints contribute no work. Check corner compatibility, open-normal pressure
treatment, endpoint reaction-work equality and trapezoidal force-displacement
work. Keep phase active throughout the domain.

Compare stress/phase/rate on identical physical axes and averaging kernels,
connected contours, centerline/flanks, initially intact/remote damage, reactions
and separate physical/BE energy terms. A lower remote increment alone does not
qualify propagation or accuracy. No AMR, transport, RSF, moving geometry, limiter
or physical-law change, and no further case starts automatically.
