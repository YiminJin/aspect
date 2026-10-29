# Angular flank activation — saved-state diagnosis, 20 October 2026

**Onset follows the combination of mechanically generated oblique driving and
the weak fracture-resistance pattern near the compact cap edge.** It is neither
the global driving maximum alone nor the global fracture-force minimum alone.
The ideal smooth cap has no angular fracture-force variation at fixed radius;
the represented seed and grid-based weak averaging do. Existing aligned meshes
do not separate physical angular structure from grid bias sufficiently to claim
a mesh-independent onset angle. No new trajectory or physical change was made.

In plain language: the seed has a strongly resistant core and a relatively weak
outer rim. Loading builds two oblique driving lobes near that rim. Some rim
nodes then approach the phase-growth condition sooner than the centerline.
The rounded phase contours are the crack geometry; the colored driving lobes
are a different field. The grid changes the local resistance and the particular
nodes that activate. This supports a mechanical/weak-force explanation of the
patches, not reshaping the cap or increasing loading to remove them.

## Exact seed and its force

`bulk_cohesive/native.cc::seed` uses `GeometricFunction(.15625,1,8/3)`,
`DegradationFunction(1,128)` and `PhaseFieldProfile(...,.6)`. Within the physical
domain its distance is

    d(x,y) = sqrt(max(x−.5,0)² + (y−.5)²),   phi0 = P_N(d).

Thus x<.5 is a straight band and x>.5 is a rounded radial cap about (.5,.5) m.
There is no constant-width plateau at .6: .6 is the centerline maximum.
The actual prescribed function P_N is a **5000-point table**, followed by linear
interpolation in distance, zero beyond the last table coordinate and the existing
endpoint-preserving clamp. Native `PhaseFieldProfile` constructs it with
theta_i=i*pi/(2*4999), P_i=.6 cos²(theta_i) and composite trapezoidal integration
of dr/dtheta. Q1 then interpolates the sampled vertex values. The table, its Q1
representation and an ideal differentiable profile are distinct objects.

For these specific parameters the ideal defining equation simplifies to

    ell² P'² = A(P),   P'<=0,
    A(P)=P−h(P)/1280=P(.6−P)(1.5−P)/(1−P)²,
    dr/dtheta = 2 ell (1−P)/sqrt(1.5−P),   P=.6 cos²(theta).

Its support radius is **R=.308821593907069 m**. The native table gives
.308821593907073 m; maximum tabulated-phase difference from the smooth
reference is 1.49e-8. This is not a Gaussian, tanh or the isolated parabolic
AT1 crack profile. Its shape includes the specified cohesive degradation.

With Ef=Kf integral(phi+ell²|grad phi|²), Kf=64.00000000008 Pa, the smooth
interior fracture derivative is f_f=Kf(1−2ell² Laplacian(phi)). Consequently,

    f_band(r)=Kf h'(P)/1280,
    f_cap(r)=f_band(r)+2 Kf ell sqrt(A(P))/r,   0<r<R.

The cap curvature term is positive. The straight-band center force is 280 Pa;
the forward radial cap limit at r→0 is 496 Pa. At r=.15,.225,.288111,.298515 m,
cap point forces are 79.409, 30.435, 10.690 and 8.426 Pa. They are angle-independent
on the open forward semicircle. At the band/cap join the Hessian changes, while
the ideal phase and gradient remain continuous; a weak test there samples both
geometries. There is no added surface continuation.

At compact support P=P'=0. The inner second-derivative limit is nonzero:
P''(R−)=18.432 m⁻², and both point forces approach **6.4 Pa inside**, versus
**64 Pa outside**. This finite force jump has no delta term because the first
derivative is continuous. The table itself is piecewise linear, and a Cartesian
Q1 field has gradient jumps at cell faces. Neither has the above classical
pointwise Laplacian. In particular, using the cell-interior Laplacian of Q1
would incorrectly return 64 Pa everywhere and omit the interface contributions.

The actual native quantity is

    (g_f)_i = Kf [m_i + 2 ell² (K phi)_i],   m_i=integral N_i,
    (g_m)_i = −integral N_i Y,   g_i=(g_f)_i+(g_m)_i.

Positive initial g_f is consistent with the prescribed pre-existing-damage
obstacle: growth is unfavorable when unloaded; healing is forbidden. It is not
an initialization relaxation defect. Initial phase is bitwise unchanged during
the unloaded hold.

![Profile and fracture-force distinction](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_angular/seed-force.png)

*The point prediction is radial in the cap. Native weak densities average over
finite supports, mixing the low-force rim, intact exterior and, near ±90°, the
straight band. Even the smooth seed under the same grid-based weak tests has
angular structure. Q1 seed interpolation adds a further change. Red marks are
the later first-growing nodes, not initial damage increments.*

## Compatible angular diagnostics and first activation

The primary case is the saved 48×64/Gauss8, Lx=1.5 m trajectory. First cap growth
is step 55, **t=.1375 s**; its immediate accepted precursor is step 54, **t=.135 s**.
The forward-cap check x>.5, radius<.35 m agrees with the earlier flank-mask
onset, rather than presuming that mask excluded no earlier activation.

Arcs use r=.15, .225, .2881107643, .2985152059 and .325 m, theta in [−90°,90°]
from +x. The middle two radii pass through actual first-growing nodes; the last
is outside ideal compact support. Curves include exact grid-crossing angles,
so sampling does not miss the sharp weak-density minima at native nodes.

For an arc point z use the **same normalized weak test** for all forces:

    W_z = sum_i N_i(z) N_i / m_i,   integral W_z=1,
    qbar=integral W_z q,   Ybar=integral W_z Y,
    Fbar=sum_i N_i(z) (g_f)_i/m_i,   Rbar=Fbar−Ybar.

Phase curves show the physical Q1 field at z. qbar is diagnostic, not a new
constitutive state: Y is evaluated using each saved q and phase at its original
Gauss point, then averaged. It is **not** recalculated from qbar and arc phase.
No local history is interpolated or transferred. The missing dense step-54 CSV
is recovered from `state-0054.bin`, using fixed cell/QP geometry from step 52.
Numeric CellId ordering is used. Decoding reproduces all exported step-52
a/k/L/q values exactly; reconstructed step-54/55 weak driving agrees with the
native phase-force files within 1.30e-14 after normalization by Kf.

| First-growing angle (both signs) | r [m] | Initial/native Fbar [Pa] | Precursor Ybar [Pa] | Precursor Rbar [Pa] |
|---|---:|---:|---:|---:|
| 40.6013° | .288111 | 14.74859 | 12.61259 | 2.13601 |
| 49.3987° | .288111 | 14.04704 | 12.40034 | 1.64670 |
| 42.8789° | .298515 | 18.69558 | 16.57754 | 2.11804 |

These six nodes initially have phase .00395137 or .000978846. At step 55 their
increments are 2.02e-5, 3.08e-5 and 4.75e-5 respectively. Before this event the
cap phase and its fracture contribution are unchanged; q and Y have grown
mechanically from zero.
The onset time is the first accepted discrete step, not an independently
resolved continuous-time threshold.

![Five arcs, initialization and precursor](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_angular/angular-arcs.png)

*Blue is initialization; orange is the immediate precursor. Phase and fracture
curves overlap where unchanged. Gray solid in the fracture row uses the smooth
seed under the same weak test; gray dotted is the smooth point prediction.
Red guides identify actual first-growing angles at the two matching radii.*

## Which term selects the flanks?

At r=.288111 m, precursor Ybar peaks at **42.75°**, while fracture density is
lowest at **90°** (9.58554 Pa). At that join Ybar is only 2.14809 Pa, leaving
Rbar=7.43746 Pa. The smallest combined margin is instead **49.3987°**, 1.64670 Pa.
At r=.298515 m, Ybar and the smallest combined margin occur at **42.8789°**.
Thus one first-node set lies at the driving maximum and another is shifted by
the existing fracture pattern. qbar and Ybar maxima need not coincide because
the physical driving is nonlinear and includes h'(phi).
Outside the ideal support, r=.325 m has still higher maximum Ybar=17.0127 Pa but a
minimum remaining margin of 36.3348 Pa, so it does not activate. The radial
location and angular driving/resistance balance both matter.

Relative to the centerline at the same radius, the 49.3987° node's margin is
13.50046 Pa smaller: **5.63223 Pa** comes from lower initial weak resistance,
and **7.86823 Pa** from higher generated driving. At 42.8789°/r=.298515 the
corresponding contributions are **10.83121** and **10.37574 Pa**. This is a
linear functional accounting of the recorded state, not a counterfactual
evolution with a replaced seed.

The radial smooth force alone does not contain a ±45° weak notch. Near compact
support, however, finite weak supports mix its inside/outside force jump in an
orientation-dependent way. At r=.288111, theta=0 the point force is 10.690 Pa,
the smooth-seed/same-test force 19.374 Pa and native Q1 force 19.679 Pa. At
49.3987° the latter two are 11.544 and 14.047 Pa. Here Q1 seed interpolation
**raises**, rather than creates, the low resistance relative to the smooth
field under the same weak test. Force averaging and representation must not
be mistaken for angular pointwise material properties.

Smooth weak reference forces use integration by parts on each Cartesian Q1
cell (Laplacian(N_i)=0), with edge quadrature split at support, band/cap join
and centerline. Gauss24/32 differ by at most 3.40e-8 Pa. The tabulated/smooth
weak-force difference is 3.26e-4 Pa, far below the multi-Pa representation/
averaging effects. No cap or law was modified to obtain those comparisons.

![Driving with phase contours and numerical comparisons](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_angular/driving-geometry.png)

*White contours are phase geometry at .001,.01,.1,.3,.5; the ring is the original
endpoint. Color is actual Gauss-point Y on a common scale. Red crosses identify
the six actual first-growing nodes, shown as upcoming locations on the precursor
panel. The oblique colored lobes are not two prescribed crack tips or seed bulges.*

## Numerical sensitivity and one next experiment

At common t=.135 s and r=.288111 m, the square 32×32, fine 32×64 and wide 48×64
cases have Ybar maxima near 40.60°, 43.75° and 42.75°. Their first-growing cap
node sets span ±40.60–49.40°, ±40.60–57.99° and ±40.60–49.40°, at .1425,
.1400 and .1375 s. Fine/wide initial weak forces agree; the changed domain
changes mechanical driving. Square/fine change both represented curvature and
mechanics/averaging. These results support an oblique discrete response but all
meshes share the same Cartesian directions. The fine G6/G8 agreement excludes
that quadrature explanation; it does not exclude grid orientation bias.

Recommend **one 45° near-tip mesh-orientation control** on the same physical
wide specimen: fixed physical endpoint/continuum seed, global shear tensor S,
boundary conditions/loading, material, timestep and tolerances; matched local
cell shape/size and fresh unloaded histories. Rotate the grid relative to those
physical data, not the whole physical problem together with the grid. Compare
initial weak forces, precursor angular q/Y/R and first-growing locations in
physical coordinates. This can test whether lobes follow physical loading or
the grid. Keep geometry fixed during the trajectory.
For cross-orientation fields use a common physical averaging kernel, alongside
each native KKT test: rotating the diagnostic test together with the grid would
itself change its averaging of the compact-support force jump.

This is not an existing input switch: the current Cartesian column observer and
source/projection assumptions require targeted qualification for a conforming
rotated interior patch and its transition. Keep source/coverage guards intact;
identify any necessary representation dependency before adopting a different
discretization. Do not silently relax projection tolerances. No orientation
test, cap reshaping, increased drive or further propagation run is started here.

The original energy-screen failure (wide 2.2328%, earlier dt/2 still above 1%)
and its work-time-discretization explanation remain; this diagnosis does not
qualify physical energy, stability or propagation. BP3, old evidence, production
defaults and `strict_reference_match_passed=false` are preserved. Entry is
f1f9f29cf plus retained local work. Scripts, actual arc/node data, saved-state
decoding checks, source/input hashes and figures are in
`benchmarks/reconstructed_fault/particle_free/evidence/bulk_angular/`.
