# Fracture scales and one propagation-focused bulk test

21 October 2026; f1f9f29cf plus preserved local work.
[Pre-run selection](BULK_FRACTURE_SELECTION.md),
[compact data and figures](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture/).
One fresh released-side trajectory uses the qualified four-rank, fixed-mesh
bulk path. No spatial activation gate, stress clipping, law change, transport,
RSF, AMR or BP3 modification is introduced.

**Result:** the new scales replace flank-first/remote-island activation with a
seed-connected softening zone. High-phase core advance remains small, and the
hold has distinct admissible KKT continuations. This is a useful propagation
specimen candidate, not quantitative or unique physical propagation qualification.

## Units and competing physical scales

Source units are SI: G/q/tau/C in Pa, eta in Pa s, Gc in J/m2=Pa m, ell in m;
a/k/phi/h are dimensionless and w is engineering shear rate [1/s]. Observer
I=int h dy [m] is a compliance integral, not geometric width; Z=G int a dy
[Pa m] adds no surface energy. Two-dimensional work/storage is J/m per
out-of-plane thickness; strip quantities are J/m2.

In this bulk option, configured cohesion C is the **strength drop**, not total
peak traction: Kf=Gc/[(8/3)ell], m=2G Kf/C², residual R=mu p_N=1000 Pa and
tau_p=R+C. The friction coefficients are both .1, p_N=10000 Pa, and eta/G=100 s.
Computed mechanical pressure does not replace p_N. Every positive bulk update
uses its actual Delta t=.00125 s; the inherited initial-time-step entry does
not create an extra constitutive advancement.

| Quantity | Retained released specimen | Selected fresh specimen |
|---|---:|---:|
| G [Pa]; eta [Pa s] | 1e6; 1e8 | unchanged |
| Gc [J/m2] | 26.6666666667 | 1 |
| tau_p / tau_r [Pa] | 2000 / 1000 | 3000 / 1000 |
| G Gc/(tau_p-tau_r)² [m] | 26.6667 | .25 |
| Gc/(tau_p-tau_r) [m] | .0266667 | .0005 |
| ell [m]; Kf [Pa]; m | .15625; 64; 128 | .046875; 8; 4 |
| Initial support width [m] | .6176432 | .1852930 |
| Straight seed length / remaining ligament [m] | .5 / 1 | 1 / .5 |
| Domain [m] | 1.5 x 1 | unchanged |

Top/bottom v_x=.018(y-.5) on (.05,.175] s, then hold to .185 s; lateral ux is
natural and uy restrained. Delta t=.00125 s, Gauss8/source3, unlimited DGQ2
and the native FE source complement are unchanged.

The original slip scale exceeds the .00225 m grip motion almost twelvefold.
For an ordering estimate, E'=4G and K_II=F(tau-R)sqrt(pi a) give excess
propagation thresholds 8240.5 / 1128.4 Pa for retained/new cases with F=1,
compared with virgin drops 1000 / 2000 Pa. The plane-strain relation is standard
([Bower, section 9.4.3](https://solidmechanics.org/Text/Chapter9_4/Chapter9_4.php));
the unknown geometry factor and partial damage prevent treating this as a
computed J integral. Only two new process scales fit the remaining ligament;
boundary independence is untested.

The same compact seed family has peak .6 and no flat core. With distance r to
the straight seeded interval, phi=.6 cos²(theta) and
dr/dtheta=2ell(1-phi)/sqrt(1.5-phi). The cap is the same radial prescription.
Both Gc/strength-dependent degradation and physical seed width are rebuilt
from configured coefficients; histories start unloaded. Material remains
eligible for phase growth everywhere. The first ell=.0625 choice put m=3 at
the existing strict g''(0)>0 guard; startup stopped before initialization.
That input/log is preserved. Selecting m=4 gives ell=.046875 analytically;
the guard and all numerical tolerances remain intact.

## What the strip establishes

The independent 256-cell Q1/Gauss8 strip enforces equilibrium, Coulomb stick/slip,
BE, irreversibility and prescribed displacement. Every update is checked
against the unchanged C++ point law.
Scaled discrepancies are below 1e-14 for the prescribed .185 s history.
Complete reduced phase actions, including the eliminated traction response,
agree with independent admissible directional differences to 5.98e-9.

For the stationary profile family with peak A and uniform forward-sliding q,

    q(A) = C(1-A)/sqrt(1+A),
    ell² phi_r² = phi - A h(phi)/h(A),
    delta_el = q int(h)dy/G,  Ec = q² int(h)dy/(2G).

The excess-work area int q d(delta_el) tends to Gc, with final elastic slip
[3pi/(2sqrt(2))]Gc/C, not simply Gc/C. Numerical areas are 26.66663854 and
.9999989452 J/m2; the 512/1024-point integration difference is 5.37e-11 scaled.
The fully developed AT1 profile (1-|y|/(2ell))² has
int(phi+ell² phi_y²)dy=(8/3)ell. Thus **Gc normalizes full-band fracture storage**.
It is not total traction-slip work including friction, viscosity or elastic
storage. The initial .6 profile already stores .5467498155 Gc per straight
unit length; that pre-existing energy is subtracted from subsequent work.

The stationary family is a calibration, not a substitute for an irreversible
history path: changing its peak may require local unloading/sticking. Actual
BE strip histories report bulk/cohesive storage, initial fracture energy,
R int|dk|dy, both viscosities, endpoint/trapezoidal work and every numerical
remainder separately. The .185 s histories pass. Under a separately extended
strip calibration load, continuation stops at .1775 s after .17625 s acceptance.
The failed trial has negative Newton-free curvature (-.03549), including a
feasible mode; it is not a converged KKT state. This exposes a nonconvex
continuation limitation, not a verified physical instability or complete
post-softening fracture-energy measurement. No jump rule or regularization
is added to get through it.

## Native result and interpretation

The four-rank trajectory completes 148 steps to .185 s in 1906.10 s (31.77 min;
7200 s bound). The 96x128 mesh has Delta x=.015625 m, Delta y=.0078125 m:
3/6 cells per ell, 23.72 across seed support. Actual counts: 12,288 cells,
372,233 native DoFs (124,228 velocity/pressure/phase), 331,776 DG coefficients,
110,592 fixed particles. Observed peak rank RSS is 1255 MiB; sum of rank peaks
3711 MiB is not a simultaneous node peak. Assembly/linear/history/output cost
903.4/647.7/40.4/133.0 s; communication 60.4 s. This is not a speedup comparison.

Fresh initial L2 representation errors are .5920% whole-domain / 2.0576% cap,
versus .08529% / .18849% retained. Bounds are [0,.6], unloaded phase drift is
exactly zero, and minimum unloaded weak fracture/(mass Kf) is +.11245. Native
sampling agrees with the independently generated continuum table to 2.95e-15.
Initial fracture storage is .56941370 J/m; continuum prescription gives
.57106730 J/m. These are representation differences, not physical-time growth.

| Measured response | Retained | New |
|---|---:|---:|
| First growth bracket [s] | (.12625,.12750] | (.13375,.13500] |
| First initially-zero node bracket [s] | (.13750,.13875] | (.14375,.14500] |
| Far-ahead window activation [s] | (.14500,.14625] | (.17375,.17500] |
| Maximum phase increment at .185 [1] | .018182 | .292627 |
| Centerline phi=.1 contour change [mm] | .0229 | 231.14 |
| Centerline phi=.3 / .5 changes [mm] | 0 / 0 | 49.79 / 1.005 |
| Maximum detached components at phi=.001/.01 | 1 / 1 | 0 / 0 |
| Added fracture storage [J/m] | .159873 | .040073 |

Onset rows use increments >1e-12; with >1e-5 the retained first-cap bracket
is (.12750,.12875], while the new first-cap bracket is unchanged.

Retained onset is on both flanks near (.719,.305/.695), whereas new onset is
on the centerline and adjacent nodes at x=1.078125 m, inside the initial cap.
The new initially-zero node is (1.09375,.5), just beyond its .0926465 m support.
The x-time diagram shows the retained disconnected far activation and later
merger separately from the new connected progression. All reported new phase
components remain seed-connected at the stated thresholds and accepted times.
Far-ahead positive phase in the new case is consequently not labeled a detached
nucleus. At .185, sampled physical shear stress peaks at 2999.53 Pa; maximum
at still-intact quadrature points is 2985.65 Pa, below the 3000 Pa homogeneous
threshold. The driving/resistance profiles show the role of profile gradients.

The low/mid-phase zone extends into initially intact material, but the .3
contour still ends inside the old cap support and the .5 core moves less than
one cell. Thus this is process-zone development, not demonstrated mature
crack-core propagation. The largest single-step phase increment is .085608
at (1.25,.5), in (.17375,.17500] s. That rapid interval is preserved with a
preceding native accepted-127 checkpoint; its speed is not validated.

New cumulative work is W_BE=3.50835692 and W_tr=3.47680323 J/m. Storage change
excluding the initial seed is 2.89991723 J/m; bulk/cohesive viscosity and friction
are .00316951/.000097649/.57377647 J/m. Signed trapezoidal remainder is
-1.57625e-4 J/m; sum of absolute step remainders / sum absolute work is
**.004534%**, so a small signed result is not hiding cancellation. The exact BE
ledger separately retains stress/traction/phase/fracture numerical terms
(.03064749, .00278868, -.00353395, .00149384) J/m and phase/mechanical residual
works (1.73e-12, -2.82e-13) J/m. This work balance does not validate response
time accuracy or choose a branch.

During the first hold step, phase increases .0158369 despite zero boundary work;
stored energy supplies friction and fracture changes. However, **another hold
candidate exists**: keep phase fixed, set displacement increment zero, scale
full T/a/p by beta and use the actual point return for q. Its minimum scaled
weak phase gradient is +2.86881e-6, projected KKT is zero, predicted free
mechanical residual <=1.52e-15 and slip histories are unchanged. Native point
stress checks agree to 3.42e-15 scaled. This is a read-only candidate calculation,
not a second simulator trajectory or a stability qualification. The actual
hold therefore exhibits a branch-selection issue; energy/KKT closure alone
does not establish that its extra advance is uniquely physical.

Native mechanics/KKT/source maxima are 8.79e-10 / 6.53e-8 / 6.60e-15, within
unchanged 1e-9 / 1e-7 / 1e-12 limits; maximum ledger error is 3.49e-15 J/m.
The existing full native tangent check gives 2.68e-10 under its original scaling.
All 148 publications occur once. Nine original-QP snapshots independently check
full plane-strain stress/storage, cohesive relation, friction and accumulated
history. Core equations/binary, baseline plugin and all 34 protected BP3 files
remain hash-identical. No new restart or general propagation qualification is
claimed from unchanged infrastructure.

[PNG mechanism maps](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture/physical-mechanism.png),
[x-time diagram](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture/activation-x-time.png),
[early profiles](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture/early-physical-profiles.png)
and [GIF](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_fracture/propagation-focused.gif).
The inputs prescribe a longer, narrower seed and recalibrated fracture scales;
the pictures show calculated physical stress, phase and slip activity. They do
not prescribe an advancing segment. Temporal/spatial accuracy and branch
stability remain untested here. The smallest next task is a bounded timestep
and branch/stability comparison around .1725–.17875 s, before any longer run.

## Data definitions and retained limitations

The CSV files encode seconds, metres, Pa, J/m, J/m2 and rate units in their
headers. Centerline phase is the physical Q1 value at x-cell centers, y=.5.
The band rate averages the two adjacent normal rows. The plotted cross-fault
rate is the x-cell-averaged integral of |w| over the whole height [m/s]; it is
slip activity, not signed surface velocity. Stress/Y maps use positive native
Gauss8-weighted cell means. Early profiles apply K(r)=(1-r²/a²)^3 for r<a,
a=.03125 m: area-weighted cell means for stress/Y, and native integrated forces
with matching nodal masses for compatible mechanical/fracture force comparison.
No constitutive state is evaluated from interpolated local histories.

Activation thresholds are increments >1e-12 and >1e-5, bracketed by adjacent
accepted times. Phase contours are .001,.01,.05,.1,.3,.5; first centerline exit
and seed-component extent are separate from detached components and mergers.
The connected-component raster subdivides each physical cell four times.
Selected onset/final states give identical component counts at eight subdivisions.
The reporting-only far-ahead window is x>tip+.4 m; it never enters the PDE.
Saved temporal resolution is Delta t=.00125 s; the GIF shows every accepted
state from .12 to .185 s (frame stride .00125 s). No propagation speed is inferred.

The prior 35.86% failed growth-spatial target, endpoint energy-screen failures,
branch findings and strict_reference_match_passed=false remain. This different
physical specimen has no timestep-halving or spatial-reference qualification.
Bulk source/projection, independent physical-stress/energy checks, native
residual/KKT/irreversibility and once-only publication remain required.
Particles stay fixed and domain-free; particle infrastructure is still present.
Solver archives/checkpoints and plotting caches are retained separately under
build-particle-free/bulk-fracture. Deliverables are PNG, one GIF and compact CSV;
no routine PDF/SVG/MP4 copies are generated. No further trajectory is automatic.
