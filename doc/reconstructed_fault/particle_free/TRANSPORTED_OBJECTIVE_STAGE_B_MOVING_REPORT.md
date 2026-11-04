# Stage B completion: moving profile, transport and layered relaxation

2 November 2026. The remaining Stage B comparison is complete. Independent
physical and complete separated histories now evolve on an Eulerian Cartesian
DGQ2 mesh while a prescribed material fault translates, shears, changes its
normal profile and crosses a viscosity layer. Twenty numerical criteria pass;
a further eight consolidation/profile/space-selection checks pass. The
[first-part evidence](TRANSPORTED_OBJECTIVE_STAGE_B_FIXED_REPORT.md), including
its rejected q6 quadrature candidate, remains unchanged.

**Selected for the first native mature implementation:** existing trial T in
DGQ2, a separate complete residual R in DGQ2, and the auxiliary surface scalar
memory. Keep direct physical stress as the independent regression reference.
This decision is for the declared constant-G, affine Cartesian history space;
it enables no native configuration. Native residual/tangent/lifecycle/restart
integration is Stage C. Cohesive native-remainder/lifting and shape-work gates
remain separate.

[Comparison PNG](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving/comparison.png) ·
[vector PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving/comparison.pdf) ·
[results](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving/results.json) ·
[selection](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving/qualification.json) ·
[evidence index](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving/README.md).

## Model, motion and complete source

The [protocol](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/STAGE_B_MOVING_PROTOCOL.md)
precedes implementation/runs. The comparison uses G=3MPa, initial ell=.06m,
initial37° infinite through-going fault, observation square [-.4,.4]²m and .8s.
It has no physical tip, artificial continuation, normalizer floor, limiter,
physical damping adjustment, RSF return or solved mechanics.

Prescribe the divergence-free material velocity v=(cx+.4y,.12)m/s, with cx=.24
for the main study and−.24 for the reverse check. Its material map is

    x(t)=d(t)+F(t)X,
    F=[[1,.4t],[0,1]], d=(cx t+.024t²,.12t), det(F)=1.

Bulk histories follow this v; the curve is the material image of alpha t0.
Set g=|F t0|, t=F t0/g, n=rot90(t), z=(x−d)·n, alpha=(x−d)·t/g. The carried
profile is

    z0=g z,   chi=g exp(−(z0/ell)²)/(ell sqrt(pi)),
    D_v chi=(t·L t)chi,       D_v alpha=z(t·L n+n·L t)/g.

Width changes physically with the material deformation from .060000 to
.0517416m; the normal is obtained from geometry, not stress rotation.
Full normal integrals at0/.4/.8s differ from1 by at most3.6e−15.
An independent pullback of the material scalar h=exp(−(z0/ell)²) agrees within
9.44e−16. Applying an old-cell Eulerian maximum instead leaves a trailing
scalar defect .843936. This is a test of the **prescribed material profile**,
not a claim that native phase obstacles have already been transferred.

Local physical relaxation is r_b=.5+.3tanh(y/.08m) per second and eta=G/r_b,
spanning3.75–15MPa s. Surface r_Gamma is explicitly the material-P1 L2
projection of that surface trace on alpha∈[-1.2,1.2]m. Its common physical
measure g d_alpha cancels in this projection. This **auxiliary rate projection**
is a benchmark definition, not a change to native material-mixture/viscosity
averaging. The eventual native operator must use its actual chosen surface
coefficient consistently in the same complete transformation.

Prescribe s=.025(1−exp(−.35t))/.35 [1+.4sin(6alpha)]m and
V=D_Gamma s+r_Gamma s. It is a nonuniform analytic surface-history fixture,
not numerically evolved RSF age or native surface transfer. Initial s=R=0.
Initial xx,yy,xy vary independently with both material coordinates; zz=−xx−yy
is nonzero. Sources and every error/storage contraction retain zz.

With S=sym(t⊗n), W=skw L, and physical mature Qgeo=0, independently evolve

    D_v tau=[W,tau]−r_b tau+2G(D−chi V S),
    D_v T=[W,T]−r_b T+2G D,
    D_v R=[W,R]−r_b R+Q_geometry+Q_extension+Q_rate,
    tau_sep=T−2G chi s S−R,

where

    Q_geometry=−2G chi s[(t·L t)S+S_dot−[W,S]],
    Q_extension=−2G chi S (D_v alpha) s_alpha,
    Q_rate=−2G chi s S(r_b−r_Gamma).

R is never assigned from the direct result. These are the full constant-G
Stage A product/transport/coefficient terms. The direct constitutive equation
contains neither a surface relaxation substitution nor artificial geometry
forcing. Changing projected r_Gamma in the auxiliary representation does not
authorize changing physical local r_b.

## Numerical implementation and independent reference

[moving_model.py](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/moving_model.py)
owns prescribed geometry/profile/coefficient functions and exact material maps.
[stage_b_moving.py](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/stage_b_moving.py)
implements the nine-mode Cartesian DGQ2 weak advection form: volume gradient,
upwind faces, exact physical inflow and local projected r_b multiplication.
It follows the existing PF4 `dg_transport.cc` volume/face structure; it is a
standalone coefficient experiment, not a new ASPECT owner/API. No particle is
used as authoritative or replacement history.

For the reference, y(t)=Y+.12t gives an independent log(cosh) antiderivative of
local r_b. Constant W gives a component propagator. Smooth forcing is integrated
on exact characteristics with temporal Gauss32/48/64. The direct physical RHS
and each residual source have separate integrals; reconstruction is not used to
define the residual. An independent real-component RK4 checks physical stress.

Reference quadrature differences are below3.9e−16G, independent RK4 discrepancy
1.06e−15G, continuous direct/separated closure2.74e−16G. Geometric alpha/profile/
normal derivatives checked by coordinate finite differences have errors
1.96e−11 /3.91e−11 /4.69e−11 at the declared scales, below2e−9. Surface
projection96 versus128 differs by4.28e−15/s. Sampled r_Gamma remains positive
(minimum.29231/s); maximum sampled |alpha|=.66911m is within its declared domain.
Constant/linear DG advection defects are at most2.42e−13; mass defect5.56e−15.

The spatial experiment integrates this coefficient reference with RK4 and a
predeclared DG CFL. This suppresses temporal contamination; it does **not**
replace the selected native lagged finite-rotation Lie convention. That
integrator has its own exact-characteristic80/160/320-step study below.

## Independent refinements

Tensor errors use full3D L2/(G sqrt(area)); normal/shear use their corresponding
volume RMS divided by G. The nonzero reference fixes the denominator.

| Cells/direction | RK4 steps | Direct physical tensor error | Complete separated error | Best Q2 residual projection error |
|---:|---:|---:|---:|---:|
| 8 | 84 | 4.36986e−3 | 5.68336e−4 | 4.24461e−4 |
| 16 | 167 | 6.98609e−4 | 9.21046e−5 | 6.22916e−5 |
| 32 | 333 | 9.36991e−5 | 1.25408e−5 | 8.10972e−6 |

Orders are2.6450/2.8984 direct and2.6254/2.8766 separated, both passing the
predeclared>2 bound. Both converge toward third-order Q2 behavior. At N16,
halving the time step changes the physical solution by2.59e−10G direct and
2.50e−10G separated, far below10% of their spatial errors. The test therefore
does not mislabel time error as normal-history resolution.

At N32, direct/separated tensor RMS errors are281.10/37.62Pa; normal errors
13.16/12.72Pa; shear errors198.28/22.98Pa; zz errors2.40Pa in both. Sampled
maximum tensor errors are .00134252G/.000207063G: an integrated bound is not a
pointwise certificate. Reference bulk storage is79100.602818J/m; relative
storage errors7.37e−7 direct /4.91e−8 separated. This is a final-state storage
comparison, not a coupled boundary/friction/fracture-work balance.

| Characteristic Lie steps | Direct error/G | Separated error/G | Discrete closure difference/G |
|---:|---:|---:|---:|
| 80 | 7.03986e−4 | 8.77519e−4 | 3.16859e−4 |
| 160 | 3.52206e−4 | 4.38983e−4 | 1.58400e−4 |
| 320 | 1.76156e−4 | 2.19547e−4 | 7.91928e−5 |

Both material updates show first order (ratios1.9988–1.9995), retaining the
finite-step direct/separated splitting difference rather than hiding it by
reassigning R. Stage C must derive the actual declared native discrete residual,
source, tangent and publication, using the same accepted anchor everywhere.

Quadrature is varied at fixed N8 and fixed84 steps:

| Source/reaction rule versus12 | Direct solution difference/G | Separated difference/G |
|---:|---:|---:|
| 4 | 1.03859e−5 | 2.26592e−6 |
| 6 | 1.21777e−8 | 3.27912e−9 |
| 8 | 6.97958e−12 | 3.02009e−12 |

The selected rule8 meets the unchanged1e−8 bound. These are this fixture's
quadrature results; the first-part rule6 failure remains preserved and is not
reclassified. Reversing horizontal motion at N16 gives direct/separated errors
7.79875e−4 /1.15897e−4, below the declared .01 bound. No full second refinement
matrix was needed. The main centre moves .20736m horizontally and .096m
vertically:8.2944/3.84 fine cells and3.456/1.6 initial widths. Spin is .16rad,
integrated symmetric-strain norm .22627; the sampled reference stress norm/G
reaches .45850. These synthetic controls do not establish an Earth-scale
finite-strain elastic regime or physical event accuracy.

## Missing-term and heterogeneous-relaxation findings

Independent source-integral omission controls retain the following tensor
errors at the same G scale:

| Omitted term | Scaled tensor error | Dimensional RMS |
|---|---:|---:|
| Geometry/profile/orientation | .00660185 | 19.81kPa |
| Surface-extension gradient | .000214253 | .643kPa |
| Local/projected relaxation mismatch | .00436608 | 13.10kPa |

Each exceeds the declared1e−5 detection bound. A commutator-only residual is
therefore incomplete under the selected motion/layering. The body path through
y=−.06→.036m (and two neighboring paths) also tests nonconstant relaxation.
Replacing the path coefficient with its initial homogeneous value produces
.0127997G error. Independently integrated viscous loss plus final storage closes
the **zero-forcing material relaxation** balance to2.56e−16 relative to initial
storage. No closing residual is invented; this is not the full coupled
mechanical energy balance required in Stage D.

## History-space decision and measured costs

The smallest-space screen is recorded before its focused probe in the protocol.
[Independent best-projection results](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving/space_screen.json)
show that a P1 residual on N8 gives residual error1.33932e−3 and full reconstructed
tensor error1.34552e−3, above the same1e−3 target even before transport error.
Q2 gives best reconstructed error4.43636e−4. P1 here was a **capacity screen**,
not a transported P1 solve or a claim of its temporal/spatial order. It supplies
no reason to add a separate native lower-degree history abstraction.

| First tested passing configuration at1e−3 target | Scalar tensor coefficients | Raw tensor bytes |
|---|---:|---:|
| Direct physical DGQ2, N16 | 6912 | 55296 |
| Complete T+R DGQ2, N8 | 3456 | 27648 |
| Best P1-R/Q2-T capacity at N16 (not transported) | 9984 | 79872 |

The analytic surface memory is not free native information. A separate Q1
snapshot screen uses16/32/64 material intervals (17/33/65 stored scalars).
Its induced B-extension RMS/G errors are .00150776/.000395781/.0000989008,
with convergence orders1.92964/2.00065. The finest meets the same1e−3 target;
count65 scalar coefficients in addition to the separated tensor table. These
are interpolation-capacity measurements, not evolved native scalar/ALE or
source-derivative qualification. The surface protocol's word “ratios” was
corrected to “orders”: second-order raw ratios approach4, log2 orders approach2;
the initial checker already used the correct expression and the original
wording is preserved. No numerical error bound was relaxed.

**Decision:** use DGQ2 for the complete independent residual in the first native
mature extension, alongside the existing DGQ2 trial tensor and scalar surface
memory. The accurate analytic localization lets this form meet the measured
target on a coarser history mesh; no enrichment is forced by the measured R.
Retain the independently evolved direct tensor for regression. This is the
smallest tested complete choice at the declared benchmark target, not a proof
of a globally optimal basis or permission to coarsen the mechanics mesh.

At a common externally fixed mesh the generic separated form still has twice
the tensor coefficients of direct DG. The prototype deliberately stores all
three in-plane components for each full3D deviator, matching the existing owner
layout. In these initialized controls the residual's zz remains zero by its
source symmetry; no new reduced-component native API is selected from that
special fact. Direct storage's same-mesh advantage remains visible if later
coupled physical resolution fixes a finer mesh. No production performance claim
is made from the minimum isolated-history counts above.

Both representations were advanced together. Their combined N8/16/32 solves
cost1.11/3.07/13.84s; the full numerical study about33s, one Python process and
one BLAS thread. These are coefficient-fixture costs, not separate per-law
native timings or MPI/server estimates. Full typed results include cost,
errors, criteria, source hashes and commands; there are no large field sequences.

## Completion boundary and next action

Stage B now resolves its representation blocker in bounded fixed and moving,
constant and heterogeneous-coefficient controls. It does not integrate that
state into ASPECT. Surface s and curve/profile motion are prescribed analytic
functions here; native surface transport, solved kinematics, physical
irreversibility, MPI/AMR, rejection, restart, contact/RSF and evolving-phase
shape-work remain Stage C/D work. The separate cohesive pilot retains its native
FE remainder and lifting; do not transplant mature auxiliary memory into that
physical state or assume the mature history choice completes its derivation.

Next justified action is Stage C: add the selected R owner to the native mature
history transaction and consistent residual/source/tangent/observer path,
first on the declared affine fixed mesh, then integrate moving maps and native
surface state with lifecycle checks. Keep the total-spin/constant-G convention,
legacy defaults and old checkpoint rejection; do not merely remove a guard.
The existing cohesive shape/measure-work conflict remains isolated and visible.

Reproduce, sequentially from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/stage_b_moving.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/screen_moving_spaces.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/screen_surface_memory.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/qualify_stage_b.py
```

Python3/NumPy/Matplotlib only. No source build, native timestep, production
checkpoint change/conversion, server submission or long scientific trajectory.
[Preservation record](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving/verification.json)
checks current source, protected binaries/checkpoints, server outputs, cleanup
critical files and the entire prior B1 evidence. Historical cleanup losses and
failed BP3/RSF physical screens remain unchanged.
