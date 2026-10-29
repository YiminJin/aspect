# Coulomb direction selection: bounded point qualification

Completed on `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local
work. **The experimental point return passes; direct native adoption is not
yet supported.** Mechanical confinement/contact semantics, strength calibration
and a non-associated stability acceptance rule require decisions first. No
native constitutive option, specimen step, mesh, default or BP3 behavior changed.

## What the current radial law permits

Actual native callers confirm that `in-plane radial tensor` passes a null
fixed-direction pointer to `TensorReturn::update`; only the fixed-tensor control
passes S0. The radial projection is the full two-component in-plane trace-free
space, not a fixed xy projection. Full3D plane-strain T/stress, including zz,
remain. A/K are physical tensor histories, not stored directors.

Bulk eligibility excludes mature-fault and reconstructed-surface paths. The
legacy normal-lock threshold printed in parameters is not consulted here;
the scalar xy projections are diagnostics and do not feed mechanics. Phase
evolves throughout the bulk, subject to its physical obstacle/bounds and AT1
terms. There is no phase-growth mask or phase-gradient direction lock.

The current return maximizes effective shear from tau−Q against the constant
R=mu p_N=1000 Pa. Changing this scalar resistance does not introduce a Coulomb
plane angle. Physical stress, the orientation of net cohesive flow W, and the
geometry of phi contours are distinct. A signed shear tensor has two equivalent
orthogonal plane labels; neither is a unique crack normal. Freedom is in-plane
and small-strain, not unrestricted 3D or finite-rotation physics.

## Archived diagnostics and the pressure distinction

Four same-rank restore-only reads use radial133/.1425 s, radial176/.169375 s,
radial177/.170 s and fixed176/.169375 s. A public FEValues observer reads solved
pressure after complete restore; existing termination checks return **before any
physical step**. Checkpoint hashes/publications remain unchanged. Total native
read time is 24.15 s. Stress and W are the archived Gauss8 cell means; pressure
uses the identical rule. Principal axes are those of the cell-mean Cauchy tensor,
not constitutive evaluations on averaged histories.

The retained mechanical weak stress is sigma=tau−p_mech Id, tensile positive.
Near-tip mean p_mech ranges are ±1089.35 Pa before activation and ±1450.60 Pa
at the radial target; matched fixed values reach ±1525.64 Pa. **p_N=10000 Pa
sets friction resistance; it is not added to those mechanical weak loads.**
Released lateral normal velocity gives natural total traction, so adding a
pressure constant changes that traction reference, not just a pressure gauge.

None of 2,166 sampled cell-mean states per snapshot is compressive in every
in-plane direction. This does not count individual quadrature-point contacts,
but prevents blanket qualification of this specimen for the deliberately fully
closed point option. At the illustrated target cell (1.089844,.496094) m,
p_mech=−111.4 Pa. The plotted +10000 Pa offset is **hypothetical added mechanical
confinement**, holding archived tau for a traction diagnostic; it is not a
reequilibrated solution or an inferred native background stress. Choosing such
a background requires explicit stress/boundary-traction semantics.

![Stress, flow and candidate planes](../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_direction/directions.png)

White contours are actual phase geometry. Black marks the major in-plane
compression axis; cyan gives equivalent shear representations of measured net
W, only where defined. Orange gives bare-physical-stress Coulomb candidates,
dashed where normal pressure is nonpositive. These candidates omit Q; the
actual current law uses tau−Q. The bottom panels separate mechanical pressure,
the constant-resistance reference and hypothetical confinement, and show the
independent analytical-angle tests. [Compact selected directions CSV](../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_direction/archived-directions.csv),
archived-cells.csv and traction-angle.csv preserve values, coordinates and units.

## Closed-contact experimental point law

[Derivation and complete update](COULOMB_DIRECTION_FORMULATION.md) were written
before implementation. Use pn=p_mech−n.tau.n and
f(theta)=|(tau−Q):S(theta)|−c_r−mu pn<=0. The bounded option rejects tensile
trial/returned states explicitly; it does not clip pressure or hide them with a
phase mask. Flow on each active plane is tangential, with zero dilation and no
opening/volumetric variable. Both conjugate planes may be active. Selecting one
arbitrarily merely to tilt a path is not the implemented closure.

Keep beta, Tplus, h(phi), A/Q identity and irreversibility. The new increment
adds to beta Aold; K keeps its old tensor and L its net tensor-strain meaning.
No accepted tensor rotates or disappears. Plane-slip sum can exceed net Ldot;
friction power is sum(c_r+mu pn_k)w_k=(tau−Q):W>=0, rather than R Ldot.
The existing storage, viscous terms, intact driving limit and constitutive BE
work identity remain. The point fixture prescribes admissible phi; it does not
solve a phase-evolution problem.

The zero-dilation assumption is non-associated. Friction/dilation are distinct
in standard [Mohr–Coulomb conventions](https://docs.itascacg.com/itasca930/common/models/mohr/doc/modelmohr.html);
this tensor-memory update is the separately derived experimental law here.
Its consistent tangent has measured major asymmetry **.357573 G**. Consequently
the current incremental potential and symmetric mechanically relaxed Hessian
certificate do **not** carry over. Local energy closure is not a substitute.
Native integration would require the pressure derivatives of stress and Y,
the full nonsymmetric coupled action, and a reviewed stability criterion.

Residual cohesion c_r=0 is a **new residual-friction parameter**, not setting the
retained 2000 Pa bulk strength-drop input to zero. Retain G/eta/Gc/ell/m=4.
At the explicitly controlled physical pressure 10000 Pa:

| mu | Coulomb plane angle to major compression axis | Pure-shear residual [Pa] | Intact phase threshold [Pa] |
|---:|---:|---:|---:|
| .1 | 42.144703° | 995.037 | 2955.332 |
| .6, diagnostic material | 29.518122° | 5144.958 | 6539.320 |

Angle is alpha=45°−atan(mu)/2 to the compression **axis**, or 45°+atan(mu)/2
to its normal plane. Plane tangents are axes modulo180°; angles to global x,
normal directions and shear signs are explicit in angles.csv. The .1/.6 offsets
from maximum-shear planes are 2.855297°/15.481878°. With nonzero Q, use the full
generalized traction criterion, not the bare-stress angle formula. Rotated and
reversed-shear tests check both conjugates. Symmetric virgin loading activates
both and leaves the net flow radial: even tilted Coulomb planes need not select
a tilted phase path.

Keeping m=4 does not preserve the old 1000/3000 Pa strength pair under a new
pressure law. Matching that pair at p=10000 Pa, mu=.1 would instead need
c_r=4.987562 Pa and m=3.841535, with corresponding seed/degradation review.
That recalibration is **not applied**. At mu=.6 the same residual target would
require negative c_r and is inadmissible with this reference pressure.

## Point evidence and native readiness

| Check | Measured maximum |
|---|---:|
| 24 known-principal-stress plane tests, sin angle error | 1.23e-16 |
| 15 independent 70-digit trigonometric equation solves, scaled Q defect | 8.91e-16 |
| Constitutive BE ledger / plane-power identity | 4.89e-16 / 2.06e-15 |
| Complete strain/phase/pressure action FD | 8.76e-9 |
| Plane-direction / normal-traction FD | 2.12e-8 / 2.13e-9 |
| Intact right-phase FD, including tiny h | 1.51e-9 |
| Coordinate covariance / reversed shear | 1.71e-14 / exact |
| Constant-resistance radial identity | exact |

Existing bounds remain 1e-12 for scaled identities and 1e-6 for FD actions.
All 84 cases exercise stick 32, plus-only 5, minus-only 4 and two-plane 43.
Derivatives include changing directions, multipliers, system slip, margins and
energy/dissipation. Yield ties use a declared sticking generalized derivative;
classical derivatives are not claimed across active-set changes.

The 60-step non-proportional point history preserves incoming state through
discard/retry, publishes 60 accepted copies, and restarts at 30 with the next 30
states bitwise identical and zero response difference. Tensile/unconfined input is refused. A separate backstress-release
control also rejects a noncompressive returned state despite its closed trial,
including the zero-friction identity path. The original constant-R point fixture also passes unchanged,
including its owner/restart controls. No Coulomb native checkpoint conversion,
coupled phase, global stability or propagation qualification is claimed.

**Recommendation:** point evidence supports a narrowly scoped native-operator
prototype only after choosing pressure/confinement and calibration semantics.
First verify a fully confined, fixed-phase small block's nonsymmetric mixed
tangent and pressure couplings. Phase-active adoption additionally needs a new,
reviewed acceptance/stability formulation; do not disable the retained Hessian
guard to bypass that dependency. No such native run is begun here.

The 52 entry source/artifact hashes, production defaults, BP3 and all earlier
P5c/P4 evidence remain. Existing spatial, temporal and boundary limitations and
**strict_reference_match_passed=false** are unchanged. New files are benchmark
point/restore diagnostics, build targets and this documentation. Full copied
checkpoints stay under build-particle-free/coulomb-direction; the compact
[review archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_direction.zip)
contains the PNG, CSV/JSON, source context, commands and checksums.
