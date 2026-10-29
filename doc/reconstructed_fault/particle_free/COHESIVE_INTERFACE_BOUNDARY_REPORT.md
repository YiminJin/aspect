# Boundary mechanism and finite-strip continuation

**Completed:** the refined native trajectory advances the fully decohered left
region **3.2135 m** by 2.58 s, through a ~1 m cohesive zone. The focused temporal,
physical-work and lifecycle checks pass for the declared finite-boundary problem.

The strong normal-pressure dependence is a **finite-wall, long-wavelength
incompressibility effect**, not a numerical pressure-gauge drift. An independent
mixed-strip calculation predicts a 635 Pa period-induced normal response to
.020 Pa RMS, at the same physical history. Native geometric reflection reproduces
the predicted flank swap. The longer experiment retains the original off-centre
fault and treats its periodicity and walls as physical boundary conditions.

[Selection](COHESIVE_INTERFACE_BOUNDARY_SELECTION.md) ·
[indexed figures/data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_boundary/INDEX.md) ·
[compact archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_boundary.zip) ·
[changed files](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_boundary/changed-files.json).
Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532`, branch
`pf-rsf-particle-free`, plus preserved local work. Production/BP3 and earlier
evidence are protected. The cohesive–RSF law and single fracture inventory are
unchanged. This remains the isolated native mixed/Schur component, not production
Stokes integration or a free-path solver.

## Why the two responses differ

Let b and t be bottom/top wall distances, H=b+t, d=t−b, and m(x) the accepted
relaxed-jump/Maxwell-memory field in metres. Perturbation displacement is zero
at both walls; their normal displacement is fixed. The cross-section flux
F=∫u_x dy satisfies F′=0 by incompressibility and closed normal contact. A slow
slip perturbation alone produces flux dm/2. Pressure-driven parabolic flow
adds −H³ p′_x/(12G). Thus nonzero horizontal modes need

    p′_x ≈ C(m−m̄), C=6Gd/H³,
    N(k) ≈ −i C/k,
    K(k→0,k≠0)=G/H+D, D=3Gd²/H³.

The **zero** mode permits a constant flux and has K(0)=G/H, N(0)=0. It must
not be replaced by the nonzero-mode limit. The exact biharmonic layer DtN
reference verifies these limits. High wavenumbers recover the equal-material
half-space response K≈G|k|, N≈0. Centring gives d=0 and eliminates normal
coupling by symmetry; reflection changes its sign. Increasing wall distances
equally at fixed physical offset reduces C as H⁻³. The H=2/H=4 reference
controls scale wall displacement with H, keeping the background Maxwell stress
history fixed; they do not retain an inequivalent wall speed.

For the same compact physical m, let A=∫m dx and J=∫(x−4)m dx. Changing the
period from 8 to 16 m gives, on the common interval,

    Δpn ≈ C[A(x−4)−J]/16,
    Δtau ≈ −D A/16.

With b=.4, t=.6, C=1.2e6 Pa/m² and D=1.2e5 Pa/m. The first expression is a
pressure ramp; the second is a small nearly uniform shear shift. At saved 2.47 s,
A=.00415126 m² and J=−.00163424 m³. The native same-memory shear change is
31.109 Pa RMS; the prediction is −31.134 Pa, with .0833 Pa RMS discrepancy.
The native normal change is 635.423 Pa RMS, predicted to **.0195 Pa RMS**
(maximum discrepancy .0773 Pa). Across common movable nodes it spans roughly
−948 to +1193 Pa. The sharp Q1 Galerkin reference's alias-tail change is
2.97e−7 relative. Relevant low-mode native/reference differences are far below
the observed boundary effect. High-mode mechanical discretization errors remain
explicit in the mode table.

The native algebraic pressure pin only removes the linear-system nullspace.
The code subsequently removes global mean p′ and adds **physical p0=10000 Pa**,
using pn=p0+p′−tau_yy in friction. Both original domains already have global
mean p′≈0. Removing the spatial ramp, prescribing friction pressure or changing
p0 would change the physical problem. No such change is made.

The accepted 8/16 m trajectory differences also include their different m,
damage, rate and age. The exported exact decomposition separates
`(N16−N8)m8` from `N16(m16−m8)`, and similarly for K, at common Gauss points.
Fixed-state friction-pressure changes are evaluated at the unchanged reference
damage/rate/age; they are labelled diagnostic partial variations, not new
constitutive states. Pressure raises right-flank resistance while relieving it
on the left; shear and nonlocal history redistribution compete with that effect.
The right cohesive-zone integral changes 39.7%, while the established left
full front changes only 22.1 mm. A fully decohered-front threshold and an
unsaturated cohesive flank are different response measures; neither a small
front difference nor energy closure implies pressure independence.
Using common represented Q1 flank tests (integral 1.9375 m), the same-state
normal changes average −656 Pa left and +901 Pa right. The corresponding
shear-minus-friction load perturbation is only **+8.84 Pa left**, versus
**−46.55 Pa right**: pressure relief offsets much of the shear reduction on
the left, while the terms compound on the right. The subsequent memory change
is separately resolved; the complete decomposition closes to 3.74e−11 Pa.

## Native control and specimen interpretation

The one new control is the 8×1 m fault reflected from y=.4 to .6 m, with the
same seed and **same positive loading**. Proper 180° rotation maps physical
coordinates (x,y)→(8−x,1−y), preserving the signed jump convention. It swaps
the normal ramp and developed flank without prescribing normal pressure.
The control passes the existing comparison allowances: maximum damage mismatch
2.14e−14, jump/memory 1.43e−17 m, surface pressure 3.82e−11 Pa, and actual bulk
stress/pressure 2.49e−10/5.85e−11 Pa. Rotated K/N differences are 7.14e−16 and
2.10e−13 relative. Runtime is 283.83 s; observed peak RSS 925 MiB.

The centred/reflected configurations are diagnostics, not replacements for the
longer specimen. The latter retains **L=16 m, H=1 m, y=.4 m**, physical
x∈[−4,12], the mature seed [2.5,5.5] and Γ=[−3.5,11.5]. Top/bottom have imposed
relative shear motion and zero normal displacement; horizontal boundaries are
periodic. The physical experiment is a **finite-height periodically repeated
fault specimen**, with p0 fixing the mean confinement. It does not model an
isolated fault in a distant-boundary body. At fixed H and offset, increasing L
alone retains the pressure-step/ramp mechanism; a distant-body interpretation
would require a different wall-distance/traction study. No domain-independence
requirement is imposed on these deliberately physical walls.

Retained parameters: G=1e6 Pa, η=1e8 Pa s, Gc=1 J/m², peak=3000 Pa;
RSF direct/evolution coefficients .005/.01, μ0=.1, V0=5e−6 m/s, Dc=.001 m.
Radiation uses G/(2√(G/2670)). The mesh uses mixed Q2/Q1 bulk mechanics,
Q1 signed jump and independent surface Gauss histories. Physical stress is
reconstructed from the accepted Maxwell memory, not identified with jump alone.

## Continuation protocol

The run used the retained accepted checkpoint at 2.2975 s, before the old loading end.
Only its future boundary plan changed: loading continued to 2.55 s, then hold
to 2.58 s. A guarded opt-in permits this forward extension only while the accepted
clock precedes the old endpoint. It does not convert histories or reinterpret
an accepted hold. The first wrapper attempt supplied `16` instead of checkpoint
string `16.0`; the unchanged strict configuration guard rejected it before
advancement. The preserved attempt is an input-format failure; matching the
original string fixes it without weakening that guard.

The mesh retained Δx=.0625 m, Δy=.0125 m and physical material/history conventions. Baseline
Δt=.00125 s through 2.45 s, then .000625 s; the checkpoint's pending first
interval remained .0025 s. A pre-event checkpoint was preserved and the new
window halved separately. The ≥1.5 m fully decohered-advance target was exceeded,
with a resolved cohesive zone and continued sliding during the .03 s hold. Existing contact,
curvature, derivative, weak, energy, remote-strength and buffer limits remain.
All original early spatial and boundary findings remain valid limitations.

## Developed propagation and temporal uncertainty

Both runs reached 2.58 s. The half-step run begins from the same accepted
2.42375 s checkpoint: its pending first .00125 s interval is shared, followed
by .000625 s to 2.45 and .0003125 s afterward. Comparisons use common physical
coordinates/times without branch reflection or history interpolation.

| At 2.58 s | Baseline | Half step | Absolute difference |
|---|---:|---:|---:|
| Left full advance (m) | 3.215226 | 3.213516 | .001710 m |
| Right full advance (m) | .097368 | .097372 | .0000037 m |
| New fracture inventory (J/m) | 4.206365 | 4.203567 | .002798 J/m (0.0666%) |
| Integrated absolute jump (m²) | .007530153 | .007529346 | 8.07e−7 m² (0.0107%) |
| Peak jump (mm) | 1.248884 | 1.249253 | .000369 mm |
| Wall reaction (kN/m) | 32.763939 | 32.764742 | .000802 kN/m |

There are **51 complete newly decohered fine cells** on the left (3.1875 m),
with the continuous front farther inside the next cell. The initial seed is
3 m long; its fracture inventory is unchanged. Left a≥.01 reaches x=−1.71875 m,
while the full front is x=−.71352 m. This is developed advance, distinct from
the ~1 m partial-damage zone ahead. No detached a≥.01 component appears on Γ.
Maximum phase increments decrease .02281→.01143 per step. Peak slip velocity
is .02438 m/s; it is distinct from front motion.

Final growth-relative L2 difference is .1384%; full 3D physical deviatoric-stress difference is
.0342%, and bulk normal-pressure RMS difference is .736 Pa. All selected temporal
targets pass. At the actual refined memory, the independent continuum-layer
solve gives near-tip native coupling RMS errors **1.10 Pa work-adjoint shear
load/.00210 Pa projected normal traction**, with maxima 3.72/.00269 Pa. These use
the same Q1 Galerkin projection as the implemented coupling. This measures bulk mechanical approximation
at fixed Q1 memory; it does **not** establish raw point-trace accuracy or convergence of the surface damage
space or erase the historical 65.4% early spatial failure.

![Actual accepted growth and sliding](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_boundary/longer-space-time.png)

*Imposed: fixed Γ, mature seed and finite-wall loading followed by displacement
hold. Computed: mixed mechanics, current-pressure RSF, signed jump and irreversible
surface damage. The cyan curve measures full decohesion; the wider coloured zone
includes partial damage. Free paths and off-path nucleation are untested.*

## Physical work and lifecycle

Use the raw trapezoidal work remainder, including changes of bulk/bond storage,
new fracture, friction, radiation and Maxwell loss. No background subtraction
enters acceptance. The initial 3 J/m seed inventory is counted once and excluded
from new fracture.

| Interval | Absolute remainder baseline→half (J/m) | / new fracture | / sliding loss |
|---|---:|---:|---:|
| 2.45–2.58 s | .020286→.010152 | .649→.325% | .501→.251% |
| Hold 2.55–2.58 s | .007472→.003741 | .763→.382% | .546→.274% |

Signed remainders have the opposite sign and essentially the same magnitude;
there is no hidden timestep cancellation. During the refined hold, external
work is zero, bulk storage falls **2.37022 J/m**, bond storage rises .00639,
new fracture gains .97850, friction dissipates 1.04217, radiation .32443 and
Maxwell .02248 J/m. Sliding and growth persist using stored mechanical energy;
holding displacement does not remove stress. Exact BE ledger discrepancies
remain ≤1.76e−12 J/m, separately from these physical-work errors.

All accepted histories publish once. The shared past through 2.44875 s is
bitwise; at the old 2.45 endpoint the changed clock-snapping landmark differs
1.47e−14 s, damage 2.37e−13 and shear 6.57e−10 Pa. No clock or history is rewritten.
The baseline command hit its **1500 s wall budget** at accepted 2.57625 s;
a separate native restart from 2.574375 completed the endpoint. Its three
replayed accepted states match **all physical columns and histories bitwise**.
An attempted future-load change after an accepted hold fails before publication.

Weak residuals stay <1e−8 Pa, accepted derivative errors <1e−6 and the retained
tested curvature stays positive. Contact remains ≥7.72 kPa; final bulk Gauss6
horizontal compression stays ≥7.33 kPa. Uninstalled-Γ weak shear stays below
2574 Pa, versus the 3000 Pa intact peak. No physical/solver guard was waived.

| Native command | Wall time | Setup time |
|---|---:|---:|
| Reflected diagnostic | 283.83 s | 131.07 s |
| Longer, administrative stop | 1500.68 s | 745.90 s |
| Checkpoint completion | 848.75 s | 755.92 s |
| Half-step event/hold | 2077.57 s | 740.20 s |

Including the two short rejected-input checks, total native cost is **78.6 min**.
Observed peak RSS is ~2120 MiB (reflected case: 925 MiB). Runs were sequential,
one rank/thread, with preserved checkpoints. Repeated fixed-operator rebuilding
and full derivative checks are substantial costs; no full-event timing follows.

## What is established and what remains

The boundary mechanism is explained and independently reproduced. A native,
time-controlled **finite-height periodic** cohesive–RSF trajectory now demonstrates
resolved full decohesion, a sustained cohesive zone and sliding/growth in a hold.
The earlier domain differences remain physical differences between these specimens;
they were not removed by pressure fitting, lagging or a centred replacement.

Spatial accuracy of the evolving surface still needs a focused reference.
The bulk is quasi-static with radiation damping, so the measured model front
kinematics are **not validated physical rupture speeds**. The refined hold
secant is 28.95 m/s; the nominal shear speed used only in the radiation coefficient
is 19.35 m/s. This comparison supplies no wave-causality or supershear qualification.
A physical-speed interpretation requires an explicit bulk-dynamics study.

Recommend a cheap surface/jump-resolution comparison using the existing analytical
mixed strip before further quantitative front claims. A distant-body specimen
requires a separately stated wall-distance/traction problem. Production/BP3,
all 1,051 protected files and 11 earlier archives retain their entry hashes.
No further loading, free path, bend sliding, MPI/AMR or transport starts here.
