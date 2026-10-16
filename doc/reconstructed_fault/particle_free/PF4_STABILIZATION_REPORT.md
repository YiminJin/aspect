# PF4 timestep, scaling and finite-width comparison — 16 October 2026 UTC

**All four selected tasks are complete.** Keep unlimited DG as the resolved
smooth baseline; retain both WENO paths as experiments. Scaling is substantially
less diffusive than these WENO paths on the signed pulse, but still less accurate
than unlimited and requires a justified envelope. The resolved finite-width
control provides **no evidence that more elaborate stabilization is needed**.
Neither WENO path is temporally qualified by this bounded comparison.

Worktree/branch/HEAD remain `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free` /
`3232056cde12dbae17ef4265ecb804f9098b88f1` plus incoming work. Root AGENTS.md and
selected design/spec sections were read; no nested override applies. Incoming
instructions, derivation, physical laws/defaults and PF0–PF4 evidence are preserved.
No commit, push, PF5, coupled BP3 or propagation work. Earlier
[limiter results](PF4_LIMITER_REPORT.md) remain unchanged historical evidence.

## Fixed comparison and scaling formula

Use the same serial affine Cartesian DGQ2 space, native Q2 velocities, upwind
material operator, Gauss3 mass, Gauss4 transport/inflow, Gauss8 initial projection
and SSPRK3 stage schedule. Outer time/history remain unadvanced; only independent
scratch tensors are transported to T=.2s. No initial limiting.
These pure-advection controls retain the earlier diagnostic choice r=0 and zero
volume source; the native Maxwell rate and other physical laws are unchanged.
WENO thresholds
KXRCF1 and safeguarded Fu–Shu.1, reconstruction weights, smoothness and quadrature
are unchanged; there is no new threshold search.

For benchmark tensor `B=(1000,-400,700)Pa`, zz=-(xx+yy), use

```
<A,B>3 = Axx Bxx + Ayy Byy + (Axx+Ayy)(Bxx+Byy) + 2 Axy Bxy
s(A) = <A,B>3 / |B|3²,       |B|3² = 2.5e6 Pa²
Anew = Amean + theta*(A-Amean),       0 <= theta <= 1.
```

The signed pulse has the known scalar envelope -1≤s≤1. At the specified
Gauss-Lobatto9×9 control points, choose the **largest common theta**:
start at1 and take the minimum of `(1-smean)/(s-smean)` for upper violations
and `(-1-smean)/(s-smean)` for lower violations. All tensor coefficients/modes
use this one factor. The target physical tensor mean is unchanged; there is no
component clipping, neighbor reconstruction or chi filtering.

A comparison allowance128 machine epsilons suppresses spurious constant-state
activation. Means outside the exact envelope are counted with maximum excess;
excess beyond this allowance rejects the candidate explicitly. An out-of-envelope
mean cannot be repaired by scaling about it and is never clipped. **Every actual
scaling stage-cell mean in this family was inside the envelope: count0, maximum
excess0.** A deliberate mean1.1 fixture is rejected. These bounds/reference exist
only in the private benchmark experiment, not as production stress admissibility.

For the actual finite-width analytic pulse, with x*=x-.4t,

```
g(x*;x0) = .5*(1+tanh((x*-x0)/delta))
f(x,t)  = g(x*;.25) - 2*g(x*;.5) + g(x*;.75)
delta   = ell/ln(9) = ell/(2*atanh(.8)),       ell = .15625 m.
```

Ell is read from the existing phase handler; it is not duplicated or changed.
Each underlying transition has10–90% width ell. The **entire overlapping pulse**
is fixed across meshes and used for source/inflow and independent error integrals.
Its actual peak is918.505516Pa, not1000Pa. Its three actual analytic transition
widths are .139199795/.135360618/.139199795m. The stress-width choice is a
prescribed control at an existing physical length, not a derivation that the
constitutive law imposes this tangential stress width.

Meshes are16×32 and32×64: ell spans2.5 and5 tangential cells. On32×64, use103/206/
412 steps, dt=.001941748/.000970874/.000485437s, for sharp and finite-width
pulses; Gaussian smooth baseline uses205/410/820 steps and velocity(.4,.2).
Pulse velocity is(.4,0). Projections are byte-identical across timestep levels.
Fields are also compared at common times .0504854/.100971/.151456/.2s.

## Direct limiter-call norm measurement

Let Q(A)=integral|A|3², with the full embedded tensor metric above. Each limiter
call records Qbefore,Qafter and
`deltaQ=integral(2<A,dA>3+|dA|3²)` at Gauss3. This stable difference avoids
subtracting large norms; actual L2 changes are deltaQ/(sqrt(Qbefore)+sqrt(Qafter)).
The scalar-envelope limiter obeys the exact cell identity
`deltaQ=(theta²-1)*integral|A-Amean|3²≤0`. Independent non-proportional tensor
checks verify this identity as well as the reconstruction.

Raw stage-call sums are **not** net physical dissipation. For SSPRK3 write
`zi` for each raw stage and `yi=L(zi)`, with E(u)=u+dtF(u). The exact quadratic
identity weights direct limiter changes by `(1/6,2/3,1)`. More explicitly,

```
Q(y3)-Q(un) = delta1/6 + 2*delta2/3 + delta3
 + (e0+e1)/6 + 2*e2/3
 - Q(un-E(y1))/8 - 2*Q(un-E(y2))/9,
e0=Q(E(un))-Q(un), ei=Q(E(yi))-Q(yi), i=1,2.
```

The remaining terms combine Euler advection/boundaries and SSP convex mixtures.
A separate64-case independent full-tensor algebra check has scaled defect
3.65e-16. These are algorithm ledgers, not a mechanical-energy balance or
counterfactual attribution of the effect of disabling a limiter.

At32×64, original dt, sharp pulse (all entries relative to initial Q):

| Path | Raw sum of call changes | RK-weighted direct limiter term | Net Q change | Remaining transport/RK term |
|---|---:|---:|---:|---:|
|KXRCF/WENO|−12.3202%|−6.2941%|−28.9484%|−22.6542%|
|Fu–Shu/WENO|−12.3162%|−6.3009%|−28.9033%|−22.6024%|
|Scaling|−18.9651%|−9.5954%|−12.5489%|−2.9535%|

Thus the larger total WENO norm loss is not a larger direct-call contraction.
Limiting changes the fields subsequently seen by the transport/RK operations.
WENO is not locally norm-contracting: even the sharp FS run has103168 positive
cell-call changes despite negative global call totals. For the finite-width
KXRCF run, **257/309 global calls increase Q**, with RK-weighted total+0.006651%.
Scaling has no positive change beyond roundoff and is idle on every finite-width
case. Norm/boundary reference changes are kept separate: at original dt the
finite-width unlimited Q change is−6.4761Pa²m², of which the analytic open-boundary
change is−3.0115 and the numerical excess−3.4646. It is not a closed-domain test.

## Sharp-pulse accuracy, amplitude and broadening

At32×64, original dt, sampled extrema use the same GL9×9 rule as PF4. Norms/errors
include zz and both xy terms. Signed-reference error integration is split at the
three analytic jumps before Gauss24/48 quadrature.

| Path | Relative tensor L2 | L1, Pa m² | Sampled over/undershoot | Q loss | Three10–90% widths, m |
|---|---:|---:|---:|---:|---|
|Unlimited|.17755|67.331|16.503/15.822%|3.574%|.02353/.02353/.02353|
|KXRCF/WENO|.37629|227.178|.0161/.0241%|28.948%|.09967/.09985/.09695|
|Fu–Shu/WENO|.37658|227.125|.0488/.0223%|28.903%|.09686/.09671/.09662|
|Scaling|.22675|91.211|≤1.4e-13% /≤1.4e-13%|12.549%|.02655/.05008/.04761|

Plateau peak attenuation is zero to roundoff for all four fine paths; lower
overshoot does not imply lower full-field error or preserved front sharpness.
Scaling roughly halves WENO's norm loss and substantially reduces smearing, but
its middle/trailing fronts are still wider than unlimited. On16×32 its L2 error
is.31795 and Q loss24.245%, versus WENO .4714–.4718 and approximately43.0% loss.

Scaling enforces the requested **control-point** envelope, not a continuous
maximum principle: exact one-dimensional quadratic trace extrema reveal a
.00176% overshoot between GL9 points at original dt (.00160% at dt/4). Numerical
transverse variation remains≤7.24e-10Pa across these one-dimensional controls.
Peak attenuation uses polynomial trace extrema rather than relying solely on
sample locations. The known global envelope also permits local ringing inside
[-1,1]; it is not a local TV bound or a general tensor constitutive constraint.

## Fixed-mesh timestep sensitivity

Endpoint full-tensor differences are normalized by sqrt(initial Q), with identical
mesh/physics/initialization at all levels:

| Case/path | dt versus dt/2 | dt/2 versus dt/4 |
|---|---:|---:|
|Gaussian unlimited|8.465e-9|1.058e-9|
|Sharp unlimited|2.685e-5|3.344e-6|
|Sharp KXRCF/WENO|.014138|.131572|
|Sharp Fu–Shu/WENO|.013526|.129804|
|Sharp scaling|.0006057|.0002812|
|Finite-width unlimited = scaling|1.342e-7|1.677e-8|
|Finite-width KXRCF/WENO|.0005237|.051660|
|Finite-width Fu–Shu/WENO|.00006014|.00006550|

Unlimited differences decrease by approximately8, consistent with third-order
fixed-space SSPRK3. Scaling sharp differences decrease approximately2.15;
maximum dt/2-versus-dt/4 difference over common times is.03615%, compared with
12.98–13.16% for sharp WENO. No third-order switching-limiter claim is made.
The bounded WENO family is not temporally resolved; these results do not prove
that a finer-step limit cannot exist. No dt/8 expansion or retuning followed.
Sharp final L2 errors at dt/4 are.17755/.43626/.43250/.22631 in table path order.

A named held-field diagnostic applies the unchanged limiter12 times at fixed
physical time, fixed detector velocity and boundary data, **without transport**.
Starting from the dt/4 final sharp field, KXRCF and FS change the tensor field
by5.51% and5.06% of initial norm; scaling changes it exactly0. Finite-width held
changes are.14725%/.00573%/0. This independently verifies non-idempotent WENO
map effects and application-frequency sensitivity, without claiming it explains
every switching event. The held candidates are never committed or used to
continue the measured trajectories. All first held maps are checked independently.

## Finite-width result, including actual overlapping-front amplitudes

| Path / timestep | Relative tensor L2 | Positive peak attenuation | First-front broadening |
|---|---:|---:|---:|
|Unlimited = scaling, dt|.00042764|.003709%|+.02890%|
|KXRCF/WENO, dt|.00055178|.003709%|+.05534%|
|Fu–Shu/WENO, dt|.00044103|.003709%|+.03989%|
|Unlimited = scaling, dt/4|.00042762|.003715%|+.02889%|
|KXRCF/WENO, dt/4|.051565|.009752%|+15.0868%|
|Fu–Shu/WENO, dt/4|.00047542|.003715%|+.03194%|

Negative peak attenuation is0: small dispersive undershoot remains. At dt/4,
unlimited polynomial undershoot relative to the **actual analytic peak** is
.002024%; sampled undershoot is.000966%. Finite-width numerical maxima remain
well inside the benchmark envelope±1000Pa, so scaling is exactly inactive and
its fields are byte-identical to unlimited on both meshes and all timesteps.

Actual finite-width front crossings are compared with the full analytic pulse,
using its peak P: levels .1P/.9P, .8P/−.8P, −.9P/−.1P on the three monotone
branches. Quadratic roots and trace jumps are used, with no mesh-dependent
smoothing. Unlimited dt/4 widths are .139240012/.135314468/.139216483m against
the analytic widths above. KXRCF's first width grows to.160200533m despite very
small peak attenuation: a peak-only assessment would miss the field damage.
On16×32 unlimited finite-width error is.0031502; refinement to32×64 gives rate2.881.
The coarse FS finite-width error is.103714, illustrating resolution dependence.

For this five-cell-wide physical-length control, reducing dt barely affects
unlimited and stabilization adds no accuracy benefit. This supports resolving
finite histories before adding more elaborate stabilization; it does not
establish an acceptable mechanical overshoot tolerance or qualify other widths,
coefficients, meshes or coupled histories.

## Verification, cost and remaining scope

**149/149 final transport/ledger checks,45/45 scaling/algebra checks, and77/77
retained WENO/local-snapshot checks pass.** Independent references check every
captured first-call reconstruction and direct norm delta, target means, largest
common theta, constants/zero, affine and Q2 including x²y², non-proportional
components, in-plane tensor-basis covariance, ordering, idempotence and rejection.
The maximum measured stage mean error is1.07e-15 in the common nodal scale.
All incoming and accepted DG/phase/Z/I remain unchanged. Native cohesive/Maxwell
628 assertions/8 cases and PF1's94 identities pass; both production DG translation
units compile without unity/PCH. Full PF2/PF3 native regressions were not repeated.

Fresh original PF4 operator, loads, metrics and publication/lifecycle records
are byte-identical. Native independent operator errors remain≤1.56e-15 scaled.
New base-step unlimited and WENO signed fields are byte-identical to prior PF4/
limiter evidence. Adding explicit mean-excess logging changes **none** of the36
measured final fields; both preliminary and final families are preserved.

Original-dt32×64 advancement costs (seconds, including norm instrumentation,
excluding diagnostic field export/setup): sharp unlimited .407, KXRCF2.548,
FS3.597, scaling1.201. Per limited path, norm measurement costs.111–.117s;
scaling reconstruction itself .0067s, versus WENO .877/1.449s. These are short
serial instrumented measurements, not general performance estimates.

Final16/base,32/base,32/dt2,32/dt4 runs take12.087/39.292/118.492/164.762s;
peak child RSS258228/448120/447808/447652KiB. Every run is bounded180s with at
most two native simulations at once. All final runs share core SHA256
`3ee53a8645b2975db3e21b7de1e9304754531fc9e7c9a17b6893a2e073881718`, GCC12.4,
deal.II9.6.2, Trilinos14.2, OpenMPI5.0.10 Release, one thread/rank.

`dg_limiter.cc` adds the private scaling branch and optional norm/capture diagnostics;
`dg_history_internal.h` holds only scratch stencil/statistics additions. The
existing transport/SSPRK3 numerical code is unchanged. `stabilization_checks.cc`
and `assess_stabilization.py` own the bounded fixtures and independent references;
matched design/spec addenda record only this selected extension.

Particle dependencies are unchanged: startup fixed-particle validation, H/CPDI/
domain initialization, support/association, surface material projection and
derived sampling still require particles. Limiting/advancement read no particle
properties. Accepted publication, surface-history/phase-bound transport, restart,
AMR/MPI/periodic/moving-geometry/coupled transport remain guarded or unimplemented.
No physical laws, default parameters, spin convention, generic projector or
localization have changed. No mechanically meaningful general stress envelope
has been established.

[Portable evidence and replay commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_stabilization/README.md)
include full settings, per-call data, common-time fields, costs, source identities,
preservation checks and a [measured comparison figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_stabilization/comparison.png).
**Stop here:** retain unlimited for resolved smooth/finite-width controls and all
limiters as experiments. No additional stabilization, production adoption or
coupled/propagation stage is authorized by these results.
