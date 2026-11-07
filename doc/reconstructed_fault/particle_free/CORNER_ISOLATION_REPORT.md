# Upper-left corner isolation

**The experiment separates an inflow-model error from the large coupled corner
amplification.** Native DGQ2/SSPRK3 with exact exterior data reproduces the
prescribed discrete solution within **8.1e-15 relative L2**. Frozen accepted-interior
trace gives a persistent boundary bias, but does not reproduce the large stress
growth. With that same held-trace inflow retained, replacing only the zero top
traction by a compatible shear-background traction reduces the coupled temporal
stress difference from **22.36% to .608% globally**, and **61.89% to .620% in the
upper-left corner**. The peak sampled stress falls from 58.8 to 2.34 Pa.

This establishes sensitivity to the coupled mechanical boundary problem; it
**does not uniquely separate mixed-boundary corner regularity from inflow
feedback in the original non-affine velocity field**. The compatible loading
also changes fault slip substantially. It is a diagnostic control, not a silent
repair of the original free-top physical problem.

[Protocol](CORNER_ISOLATION_PROTOCOL.md) ·
[Evidence/reproduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_isolation/README.md) ·
[Figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_isolation/corner-isolation.png) ·
[Original shear report](SHEARED_MOVING_FAULT_REPORT.md)

## Preserved state and implemented experiment

Root `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, revision
`cf06525471ce89789f1ac10905dc79f3aac7562a`. Existing dirty research documentation,
untracked/manual work, outputs and checkpoints remain. The six current binaries,
63 protected artifacts and all four original shear trajectories match their
recorded entry hashes. Native source, CMake, tests and default parameters are
unchanged. No production/server job or checkpoint conversion ran.

A separate benchmark DSO under `transported_objective/corner_inflow/` uses the
public native `TensorAdvection` and `TensorField` operations, the existing finite
rotation helper and `mature_stress_endpoint`. It constructs scratch numerical
tensors, not accepted physical histories. Its compatible-traction plugin uses
the ordinary ASPECT boundary-traction interface. The original solver and material
transaction still own all coupled evolution. Build used at most two workers;
native executions used four MPI ranks sequentially. Python/BLAS used one thread.

## 1. Prescribed affine-flow reference

Domain [-2,2]², v=(.4+.5y,0) m/s, final time .4 s. Initial physical deviatoric
stress is (xx,yy,xy)=f*(1,-.3,.45) Pa with
f=1+.15x+.1y+.08xy, explicitly continued outside the domain. Its affine
characteristic pullback remains in Cartesian Q2, including the generated y²
term. Full 3-D plane strain supplies zz=-xx-yy in every norm and energy.

The advection-only reference is the initial tensor at (x-(.4+.5y)t,y).
For Maxwell/Jaumann evolution use G=eta=10, no localized slip,
W_xy=.25 s^-1 and D_xy=.25 s^-1. Writing
m=(tau_xx+tau_yy)/2 and z=(tau_xx-tau_yy)/2+i*tau_xy gives

```
m(t) = exp(-t) m0(characteristic foot)
z(t) = exp(lambda*t) z0(foot) + 5i*(exp(lambda*t)-1)/lambda
lambda = -1-.5i
```

An independent discrete recurrence supplies the split reference:
r=exp((-1-.5i)*dt), F=5*(1-exp(-dt)),
z_n=r^n*z0(foot)+iF*(1-r^n)/(1-r). The native update transports, rotates,
then applies exponential endpoint forcing. Constant spin/coefficients keep the
result in Q2, so the scratch combined rotation/reaction projection is algebraically
equivalent to separate projections. This tests the native numerical kernels and
boundary treatment; it is not a duplicate mature timestep driver.

During each transport interval, exact inflow continues the **start-of-interval
split state** along characteristics. Supplying the fully reacted continuous
stress at SSPRK stage times would double-count/reorder the reaction. Held trace
instead freezes the accepted interior polynomial at the boundary, matching the
existing mature inflow convention. Both use the unchanged SSPRK3 CFL subcycle.

At 32², dt=.1/.05/.025 s tests both inflows and both laws; at 64² only dt=.025 s
is added. These 16 scratch evolutions run inside two initialization-only native
executions and publish no physical timestep. Exact-reference threshold was 1e-9;
no threshold is imposed on held-trace bias. Upper-left corner is
x<-1.5,y>1.5; the interior window is |x|,|y|<1.5.

Final relative errors below are percentages. Held-trace values are differences
from the explicitly continued exterior reference, not a pure discretization error.

| Mesh / dt | Advection, held trace: whole / corner | Maxwell, held trace: discrete whole / corner | Maxwell, exact inflow: continuous whole |
|---|---:|---:|---:|
| 32² / 0.1 | 1.4010 / 16.9051 | 0.4638 / 3.2679 | 2.04032 |
| 32² / 0.05 | 1.4015 / 16.9677 | 0.4629 / 3.2752 | 1.02878 |
| 32² / 0.025 | 1.3882 / 16.8755 | 0.4580 / 3.2550 | 0.51655 |
| 64² / 0.025 | 1.3948 / 16.8625 | 0.4602 / 3.2525 | 0.51655 |

With exact inflow, the maximum final **discrete** reference error is 8.084e-15
across both laws and both grids. The continuous Maxwell errors halve with orders
.988 and .994, separating retained first-order forcing/Lie error from inflow.
The held-trace corner bias stays near 16.9% for advection and 3.25% for the split
Maxwell case; halving dt or h does not remove it. At 32²/dt=.025 its interior
errors are tiny compared with its corner error. Peak stresses remain bounded
(about 2.32 Pa for advection and 3.25 Pa for Maxwell), unlike the original coupled
58.8 Pa corner maximum.

The independent semidiscrete balance uses supplied upwind boundary values,
interior jumps and the native q8 face sign rule:
`dE/dt + F_supplied + D_upwind = 0`, E=integral(tau:tau/(4G)).
The largest identity defect is 2.0e-15 W/m. Substep-trapezoidal transport-balance
remainders with held trace are at most 3.04e-8 J/m accumulated over the interval;
the 64² counterparts decrease to 7.57e-9 or 5.33e-9 J/m. Exact-inflow balances
are at roundoff. Interior-trace flux is also retained separately: it is not
substituted for the supplied upwind boundary flux in this operator identity.
Small numerical transport-balance residuals do not erase the held-trace model bias.

The 32²/64² executions took 36.99/85.60 s, including simulator initialization
and scratch diagnostics. These are diagnostic costs, not production step costs.

## 2. Coupled mechanical-corner control

The original boundary prescribes affine velocity on left/right/bottom and zero
traction on top. The control retains the three velocity boundaries and sets
only top traction to the **uniform background of the same discrete stress law**,
at reference pressure p=0. For zero initial physical stress it uses

```
z_n = iF*(1-r^n)/(1-r)
t_top = (Im(z_n), -Re(z_n))
```

At initialization the separate .1 s constitutive interval supplies
(5*(1-exp(-.1)),0) Pa. This avoids an incompatible unloaded top during the
initial mechanical solve. It also retains the natural mechanical pressure
reference; no all-Dirichlet pressure-gauge change was introduced. An independent
real-component recurrence verifies the plugin's complex formula to 1.78e-15 Pa.
Fault effects remain coupled, not prescribed to the uniform background.

Both controls retain 32², 18 initial material labels at spacing .1 m, G=eta=10,
the same phase/age/fault parameters, direct ordinary DGQ2 stress, bounded Q1
profile transport, accepted-trace inflow, native quadrature and unchanged solve
criteria. Run to .4 s at dt=.05/.025. The discrete traction histories themselves
vary with dt and approach the same continuous background: their pair includes
boundary-data discretization as well as the interior scheme. Two levels do not
establish an order. No coupled mesh matrix was started.

The compatible temporal controls have bitwise-identical initial bulk/surface
exports. Across different boundary conditions, initial slip rates differ 16.82%
in whole-curve L2 relative to the compatible case. Thus original-versus-compatible
field differences are physical loading responses, not accuracy errors.

| Coupled boundary / dt | Peak sampled tensor norm (Pa) | Final net outward energy flux (W/m) | Endpoint remainder / active work | Trapezoid diagnostic / active work |
|---|---:|---:|---:|---:|
| Original / .05 | 27.93990 | -0.586214 | 7.5987% | 1.1882% |
| Original / .025 | 58.79606 | -8.017413 | 11.0005% | 5.7882% |
| Compatible / .05 | 2.34488 | 0.001354 | 6.2226% | 0.1407% |
| Compatible / .025 | 2.34288 | 0.001231 | 3.1605% | 0.0445% |

The compatible maxima move to the fault vicinity, around (-.866,-.241) m;
they are no longer at the domain corner. Its largest sampled speed stays near
1.396 m/s, whereas the original dt=.025 run reached 2.026 m/s. Label spin remains
appreciable, about-.1009 to-.1027 rad, and curve clearance exceeds .9919 m.

The upper-left corner contains 99.858% of the original timestep pair's squared
stress difference but only 1.620% of the compatible pair. Whole-domain absolute
L2 difference drops 2.04933 → .056512 Pa m; corner difference drops 2.04787 →
.007192 Pa m. The finite-fault tube difference changes little (.5934% → .5856%),
showing that the earlier global failure was largely separate from local fault
approximation error.

Near-fault central/combined-end temporal differences, percentages unless noted:

| Quantity | Original .05 versus .025 | Compatible .05 versus .025 |
|---|---:|---:|
| Tensor stress | 0.5616 / 0.6243 | 0.5525 / 0.6173 |
| Shear traction | 0.1813 / 0.3973 | 0.1552 / 0.3884 |
| Normal traction including pressure | 0.6465 / 0.7991 | 0.8307 / 0.9006 |
| Slip rate | 1.6367 / 1.4570 | 2.9375 / 5.9318 |
| Cumulative slip | 12.2784 / 16.3859 | 13.9200 / 17.4209 |

Compatible geometry Linf differences are .01624/.01652 mm central/ends,
versus .09960/.10670 mm originally. Normal traction and slip accuracy have not
all improved: cumulative slip still differs **13.92%/17.42%** between compatible
timesteps. At dt=.025, changing the boundary changes final slip-rate fields by
90.6%/220.7% relative to the compatible central/end norms, and geometry by up to
14.66 mm. The better corner behavior therefore cannot be advertised as a
behavior-preserving correction to the original physical model.

## Energy, work and verification

The unchanged mature energy reduction includes actual shear mechanical work,
Maxwell dissipation, slip work, dilatation and outward elastic-energy flux. It
uses tensile-positive normal traction tau_nn-p and frictional compression
5+p-tau_nn. Z has no physical energy. Signed and accumulated absolute balances
are retained; trapezoidal powers are a diagnostic, not an executed new integrator.

| Compatible dt | Endpoint signed / absolute (J/m) | Trapezoid signed / absolute (J/m) | Raw face minus weak mechanical power (W/m) |
|---|---:|---:|---:|
| 0.05 | -0.2144535 / 0.2144535 | 0.0037590 / 0.0048473 | -0.0017054 |
| 0.025 | -0.1088772 / 0.1088772 | 0.0001809 / 0.0015326 | -0.0022451 |

The compatible endpoint remainder nearly halves, while signed trapezoidal
cancellation would substantially understate its absolute remainder. The raw-face
work discrepancy remains about 1.7–2.25e-3 W/m and does not improve with dt;
spatial/boundary traction accuracy remains open. This is much smaller than the
original .416/.635 W/m discrepancy, not proof that it is converged. Weak
mechanical work is about 13.1 W/m and is never assumed zero.

Compatible rotation-projection energy changes are at roundoff, compared with
the original 8.62e-3 J/m loss at dt=.025. Compatible source-projection losses are
2.47e-8/1.10e-8 J/m. Net transport-energy changes are about -8.4e-5/-8.7e-5 J/m,
in contrast to the original +.2957 J/m at dt=.025. None of these net changes is
relabeled as an added physical dissipation. Native q8 surface-work residuals
are below 3.73e-13 W/m; changing saved-state quadrature to q12 shifts that residual
by 4.17e-8/9.12e-8 W/m, so native-rule cancellation is not an exact continuum balance.

All 24 intended coupled physical steps publish history and geometry once with
zero retries. Profile replay agrees within 3.89e-16, donor bounds within 2.78e-16,
age source within 1.78e-15 s, and geometry replay within 1.13e-16 m. The scratch
controls leave the accepted bulk solution and mesh unchanged and own no material
history pointer. The compatible cases took 123.16/207.36 s, with 65/106 linear
solves and 2443/3987 linear iterations. Existing native unit/lifecycle evidence
is reused; no checkpoint ownership or restart meaning changed.

Two setup defects are documented rather than hidden. The first prescribed32
attempt stopped before transport because the new checker called a norm on a
ghosted Trilinos vector; owned-vector snapshots fixed it. The first coupled .05
attempt completed eight valid steps, then an unwanted 5.55e-17 s terminal interval
because its copied-input wrapper omitted the main fixture's representable-clock
rule. Its clean rerun matches all 45 valid 0–8-step bulk/surface exports bitwise.
The superseded log, data, source/DSO where applicable and command identities are
retained; the unwanted terminal state is not used or resumed. No native tolerance
or acceptance gate was changed to obtain a pass.

## Conclusion and next justified experiment

**Implemented:** a benchmark-only prescribed inflow control, compatible discrete
background traction, independent reference/energy reductions and reproducible
four-rank runs. Native solver, defaults, constitutive convention and protected
binaries are unchanged. **Locally verified:** exact-inflow DGQ2/SSPRK3 controls,
first-order continuous forcing error, removal of severe corner amplification
under compatible mechanical loading, and 24 coupled accepted transactions.
**Physical comparison:** boundary models differ; held trace retains a model
bias and cumulative-slip, normal-traction and face-work accuracy remain limited.
**Production readiness:** not established. Original shear failures, frozen split-Q2
evidence, cohesive failures and production BP3 remain distinct. Inherited losses
of 39/45 native-pilot frames and earlier temporary logs remain unrepaired.

The original zero-top-traction fixture should not be used as a converged shear
reference. The compatible control is useful for bounded method testing, but
adopting it for a physical application requires selecting its exterior loading.
Do not infer that generic held-trace inflow is accurate from this nearly uniform
far-field control.

**Next justified experiment:** freeze/replay a saved original **non-affine**
corner velocity in a prescribed transport control, comparing exact manufactured
exterior history with held trace. This would separate inflow feedback under the
actual problematic velocity from mixed-boundary mechanics, without another
coupled refinement campaign or a change of stress rate. It is proposed, not run
here. A physical exterior/inflow model must be selected before production use.
