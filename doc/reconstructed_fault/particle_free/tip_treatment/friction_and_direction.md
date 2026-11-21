# Cross-section closure

The straight default below is retained. The user-selected, opt-in native
[material-strip closure](../FIRST_FREE_TURN_FORMULATION.md) uses physical area
per reference arclength, uniform section resistance and a coherent virgin-plane
Y0 criterion. It preserves formed planes and full tensor histories. The earlier
[single-turn proposal](../SINGLE_TURN_RSF_FORMULATION.md) and
[reference comparison](../SINGLE_TURN_RSF_REPORT.md) retain the alternative law
and historical evidence; they no longer describe the only enabled native path.

The [first native free-turn result](../FIRST_FREE_TURN_REPORT.md) accepts through
.075 s with subcell asymmetric extension, then fails the phase KKT gate at .080 s.
The reflected case has not run; a resolved free turn is not qualified.

This page describes the retained **BE default**. The selectable midpoint stage,
exact aging, endpoint publication and work identity are in
[time integration](time_integration.md); do not apply BE loss formulas to that mode.

[Guide](../TIP_TREATMENT.md)

The user expressly selected this additional constitutive assumption for one
straight, fixed-direction fault. Sections span the entire normal extent of the
specimen, including intact material and positions ahead of the initial tip:

\[
V_s=\int w(s,n)dn,\quad (\tau-Q):S=f_s,
\quad \theta_s^+=\frac{\theta_s^-+\Delta t}{1+\Delta t|V_s^+|/D_c},
\]
\[
f_s=p_N a\,\operatorname{asinh}\!\left[
\frac{V_s}{2V_0}\exp\left(\frac{\mu_0+b\ln(\theta_s^+V_0/D_c)}a\right)\right]
+\eta_{rad}V_s.
\]

The configured positive p_N is prescribed confinement, not the solved Stokes
pressure multiplier, not an inferred tensile-contact law. Mechanics fixes a
pressure gauge on its closed velocity boundary. This control supplies no
current-pressure or opening/contact qualification. The RSF constants are reused
from the existing material owner. The legacy minimum-rate convention is not
applied to this signed smooth law. Aging here is backward Euler, distinct from
the legacy constant-rate exponential age update.

Each section is initialized with `Bulk tip initial age` (seconds) everywhere,
including ahead of the initial tip. At zero slip its age increases with elapsed
time. Reconstructed segments do not own, create or reset age. Cumulative signed
section slip is advanced with the accepted V. Initial stress and cohesive strain
are zero for this fresh control; positive initial phase describes a declared
pre-existing damaged patch, not seeded damage ahead of a computed tip.

For the aligned Cartesian control, each tangential Gauss coordinate identifies
a section, with tangential weight ds. Each bulk point has dn=JxW/ds. Thus
\[
V_s^h=\sum_{q\in s}dn_q w_q,\quad
\sum_q JxW_q f_s w_q=\sum_s ds_s f_s V_s^h.
\]
The resistance must be applied uniformly to all points in that section: that is
the work adjoint of the actual integration. No normalization floor, inferred
bandwidth, w=χV ansatz, or total elastic/viscous strain is inserted in V.

Only the section dissipation fV is nonnegative. Local fw may be negative where
fault deformation reverses. The exported signed and absolute integrals expose
cancellation; multiple separated active bands share the same age and resistance
and may cancel in V. This is a physical limitation of this closure, not a reason
to clip rates or silently select one band. Bulk cohesive variance energy remains
in the point history and cannot be discarded in a one-moment surface reduction.

## Endpoint monotonicity, not steady-state stability

For positive V, with endpoint aging already eliminated,
\[
R'(V)=p_N\,\rho(V)\left[\frac aV-\frac b{D_c/\Delta t+V}\right]
 +\eta_{rad},\qquad 0\le\rho\le1.
\]
Writing d=D_c/dt, the negative part of a/V−b/(d+V) is bounded below
by −(max(0,sqrt(b)−sqrt(a)))²/d. Thus the implemented guard
\[
\eta_{rad}\ge\frac{p_N\Delta t}{D_c}
 [\max(0,\sqrt b-\sqrt a)]^2
\]
ensures monotone endpoint R and uniqueness of V+C R(V)=B for C≥0.
It also makes the section primitive convex. It neither removes b>a steady
velocity weakening nor proves convexity/stability of the coupled phase problem.
The search primitive integrates R itself, not fV. Its logarithmic-rate Gauss
quadrature is separately derivative-checked; it does not change the physical
constitutive law or mechanical/phase quadrature.


## Implementation and cancellation diagnostics

[`FaultFriction::solve_section_slip_rate`](../../../../source/material_model/rheology/fault_friction.cc)
uses a safeguarded scalar Newton/bisection solve in an asinh rate coordinate,
with the sign of the free section rate. It verifies the root to a tolerance of
32 double-precision machine epsilons times the absolute free rate, with explicit zero-rate and
zero-mobility cases. `evaluate_resistance` supplies traction, derivative, age and
separate friction/radiation parts; no minimum-slip-rate floor enters this law.
`compute_resistance_potential` uses Gauss48 in a logarithmic rate coordinate only for the
line-search functional.

`BulkTipCoupling::evaluate` first integrates the fixed-force-free point response
and mobility, solves the section root, then reevaluates every point at that common
force. The second pass independently records V_integrated, integral(abs(w)),
integral(max(w,0)) and integral(min(w,0)). The last quantity is signed negative.
`tip-sections-*.csv` retains these and the root-resolved rate, their difference,
and the independently evaluated resistance at the integrated rate. Point exports
retain w and full cohesive history, allowing separate active bands to be inspected.
A small net V alone cannot establish a single localized band or small local work.

The accepted section age/slip uses the integrated rate. Search and split
friction/radiation diagnostics use the resolved scalar law, while point work uses
f*w and the exact discrete integration adjoint. Their agreement is checked in
qualification; tiny rates need absolute resistance and Y errors as well as slip
errors. The native-basis control's near-roundoff agreement does not erase the
older tolerance-sensitivity evidence from a different solver comparison.

## Shared rheology ownership

`FaultFriction::update_state` owns the aging value and its derivative with respect
to **nonnegative speed**. `AgingMethod::backward_euler` and
`AgingMethod::constant_rate_exact` replace the former boolean argument. This is
an internal API choice; input and checkpoint method identities are unchanged.
Zero speed gives incoming age plus duration, zero duration gives incoming age,
and neither case introduces a rate floor. Exact forcing and its derivative use
`expm1` and the analytic small-argument kernel. The mature publication and timestep-predictor callers enforce Vmin themselves
and explicitly select the exact constant-rate convention. The common updater
accepts zero speed and returns `StateUpdate {age, derivative_wrt_speed}` without
publishing history. There is no mixture argument: Dc is global.

| Equation or operation | Function in the existing owner | Active caller / time location |
|---|---|---|
| Aging value and speed derivative | `FaultFriction::update_state` | Signed resistance trials; mature publication and timestep predictor; accepted bulk-tip candidate |
| Mixed mu0, a, b | `rate_state_parameters` | Coefficient, inversion, timestep, signed traction, section root and search primitive; checks law and vector sizes before indexing |
| mu(V,theta), partial mu_V at fixed theta, partial mu_theta at fixed V | `rate_state_coefficient` through positive-state public APIs | Mature material/surface friction and diagnostic APIs; no aging or history mutation |
| f(V) and total df/dV = p(mu_V + mu_theta theta_speed) + eta_rad | `evaluate_resistance` | Coupled stage at BE full duration or midpoint half duration, always from immutable incoming age; signed chain rule includes V=0 |
| V + mobility f(V) = free_rate | `solve_section_slip_rate` | `BulkTipCoupling::evaluate`; condensed resistance derivative, method-specific monotonicity admission |
| Integral from zero to trial rate of f(v) dv | `compute_resistance_potential` | Coupled search functional; Gauss48 rate quadrature and incoming age held fixed; never physical energy or dissipation |
| Accepted section age and slip | `prepare_bulk_tip_candidate` | BE aging at full duration, or exact constant-stage-rate aging at full duration; physical slip uses full dt times stage rate |
| Accepted mature surface age | `compute_surface_state_candidates` → `update_state` | Existing surface state-update timing, exact constant-rate aging and positive-rate guard |

The nonregularized state partial is b/theta, including the admitted a=0 case;
it must not be formed by multiplying mu_V by b/a. The state partial is exactly
zero for a stateless law. All three coefficient
operations accept an explicit optional age: supply seconds for rate-and-state
friction or `std::nullopt` when no state exists. The object dispatches on its
configured law, independently of whether age is supplied. Stateful evaluation
rejects missing, nonpositive or nonfinite age; stateless evaluation ignores
supplied age. Coefficient evaluation and both partials retain the finite
speed-at-or-above-Vmin contract. The signed section evaluator separately
admits zero speed. No dummy zero age represents absent state. Constant a=b=0
friction imposes
no state-splitting timestep restriction. These edge fixes do not change the
selected regularized synthetic parameters. Fixed-state partials are distinct
from the total derivative after time-discrete aging. Every Newton/line-search
trial starts from the same accepted age; accepted publication evaluates aging
directly rather than computing an unused traction.


The public interface is ordered as configuration, candidate aging, coefficient
and fixed-state partials (then initialization inversion), signed resistance,
section solve, search potential, and timestep/identity utilities. There is one
public operation per quantity. `coefficient_at_state` is the private common
validation/dispatch boundary; law-specific parameter mixing and
`rate_state_coefficient` retain the mathematics. The latter is also used by
signed resistance without imposing the mature minimum speed.

The mature mechanics caller `evaluate_reconstructed_fault_point` gathers
optional age once, then calls the coefficient and speed partial unconditionally.
Its remaining state test protects the absent history property. Other state tests
own initialization, storage, publication or timestep policy, not coefficient
dispatch. `update_state` remains explicitly stateful-only; a stateless law has
no age to evolve. See the [interface review](../FAULT_FRICTION_INTERFACE_REVIEW.md)
for caller changes and numerical comparisons.


The [free-turn recovery](../RECOVER_FREE_TURN_STEP16_REPORT.md) now supplies a
measured nonlocal-sign example: both reflected histories reach .080 s, with88
mixed-sign sections and maximum section cancellation .727%. Negative local power
is .0136% of positive power; integrated section powers remain nonnegative. Native
quadrature samples contain multiple same-sign components in some sections. The
reported counts use no phase/rate cutoff and do not infer continuous branching
from sampled signs. Old planes/measures remain unchanged and prospective age is
not reset at registration. Tiny-positive-phase registration remains an explicit
limitation of the material-plane closure and its finite solver accuracy.

### Optional Coulomb plane selection on material strips

`Bulk tip section closure = material strips Coulomb formation` selects a
direction for virgin material in the bounded midpoint single-graph control.
The existing `material strips` setting maximizes integrated intact driving
with the current trial mechanics. Both use the same bulk fracture energy,
section RSF resistance and work adjoint.

At the beginning of each positive-duration attempt, read the **accepted
endpoint physical deviatoric Cauchy stress** at the native mechanical Gauss
points. Use the same immutable full-height material strips as the section
law. For a strip of area $a_k$, form
$$
 \bar\tau_k=a_k^{-1}\sum_{q\in k} J_q w_q\,\tau_q^n .
$$
Average signed tensors before resolving traction. For tangent
$t=(\cos\alpha,\sin\alpha)$ and normal
$n=(-\sin\alpha,\cos\alpha)$, maximize
$$
 |t^T\bar\sigma_k n|+\mu_0 n^T\bar\sigma_k n
$$
using tension-positive stress. Every isotropic pressure contribution adds a
constant to the objective. The orientation therefore needs only the stored
deviator. The orientation coefficient is the configured reference friction
coefficient of the homogeneous material. The RSF resistance in the stage
solve continues to use its own rate, age and prescribed scalar confinement.
This selection supplies a plane preference. Admission of new damage still
requires the coupled fracture and mechanics equations.

Write $u=(\bar\tau_{yy}-\bar\tau_{xx})/2,\ v=\bar\tau_{xy}$.
For each shear sign $s=\pm1$, the maximizing unoriented line is
$$
 \alpha_s=\tfrac12\operatorname{atan2}(su-\mu_0v,\;sv+\mu_0u)
 \pmod\pi .
$$
Take the admissible conjugate nearest the accepted preference, counting only
lines strictly within the forward sector $(-\pi/4,\pi/4)$. A distance tie
takes the positive-shear conjugate. Zero in-plane deviatoric contrast retains
the incoming preference. Absence of an interior conjugate aborts the bounded
model. There is no clipping at a sector edge. The positive common area in
both arguments cancels.

`BulkTipCoupling::prepare_coulomb_planes` sets the attempt's angles and
measures before the first mechanical evaluation. Native $JxW/ds$ and
the work adjoint use $ds=dx/\cos\alpha$ throughout that attempt. At the
zero-clock equilibrium the seed keeps its prescribed horizontal planes.
The first accepted positive native phase in a section fixes its angle and
measure. Previously formed sections keep their frames for life, including
under unloading. Each section's age and slip have existed since initialization
and advance once at acceptance. Rejection discards the trial map and all trial
histories. Checkpoints distinguish both formation rules and preserve incoming
preferences as well as formed flags. See the
[formation comparison](../STAGE4_RECOVERY_AND_COULOMB_FORMATION.md).
