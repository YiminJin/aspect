# Bulk-tip time integration

This page describes the selectable **midpoint candidate**, alongside the retained
backward-Euler (BE) default. Qualification is recorded in
[the time-update report](../TIP_TIME_UPDATE_AND_GROWTH_REPORT.md). The physical
model, fixed section closure, native Bernstein representation, quadrature,
fracture energy and final mechanics/KKT tolerances are unchanged.

## Stage coordinates and local elimination

Write the physical duration as Δt and h=Δt/2. The algebraic velocity coordinate
is **a=h v_m**, half the physical displacement u=Δt v_m. Pressure is p_m/G and
the phase coordinate is b_m=(b_1+b_n)/2. The endpoint cone becomes
b_n≤b_m≤(1+b_n)/2. The solver uses these bounds directly, never clipping an
extrapolated phase. The zero-clock seed equilibration retains its ordinary
full [b_seed,1] obstacle.

The physical laws, with S:S=1/2 and full three-dimensional plane-strain tensors,
are

```
(tau_1-tau_n)/(2G Δt) + tau_m/(2 eta) = D3(u)/Δt - w_m S,
(A_1-A_n)/Δt + (G/eta) A_m = w_m S,
Q_m = 2G A_m/h_c(phi_m),
Y_m = G h_c'(phi_m) (A_m:A_m)/h_c(phi_m)^2,
(tau_m-Q_m):S = f_m,     V_m,s = sum_q dn_q w_m,q.
```

Here h_c is the physical compliance, not the algebraic duration h. Calling
[`Tip::update`](../../../../source/material_model/phase_field_fault/bulk_tip.cc)
with old history, D3(a), phi_m and h gives exactly tau_m,A_m,Q_m,w_m,Y_m.
`Tip::midpoint_endpoint` returns 2 tau_m−tau_n, 2 A_m−A_n and 2 phi_m−phi_n.
Analytic intact and fully damaged limits remain in the same rational elimination.
There is no accepted half step or second material law.

## Aging, signed friction and unique-root admission

For k=|V_m|/Dc, theta(s)=theta_n exp(−ks)−expm1(−ks)/k, with the continuous
zero-rate limit theta_n+s. Friction uses s=h; accepted age uses s=Δt.
This is exact for a constant stage rate, not for arbitrary varying physical V.
[`FaultFriction::signed_endpoint`, `section_endpoint` and `section_primitive`](../../../../source/material_model/rheology/fault_friction.cc)
share their existing friction and root algebra with a local exact-aging choice.
The derivative uses an expm1/Taylor kernel near zero, with no minimum-rate floor.

Because theta is a positive mixture of exp(−|V|r/Dc), r∈[0,s],
−s/Dc≤d(log theta)/d|V|≤0. In the regularized law
R'=p rho[a/|V|+b d(log theta)/d|V|]+eta_rad, 0≤rho≤1.
Thus eta_rad≥p b s/Dc is a sufficient monotonicity condition. It is checked
before root elimination. The old BE condition remains on BE only. The tested
radiation is unchanged; this admission does not qualify multiple-root RSF.

## Coupled residual, tangent and globalization

[`BulkTipCoupling`](../../../../source/simulator/solver/bulk_tip_assembly.cc)
assembles stage stress and pressure mechanics and
R_b=(H_f b_m+ell_f−integral B Y_m)/K_f. `Tip::derivative` uses h throughout;
the condensed section derivative includes exact stage-age dependence. Ordinary
ASPECT Stokes supplies the reference inverse with beta_h=1/(1+hG/eta).
The adapter scales its velocity and pressure rows with h, not physical Δt.
The row adapter and active-bound solve are in
[`bulk_tip_linear.cc`](../../../../source/simulator/solver/bulk_tip_linear.cc);
`BulkTipCoupling::evaluate` and `directional_work` own the residual/functional
pair, while the ordinary
[`solve_single_advection_iterated_newton_stokes` scheme](../../../../source/simulator/solver_schemes.cc)
owns search, final acceptance and publication.

In these stage coordinates the reduced search functional is

```
I = E_bulk(tau_m) + E_cohesive(A_m,phi_m) + E_fracture(b_m)
    + beta_h h sum_s ds integral_0^V_m R_m(v) dv.
dI = beta_h G R_u . da + K_f R_b . db_m
```

for admissible divergence-free directions. The pressure multiplier has weight
beta_h G. This is a search functional, not endpoint storage or reaction work.
The existing exact/progress directions, release-capable obstacle search and
Armijo test remain in the ordinary simulator nonlinear scheme. A scoped
multiplier-only completion handles the fixed-phase compact screen: if an
inexact full direction is rejected as uphill, test only its pressure component,
with velocity and phase unchanged. Pressure enters neither the stored energy
nor prescribed-compression RSF, so this changes no primal functional or material
history. It is accepted only if the **full unchanged mechanics and KKT gates**
are already satisfied. This is not an energy-tolerance relaxation or a general
alternative nonlinear search. Independent
coupled tangent and functional checks use the stage coordinates.

The controls impose time-independent boundary velocities, so a=h v_boundary
is exactly the stage displacement. For time-dependent user data the present
ordinary simulator constraint sampling is at the right endpoint: the candidate
then approximates the physical increment by Δt v_boundary(t_1). Such data are
**not** qualified as second-order midpoint boundary loading by these controls.
Pressure is a gauge multiplier; prescribed compression in RSF is unchanged.

## Physical work and endpoint ownership

The accepted owner stores only tau_1,A_1,b_1,theta_1 and signed slip increment
Δt V_m. Stage responses belong to the attempt. Stored rate, resistance and q
are last-stage observations, not an instantaneous endpoint law. Rejection
restores the accepted parent; publication occurs once through the existing
transaction. Physical-law archive layout is unchanged. The separate
`bulk-tip-time-update.txt` checkpoint metadata identifies BE or midpoint;
absence means historical BE. Ordinary restart cannot change methods. The
explicit benchmark common-parent import is the only selected method switch.
[`PhaseFieldFault::prepare_bulk_tip_candidate`](../../../../source/material_model/phase_field_fault/bulk_cohesive_history.cc)
constructs those endpoints from the immutable accepted owner and integrated
section rates. Method checks live at the
[checkpoint serialization boundary](../../../../source/simulator/checkpoint_restart.cc).

Endpoint cohesive storage is evaluated from A_1,phi_1, never from A_m,phi_m.
Define

```
C_c = Q_m:(A_1-A_n) - Y_m(phi_1-phi_n) - ΔE_cohesive.
D = Δt integral [tau_m:tau_m/(2 eta) + (G/eta) Q_m:A_m] dx
    + Δt sum_s ds f_m V_m.
W_phase = K_f (b_1-b_n).R_b.
W_int - ΔE_total - D - integral C_c dx + W_phase = 0.
W_ext - ΔE_total - D = integral C_c dx - W_phase + W_ext-W_int.
```

Quadratic bulk and SIPG fracture energies obey their exact midpoint chain rules.
Nonlinear cohesive energy does not. Upper-bound work remains in W_phase; it is
not discarded when saturation occurs. The physical remainder is reported before
subtracting C_c or residual work. BE numerical losses are zero in the midpoint
ledger, not reassigned to physical dissipation. Signed cumulative and accumulated
absolute errors must both be inspected.

`bulk-accepted.csv` energies are endpoints, losses/work are stage-integrated;
`tip-time-location.csv` gives parent/stage/end times. In `tip-points`, phi/tau/A
are endpoints, w/qcoh/Y are stage observations, and explicit stage_phi,
stage_tauxy,stage_Axy accompany them. Y_endpoint is evaluated from endpoint A,phi
where phi>0; it is undefined at intact points without an endpoint reaction law.
`tip-balance` contains endpoint coefficients/fracture gradients for exact spatial
work partitions, and separately stage coefficients/gradient/residual/driving.

## Accuracy and limits

Free Maxwell relaxation has amplification (1−z/2)/(1+z/2), z=ΔtG/eta.
The candidate admits z≤2, excluding sign reversal but not establishing accuracy.
It is not L-stable. No silent fallback to BE occurs. Smooth admitted trajectories
may be second order; intact birth, bound saturation, reversal and changing active
sets need separate evidence. Neither an algebraic ledger nor a small physical
remainder proves state accuracy or propagation.

Independent local equations, analytic forced Maxwell convergence, stiff sign
reversal and nonlinear cohesive chain-rule scaling are checked in
[`unit_tests/bulk_tip.cc`](../../../../unit_tests/bulk_tip.cc).
The benchmark checker tests signed friction, condensed root derivatives,
primitive derivatives, the coupled tangent and transaction behavior.
Its independent equations and observers are in
[`bulk_tip/checks.cc`](../../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/checks.cc);
[`endpoint_time_diagnostic.py`](../../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/endpoint_time_diagnostic.py)
checks conditional endpoint stationarity separately from stage acceptance.
[Hairer and Lubich's gradient-system analysis](https://www.unige.ch/~hairer/preprints/gradientflow.pdf)
provides context on energy decrease and stiff damping, not a proof for this
constrained nonlocal RSF system.
