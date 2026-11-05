# Coupled endpoint mechanics

[Overview](../MATURE_MOVING_FAULT.md) · Prerequisites: [model](model.md),
[localization](geometry.md) · Next: [state and attempted-step trace](lifecycle.md)

## Physical stress in the residual

Mechanics solves bulk momentum/incompressibility together with fault slip.
In schematic weak form, the bulk stress contribution is
$\int_\Omega (\tau-pI):\nabla\delta v\,dx$; boundary terms and constraints
retain their existing ASPECT definitions. The fault equation balances the
matched work traction against friction and radiation damping, with the existing
contact/active-set inequalities. Work traction is obtained with the same
localization and integration map used for bulk slip forcing, not by substituting
an arbitrary point sample of $t\cdot\tau n$.

[physical_history_stress](../../../../source/material_model/phase_field_fault/physical_stress.cc)
returns direct stored $\tau$ immediately, or reconstructs the separated control.
Material response and the
[reconstructed-fault assembler](../../../../source/simulator/assemblers/reconstructed_fault_stokes.cc)
combine the fixed anchor, current strain/slip and correction. The
[surface system](../../../../source/reconstructed_fault/surface_system.cc)
uses the corresponding traction, normal stress and friction response.
[Friction](../../../../source/material_model/rheology/fault_friction.cc) retains
regularized rate-state dependence on current $V$ and **incoming age**; age is
not independently advanced during Newton iterations. The example uses physical
mechanical pressure for friction, not prescribed adiabatic pressure. Pressure
conventions in production BP3 and cohesive configurations must not be inferred
from this example.

## Local tangent versus projected coupling

Write the pointwise endpoint law as $F(v,V)$, with $\bar\tau_h,\phi^+,I^+$ and
geometry fixed. For outer iterate $k$, define the stress correction

$$
 c_k(x)=\tau_{h,k}^+(x)-F(v_k,V_k)(x),\qquad
 \tau_{h,k}^+=P_hF(v_k,V_k).
$$

The inner mechanical solve uses $F(v,V)+c_k$. It holds $c_k$ fixed: its local
stress derivative is the Maxwell strain/slip derivative, along with the
implemented friction, pressure/contact and coupling derivatives. In direct
notation the frozen-coefficient stress increment is
$2\kappa D_3(\delta v)-2\kappa\mathcal L[\delta V]$.
The nonlocal projection dependence is resolved by refreshing $c_k$ **between**
inner solves. The combined process is a projection-consistency outer iteration,
not an exact monolithic projected Newton tangent.

Even with constant viscosity, the localized source need not belong to the
stress space. Therefore direct mode always uses this correction when material
transport is enabled. The separated branch retains its previous activation
rule: the outer correction is enabled for spatially varying viscosity. Constant
separated storage has its distinct analytic localization reconstruction.

`refresh_material_reaction` forms a candidate from the **same objective anchor**,
then compares corrections at Gauss `max(8, configured order)` points. It reduces

$$
 e=\frac{\max_{K,q}\lVert c_{new}-c_{old}\rVert}
 {\max(1,\max_{K,q}\lVert c_{old}\rVert,
             \max_{K,q}\lVert c_{new}\rVert,
             \max_{K,q}\lVert\tau_{h,new}^+\rVert)}.
$$

These are the existing in-plane tensor norms and absolute scale floor of one;
no new physical-energy criterion is implied. All ranks take the same maxima.
The tolerance remains $128\epsilon_{machine}$ and the limit remains 24 outer
iterations. Failure throws without publishing physical history.

## Explicit solver sequence

[solve_reconstructed_fault_stokes](../../../../source/simulator/solver/reconstructed_fault_stokes.cc)
owns the outer loop. `solve_reconstructed_fault_stokes_inner` contains the
existing coupled mechanical body once. The material owns correction data, not
solver recursion state:

```text
if terminal physical call and projected reaction is active:
    rates = current manager rates                 # private outer guess
    repeat at most 24 times:
        inner(false, &rates, false)               # solve at fixed correction
        error = material.refresh_material_reaction(solution, rates[0])
        if error <= 128 * machine_epsilon: break
    require convergence
    inner(true, &rates, false)                    # converged guess, same gates
else:
    inner(publish_history, trial_rates, reuse_preconditioner)
```

Every inner call prepares/reuses the fixed objective history, opens a private
surface nonlinear lifecycle and evaluates residuals, active sets, linear solves
and line searches with the retained criteria. A nonpublishing call returns the
converged bulk/slip scratch result and rolls back the surface lifecycle. It does
not publish constitutive histories. Other callers, including endpoint-phase
trials with permitted preconditioner reuse, bypass the new outer loop as before.

The terminal call receives the **converged private slip-rate vector**, not the
manager's earlier accepted rates. It repeats the final residual/KKT checks and
stages `pending_dg_endpoint_publication`. Only a later accepted timestep invokes
the callback that installs rates, constructs the publication reaction, commits
history and emits acceptance signals. Energy log `publication=0` identifies
outer candidates; `publication=1` identifies the publication candidate. This is
now an explicit observation argument, not a material-owned solver mode flag.
See [lifecycle](lifecycle.md) for rejection and exception ownership.
