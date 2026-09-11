# K2.3 bounded feasibility pilot: pressure preflight

## Status and accepted baseline

Baseline commit: `8ace2d1b8d71d1de6d8a3ae1249fda502c276d01`.
Continuous Q2 stress with incident-cell MPI ADD/count and the verified history
timeline are accepted for this fixed-mesh 2D work. DG compatibility and timeline
investigation are closed. No alternative transfer, audit repeat, performance
work or convergence campaign belongs to this pilot.

The user subsequently approved the proposed physical boundary variant and a
matched homogeneous control. Both bounded runs are now complete; see
[the pilot report](stage_K2_3_pilot_report.md). The preflight reasoning below
records why the boundary choice needed explicit approval. K2.2's reference
remains provisional and Gate K2 remains unmet.

## Existing pressure convention and the mismatch

The inherited K2 fixture has zero gravity, x-periodic boundaries, prescribed
velocity on both top and bottom, and no prescribed traction. On the 0.25 by
1 m box (y in [-0.5,0.5]), both walls prescribe

```
u_x = y * 1e-4 * (1 + 0.2*min(t/4,1));  u_y = 0.
```

It sets `Surface pressure = 1000` Pa but `Pressure normalization = volume`.
The latter means **zero volume-mean bulk pressure**, not 1000 Pa. This follows
from the parameter documentation in `source/simulator/parameters.cc` and the
actual adjustment in `Simulator::normalize_pressure` in `helper_functions.cc`.
`compute_initial_pressure_field` also applies that normalization. The separate
adiabatic-model pressure is 1000 Pa in this zero-gravity fixture and supplies
friction in the completed prescribed-pressure runs.

In true-normal-stress mode, the actual point response uses

\[
\sigma_n=p_{\rm FE}-\boldsymbol\tau:\boldsymbol N,\qquad
\boldsymbol\sigma=\boldsymbol\tau-p_{\rm FE}\boldsymbol I.
\]

Compression is positive in sigma_n. The field named `dynamic_pressure` in the
point input is the physical FE pressure in Pa; the constitutive code does
**not** add the adiabatic pressure. Homogeneous solver directions use scaled
pressure: delta p_physical = pressure_scaling * delta p_solver. Conversion is
performed in the condensed helper before the physical G action; it is not a
pressure-reference offset.

The coupled solver deliberately normalizes private bases/trials only in
prescribed-pressure mode. A pressure shift in true mode changes friction and
is not a gauge operation. Its verified pressure-complement treatment is also
restricted to eligible prescribed-pressure configurations. Thus toggling the
friction flag neither imposes nor preserves a 1000-Pa physical confining load.
Changing `Pressure normalization` to `surface` alone would only change the
initial pressure in this true-pressure coupled path, not impose a maintained
physical loading condition.

The current closed velocity boundary conditions supply no normal-traction
datum. There is an additional structural warning: the constant pressure test
in incompressibility still yields a redundant zero-flux row, whereas absolute
pressure now enters the surface law. Enabling that dependence does not itself
justify treating an arbitrary pressure offset as a harmless gauge. A pilot
needs an explicit physical pressure/loading choice before a numerical outcome
can be interpreted. No solve was used to select an arbitrary branch.

## Explicitly approved supported boundary variant

Use the existing boundary-traction function to prescribe **top normal total
traction -1000 Pa**, retain top tangential velocity, and retain both bottom
velocity components. Leave x periodicity and the mesh unchanged. With outward
top normal e_y, this fixes p - tau_yy = 1000 Pa on the top, while fault sigma_n
is determined by the coupled bulk solution and may vary. It changes the top
normal velocity condition: local u_y there is no longer prescribed, though
incompressibility and the other boundaries require zero net top flux. This is
an explicit physical boundary change from K2.2, not an equivalent gauge choice.

The approved parameter overrides use existing component selectors supported
by ASPECT:

```prm
set Pressure normalization = no
subsection Material model
  subsection Phase field fault
    set Use adiabatic pressure in fault friction = false
  end
end
subsection Boundary velocity model
  set Prescribed velocity boundary indicators = bottom:function, top x:function
end
subsection Boundary traction model
  set Prescribed traction boundary indicators = top y:function
  subsection Function
    set Function expression = 0; -1000
  end
end
```

Keep `Surface pressure = 1000` for the existing adiabatic initial pressure,
but the normal traction, not a post-solve pressure shift, would establish the
physical datum. No solver pressure projection or production correction is
proposed. A comparison to K2.2 must disclose the changed normal boundary
condition. Retaining impermeable top and bottom while imposing an absolute
mean pressure would instead need a separately reviewed coupled constraint;
it is not implemented by the current normalization switch.

## Intended feedback and bounded execution after approval

Start from the existing 32x128 coarse fixed-profile fixture, 0.5-s maximum
steps, and the same compact 5% initial Theta bump. Run initialization and at
most two real steps, stopping at 1 s. Preserve the converged initial phase
field with the existing benchmark constraints. Keep the accepted surface
domain rule, support, full I_h, histories, tolerances, iteration budgets and
initialization semantics. No history refresh for visualization or new Maxwell
evaluation on already committed history is allowed.

Intended feedback is the bulk pressure/normal deviatoric stress changing
mu*sigma_n, thereby changing V and the coupled bulk state. The existing G
action contains 2*kappa_bulk*(S+mu*N):delta strain_rate - mu*delta p.
K_V differentiates at fixed bulk unknowns; S:N=0 removes a direct slip-induced
normal-stress derivative. Mechanics uses committed Theta before the split
history update. No Jacobian change is proposed.

Measure the actual parent-P0 bulk/history plus surface-quadrature constitutive
normal stress, not a normal-column substitute. Use the preserved accepted
surface weak terms before history publication for q, cohesion, friction,
damping and balance. Report pressure and sigma_n extrema and variation, total
and nonuniform V/slip response, and active/free-node counts at accepted times.
Genuine convergence must include final bulk and projected-surface criteria and
fresh returned-direction residual checks. Active-node balances must be assessed
with the lower-bound inequality, not assumed to vanish as on free nodes.

Recheck initialization retention and accepted updates with existing machinery;
do not repeat the completed transfer/timeline audits. Provide bulk/fault VTU
and sampled profiles, explicitly distinguishing published old-history FE
composition from committed particle stress. A symmetric response with no
resolved sigma_n variation is a symmetry/feasibility result, not demonstrated
nonzero normal-feedback coverage. No asymmetry will be added automatically.

For this feasibility pilot only, the user explicitly permits provisional
omitted fraction <=1e-4 (original target 1e-6); actual slip-normalization error
must separately remain <=1e-4 at measured locations and accepted times. Measure
both using existing profile analysis. No support widening or I_h renormalization
is permitted. Fixed support remains an independent approximation.

The same-mesh prescribed-pressure replay took 38.01 s. Budget this pilot at
120 s hard wall time, initialization plus at most two steps, with no automatic
retry. True-pressure convergence cost is unmeasured. Total exploratory test
execution must remain below 600 s; ask before extending either bound. Stop on
a modeling/production correction requirement or an acceptance failure. No
expensive comparison run or K2.3 convergence campaign is authorized.

The boundary decision is now closed. Review the bounded pair's measured
feedback before authorizing further work; no convergence campaign follows
automatically from feasibility.
