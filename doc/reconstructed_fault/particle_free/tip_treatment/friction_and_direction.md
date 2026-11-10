# Cross-section closure

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

[`FaultFriction::section_endpoint`](../../../../source/material_model/rheology/fault_friction.cc)
uses a safeguarded scalar Newton/bisection solve in an asinh rate coordinate,
with the sign of the free section rate. It verifies the root to a tolerance of
32 double-precision machine epsilons times the absolute free rate, with explicit zero-rate and
zero-mobility cases. `signed_endpoint` supplies traction, derivative, age and
separate friction/radiation parts; no minimum-slip-rate floor enters this law.
`section_primitive` uses Gauss48 in a logarithmic rate coordinate only for the
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
