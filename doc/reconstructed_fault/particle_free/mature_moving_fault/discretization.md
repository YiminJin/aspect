# Discrete transport, rotation and projection

[Overview](../MATURE_MOVING_FAULT.md) · Prerequisite: [model](model.md)
· Next: [geometry](geometry.md)

## Spaces and shared mass projection

The supported fixture uses native continuous Q2 velocity, Q1 pressure and Q1
phase on a fixed Cartesian mesh. Stress has its own scalar `FE_DGQ(2)` space
for each independent in-plane component: nine coefficients per cell/component,
without continuity constraints across faces. `TensorField` owns these vectors
and a shared DoF space; it assigns no physical meaning or acceptance status.

For each cell $K$, basis $N_i$ and evaluated tensor component $f_c$, require
$\int_K N_i((P_h f)_c-f_c)\,dx=0$. Thus

$$
 M_{ij}=\sum_q w_q N_i(x_q)N_j(x_q),\quad
 b_{ic}=\sum_q w_q N_i(x_q)f_c(x_q),\quad
 a_c=M^{-1}b_c,\quad (P_h f)_c=\sum_j a_{jc}N_j.
$$

[TensorField::project](../../../../source/material_model/history/tensor_projection.cc)
implements this algebra once. Its single-field overload uses the multiple-field
kernel with one field. The latter builds/inverts one mass matrix per owned cell
and applies it to every field/component RHS; separated reaction supplies two
fields in one call. RHS indexing is field-major, then deal.II symmetric component
order. The loop order is quadrature, test basis, mass/trial basis, component.
The caller supplies quadrature and cell selection. The material adapter
[project_material_tensor](../../../../source/material_model/phase_field_fault/material_tensor_projection.cc)
explicitly receives profile state, accepted velocity and a tensor evaluator.
Reaction sampling instead reads current velocity gradients and fixed anchor
values through `project_mature_reaction`.

For rotation/reaction the rule is tensor-product Gauss with
`max(8, FE bulk normal quadrature points)` points **per direction**. This is
quadrature-defined projection; orthogonality claims refer to that measure.
For variable viscosity, $P_h[\beta\tau+2\kappa D_3-2\kappa\mathcal L[V]]$
is generally different from multiplying projections of its factors. Even
constant $\kappa$ does not put $\mathcal L[V]$ into DGQ2. Its profile ratio,
changing frame and finite support can be nonpolynomial or discontinuous.

Each rank assembles only owned cells. Callback evaluation is synchronous and
must fill every supplied field/point; it cannot call MPI collectives. After all
cells, owned vectors compress with `insert`, then fill owned/ghost vectors in
the same shared space. Every rank participates, including ranks with no selected
cells. Nothing accepted is mutated. Material energy observers integrate the
local field before projection and the returned field after projection; global
energy reductions happen outside callbacks.

## Advection

With accepted velocity held fixed, transport solves
$\partial_t a+v^-\cdot\nabla a=0$ componentwise. Integration by parts gives the
DG volume action $\int_K a(v^-\cdot\nabla N_i+N_i\nabla\cdot v^-)dx$ and face
flux $-\int_{\partial K}N_i(v^-\cdot n)\widehat a\,ds$. The upwind trace
$\widehat a$ comes from the donor cell, or selected inflow data. The divergence
term is needed for the **material**, nonconservative transport equation; this
is not a conservative stress-density remap.

[TensorAdvection](../../../../source/material_model/history/tensor_advection.cc)
and its [distributed implementation](../../../../source/material_model/history/tensor_advection_distributed.cc)
construct the operator and apply SSPRK3. For a substep $\delta t$ and its affine
operator $L$ (including boundary forcing),

$$
 a_1=a_0+\delta t L(a_0,t),\qquad
 a_2=\tfrac34a_0+\tfrac14[a_1+\delta tL(a_1,t+\delta t)],
$$
$$
 a_3=\tfrac13a_0+\tfrac23[a_2+\delta tL(a_2,t+\delta t/2)].
$$

The moving material caller supplies zero reaction/source here. It chooses
$n=\max(1,\lceil\Delta t\,r_{adv}/0.1\rceil)$ equal substeps, at most 4096;
the operator's existing admissibility bound is 0.125. This engineering bound is
not a sharp stability theorem. Transport assembly uses its retained Gauss3 mass
and caller-selected integration order (at least eight for moving material).
Owned-row MPI transport reads synchronized neighbor coefficients at each stage.

Zero tensor inflow is the default. Optional `accepted trace` snapshots the
accepted interior trace at each boundary load point and freezes it throughout
the substeps. It is a boundary model, not an exact general exterior solution.
The bounded examples use compatible short translated compact fields. Neither
choice changes the profile's separate pullback rule.

## Rotation and reaction ordering

[transport_material_history / rotate_material_history](../../../../source/material_model/phase_field_fault/stress_evolution.cc)
use the same accepted velocity supplied by `MaterialHistoryInputs`. Let $A_h$
be completed advection and $Q=\exp(\Delta t W^-)$ the finite rotation:

$$
 \bar\tau_h=P_h[Q A_h(\tau_h^-)Q^T],\qquad
 \tau_h^+=P_h[\beta_b\bar\tau_h+2\kappa_b(D_3(v^+)-\mathcal L^+[V^+])].
$$

Geometry/profile staging occurs between transport and rotation so subsequent
localization/normalization reads candidate geometry. The finite 2-D rotation
in [objective_stress.h](../../../../include/aspect/reconstructed_fault/objective_stress.h)
is exact for its frozen local spin. Spatially varying rotation followed by
projection is not exact field transport, and the full accepted-gradient
Lie/endpoint method is not a globally exact rotation or a second-order coupled
integrator. It retains lag, splitting, spatial projection and boundary errors.
The outer loop repeats the **reaction**, never advection or rotation.

The specialized native-source projection retains Gauss3 differentiation and
extended-precision roundoff bounds; stationary objective preparation retains
an exact zero-spin coefficient-copy branch. Formed-side cohesive spaces and
legacy geometry-only shear corrections likewise have different contracts.
See [intentional alternatives](alternatives.md), rather than forcing these
through the common moving-field interface.
