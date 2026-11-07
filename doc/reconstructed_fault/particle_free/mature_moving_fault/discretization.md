# Discrete transport, rotation and projection

[Overview](../MATURE_MOVING_FAULT.md) · Prerequisite: [model](model.md)
· Next: [geometry](geometry.md)

## Spaces and shared mass projection

The fixture uses native continuous Q2 velocity and Q1 pressure on a fixed
Cartesian mesh. Native phase uses a semantic composition with canonical
composition FE settings: DGQ2 is the development selection, Q1 a comparison.
Legacy CPDI retains separate Q1 storage. The [phase contract](../DGQ2_PHASE_FIELD_DESIGN.md)
and [shared assembly report](../SHARED_SCALAR_DG_REPORT.md) describe mapped
scalar assembly through the ordinary cell/interior/boundary advection classes and the still-Cartesian coupled stress admission. Stress has its own scalar `FE_DGQ(2)` space
for each independent in-plane component: nine coefficients per cell/component,
without continuity constraints across faces. `TensorField` owns these vectors
and a shared DoF space; it assigns no physical meaning or acceptance status.

Cartesian admission permits coordinate roundoff from non-dyadic refinement:
each expected vertex coordinate must agree within 16 machine epsilons times
`max(abs(lo),abs(hi),hi-lo)`, with strictly positive cell widths. This does not
permit physically skewed cells or change quadrature/solver tolerances. The
[layered campaign](../LAYERED_BULK_CAMPAIGN_REPORT.md) records the measured
roundoff defect, a distorted-cell rejection and unchanged coarse-grid results.

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
choice changes the independently selected phase transport operator.

## Native phase transport and growth

The physical phase has its own scalar spatial operator in ordinary composition
storage, with a special semantic type rather than chemical mixing. With DG
selected, ordinary `solve_native_phase_transport` dispatch passes the accepted
phase and full accepted velocity to `PhaseFieldHandler::transport_dg_phase`.
The Q1 remapper no longer dispatches DG transport. Upwind fluxes and the divergence source
advance the material equation, using Lobatto5 weak transport, exact Q2 mass,
SSPRK3, mean-admissible subcycling and whole-cell bound scaling. The compact
mature fixture prescribes zero phase on every incoming boundary. It stores the
result directly in the candidate native phase block, without projection to Q1.

In the separate supported fixed-fault cohesive path, the phase handler supplies
the SIPG fracture energy/residual/tangent and the endpoint solver adds the existing
mechanical energy and normalization adjoint. Positive-basis gap constraints define
the discrete obstacle; the phase equilibrium contains no invented time-mass term.
The transport limiter is not a final correction to this equilibrium. The
[operator contract and source map](../DGQ2_PHASE_FIELD_DESIGN.md) give equations,
quadrature, trace choices and ownership. Static refined faces have operator tests;
accepted DG phase AMR transfer is explicitly unsupported.

## Rotation and reaction ordering

[transport_material_history / rotate_material_history](../../../../source/material_model/phase_field_fault/stress_evolution.cc)
use the same accepted velocity supplied by `MaterialHistoryInputs`. Let $A_h$
be completed advection and $Q=\exp(\Delta t W_b^-)$ the finite rotation:

$$
 \bar\tau_h=P_h[Q A_h(\tau_h^-)Q^T],\qquad
 \tau_h^+=P_h[\beta_b\bar\tau_h+2\kappa_b(D_3(v^+)-\mathcal L^+[V^+])].
$$

Geometry/profile staging occurs between transport and rotation so subsequent
endpoint reaction localization/normalization reads candidate geometry. **Spin
subtraction does not read that candidate.** `prepare_material_history` samples
accepted velocity gradient, committed rate, phase, normalization and segment
frame before staging. `AcceptedSpinSamples` stores unprojected angular rates by
CellId and Gauss order, with accepted-parent/step identities. The projection
callback receives CellId and quadrature index explicitly. Both direct stress
and separated $T$ and $M=R+P_h B$ use the same frozen rates. The snapshot is
Eulerian and is not advected. A retry forms a new angle from its actual interval;
Newton changes no spin and adds no spin derivative to the Jacobian.

The finite 2-D rotation
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

The stationary bulk-spin anchor uses the same accepted sampling and stable
$QBQ^T-B$ increment for the residual. Its coefficient-copy shortcut tests the
selected angular rates for exact zero, including the converse case of zero
total spin with nonzero fault spin. Total-spin arithmetic retains its prior
helper and shortcut. Native FE gradients need not cancel a nonpolynomial
localized source exactly: that residual is a spatial error, not a reason to
project or smooth either spin before subtraction.
