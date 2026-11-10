# Retained alternatives and deliberate distinctions

Current support is native DG Bernstein phase and owned-row stress transport on
all ranks. Q1 remap/MacCormack, CPDI and nodal-phase descriptions below are
historical comparisons only; their implementations are retired. The old
`Material profile transport` name is parsed solely for checkpoint identity and
its retained support admission. It does not select an algorithm. See the
[active reading path](../ACTIVE_RESEARCH_MODEL.md).

[Overview](../MATURE_MOVING_FAULT.md) · Prerequisite: [direct model](model.md)
· Return to [configuration/code map](configuration.md)

## Separated storage

The default `separated` representation retains auxiliary $T_h,R_h$ and surface
$Z$. Physical access reconstructs
$\tau=T_h-R_h-B$, where $B=2\mathcal L[Z]$ has stress units. A zero initial $R$
does not justify omitting its subsequent evolution. During material motion,
[stress_evolution.cc](../../../../source/material_model/phase_field_fault/stress_evolution.cc)
transports $T_h$ and $U_h=R_h+P_hB$; it rotates these complete fields and subtracts
mapped $B$ to obtain the residual anchor. Transporting $R$ alone would omit
profile/geometry effects.

At the endpoint, surface memory obeys
$Z_j^+=\beta_s(X_j)Z_j^-+\kappa_s(X_j)V_j^+$. The shared reaction projector
returns two fields from one mass inversion per cell:

$$
 T_h^+=P_h[\beta_b\bar T_h+2\kappa_bD_3],\qquad
 R_h^+=P_h[\beta_b(\bar R_h+B^-)+2\kappa_b\mathcal L[V^+]-B^+].
$$

Here the B terms mean the fixed staged profile/normalization evaluated with
incoming and updated memory respectively, not an independent second geometry
update. Bulk and surface coefficients may differ spatially. This reconstruction
and its analytic localization remainder need not equal a directly projected
physical stress. These are discrete representation comparisons, not proof that
two coarse solutions are identical. Direct-mode diagnostic $Z$ does not enter
stress, resistance or elastic energy.

## Why some projection code remains separate

The common moving-field mass/RHS algebra is in
[TensorField::project](../../../../source/material_model/history/tensor_projection.cc).
Nearby routines were inspected; the following distinctions are intentional:

| Routine | Reason not to replace it blindly |
|---|---|
| `make_dg_history_trial` in [dg_history.cc](../../../../source/material_model/phase_field_fault/dg_history.cc) | Native Gauss3 strain-source projection, cached inverse and extended-precision roundoff estimates depend on differentiation operands and exact containment of native Q2 strain. This is more than evaluated-field projection. |
| `prepare_objective_dg_anchor` in [dg_objective.cc](../../../../source/material_model/phase_field_fault/dg_objective.cc) | Stationary separated rotation has an exact zero-spin coefficient-copy branch, and its retained componentwise publication ordering. Routing every cell through projection would destroy that identity. The moving direct branch does not use this algebra. |
| Geometry-only correction in [dg_geometry.cc](../../../../source/material_model/phase_field_fault/dg_geometry.cc) | A retained control projects a scalar shear correction between two geometry maps and adds it to old coefficients; it is not the moving material endpoint transaction. |
| Cohesive tensor/history spaces | Formed-piece/material-side partitions, native remainder and lifting have their own measures and persistence. Replacing those with an uncut DGQ2 projector changes the space and physics. |

The local tensor-space projector is numerical infrastructure; representation
reconstruction remains confined to the material owner and physical-stress access.
There is still one coupled mature mechanical solver.

## Other histories are not interchangeable

Native FE history stores displacement-like Maxwell memory in a mixed FE velocity
block and derives strain; DG mode uses that block only for phase. Production BP3
retains its selected legacy/default conventions and validation scope. Cohesive
history additionally stores physical elastic slip, formed-side tensors and
native/lifting memory with recoverable cohesive energy. Mature auxiliary $Z$ or
$R$ is not a cohesive energy inventory. No checkpoint conversion or production
restart follows from the current consolidation.

The guide's supported moving case has fixed Cartesian mesh, constant $G$,
positive prescribed spatial viscosity and growth disabled. Optional bounded Q1
transport improves retained profile comparisons but is not conservative and
does not remove finite-end tensor approximation error. Full coupled energy/flux
accuracy, moving AMR/birth and independent cohesive tip/energy gates remain open.
See [STATUS](../STATUS.md), [frozen finite-end evidence](../FROZEN_FINITE_END_REPORT.md)
and the [consolidation report](../STRESS_CONSOLIDATION_DOCUMENTATION_REPORT.md).
Inherited cleanup losses, including the recorded 39/45 native-pilot frames and
temporary logs, are not repaired or recreated by this work.
