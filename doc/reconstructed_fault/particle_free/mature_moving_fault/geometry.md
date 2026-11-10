# Geometry, profile and localization

Current support is native DG Bernstein phase and owned-row stress transport on
all ranks. Q1 remap/MacCormack, CPDI and nodal-phase descriptions below are
historical comparisons only; their implementations are retired. The old
`Material profile transport` name is parsed solely for checkpoint identity and
its retained support admission. It does not select an algorithm. See the
[active reading path](../ACTIVE_RESEARCH_MODEL.md).

[Overview](../MATURE_MOVING_FAULT.md) · Prerequisite: [discretization](discretization.md)
· Next: [mechanics](mechanics.md)

## Material labels and support

`stage_material_geometry` in
[material_geometry.cc](../../../../source/material_model/phase_field_fault/material_geometry.cc)
samples accepted velocity and its gradient at accepted fault vertices using
MPI remote-point evaluation. All requested curve points must be covered by the
native FE mesh. `ReconstructedFaultManager::stage_material_motion` in
[manager.cc](../../../../source/reconstructed_fault/manager.cc) clones the
geometry and generic per-label properties, then applies

$$
 X_j^+=X_j^-+\Delta t\,v^-(X_j^-),\qquad
 F_j=I+\Delta t\nabla v^-(X_j^-),\qquad
 w_j^+=w_j^-/\lVert F_j^{-T}n_j^-\rVert.
$$

A nonpositive determinant is rejected. Support normals are geometric covectors;
segment tangents/normals are rebuilt from the moved vertices. They are not
rotated with the stress-spin tensor. The same ordered labels carry age, slip and
memory; relative fault slip $V$ does not advect them. There is no birth,
resampling, tangential redistribution or AMR in this admitted calculation.

The private manager is swapped into the live geometry only after staging. The
candidate native phase profile is bound with it, normalization and surface temperature are
rebuilt, and old geometry is retained for rollback. Binding exposes a coherent
candidate to mechanics; it is not acceptance.

## Profile and material irreversibility

For the historical continuous Q1 selection, `transport_material_profile` (retired source)
visits owned Q1 vertices $x_i$ and pulls back the accepted field:
$\phi_F(x_i)=\phi_h^-(x_i-\Delta t v^-(x_i))$. The default is this `Q1 remap`.
Remote-point averaging handles points shared by cells. Missing departure
coverage supplies the existing zero incoming profile; it is not extrapolated
from an arbitrary endpoint. This is an interpolation remap, not a conservative
phase-density scheme.

Optional `bounded MacCormack` evaluates the reverse pullback of $\phi_F$ at
$x_i+\Delta t v^-(x_i)$ and computes

$$
 \phi_C=\phi_F+\tfrac12(\phi_h^- -\phi_R),\qquad
 \phi^+=\operatorname{clamp}(\phi_C,\phi_{donor,min}^-,\phi_{donor,max}^-).
$$

The bounds are extrema of accepted donor-cell vertices, including every cell
admitted by the remote-point search. Missing reverse coverage uses the forward
value without error correction. The resulting values must be finite and in
$[0,1)$. In this growth-disabled configuration, the **same transported field**
is the physical phase and the material irreversibility obstacle. An Eulerian
maximum with the old field would leave a damage trail and is deliberately not
taken. The optional profile/inflow extension requires `full normal cross-section`
association; stationary seed support could clip a separately transported profile
whose tails still contribute to normalization.

The nodal pullback and independent donor-range clamp do not enforce the
integrated material-transport balance
$d\int\phi\,dx/dt=\int\phi\nabla\!\cdot v\,dx-\int_{\partial\Omega}\phi v\!\cdot n\,ds$.
Bounds and the transported obstacle therefore do not establish conservation
when the physical divergence and boundary terms vanish. The
[layered campaign](../LAYERED_BULK_CAMPAIGN_REPORT.md) measures 7.04% profile-mass
growth on its coarse small-timestep case and 2.39% on the refined mesh, with
substantial stress/slip sensitivity. This remains a profile-discretization
limitation; the campaign preserves the operator while assessing its accuracy.

## Actual association and extension

`project_to_bulk_source` first uses normal-profile association. In
[project_to_normal_profiles_unchecked](../../../../source/reconstructed_fault/utilities.cc),
for every segment it computes the un-clamped tangential coordinate $\xi$.
Only $0\leq\xi\leq1$ and normal distance within the interpolated support width
are admitted. Within one fault the smallest admitted normal distance wins;
admission to multiple faults is unsupported and throws. In particular, this is
**not** a globally nearest-point map with endpoint clamping. A point beyond an
interior finite endpoint can have no source even when its Q1 phase is nonzero.

`project_to_bulk_source` also contains separately qualified boundary-completion
branches. The moving interior finite-curve configuration does not enable these
branches or invent continuation beyond its physical ends. Full-normal support
extends transverse coverage, not the tangent interval. The resulting finite-end
source jump is a principal source of uncut-DGQ2 error in the
[frozen comparison](../FROZEN_FINITE_END_REPORT.md).

For an admitted segment $j$ and $\xi$, interpolate nodal values and normal
integrals **separately**:

$$
 a(\xi)=(1-\xi)a_j+\xi a_{j+1},\quad
 I(\xi)=(1-\xi)I_j+\xi I_{j+1},\quad
 \texttt{localize\_surface\_values}(a)=\frac{2h(\phi)a(\xi)}{I(\xi)}S_j.
$$

[localize_surface_values](../../../../source/material_model/phase_field_fault/dg_objective.cc)
returns zero outside support. Inside, it requires finite values and positive
$I$. $S_j=\text{shear sense}\,\operatorname{sym}(t_j\otimes n_j)$ is in the
spatial frame, so its off-diagonal component for an axis-aligned fault is one
half. The explicit factor two is essential. With $a=V$ (m/s) the result is
$2\mathcal L[V]$ (1/s); with $a=Z$ (Pa m) it is $B$ (Pa). The shared operation
has no separate memory law and no additional division by $I$.

`compute_normalization_integrals` integrates profile compliance along the
admitted normal rays and stores nodal $I$; localization uses the current
profile and those integrals. The fixture selects the retained `cell intervals`
backend, surface subdivisions eight and existing integration tolerances.
[Normalization code](../../../../source/material_model/phase_field_fault/normalization.cc)
and native sampling use the matching geometry/profile data. The same shape
functions and measures must be used in the mechanical work adjoint: replacing
association, normalization or support in just one consumer breaks this contract.

For native DGQ2 phase, ordinary simulator field dispatch transports the semantic
phase composition before `stage_material_geometry` receives and binds it. All
nine scalar modes per cell enter sampling and derivatives. A shared DG point
selects the lexicographically greatest persistent CellId before nonlinear
localization; cell-clipped rays use the same rule. See the
[phase contract](../DGQ2_PHASE_FIELD_DESIGN.md). Scalar mapped-mesh support does
not remove the Cartesian restriction of coupled tensor history.
