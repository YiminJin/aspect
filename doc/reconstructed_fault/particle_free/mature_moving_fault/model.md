# Model and variables

[Overview](../MATURE_MOVING_FAULT.md) · Next: [discrete update](discretization.md)

## Physical equation

For a mature frictional fault, the selected bulk-corotational material law is

$$
 (\partial_t+v\cdot\nabla)\tau-W_b\tau+\tau W_b
 +\frac{G}{\eta_b}\tau
 =2G\bigl(D_3(v)-\mathcal L[V]\bigr),\qquad
 W_b=\operatorname{skw}(\nabla v-J_f),\quad J_f=\chi\widetilde V(s\otimes n).
$$

Here $(\nabla v)_{ij}=\partial_j v_i$. The stress is the deviatoric physical
stress, in the spatial frame; pressure is a separate mechanical unknown.
$\mathcal L[V]=\chi\widetilde V\widetilde S$ connects this notation to the
manuscript's extended fault rate and shear tensor. The material derivative uses
the full material velocity. The explicit `bulk slip subtracted` selector removes
only the regularized jump gradient from the constitutive spin. The retained
`total material` default uses $W=\operatorname{skw}\nabla v$ and is a different
constitutive law; historical total-spin results remain controls. Direct storage removes the auxiliary
localization reconstruction $T-R-B$; it does **not** remove physical objective
rotation. This is the small-strain Maxwell model with an objective stress rate,
not a finite-strain hyperelastic model.

The in-plane symmetric tensor stores $xx,yy,xy$ in deal.II component order.
Complete it with $\tau_{zz}=-\tau_{xx}-\tau_{yy}$ and zero $xz,yz$. For velocity
strain, embed the 2-D symmetric gradient in 3-D and subtract one third of its
3-D trace: $D_3=\operatorname{sym}\nabla v-\operatorname{tr}(\nabla v)I_3/3$.
Consequently the energy contraction of stored tensors is
$a:b+\operatorname{tr}(a)\operatorname{tr}(b)$, with the first contraction
in-plane; using a two-dimensional trace-free projection would change the law.
The elastic energy density is $\tau:\tau/(4G)$ with the completed tensor.

## Quantities at an attempted step

Superscripts $-$ and $+$ denote accepted parent and candidate endpoint. Bars
mean transported/rotated data fixed during the mechanical solve.

| Quantity | Units; location | Time/frame and source |
|---|---|---|
| $\tau_h$ | Pa; bulk DGQ2 coefficients | Accepted or endpoint spatial stress; `DGHistory::primary` in direct mode |
| $v,\nabla v$ | m/s, 1/s; bulk native FE | Accepted $v^-$ drives transport/spin; current $v^+$ drives strain forcing |
| $V$ | m/s; fault nodal scalar | Committed accepted rate for spin; private endpoint iterate for reaction. Never $Z/\Delta t$ |
| $D_3,W_b$ | 1/s; bulk quadrature | Endpoint strain, accepted spin respectively |
| $\phi$ | dimensionless; continuous Q1 | Transported endpoint physical profile and irreversibility obstacle |
| $h(\phi),I,\chi=h/I$ | 1, m, 1/m | Endpoint profile compliance, normal integral and localization |
| $S$ | dimensionless; extended fault frame | `shear_sense*sym(tangent outer normal)` on the associated segment |
| $G,\eta_b$ | Pa, Pa s; material samples | Constant $G$; positive spatial multiplier times reference viscosity |
| $\beta_b,\kappa_b$ | 1, Pa s | Endpoint coefficients sampled at bulk quadrature points |
| Age $\theta$, cumulative slip | s, m; material fault labels | Accepted incoming age enters friction; updated once on acceptance |
| $Z$ | Pa m; fault nodes | Auxiliary Maxwell slip memory; still evolved diagnostically in direct mode |

For the supported mature profile the compliance is
$h=m\phi(1+\phi)/(1-\phi)^2$ (the material's dimensionless degradation coefficient
$m$ is not a mass). No normalization floor or profile cap is inserted. Details
and support are in [geometry](geometry.md).

## Endpoint reaction

Freeze the endpoint forcing during the reaction interval. Integrating the
linear Maxwell relaxation gives

$$
 \beta_b=e^{-G\Delta t/\eta_b},\quad
 \kappa_b=-\eta_b\operatorname{expm1}(-G\Delta t/\eta_b),\quad
 F=\beta_b\bar\tau_h+2\kappa_bD_3(v^+)-\kappa_b\bigl(2\mathcal L[V^+]\bigr).
$$

`compute_maxwell_coefficients` calls $\kappa$ `eta_ve`. It has viscosity units;
$\kappa/G$ has time units. `Rheology::mature_stress_endpoint` in
[stress_law.h](../../../../include/aspect/material_model/phase_field_fault/stress_law.h)
expresses precisely this local law. `make_material_reaction` in
[stress_reaction.cc](../../../../source/material_model/phase_field_fault/stress_reaction.cc)
samples $\eta_b(x)$, native strain and localized rates, then projects the complete
field $F$. The actual update is $\tau_h^+=P_hF$, not the unprojected point law.
The surface Maxwell factors use viscosity at surface quadrature points, not a
separately homogenized bulk coefficient. The multiplier is evaluated at the
attempt's endpoint time.

An illustrative uniform relaxation has $D_3=0,V=0,W=0$ and no transport change:
$\tau^+=e^{-G\Delta t/\eta_b}\tau^-$. A uniform tensor lies in DGQ2, so its
projection is the same tensor up to roundoff. This checks the sign and units;
it says nothing about a narrow finite-end localization source, whose projection
is generally not exact. See the [analytical fixture](configuration.md#checks-and-outputs).

## Spin closure and its limits

The shared localization sample gives $2\operatorname{sym}J_f=2\mathcal L[V]$.
It interpolates $V$ and $I$ separately before division, using the same active
segment, signed tangent, normal and finite-end support as mechanics. Inactive
support gives $J_f=0$. No viscous spin is inferred from Maxwell relaxation.
For horizontal positive slip, $J_{01}=q$ and
$\omega_b=(L_{10}-L_{01}+q)/2$; pure slip $L_{01}=q$ has zero bulk spin.
Curvature and variations of slip generate regular gradients that remain:
$v=H(\rho)j$ implies $L-J_f=H\nabla j$.

Under an observer rotation, $J_f^*=QJ_fQ^T$ and
$W_b^*=QW_bQ^T+\dot Q Q^T$. The spin correction is objective and does no
stress work. With full 3-D contractions,
$D_t[\tau:\tau/(4G)]+\tau:\tau/(2\eta_b)=\tau:D_3-\tau:D_f$.
This selected small-elastic-strain closure is not an elastic polar rotation,
a sharp-tip qualification, or a finite-strain hyperelastic law. Particle
histories and evolving cohesive histories reject this selector because their
kinematics and source rates have not been derived for it.
