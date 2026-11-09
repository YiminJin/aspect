# Discrete equations and executable operations

[Guide](../TIP_TREATMENT.md) · [current integration evidence](../TIP_SOLVER_INTEGRATION_REPORT.md)

This page describes the integrated `Bulk tip solver = ASPECT` route. The older
`benchmark direct` route is an explicit small reference, with its original
add-only search and replicated algebra. Its convergence is not the definition
of the corrected algorithm. Implementation and verification are distinct; consult
the report for the configurations actually tested.

## One accepted step: reading path

1. [`Simulator::solve_single_advection_iterated_newton_stokes`](../../../../source/simulator/solver_schemes.cc)
   selects the integrated route and borrows the immutable accepted bulk owner.
   All auxiliary compositions and temperature are static. Initialization is a
   zero-clock operation; it advances neither Maxwell memory nor section age.
2. [`internal::BulkTipCoupling`](../../../../include/aspect/simulator/solver/bulk_tip.h)
   prepares cell Bernstein coordinates, the accepted obstacle, fixed boundary
   displacement, and the authoritative distributed fracture operator.
3. [`BulkTipCoupling::evaluate`](../../../../source/simulator/solver/bulk_tip_assembly.cc)
   integrates the free section rate and mobility, eliminates section resistance,
   evaluates physical point responses, and assembles residuals and tangents.
   Its lifetime is one physical attempt; evaluation never publishes history.
4. The ordinary solver scheme owns nonlinear convergence and functional line
   search. [`Simulator::compute_bulk_tip_search_direction`](../../../../source/simulator/solver/bulk_tip_linear.cc)
   solves a linearized box problem, allowing active constraints to enter **and
   leave**. Flexible Krylov applies the exact coupled tangent. Its approximate
   block preconditioner invokes ordinary ASPECT Stokes assembly, AMG construction
   and `solve_stokes()` for velocity/pressure, and the phase fracture matrix for
   the phase block. It does not hand phase unknowns to `solve_stokes()`.
5. Every trial recomputes the full point, section, mechanics and phase response.
   Only final physical residual/KKT and energy checks permit the ordinary
   pending-publication transaction to install all histories together. A rejected
   attempt leaves the accepted owner intact; the simulator restores fields.

## Coordinates and point elimination

Write \(u=\Delta t\,v\), \(\pi=p/G\), and \(\phi=EB\), where E maps independent
cell Bernstein coefficients B to native phase coefficients. E is the identity
for the active native Bernstein fixtures; only the retained nodal route uses
a conversion. See the [basis guide](../NATIVE_DG_BERNSTEIN.md). Material
quadrature evaluates the same polynomial directly with positive Bernstein basis
functions, avoiding cancellation near intact values. The polynomial space and quadrature rule are unchanged. The displacement
boundary value is \(u_D=\Delta t\,v_D\). Corrections have homogeneous velocity
boundary values. One pressure coefficient fixes the outer operator's gauge;
frictional compression \(p_N\) is separately prescribed, not that gauge.

For accepted parent \((\tau^-,A^-,\phi^-)\), define \(e^-=2A^-:S\),
\(\beta=(1+\Delta tG/\eta)^{-1}\) and \(P=\beta(\tau^-+2G D_3u)\).
The fixed direction has \(S_{xy}=S_{yx}=1/2\), \(S:S=1/2\). The full 3-D
plane-strain deviator \(D_3\) retains its zz component.

<a id="point"></a> **(P) Stable endpoint elimination**
\[
c=m\phi(1+\phi),\quad d=(1-\phi)^2,\quad H=\frac{c}{c+d},\quad
F=P:S+G\beta e^- - f_s,
\]
\[
e=HF/G,\quad q=dF/(c+d),\quad A=eS,\quad Q=2qS,
\quad \tau=P-2G(e-\beta e^-)S,\quad
w=(e/\beta-e^-)/\Delta t,
\]
\[
Y=\frac{m(1+3\phi)(1-\phi)}{(c+d)^2}\frac{F^2}{2G}.
\]
[`BulkCohesive::Tip::update`](../../../../source/material_model/phase_field_fault/bulk_tip.cc)
uses `denominator=c+d`, `fraction=H`, `force=F`, and `driving_factor` for the
coefficient preceding \(F^2/(2G)\). The old local name `z` for c+d has been removed;
it must not be confused with the force-like symbol in older derivations.
No evaluation of infinite h or a 0/0 product is required at either endpoint.
`Tip::derivative` differentiates this complete response at fixed accepted parent.

<a id="section"></a> **(S) Section closure and exact condensation**
\[
V_s=\sum_{q\in s}dn_q w_q=B_s-C_s f_s,\quad
C_s=\sum_{q\in s}\frac{dn_q H_q}{G\beta\Delta t},\quad f_s=R_s(V_s).
\]
Here B_s denotes a free rate, **not** a Bernstein coefficient. Actual discrete
measures satisfy \(JxW_q=ds_s\,dn_q\). Only fault deformation w is integrated;
ordinary elastic and viscous deformation are excluded.
[`FaultFriction::section_endpoint`](../../../../source/material_model/rheology/fault_friction.cc)
solves \(V_s+C_sR_s(V_s)=B_s\). At the current solution,
\[
\delta f_s=\frac{R_s'}{1+C_sR_s'}
 \sum_{q\in s}dn_q\,\delta w_q\big|_{f_s\;fixed}.
\]
This retains all mechanics/phase cross derivatives, including \(\delta C_s\).
The exact tangent is a local sparse matrix plus section integration/adjoint
products. Only the small section reductions are replicated across MPI ranks.

## Weak equations, scaling and tangent blocks

<a id="residual"></a> **(R) Physical residual in solver coordinates**
\[
(R_u)_i=\int_\Omega[(\tau/G):D_3N_i-\pi\nabla\!\cdot N_i]\,dx,
\qquad (R_\pi)_j=-\int_\Omega N_j^p\nabla\!\cdot u\,dx,
\]
\[
R_B=\frac{1}{K_f}E^T\left[\partial_\phi E_f-\int_\Omega N^\phi Y\,dx\right],
\qquad K_f=\frac{G_c}{(8/3)\ell}.
\]
Natural fracture boundaries contribute no prescribed flux. Closed prescribed
velocity boundaries supply reaction work; no external traction term is present
in this supported fixture. Pressure does not enter R_B or the prescribed RSF
compression. The phase variational inequality is
\[
B^-\le B\le1,\quad R_{B,i}=0\text{ if free},\quad
R_{B,i}\ge0\text{ at the lower bound},\quad R_{B,i}\le0\text{ at the upper bound}.
\]
The Bernstein cone ensures polynomial bounds and irreversibility, but is a
sufficient cone narrower than all pointwise-admissible Q2 polynomials.

For AT1, the fracture functional is the native cell potential plus SIPG gradient
energy. Writing its gradient coefficient as \(k_f=2G_c\ell/(8/3)\),
\[
E_f=\int_\Omega K_f\phi+\tfrac12 k_f|\nabla\phi|^2\,dx
-\sum_F\int_F\{k_f\nabla\phi\}\cdot n[\phi]\,ds
+\tfrac12\sum_F\int_F\sigma_F[\phi]^2\,ds.
\]
The derivative includes both consistency/symmetry face terms and the penalty;
it is not a cell-only Laplacian. The authoritative implementation is
[`PhaseFieldHandler::assemble_dg_fracture`](../../../../source/phase_field/fracture.cc)
and [`PhaseField::DGOperator::assemble_sipg`](../../../../source/phase_field/dg_sipg.cc).
The compatibility `evaluate_dg_fracture` observer still gathers for old consumers;
the active solver consumes distributed owned rows and the actual cell/face stencil.

Before condensation the blocks are mechanics–phase–section; after condensation,
\[
J=\begin{pmatrix}J_{uu}&-D^T&J_{uB}\\-D&0&0\\J_{Bu}&0&J_{BB}\end{pmatrix}
 +\sum_s a_s\frac{R_s'}{1+C_sR_s'}c_s^T.
\]
Here c_s integrates the fixed-force rate derivative; a_s is the force derivative
of mechanical/phase residuals. \(J_{uB},J_{Bu}\) and the section products are
nonzero. The retained nodal route applies dual/primal E transformations to phase rows/columns; native Bernstein assembly already has those coordinates.
The phase scaling K_f and mechanical scaling G are algebraic only. For the
restricted functional, row weights are \(\beta G\) and K_f, respectively.
The functional-weighted tangent is checked for symmetry on the free subspace.
The retained Krylov implementation uses the stated solver coordinates directly;
a tested alternative scalar block scaling did not improve iteration counts and
is retained only as benchmark evidence in the report.


Each rank owns cell quadrature histories, native FE rows and local section
support. Assembly adds ghost contributions and compresses collectively. No
all-pairs phase graph or globally replicated FE vector is required. The reference
Stokes inverse handles its usual pressure nullspace; the adapter reconstructs
the omitted continuity row using partition of unity and subtracts the output
pressure gauge. This is distinct from changing the prescribed compression.

## Exact operator and approximate inverse

The retained preconditioner is block diagonal. A triangular candidate added the
exact phase-to-mechanics block but increased measured wall time and was rejected.
No physical tangent block is dropped from the outer solve.

| Operation | Owner and actual work | Mathematical role |
|---|---|---|
| Full J action | `BulkTipCoupling::vmult`: distributed cell/face matrix and section integration/adjoint | Exact physical derivative, including cross-blocks |
| Mechanical reference inverse | `Simulator::apply_bulk_tip_stokes_inverse`: ordinary ASPECT assembly, AMG and `solve_stokes` | Approximate coupled Krylov preconditioner; preserves pressure quotient and coordinate scaling |
| Positive phase inverse | `BulkTipCoupling::phase_search_matrix` and AMG | Fracture-plus-mass approximation; not the possibly indefinite J_BB |
| Shifted phase minimizer | `PhaseField::solve_obstacle` | SPD bound-constrained search subproblem |
| Mechanical correction after imposed phase change | Exact restricted J action with FGMRES and ASPECT Stokes preconditioning | Includes induced section-force correction and homogeneous incompressibility |
| Search ordering and acceptance | `Simulator::solve_single_advection_iterated_newton_stokes` | Fresh physical response, restricted-functional Armijo and final physical gates |

The exact-search linear tolerance remains
`max(1e-16, min(1e-10 * ||rhs||, 1e-10 * minimum_phase_mass))`; the ordinary
Stokes reference inverse retains 1e-4. Neither is an adaptive forcing policy.
The search-order merit is
\(r=\max(\|R_m\|/\epsilon_m,\mathrm{KKT}/\epsilon_\phi)\).
The Newton retry condition \(r\le\max(10,0.1r_{last\,Newton})\) does not
change acceptance at r≤1. Line search remains on the restricted functional,
with the existing scaled-residual tie-break for roundoff-sized energy changes.

## Bounds, search and physical acceptance

<a id="bounds"></a> **(B) Linearized box subproblem**
At nonlinear x, first solve stationarity for \(Jd+R\), subject to
\(B^-\le B+d_B\le1\). Classification uses both current primal values and
multipliers through [`PhaseField::box_constraint`](../../../../include/aspect/phase_field/box_constraints.h).
Every pass reclassifies both signs; previously constrained coordinates may
release. Fixed-bound displacements enter **all** coupled right-hand-side blocks.
The free equations are solved with flexible Krylov, not the SPD obstacle solver.

At the first iteration of each physical step, try the exact coupled Newton VI.
If it fails to supply an accepted correction, prefer the positive phase-search
family on subsequent iterations, testing its least restrictive metric first.
Retry exact Newton when the scaled mechanics/KKT residual falls tenfold since
its previous attempt, or is within one decade of the unchanged final gates.
Successful exact corrections restore exact-first ordering; an exhausted shifted
family also falls back to exact Newton. This is a search-order policy, reset at
each physical step. Every trial still receives fresh bounds, coupled correction,
constitutive evaluation and globalization; no active set or direction is cached.
See the [efficiency report](../TIP_EFFICIENCY_AND_GROWTH_REPORT.md) for validation
and rejected variants. The positive phase search metric is
\(P_B=H_f/K_f+\mu M_B\), with \(\mu>0\). The existing SPD obstacle solver
minimizes \(R_B\cdot(y-B)+\tfrac12(y-B)^TP_B(y-B)\) over the true bounds.
The imposed phase displacement y−B then enters an **exact coupled mechanical**
linear correction, including the condensed section derivative. This is a search
direction, not a replacement constitutive update or a one-pass partitioned solve.
Increase the search metric when needed and recompute the entire nonlinear
response at each trial. A backtracked step satisfies the Armijo test on the
[restricted incremental functional](energy_and_validation.md#restricted-incremental-stationarity-functional),
with derivative \(\beta G R_u\cdot d_u+K_fR_B\cdot d_B\) on incompressible
variations. This functional is not the physical work ledger. Full residuals are
recomputed after every trial; there is no post-clipping certificate reuse.
Prescribed active rows are imposed exactly inside the linear solve and the
complete linearized residual is rechecked afterwards. Bound-directed interpolation
uses the prescribed target to avoid cancellation of tiny positive accepted
obstacles in B+(Bminus−B); arbitrary infeasible directions are not clipped.
When functional differences fall below summation roundoff, the line search also
requires decrease of the scaled physical residual. Resolvable decreases obey
Armijo; the tie-break does not weaken final physical gates.
Final gates retain the input mechanical and phase tolerances and physical ledger.
Nonlinear convergence is not a constrained stability or uniqueness certificate.

Zero-load fracture equilibration has an SPD quadratic contract. It uses the
existing [`PhaseField::solve_obstacle`](../../../../source/phase_field/obstacle.cc)
with a positive search mass, then verifies the **unmodified** fracture KKT. The
result becomes the initial obstacle before physical aging; no accepted loaded
state is smoothed.

## Equation-to-test trace

| Equation | Executable entry | Independent check |
|---|---|---|
| P: tau/A/Q/Y endpoint | `Tip::update`, `Tip::derivative`, `Tip::account` | [point tests](../../../../unit_tests/bulk_tip.cc): endpoint/reversal/finite derivatives/power |
| S: signed section rate, RSF, age | `FaultFriction::signed_endpoint`, `section_endpoint` | [checker](../../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/checks.cc): signed/condensed derivative, discrete power identity |
| R: DG fracture | `PhaseFieldHandler::assemble_dg_fracture` | checker SIPG energy/Hessian derivative and compatibility observer |
| R: full coupling | `BulkTipCoupling::evaluate`, `vmult` | checker distributed coupled directional derivative; benchmark oracle comparisons in report |
| B: primal/dual release | `Simulator::compute_bulk_tip_search_direction`, `PhaseField::box_constraint` | point test “Coupled box iteration releases incorrect lower and upper bounds” |
| Physical lifecycle | accepted bulk owner, pending simulator publication | checker rejection/retry, checkpoint and rank-change checks |

Mechanical quadrature remains configured Gauss6; fracture remains tested Gauss5
on cells and faces. Polynomial fracture terms are integrated exactly on the
constant-coefficient affine fixture; rational cohesive factors are not generally
exact under Gauss6. Neither this integration nor solver convergence establishes
quadrature convergence, moving-history transport, or general tip propagation.
