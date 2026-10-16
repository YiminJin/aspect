# Verification cases and evidence

These are specifications, not completed tests. The repository session supplies
real source paths, executable commands, and measured results. Run cases relevant
to the current phase and implementation scope.

## Measurement rules

Use a declared scaled defect:

$$
E=\frac{\|q-q_{\rm ref}\|}
{\max(q_{\rm scale},\|q_{\rm ref}\|)}.
$$

`q_scale` is a fixed meaningful scale with the same units, chosen before
observing results. Do not divide a near-zero answer by its own norm. Record
absolute errors too. Specify volume/surface measures and norms; tensor norms
include the reconstructed plane-strain `zz` component.

For well-scaled double-precision local identities, target `E <= 1e-12`.
For assembled adjoint identities with modest conditioning, target `E <= 1e-10`.
If conditioning, solver tolerance, or reductions set a larger floor, quantify
and justify it before judging the result. These are not PDE error thresholds.

For convergence, use at least three levels with non-target errors made smaller.
Declare the expected order for the actual discretization and regularity; inspect
errors and slopes. Explain pre-asymptotic behavior. Nonsmooth tests need an
appropriate weaker norm rather than the smooth formal order.

Use fixed seeds. One CSV/JSON summary per case family is enough. A failure
should produce a failing assertion/nonzero exit, not just a plot. Solver success
is insufficient. A reference calling the same unverified operator as the
implementation is not an independent validation.

## Case matrix

| ID | Main question | Earliest phase |
|---|---|---|
| V0 | What does the unchanged branch do? | PF0 |
| V1 | Are relaxation, units, tensor conventions, initialization correct? | PF1 |
| V2 | Are product-rule and material-coefficient terms retained? | PF1 |
| V3 | Does transport commute with extension in the claimed cases? | PF1 reference; PF4 discretization |
| V4 | Can current-profile-only closure lose mature memory? | PF1 |
| V5 | Does storage retain the native FE source? | PF2 |
| V6 | Are extension and fault loading work adjoints? | PF2 |
| V7 | Are phase energy, residual, and tangent consistent? | PF3 |
| V8 | Is a narrow known kernel integrated accurately? | PF3 |
| V9 | Are accepted histories, rollback, restart consistent? | PF2; PF5 |
| V10 | Does DG solve the intended material equation? | PF4 |
| V11 | Does limiting help without damaging smooth accuracy? | PF4 if needed |
| V12 | Are geometry/AMR/MPI transfers correct? | PF5 |
| V13 | Does bounded coupled propagation agree under refinement? | PF6 |

## V0: retained baseline

Use one small existing propagation/native-history fixture. Record base commit,
dirty state, executable identity, input, mesh/mapping, quadrature, boundary
conditions, time-stepping limits, and filters. Save a compact accepted-state
summary. Establish whether the path is cohesive or mature and whether it uses
spin. No full event is needed. Distinguish build blockage from scientific failure.

## V1: local Maxwell and initialization

For frozen geometry, constant coefficients, and constant imposed strain/slip
rate, use the analytic solution:

$$
\beta=e^{-\Delta t/\lambda},\quad \kappa=\eta(1-\beta),\quad
\tau^{\rm tr}_{+}=\beta\tau^{\rm tr}_{-}+2\kappa\dot\varepsilon,\quad
s_{+}=\beta s_{-}+(\kappa/G)V.
$$

Test small, order-one, and large `dt/lambda`, zero forcing, and nonzero compatible
initial stress/slip. Verify physical stress, cohesive traction, units, trace,
and `S:S = 1/2` factors. Check cancellation in `1-beta` at small steps.
Include initial data producing narrow trial history; this prevents an
unsupported universal-smoothness claim.

## V2: evolving localization and coefficients

Prescribe a normalized profile with changing width and a nonzero slip moment.
Evaluate the derivative of reconstructed stress independently. Confirm the
sign of `Q_geo = -2G C[s_elas]`. Contrast with a pure amplitude change of `h`
that leaves normalized `chi` unchanged. The omitted-product-rule defect is
nonzero in the first case and cancels in the second.

Check the cohesive physical phase increment at fixed trial stress/elastic slip,
and separately the mature numerical remap at fixed physical state. These have
different expected stress changes.

Include a normally varying relaxation rate and a separate varying-modulus case
against `eq:variableG` for the explicitly stated law. At fixed geometry/phase,
check the cohesive power balance `eq:fixedenergy`. This does not verify general
moving-surface energetics.

## V3: transport and the commutator

Start with smooth prescribed fields and independent characteristic solutions:

1. Frozen profile without advection/spin: extension commutes.
2. Profile and surface labels rigidly translating with material: extension
   commutes under compatible definitions.
3. Rigid rotation with co-rotating profile and slip tensor: the Jaumann rate
   vanishes for pure tensor rotation, apart from imposed relaxation.
   Label this an objective-law test; do not silently alter a non-objective law.
4. Material translating through a spatially fixed narrow profile:
   `velocity dot grad(chi)` generally contributes.
5. Along-fault variable slip memory with differing bulk/surface tangential
   transport: `b dot grad_F(s)` contributes.

Compare direct stress with the correct transformed equation. Retain residual
history in the mature test and redistribution in the cohesive test. An omitted
nonzero term is a negative control: detect its error rather than accept it.

## V4: old profiles and mature closure

Apply slip in two intervals with distinct nonproportional normalized profiles,
such as separated centers. Compute mature history with an independent
exponentially weighted characteristic integral. Compare direct stress; trial
plus auxiliary memory plus residual; and trial plus auxiliary memory alone.

The first two agree at their declared error level. The third retains a finite
model discrepancy as numerical error shrinks. Establish that expected nonzero
discrepancy analytically before testing. Do not assign cohesive resistance to
auxiliary memory to improve the comparison. This is a closure test, not a
comparison of the superiority of particles and DG.

## V5: native FE source

On affine Cartesian cells, prescribe Q2 velocity with cross terms, not only
affine velocity. Check DGQ2 source representation using adequate quadrature;
include a mode lost by DGQ1 as a negative control. Then use the actual production
mapping and quantify projection error.

Compare constitutive weak loads and accepted history generated by the same
source. Verify full rank of the chosen DG mass matrix. `2 x 2` Gauss points
cannot produce a full-rank nine-DoF DGQ2 scalar mass matrix.

## V6: discrete work adjoint

For independent surface coefficients and symmetric deviatoric bulk tensors,
compare `integral_B tau:L[V]` and `integral_F (L* tau) V` using the actual
mass/projection conventions. Include an oblique fault and a completed boundary
profile if supported; include curved Jacobians only for implemented geometry.
Trace failures to quadrature, association, normalization, ownership, or adjoint.
Centerline sampling is not an exact reference.

## V7: phase energy, derivatives, and bounds

At fixed incoming trial stress, elastic slip, and geometry, compare directional
energy differences with the residual, and residual differences with the tangent.
Include projected `I_h` derivatives. Sweep perturbation size to separate
truncation from cancellation error.

Use an interior admissible state for ordinary derivatives. Check active bounds
and irreversibility separately through the variational inequality/KKT residual.
Check cohesive stress redistribution after physical phase change.
Handle inactive and nearly singular profiles explicitly. Derivative consistency
does not prove a nonconvex stationary point is a stable minimum.

## V8: quadrature versus representation

At fixed mesh, increase integration accuracy for a known sharp `chi` and compare
normalization, energy, and weak loads with an independently resolved integral.
Include a profile with important width smaller than a cell.

Vary independent phase/history spaces separately. Report error that converges
with quadrature and error persisting from missing degrees of freedom. For CPDI
versus FE integration, name both discrete functionals and study their refinement;
roundoff agreement between different discretizations is not expected.

## V9: accepted state, rollback, and restart

Repeat a trial evaluation with fixed accepted input and require identical
output. Reject a step and verify every accepted bulk/surface/phase history is
restored. Accept once and verify one publication, not one per Newton iteration.
Compare restart with uninterrupted advancement at the same time-step sequence.
Inspect auxiliary/residual histories too. State whether equality is exact,
roundoff-level, or limited by a documented transfer/serialization operation.

## V10: DG material transport

Transport a smooth tensor pulse with a known characteristic solution and
prescribed velocity. Refine space and time separately. Check symmetry/trace,
inflow, and the selected rotation convention.

Use manufactured velocity with nonzero divergence to distinguish material
transport from density transport: a spatially constant tensor under pure
material advection remains constant. This is an operator test, not a claim
that the physical Stokes solution is compressible.

Then transport signed nonsmooth data without arbitrary fixed stress-component
bounds. Measure overshoot/undershoot, norm error, and dissipation. These data
establish whether a limiter is needed.

## V11: limiter, if needed

Compare the same input with limiter disabled/enabled. Verify cell means,
constant states including zero, smooth order, affine reconstruction where
applicable, degenerating stencils/weights, and finite outputs. Check tensor
symmetry/trace and covariance under a consistent change of tensor basis on a
fixed spatial stencil. This does not prove arbitrary mesh-rotation invariance.

For historical code inspect detector activation, constant-candidate weight rank,
quadrature/reconstruction degree, and negative-weight treatment before reuse.
Check stencil/ghost availability under AMR/MPI. Demonstrate a measured benefit
on a troubled case without unexplained smooth-order loss. Do not describe WENO
as universally bound- or energy-preserving.

## V12: transfers and geometry

For a mature representation change at fixed trial history, verify

$$
\tau_{\rm res}^{+}=\tau_{\rm res}^{-}
+2G\mathscr L^{-}[s_{\rm mem}^{-}]
-2G\mathscr L^{+}[s_{\rm mem}^{+}],
$$

preserving physical stress at the relevant error level. A coordinate relabeling
with unchanged physical extension causes no physical change. Contrast with
physical cohesive phase evolution.

Compare serial and a small multi-rank case, including partition boundaries.
Test refinement/coarsening and the material phase lower bound; quantify
projection error rather than demand impossible exactness after coarsening.
If surface birth is required, test its selected rule explicitly; the remap
identity does not supply that rule.

## V13: bounded coupled propagation

Use matched small-case inputs and record unavoidable functional differences.
Compare shear/normal traction, `V`, `Theta`, `V*Theta/D_c`, phase increment,
stress/history norms, cumulative slip, work/energy defect, and nonlinear
iterations. Add runtime/memory and remaining particle dependencies.

Judge against a predeclared error budget and refinement trend. Particle output
is a control, not automatically ground truth. Start with a near-onset or short
transient; a full BP3 event is unnecessary. A smoother plot alone is insufficient.
