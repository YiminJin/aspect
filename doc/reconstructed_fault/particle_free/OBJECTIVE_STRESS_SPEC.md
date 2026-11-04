# Objective stress rotation: audit, implementation, and qualification

This is a task specification to reconcile with live source and governing
documents. It does not assert that the current implementation has been audited.
The user selects adding stress-spin terms; legacy trajectories remain controls.

## 1. Fix the meaning of every derivative and spin

For the candidate Jaumann convention, define the velocity gradient by
`L_ij = partial v_i / partial x_j`,
`D = (L + L^T)/2`, and `W = (L - L^T)/2`.
For a spatial symmetric tensor A,

    D_t A = partial_t A + v . grad A
    J_W[A] = D_t A + A W - W A.

Thus an isotropic, constant-parameter Maxwell branch would satisfy

    J_W[tau] + tau/lambda = 2 G D_branch_dev,
    lambda = eta/G,

or, equivalently,

    D_t tau = 2 G D_branch_dev - tau/lambda + W tau - tau W.

This sign convention must be checked against actual FE gradient indexing and an
independent rigid-rotation solution. In 2D retain the embedded full 3D
plane-strain stress/deviator convention; do not silently replace it with a 2D
trace-free tensor. Hydrostatic pressure commutes with W; that does not remove
the need for correctly framed normal-traction evaluations.

The branch forcing is a symbol to resolve from the existing kinematic law. In
the production mixture it may involve a localized slip strain rate; in the
newer displacement-jump interface the bulk forcing may be smooth FE strain.
Do not assume that both routes have the same volume plastic-rate subtraction.
Do not alter G, eta, localization, friction, contact, or damage to offset effects
of the new term. Variable coefficients and moving structural fields require
their existing product/transport terms.

Audit the manuscript's objective operator. A generic corotational derivative
could use a spin Omega different from W. Jaumann, Green–Naghdi, and logarithmic
rates are constitutive choices; they are not interchangeable integration
options. If an existing, explicit convention selects another spin, identify
the conflict rather than silently substituting Jaumann.

## 2. Distinguish localized slip from material rotation

For an illustrative flat band with localization chi, unit slip vector s and
normal n, a simple regularized slip gradient is

    L_slip = chi V (s tensor n),
    D_slip = chi V sym(s tensor n),
    W_slip = (chi V/2) (s tensor n - n tensor s).

This is a local kinematic illustration, not the full curved-fault extension
operator. Its coefficients/units must match the implemented localization.
Specifying only D_slip does not determine an elastic/inelastic spin split.

Determine whether the computed L contains this regularized jump gradient,
whether the physical material uses total W, and what spin the existing bulk,
fault-elastic, and structural histories are assumed to follow. In particular,
check whether large V/width gradients would rotate material histories merely
because a sharp jump was represented by a diffuse band. A width dependence is
a finding to investigate, not automatic permission to remove part of W.

Do not automatically adopt W - W_slip. Such a choice needs a kinematic and
constitutive derivation (including any finite elastic distortion assumptions),
objective transformation rules, and a compatible energy statement. If total W
is the selected Jaumann spin, state that assumption and test its consequences.

Distinguish spatial tensor rotation, change of components into a segment frame,
actual evolution of a fault frame, and numerical transport between quadrature
points. Apply each once. Frozen physical geometry in a benchmark does not
justify rotating scalar histories or silently evolving the fault orientation.

## 3. Audit all independent histories before writing the update

List every stress-like tensor, its constitutive role and coordinate frame:
physical branch stress; DG coefficients; the full native FE gradient remainder;
geometric/cohesive tensor corrections; physical recoverable elastic tensor;
diagnostic/auxiliary memory; and particle fields if still used by observers.

Derive the evolution of the actual stored independent variable. If physical
stress is reconstructed as an independent tensor plus geometry/strain terms,
derive the transformed history equation from that reconstruction. Rotating
only the stored DG term can be incorrect. So can treating every coefficient
as a physical stress tensor. If rotation varies spatially, reconstruction,
pointwise rotation, and projection generally do not commute; quantify this.

Scalar age, accumulated scalar slip, scalar damage, and pressure do not receive
tensor commutators. Their values still depend on consistently framed mechanics
and appropriate transport. Resolve signed slip/frame conventions explicitly.

Find where the native velocity gradient and W are sampled, and whether they
are accepted, lagged, or current trial values. Include the nonlinear feedback
through v, stress, pressure/contact, and RSF. If W depends on the current
velocity, the residual must include that dependence and the Jacobian or outer
iteration must address it. A lagged iteration is permissible when specified
and converged; do not call it a consistent Newton tangent.

## 4. Integrate rotation without inventing dissipation

For constant W during a rotation substep,

    R = exp(Delta t W),
    tau_rot = R tau_old R^T.

This exactly integrates that frozen-spin rotation and preserves symmetry,
trace, eigenvalues and quadratic tensor norm. It is a preferred building block
over an unchecked forward-Euler commutator. It does not make the complete
coupled step exact or select a finite-strain constitutive model.

For constant G and lambda along a material characteristic, the variation of
constants formula for a prescribed spin and forcing is

    tau(t1) = beta R(t1,t0) tau(t0) R(t1,t0)^T
              + 2G integral[t0,t1] exp(-(t1-s)/lambda)
                  R(t1,s) D_branch_dev(s) R(t1,s)^T ds,
    beta = exp(-Delta t/lambda).

Here R(t,s) is the spin propagator; for constant W it is exp((t-s) W).
This formula exposes why rotating only the old stress and retaining an
incompatible unrotated forcing is not generally the full solution. Derive a
corotating-frame, exponential, or appropriately split update that is compatible
with the existing temporal scheme. Do not silently switch its Maxwell
coefficient convention. Verify both zero-spin and zero-forcing limits.

State transport/rotation/relaxation ordering and temporal accuracy. A symmetric
split alone does not make an otherwise first-order method second order. An
exact frozen-W rotation does not guarantee discrete covariance for arbitrary
time-dependent observer rotations; test and report the actual convergence.

Evaluate trials from the same accepted state. No history is repeatedly rotated
by Picard/Newton iterations, retries, observations, or restart. Preserve the
accepted solution exactly under rejected trials and publish only once.

## 5. Separate objective energy balance from finite-strain elasticity

For a single homogeneous isotropic branch with
psi = tau:tau/(4G), the algebraic identity

    tau : (W tau - tau W) = 0

means rotation contributes no direct change in this quadratic storage. With
finite Maxwell viscosity and D_branch_dev=0, storage decreases only through
the specified viscous relaxation. This is a local identity, not a proof that
the complete model has path-independent finite-strain elastic energy.

For the actual coupled law use the specified branch/cohesive energies and their
structural-frame dependence. Rotate anisotropic structural information
consistently where the selected model requires it. Separate pure rotation,
Maxwell loss, friction, radiation, fracture, numerical projection/splitting
error, external work, and any transport flux. Do not create “spin dissipation”
to absorb a residual or relax the existing physical-energy checks.

Jaumann objectivity does not eliminate the known finite-shear limitations of a
hypoelastic law. Measure elastic stress/strain scale, deformation-rate times
Maxwell time, accumulated relevant shear/rotation, and rotation per step in the
actual constitutive branches. Total slip-derived strain above 1 alone neither
proves failure nor establishes validity. A universal strain limit of 0.1 is
not an acceptance rule. If the intended parameter range exposes a constitutive
limitation, report it; do not claim that timestep refinement fixes it.

## 6. Required bounded qualification

Define tolerances and expected convergence from the independent reference,
floating-point scale, spatial projection and current temporal order. Do not
choose tolerances after inspecting a failing result.

| Test | Independent expectation / question |
| --- | --- |
| Zero-spin constitutive control | Recover the existing Maxwell/RSF branch at W=0; exercise nonzero stress and forcing. Legacy option reproduces unchanged fixtures to their established precision. |
| Homogeneous rigid rotation, eta infinite | For non-isotropic prestress, tau(t)=R(t) tau0 R(t)^T. Principal stresses and energy remain constant. Include a finite angle and inspect the full tensor, not only its norm. |
| Rigid rotation plus Maxwell relaxation | tau(t)=exp(-t/lambda) R(t) tau0 R(t)^T. Check energy decrease against integrated viscous loss; retain full plane-strain components. |
| Simultaneous deformation and rotation | Compare with an independent high-accuracy integration of the selected ODE or an available analytic solution; demonstrate timestep convergence and correct forcing rotation. |
| Superposed time-dependent rigid motion | Transform velocities, geometry, structural frames, stresses, and boundary data of a manufactured case consistently. Scalar work/energy and constitutive results should transform as specified. A fixed coordinate rotation alone does not test the spin term. |
| Transport plus rotation | Use a smooth nonuniform tensor field with a known characteristic solution. Separate advection, rotation, projection, and splitting error; a spatially uniform test cannot establish this. |
| Simple/cyclic shear | Check numerical convergence to the selected constitutive model and assess its finite-shear behavior separately. A numerically correct hypoelastic solution is not automatically a finite-strain physical reference. |
| Fault-frame coupling | Rotate a prestressed fault/host fixture consistently; check shear and normal traction, scalar RSF/contact outputs, structural tensor treatment, and signed slip. If a diffuse slip gradient enters W, perform a focused width/localization comparison at fixed physical slip. |
| Residual/tangent | Independent directional derivative check for changed residual/Jacobian terms, or demonstrate convergence of the declared lagged outer iteration. Keep pressure coupling. |
| Lifecycle | Forced rejection/retry, same-rank restart and a small MPI repartition test if affected. All accepted physical histories, independent memory and reconstructed fields must agree within declared tolerances. |
| Bounded BP3-like comparison | Same inputs and initial histories, isolated outputs, spin off/on. Report stress/normal traction, V, theta, work/energy and solver cost. This is a sensitivity comparison, not an automatic validation against the official small-strain BP3 problem. |

These tests are required by their applicable scope. Reuse existing meaningful
tests and independent references; do not duplicate suites or conduct full
earthquake trajectories to verify a local rotation identity. If the selected
configuration disables transport, explicitly report that limit and keep
transport-enabled combinations unqualified until their test is available.

## 7. Restart and benchmark interpretation

Identify the spin convention/integrator in parameters, output metadata and
checkpoint compatibility checks. Missing metadata in old checkpoints must map
to an explicit documented legacy interpretation, not silently to the new law.
An old checkpoint can serve as the initial state for a deliberately labeled
sensitivity branch; it cannot make earlier history a spin-enabled trajectory.
Existing production continuation remains on its recorded law unless separately
selected. Do not convert or overwrite the protected server payloads.

The BP3 reference is an infinitesimal-deformation elastic problem. A finite
rotation correction may change that problem. Preserve a clearly identified
benchmark-compatible route and quantify differences for the new model rather
than treating every mismatch as a numerical bug or changing parameters to fit.
Check actual compressibility, boundary and normal-stress conventions as well.

## Primary background references

- ASPECT viscoelastic material documentation: Jaumann convention and existing
  rotation/advection treatment. Inspect local source before reusing it; this
  documentation does not establish correctness for reconstructed-fault histories.
  https://aspect-documentation.readthedocs.io/en/latest/doxygen/classaspect_1_1MaterialModel_1_1Viscoelastic.html
- Schrank et al. (2017), *A comparative study of Maxwell viscoelasticity at large
  strains and rotations*, GJI 211, 252–262, doi:10.1093/gji/ggx297. Their simple
  shear comparisons motivate checking rate and deformation regimes; their
  thresholds are not universal criteria for this fault model.
  https://doi.org/10.1093/gji/ggx297
- MOOSE objective stress-rate documentation: integration choices and large-shear
  limitations. A different rate is a model choice, not a drop-in accuracy fix.
  https://mooseframework.inl.gov/modules/solid_mechanics/ObjectiveStressRates.html
