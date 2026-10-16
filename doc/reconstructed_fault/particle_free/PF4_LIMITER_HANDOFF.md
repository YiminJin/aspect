# PF4 limiter comparison: source reuse and detector choice

Prepared 16 October 2026 UTC for `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`.

## Assignment

Execute the bounded PF4 limiter comparison. Use the supplied `limiters.cc` as the primary implementation reference for a compact, mean-preserving WENO reconstruction of the independent DGQ2 tensor history. Use a Fu-Shu-inspired detector with tensor RMS normalization and a polynomial-discrepancy safeguard as the main experimental candidate. Keep a KXRCF-style detector with the SAME reconstruction as a comparison baseline, plus the existing unlimited control. This selects an experiment, not a production default or a constitutive change.

Read the applicable AGENTS.md and current PF4 report first. Preserve all existing work and evidence. The reviewed PF4 report described a serial, fixed affine Cartesian mesh, prescribed native Q2 velocity, independent DGQ2 history, SSPRK3, explicit inflow, immutable incoming state, and no accepted-state publication. If those assumptions have changed, reconcile the current source and report before editing. Do not expand this task into AMR/MPI transport, persistence, native surface-history transport, moving geometry, particle removal, or coupled BP3/propagation runs. Keep existing unsupported-path guards.

## References and what to reuse

1. Supplied `limiters.cc`, particularly `apply_WENO_limiter_to_dg_solutions` and `compute_KXRCF_indicators`. This is a different implementation from the earlier `elasticity.cc`.
2. Fu and Shu (2017), *A new troubled-cell indicator for discontinuous Galerkin methods for hyperbolic conservation laws*, JCP 347, 305-327, DOI 10.1016/j.jcp.2017.06.046. Read equations (2.3)-(2.5), Remark 2.2, and Remark 3.1.
3. Zhong and Shu (2013), *A simple weighted essentially nonoscillatory limiter for Runge-Kutta discontinuous Galerkin methods*, JCP 232, 397-415. Author manuscript: https://users.math.msu.edu/users/zhongxh/paper/wenolimiter.pdf . Read the whole-polynomial reconstruction and its two-dimensional extension.

The supplied source already extrapolates immediate-neighbor polynomials to the target cell, integrates those extrapolated polynomials over the TARGET cell, shifts their means to the target mean, and combines them with nonlinear weights. Reuse this structure. It avoids a 5x5 stencil and the old data-dependent linear-weight solve. On a uniform rectangular grid it needs the target and four face neighbors.

Retain the complete Q2 polynomial, including xy, x^2 y, x y^2, and x^2 y^2 modes. Q2 is not total-degree P2. Do not convert candidates to linear functions or discard mixed modes to simplify the port.

Important source findings:

- `epsilon = 1e-6 * sum(beta)` becomes zero if every candidate is constant. The weights then divide by zero and their normalization can produce NaNs. Piecewise constant data can be flagged by the detector, so this is not merely an unreachable uniform-state case.
- The scalar routine computes separate weights for each advection field. A tensor adaptation needs one mask and one set of weights for all tensor components.
- The reconstruction mass matrix uses the configured advection quadrature. Its rank cannot be inferred from this file alone. Verify the actual rule; do not repeat the earlier claim that this routine necessarily uses a 2x2 mass rule. The fixed Gauss2 rules are in the KXRCF detector.
- The detector's two-point volume sampling is not an exact DGQ2 maximum. Its whole-face integrated velocity sign does not resolve a face with mixed inflow and outflow.
- The KXRCF face-once ownership code stores only locally owned cell accumulators and shows no exchange of remote contributions. It is not evidence of correct MPI detection.
- The reconstruction skips all boundary faces, including periodic boundary faces, despite having later periodic-neighbor code. Audit rather than importing that behavior.
- Existing coarse/fine branches are useful reference code, not AMR qualification. Do not enable them in the PF4 transport path.

This review was static plus small algebraic checks; the uploaded historical source was not compiled.

## Tensor notation and invariants

Let A_K be the independent DG tensor polynomial on cell K. In the existing plane-strain representation reconstruct A_zz = -(A_xx + A_yy), with A_xz=A_yz=0. Use

    ||A||_3^2 = A_xx^2 + A_yy^2 + (A_xx + A_yy)^2 + 2 A_xy^2.

Apply the same full-tensor norm to polynomial differences and derivative tensors. Use one scalar indicator and common scalar WENO weights per cell. Do not take the maximum of independent component indicators and then give each component its own weights. Do not introduce fixed stress bounds, clip signed components, or apply the limiter to chi or reconstructed localized stress.

## Main detector: explicit adaptation of Fu-Shu

For each immediate face neighbor J, let E_J A_J be its UNMODIFIED polynomial evaluated in physical coordinates over K. On the current affine Cartesian mesh, this is ordinary polynomial extrapolation; reference points outside J must not be clamped to its unit cell.

Use physical cell averages:

    <f>_K = integral_K f / |K|.

Define the patch amplitude using the original polynomials on their own cells:

    M_K = max over J in {K and face neighbors} sqrt(<||A_J||_3^2>_J).

If the entire patch is exactly zero, return zero indicators. Evaluate norms and ratios with numerical scaling sufficient to avoid overflow/underflow. Do not normalize signed tensor fields solely by norms of cell averages: a nonzero field can have zero mean. Do not insert a physical stress floor disguised as a numerical safeguard.

The Fu-Shu-inspired mean mismatch is

    I_mean(K) = sum over neighbors J ||<A_K>_K - <E_J A_J>_K||_3 / M_K.

This has the paper's unweighted neighbor-sum structure, with a tensor norm and RMS denominator replacing its scalar absolute values and own-cell mean denominator. It is an ADAPTATION, not the unchanged published detector. The averages of extrapolated neighbors belong to K; their own-cell averages are different quantities.

Add the local polynomial-discrepancy safeguard

    I_poly(K) = max over neighbors J sqrt(<||A_K - E_J A_J||_3^2>_K) / M_K.

Flag a cell if max(I_mean, I_poly) > C. Record both indicators and separate counts for cells flagged only by I_poly. Both must be computed BEFORE candidates are shifted to preserve the target mean. Using mean-shifted candidates in the detector can hide jumps.

Why the safeguard is included: on a uniform square mesh of side h, set

    p_K(x,y) = 1 + (x-x_K)(y-y_K)/h^2,
    A_K = T p_K,

with a fixed nonzero symmetric deviator T, and use the same locally centered expression in every cell. Every own-cell mean and every face-neighbor polynomial averaged over K equals 1 times T. The published mean-mismatch numerator therefore vanishes, despite nonzero face jumps. The signed face integrals in KXRCF can also cancel. I_poly detects the mismatch. This is an algebraic blind spot, not a claim that the current PF4 pulse actually contains this mode.

For identical global polynomials represented on all cells, both indicators must vanish to roundoff, including global x^2 y^2 on this mesh. A field with uniform tensor magnitude but changing tensor direction must not be reduced to a scalar norm field before detection: take norms of tensor DIFFERENCES.

Use C=0.1 as a provisional numerical choice, with a bounded snapshot sensitivity check at 0.05 and 0.2 before the transport comparison. These are proposed screening values, not published constants for this adaptation. Select one value, record why, and hold it fixed across validation meshes and cases. Do not tune per mesh or per test. Fu-Shu's two-dimensional P2 examples use thresholds 0.03 or 0.06; those scalar Euler/triangular results do not establish a DGQ2 tensor threshold. Its threshold was calibrated experimentally; the paper is not parameter-free.

The RMS scale is amplitude-based, not invariant under adding a large constant tensor background. Include one background-offset diagnostic and report this limitation; do not expand into an open-ended detector redesign if it is significant.

## KXRCF comparison baseline

Use a tensor version of the signed integrated inflow-jump indicator, retaining its norm-after-integration structure and h^((p+1)/2) scaling, with p=2. Define the reference length used to nondimensionalize h, as in the uploaded source, and document the normalization and threshold. Use the same tensor RMS amplitude as above to avoid signed-average degeneracy. Call it a KXRCF-style tensor adaptation rather than the unchanged scalar detector.

Evaluate inflow using the actual pointwise normal velocity at matching face points; zero-speed portions do not contribute. Handle the prescribed boundary inflow consistently with PF4. Do not change the face-once MPI machinery because MPI remains excluded. A cell-based serial traversal is sufficient. Use degree-appropriate quadrature. Start from threshold 1 and do not undertake a large threshold sweep.

Use the same WENO reconstruction, weights, stage schedule, initial field, and time step in this baseline and the main detector path. Keep mean-only Fu-Shu values as a diagnostic/negative control; another full production-style detector implementation is unnecessary.

## Compact WENO reconstruction

For each selected cell K, use candidates

    P_0 = A_K,
    P_J = E_J A_J - <E_J A_J>_K + <A_K>_K.

Each candidate has exactly the target mean. Reconstruct

    A_K_limited = sum_j omega_j P_j,
    omega_j >= 0, sum_j omega_j = 1.

Use common tensor weights. On an interior uniform cell, start with linear weights d_0=0.996 and d_J=0.001 for each of four neighbors, consistent with Zhong-Shu's rectangular-grid example. At a physical boundary use available genuine neighbors and give their omitted linear weight to the center; explicitly document the reduced stencil. The diagnostic boundary inflow remains the transport operator's responsibility. Do not invent an exterior polynomial from a boundary trace. Preserve existing periodic/AMR guards.

Keep the source's first/second derivative smoothness structure, expressed in dimensionless coordinates of the TARGET cell and using full tensor norms. For example, on a fixed reference square use the mean of

    ||P_xi||_3^2 + ||P_eta||_3^2
      + ||P_xixi||_3^2 + 2||P_xieta||_3^2 + ||P_etaeta||_3^2.

Declare this as a Q2 smoothness-indicator choice. These terms respond to every nonconstant Q2 mode, but they are not the full total-degree-four Jiang-Shu derivative sum. Do not claim the P2 literature proves this exact adaptation. Scale the indicators by a common nonzero patch amplitude squared before introducing a dimensionless epsilon, initially 1e-6. If all smoothness indicators vanish, use the normalized linear weights directly. Implement stable weight normalization for nearly constant and widely scaled data.

Use adequate quadrature for the squared Q2 candidates and derivatives. Gauss3 per coordinate is sufficient for these polynomial expressions on affine Cartesian cells; retain PF4's higher transport rule and its separate source/inflow quadrature choices. A direct coefficient combination is possible here, or use the full-rank DG mass projection. Preserve the physical cell mean to roundoff and retain every Q2 coefficient.

Read all candidates from one immutable stage snapshot and write into a separate candidate vector. Do not let traversal order change a neighbor polynomial already used in the same pass. Apply limiting after each SSPRK3 stage candidate, before its next use, and do not publish accepted physical history. Record extrema of the raw projected initial condition separately from changes introduced by limiting and transport. Any initial limiting pass must be explicit; never silently replace the initial projection.

## Bounded verification and comparison

First qualify the local detector/reconstruction using small algebraic fixtures:

1. Zero and nonzero constants; forced reconstruction of piecewise constant patches; all-zero and nearly zero smoothness indicators; finite nonnegative normalized weights.
2. Identical global affine and Q2 fields, including x^2 y^2: negligible indicators and forced reconstruction unchanged to roundoff. This checks extrapolation and complete Q2 preservation.
3. Physical target-cell mean preservation, tensor symmetry and trace; stencil-order and cell-order independence.
4. The locally centered mixed-xy counterexample above. The mean-only diagnostic must expose its blind spot; the safeguarded detector must flag it at the selected threshold. Do not claim the reconstruction eliminates the mode merely because it is detected.
5. A zero-crossing smooth tensor, a non-proportional multicomponent tensor, a constant-magnitude tensor with varying orientation, amplitude rescalings, and one large constant-background diagnostic.
6. Covariance under in-plane orthogonal changes of tensor basis using the full embedded 3D tensor norm and common weights. Arbitrary 3D rotations can leave the stored plane-strain subspace; do not claim to test them by discarding newly generated xz/yz entries. A generic full-tensor helper can be tested more broadly if already available.
7. One boundary stencil with prescribed inflow and a no-inflow face; relevant mixed-normal-velocity face control for the KXRCF baseline.

Then reuse the existing PF4 signed pulse and resolved smooth tensor controls. Compare unlimited, KXRCF-style + WENO, and safeguarded Fu-Shu-style + the same WENO. Add only one inexpensive mixed/non-proportional transport fixture if the local fixtures leave a concrete unresolved risk; the existing signed pulse is proportional to one fixed tensor and alone cannot qualify tensor coupling.

Report sampled overshoot and undershoot with the same sampling rule as PF4, L1/L2 full-tensor error, quadratic-norm change, front broadening, indicator/limited-cell fractions, extra cells caught by I_poly, and detector/reconstruction runtime separately. Record cell-mean changes caused by limiting separately from transport and boundary fluxes. Norm loss is not a full mechanical energy balance. Preserve the existing resolved-smooth convergence gate (finest order at least 2.5) and relevant operator/source/publication checks. Do not require every time-accuracy result for a discontinuously switching limiter to reproduce the unlimited SSPRK3 order.

Use numerical invariant tolerances tied to roundoff and quadrature, not arbitrary physical stress bounds. Report the overshoot-versus-diffusion tradeoff; no mechanically acceptable overshoot threshold has been established. If changing the indicator barely changes the oscillations, investigate the reconstruction or spatial resolution rather than repeatedly lowering C. WENO is not a strict maximum-principle guarantee.

Adaptive quadrature is not part of this limiter task. Existing polynomial quadrature suffices for these fixed-grid operations. Finer integration cannot restore history variation absent from the DG space or removed by limiting. Keep PF3's separate localized-profile integration issue separate from PF4 transport resolution.

Use the existing bounded local-run resource limits. Avoid broad parameter searches and repeated full regressions after the relevant gates pass. Do not commit or push unless separately requested.

## Deliverable and stopping point

Produce a compact PF4 limiter report with the exact detector formulas, thresholds and calibration procedure, Q2 smoothness choice, zero-data treatment, boundary behavior, source reuse/corrections, measured comparisons, and remaining limitations. Label the RMS normalization and polynomial safeguard as new adaptations, not Fu-Shu's published formula. State whether the evidence favors the safeguarded Fu-Shu candidate, the KXRCF baseline, or neither.

Keep the production limiter default unchanged. End after this comparison and its concrete recommendation; do not proceed to PF5 or coupled deployment on the strength of a detector paper alone.
