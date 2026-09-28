  # Coupled reconstructed-fault solver: Stages F–I

  ## Common architecture and authoritative specification

  Before changing production code in Stage F, update both current_design.md and specification.tex with the reconciled F–I formulation below. The current source tree remains authoritative for concrete class names and existing infrastructure. Superseded redesign documents are background only.

  Use the coupled Newton system

  \[
  \begin{bmatrix}
  A & -B \\
  G & -K_V
  \end{bmatrix}
  \begin{bmatrix}
  \delta x \\
  \delta V
  \end{bmatrix}
  =
  \begin{bmatrix}
  R_{\rm bulk}\\
  R_\Gamma
  \end{bmatrix},
  \qquad
  K_V=-\frac{\partial R_\Gamma}{\partial V},
  \]
  and the exact condensation
  \[
  (A-BK_V^{-1}G)\delta x =
  -R_{\rm bulk}+BK_V^{-1}R_\Gamma,
  \qquad
  \delta V=K_V^{-1}(R_\Gamma+G\delta x).
  \]

  All coupling APIs use these mathematical signs. Where ASPECT stores `system_rhs = -R_bulk`, the solver adapter performs that conversion explicitly.

  ### Surface residual and ownership

  For each locally owned associated particle, define

  \[
  \bm\tau^{\rm trial} =
  2\kappa\dot{\bm\epsilon}
  +\beta\bm\tau_{k-1}
  -2\kappa\upsilon^{\rm hist}\bm S,
  \]

  \[
  t^{\rm trial}=\bm\tau^{\rm trial}:\bm S,
  \qquad
  \sigma_n = p - (2\kappa\dot{\bm\epsilon}
  +\beta\bm\tau_{k-1}):\bm N.
  \]

  The history and direct slip corrections do not alter \(\sigma_n\) because \(\bm S:\bm N=0\). Assemble the weak Q1 residual

  \[\begin{aligned}
  R^\Gamma_i =&\sum_p m_pN_i^p
  \left[
  t^{\rm trial}_p
  -\kappa_p\chi_pV_\Gamma(p)
  -T^{\rm coh}p(V)
  -\mu_p(V,\Theta)\sigma_{n,p}
  -\eta^d_pV_\Gamma(p)
  \right], \\
  \chi_p=&\frac{h_p}{I_{h,p}}.
  \end{aligned}\]

  `R_Gamma` is represented as fault-major `std::vector<std::vector<double>>`, with one weak nodal vector per fault. It is:

  - assembled only from locally owned particles;
  - packed and summed across MPI so every rank receives an identical copy;
  - transient data semantically owned by `MaterialModel::PhaseFieldFault`;
  - never stored as a generic fault property or checkpointed.

  The residual evaluation also returns a particle-domain-volume-weighted strong-residual RMS, including per-fault values, for convergence diagnostics. This avoids exposing the manager’s private projection mass factors solely to compute a norm.

  ### Minimal solver-facing interface

  Introduce a narrow capability interface, `MaterialModel::ReconstructedFaultCoupling<dim>`, implemented by `PhaseFieldFault`. Stage F adds only operations needed for surface evaluation and inversion:
```
  using FaultVector = std::vector<std::vector<double>>;

  struct SurfaceResidual
  {
    FaultVector values;
    double weighted_rms;
    std::vector<double> per_fault_weighted_rms;
  };

  SurfaceResidual
  evaluate_surface_residual(const LinearAlgebra::BlockVector &bulk_state,
                            const FaultVector &slip_rate) const;

  const SurfaceResidual &
  linearize_surface_system(const LinearAlgebra::BlockVector &bulk_state,
                           const FaultVector &slip_rate);

  void
  solve_surface_jacobian(const FaultVector &rhs,
                         FaultVector &solution) const;

  double minimum_slip_rate() const;

  void validate_committed_surface_state() const;
```
  `evaluate_surface_residual()` is collective and non-committing: it may allocate scratch data but cannot modify manager-owned \(V\), committed generic properties, particle stress, cohesive history, or nonlinear state. linearize_surface_system() builds only transient residual/Jacobian data for the supplied state.

  Stage G extends this same capability with the mathematically necessary `apply_G()` and `apply_B()` operations. No placeholder B/G/condensed-solver methods are added during
  Stage F.

  ### \(K_V\) representation and inverse

  For each fault, assemble the symmetric tridiagonal block

  \[
  (K_V)_{ij} = \sum_p m_pN_i^pN_j^p
  \left[
  \kappa_p\chi_p
  +\frac{\kappa_p}{I_{h,p}}
  +\sigma_{n,p}
  \left.\frac{\partial\mu}{\partial V}\right|_\Theta
  +\eta^d_p
  \right].
  \]

  The selected friction derivative may have either sign, so \(K_V\) must not be assumed positive definite.

  Store diagonal and off-diagonal arrays for each replicated fault block. Factor them with an \(O(n)\), adjacent-pivoting tridiagonal LU representation that supports nonsingular indefinite matrices. Keep factors private to PhaseFieldFault; the condensed solver receives only `solve_surface_jacobian()`, never raw factors or a matrix
  inverse.

  At factorization:

  - reject non-finite or singular pivots with fault/vertex diagnostics;
  - estimate reciprocal conditioning from the matrix norm and repeated factored solves;
  - reject an internally classified ill-conditioned block when the estimate is at machine-precision scale;
  - after each solve, check a scaled backward residual and report the fault, residual, and conditioning estimate on failure.

  These are internal numerical-failure thresholds, not user-facing convergence parameters.

  ### Constitutive data shared with B and G

  Use one common pointwise constitutive evaluation path so Stage F residual assembly and later B/G actions cannot drift apart. Its private data includes:

  - \(\beta\), \(\kappa\), \(\chi\), and \(\upsilon^{\rm hist}\);
  - \(\bm S\), \(\bm N\), particle/fault coordinate, and Q1 weights;
  - \(t^{\rm trial}\) and \(\sigma_n\);
  - the profile-uniform surface material mixture;
  - \(\mu\) and fixed-state \(\partial\mu/\partial V\);
  - \(T^{\rm coh}(V)\), its slope \(\kappa/I_h\), and \(\eta^d\).

  Surface friction and cohesive quantities use the projected surface mixture. Bulk Maxwell coefficients continue to use the existing material-model evaluation at the relevant bulk particle or quadrature point. No raw constitutive cache is exposed publicly.

  ## Stage F — surface residual and \(K_V\)

  This stage contains no production implementation of QP coupling, \(B\), \(G\), condensation, or nonlinear solver lifecycle.

  - Promote the equations, signs, MPI ownership, state rules, and staged interfaces in this plan into both authoritative documents.

  - Register committed Theta in the manager’s generic property pool only when rate-and-state friction is selected. `PhaseFieldFault` owns its meaning and validates that every value needed by a mechanical solve is initialized, finite, and positive.

  - Do not define a universal \(\Theta_0\) construction, introduce an RSF inversion, or add generic V_init/Theta_init parameters. A benchmark or initial-condition path must supply \(\Theta_0\).

  - Hold committed \(\Theta\) fixed during every residual and Jacobian evaluation, including timestep 0. State evolution remains deferred.
  - Implement non-committing cohesive/friction/Maxwell surface evaluation, MPI-reduced weak \(R_\Gamma\), analytic \(K_V\), factorization, conditioning diagnostics, and `solve_surface_jacobian()`.

  - Remove the obsolete Maximum slip rate parameter, `Vmax` member, and upper clamping. Repository inspection confirms that Vmax has no other production use; the only unrelated occurrences are comments naming a velocity statistic.

  - Make friction operations require \(V\ge V_{\min}>0\) and evaluate the raw constitutive law without clipping. Residuals and derivatives therefore use the same law over the complete admissible interval.

  - Preserve current geometry, particle association, \(I_h\), cohesive initialization, Maxwell history, and all committed-state behavior.

  Stage-F tests:

  - Isolate mechanical-slip, cohesive, friction, radiation, and history contributions to \(R_\Gamma\) and \(K_V\).
  - Cover rate-state and rate-dependent friction, spatially varying projected compositions, nonzero cohesive history, and missing/invalid \(\Theta_0\).
  - Verify identical replicated residuals and factors on one and two MPI ranks.
  - Exercise positive-definite and nonsingular-indefinite tridiagonal systems, plus explicit singular and ill-conditioned failures.
  - Compare \(-K_V\delta V\) with centered differences of \(R_\Gamma(x,V)\) for interior admissible \(V\).
  - Re-run existing reconstructed-fault, Maxwell, cohesive, \(I_h\), and friction tests.

  Commit and review Stage F before starting Stage G.

  ## Stage G — bulk residual, (B), and (G)

  - Add manager-owned, constitutively neutral QP-to-fault associations with the same geometry-version and mesh-lifetime invalidation rules as the particle cache. No
    geometric search may occur inside a Krylov `vmult()`.

  - # Add the slip-dependent bulk stress residual using
    \[
    \bm\tau = \bm\tau^{\rm trial}-2\kappa\chi V_\Gamma\bm S.
    \]
    Use the normal Stokes weak-form assembly path and cached QP association.

  - Extend the coupling interface with:
```
    void apply_G(const LinearAlgebra::BlockVector &delta_x,
                 FaultVector &result) const;

    void apply_B(const FaultVector &delta_V,
                 LinearAlgebra::BlockVector &result) const;
```
  - # apply_G() evaluates
    \[
    G\delta x
    \sum_p m_pN_i^p
    \left[
    2\kappa(\bm S+\mu\bm N):
    \delta\dot{\bm\epsilon}
    -\mu~\delta p
    \right]
    \]
    at cached locally owned particles, followed by a fault-sized MPI reduction.

  - `apply_B()` returns \(B\delta V\), while the actual derivative of the bulk residual is \(-B\delta V\). It interpolates \(\delta V\) at cached local Stokes QPs and inserts
    the stress response associated with
    \[
    \delta\bm\tau = -2\kappa\chi\bm S~\delta V_\Gamma.
    \]

  - Add a simulator-private non-committing coupled residual evaluator returning mathematical (R_{\rm bulk}) and (R_\Gamma) for an explicit bulk candidate and fault
    vector. Any temporary use of ASPECT assembly state must restore the accepted iterate and manager state even if evaluation throws.

  - Do not alter the linear solver or nonlinear iteration in this stage.

  Stage-G tests:

  - Compare `apply_G()` with centered differences of \(R_\Gamma\) for velocity-only and pressure-only perturbations.
  - Compare -`apply_B()` with centered differences of the bulk residual for \(V\)-only perturbations.
  - Compare both actions with explicitly assembled small matrices and verify their sign conventions.
  - Verify one-rank/two-rank equivalence, cached-QP invalidation, and that repeated actions perform no new geometric search.
  - Re-run all Stage-F and earlier focused tests.

  Commit and review Stage G before starting Stage H.

  ## Stage H — exact condensed linear solve

  - In the assembled iterative block-AMG Stokes path, replace the matrix-only Krylov wrapper with a file-local condensed operator applying:
      1. \(A\delta x\);
      2. \(g=G\delta x\);
      3. \(z=K_V^{-1}g\);
      4. \(Bz\);
      5. \(A\delta x-Bz\).

  - Preserve the existing Stokes block preconditioner as an approximation to the condensed operator. Do not form \(BK_V^{-1}G\) explicitly.

  - Convert ASPECT’s stored bulk right-hand side to the documented signs and add
    \[
    BK_V^{-1}R_\Gamma
    \]
    to the condensed right-hand side.

  - After the bulk Krylov solve, recover
    \[
    \delta V=K_V^{-1}(R_\Gamma+G\delta x).
    \]
    Recovery produces a candidate update only; it does not change manager state.

  - Add early configuration errors for direct Stokes, matrix-free/GMG, melt transport, non-2-D reconstructed faults, and solver schemes that cannot supply the required assembled Newton operator.

  - Do not add joint line search, convergence, initialization, or commit behavior in Stage H.

  Stage-H tests:

  - Compare condensed operator actions with explicit \(A-BK_V^{-1}G\) on small systems.
  - Compare condensed solutions with direct solutions of the complete two-block system.
  - Verify the condensed RHS sign and ensure recovered \(\delta V\) satisfies the uncondensed fault row.
  - Exercise multiple fault blocks and an indefinite but invertible \(K_V\).
  - Verify every unsupported solver mode fails before entering the solve.
  - Re-run all Stage-G and earlier focused tests.

  Commit and review Stage H before starting Stage I.

  ## Stage I — coupled nonlinear trial and (V) lifecycle

  - At timestep 0, initialize the numerical Newton iterate on every fault with
    \[
    V^{(0)}=V_{\min}.
    \]
    This is only a nonlinear starting value. The converged timestep-0 value is committed as the physical \(V^0\).

  - Do not use any benchmark-specific physical quantity named V_init as the generic solver’s initial guess.
  - For timestep \(k>0\), start from the previous committed slip rate:
    \[
    V_k^{(0)}=V^{k-1}.
    \]

  - For rate-state friction, require the initial-condition path to have supplied positive committed \(\Theta_0\). Hold \(\Theta_0\) fixed during the timestep-0 solve. For later mechanical solves, hold committed \(\Theta^{k-1}\) fixed.

  - Apply the existing manager state machine:
      - begin nonlinear solve from the prescribed initial iterate;
      - create every line-search candidate from the current accepted \(V\);
      - accept or roll back each candidate without accumulation;
      - commit (V) only after mechanical convergence;
      - restore the previously committed \(V\) after failure or exception.

  - # Limit a proposed step only at the lower bound:
    \[
    \alpha_{\max}\min\left(
    1,~0.99\min_{\delta V_i<0}\frac{V_i-V_{\min}}{-\delta V_i}
    \right).
    \]
    There is no upper bound or upper clipping.

  - Evaluate every trial through the non-committing coupled residual interface. Rejected candidates cannot modify (\Theta), cohesive history, (I_h) history, or particle Maxwell stress.

  - # Use fixed, block-specific normalization scales over one nonlinear solve and the merit function
    \[
    \Phi =
    \frac12\left[
    \left(\frac{|R_{\rm bulk}|}{S_{\rm bulk}}\right)^2+
    \left(\frac{|R_\Gamma|_\Gamma}{S_\Gamma}\right)^2
    \right].
    \]
    Use the existing nonlinear relative tolerance independently for both blocks; both must converge. Internal machine-scale floors prevent division by a zero initial block norm and are not new user parameters.

  - On successful Stage-I convergence, commit manager-owned \(V\) only. Updating or committing \(\Theta\), cohesive traction, previous \(I_h\), and particle Maxwell stress remains Stage J.

  - Until Stage J supplies the atomic history update, keep multi-timestep production evolution guarded so the solver cannot silently advance with stale constitutive history.

  Stage-I tests:

  - Verify timestep-0 Newton starts at \(V_{\min}\), independently of benchmark initial-condition quantities.
  - Verify later solves start from committed \(V^{k-1}\).
  - Verify \(\Theta_0\) is required for rate-state friction, remains unchanged at timestep 0, and is not inferred from friction.
  - Verify lower-bound step limiting, repeated rejected trials without accumulation, accepted-current versus timestep-committed state, successful commit, and failure rollback.
  - Verify both residual blocks control convergence and the joint merit controls line-search acceptance.
  - Confirm Stage-I success changes only \(V\), not any deferred history.
  - Keep the guarded unsupported configurations from Stage H.

  ## Jacobian finite-difference verification

  Use centered differences throughout:

  \[
  J\delta y
  \approx
  \frac{R(y+\varepsilon\delta y)-R(y-\varepsilon\delta y)}
  {2\varepsilon}.
  \]

  For each direction, use a geometrically decreasing sequence of dimensionally scaled \(\varepsilon\). Require an approximately fourfold error reduction under step halving
  over at least two pre-roundoff refinements; do not validate only one hand-selected perturbation size. All \(V\) perturbations remain strictly above \(V_{\min}\).

  Verification is staged:

  - Stage F: \(V\)-only surface residual versus \(-K_V\delta V\).
  - Stage G: velocity-only and pressure-only surface differences versus \(G\delta x\), and bulk \(V\)-only differences versus \(-B\delta V\).
  - Stage H: explicit full-block action, condensed action, condensed RHS, and recovered fault row.
  - Stage I: complete coupled residual with velocity-only, pressure-only, \(V\)-only, and mixed perturbations; both friction laws; nonzero cohesive history; states near but not crossing \(V_{\min}\); one and two MPI ranks.

  All finite-difference evaluations freeze geometry, phase field, current \(I_h\), committed particle stress, committed cohesive history, and committed \(\Theta\). No finite-difference sample may commit state.
