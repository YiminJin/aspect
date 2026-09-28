  # Coupled reconstructed-fault solver: revised Stages F–I

  ## Shared architecture

  Before Stage F production changes, promote the reconciled formulation below into both authoritative documents. The implementation remains split into four separately
  reviewable stages; Stage F must contain no production (B), (G), condensed-solver, or nonlinear-lifecycle implementation.

  Use the Newton system

  \[
  \boxed{
  \begin{bmatrix}
  A & -B\\
  G & -K_V
  \end{bmatrix}
  \begin{bmatrix}
  \delta x\\
  \delta V
  \end{bmatrix}
  =-
  \begin{bmatrix}
  R_{\rm bulk}\\
  R_\Gamma
  \end{bmatrix}
  }
  \]

  with

  \[
  K_V=-\frac{\partial R_\Gamma}{\partial V}.
  \]

  Eliminating \(\delta V\) gives

  \[
  \boxed{
  (A-BK_V^{-1}G)\delta x =
  -R_{\rm bulk}+BK_V^{-1}R_\Gamma
  }
  \]

  and

  \[
  \boxed{
  \delta V=K_V^{-1}(R_\Gamma+G\delta x).
  }
  \]

  All APIs use these mathematical signs. ASPECT’s system_rhs convention is converted explicitly at the solver boundary.

  ### Responsibility split

  `MaterialModel::PhaseFieldFault` owns constitutive mechanics:

  - Maxwell coefficients and trial stress;
  - cohesive response and history correction;
  - friction-law evaluation;
  - selection of dynamic or adiabatic normal pressure;
  - pointwise surface residual and its constitutive derivatives;
  - committed material state such as \(\Theta\) and \(T^{\rm coh}\).

  A dedicated `Assemblers::ReconstructedFaultStokes<dim>` owns discretization and assembly:

  - particle/Q1 assembly of \(R_\Gamma\) and \(K_V\);
  - slip-dependent bulk Stokes residual;
  - bulk-to-fault \(G\) action;
  - fault-to-bulk \(B\) action;
  - MPI reduction of replicated fault vectors;
  - transient surface matrices and their factorizations.

  The solver owns:

  - \(K_V^{-1}\) use in the condensed operator;
  - condensed right-hand side;
  - recovery of \(\delta V\);
  - bound-active-set iteration;
  - nonlinear convergence and line search.

  ReconstructedFaultManager remains generic and owns only geometry, associations, generic properties, and committed/current/trial \(V\).

  ### Narrow material capability

  Add a narrow reconstructed-fault constitutive capability implemented by PhaseFieldFault. It accepts explicit point state and returns constitutive scalars/tensors; it does not receive global vectors, assemble weak forms, perform MPI communication, or expose `apply_B()`/`apply_G()`.

  Conceptually:
  ```
  struct SurfacePointInputs
  {
    Point<dim> position;
    std::vector<double> surface_material_fractions;

    double slip_rate;
    std::optional<double> committed_state;

    double current_I_h;
    double previous_I_h;
    double current_h;
    double previous_h;
    double previous_cohesive_traction;

    SymmetricTensor<2,dim> strain_rate;
    SymmetricTensor<2,dim> old_maxwell_stress;
    double dynamic_pressure;

    SymmetricTensor<2,dim> slip_tensor;
    SymmetricTensor<2,dim> normal_tensor;
  };

  struct SurfacePointResponse
  {
    double residual_density;
    double minus_derivative_wrt_slip_rate;

    double kappa;
    double localization_factor;
    double friction_coefficient;

    SymmetricTensor<2,dim> stress;
  };
  ```
  Stage F supplies the fields required by \(R_\Gamma\) and \(K_V\). Stage G adds only the pointwise derivative coefficients concretely required by its assembler:
  ```
  SymmetricTensor<2,dim> derivative_wrt_strain_rate;
  double derivative_wrt_dynamic_pressure;
    SymmetricTensor<2,dim> derivative_of_stress_wrt_slip_rate;
```
  These coefficients define \(G\) and \(B\), but no global action belongs to the material-model interface.

  ## Normal pressure and adiabatic-pressure option

  Add the boolean parameter:
  ```
    Use adiabatic pressure in fault friction = false
  ```
  It belongs to PhaseFieldFault, not FaultFriction. The default preserves the dynamic-pressure formulation.

  When false, use

  \[
  \sigma_n = p-\bm\tau:\bm N,
  \]

  and

  \[
  F = t-T^{\rm coh}-\mu(V,\Theta)\sigma_n-\eta^dV.
  \]

  For fixed (V) and (\Theta),

  \[
  \delta F = 2\kappa(\bm S+\mu\bm N):\delta\dot{\bm\epsilon}
  -\mu~\delta p.
  \]

  Therefore,

  \[
  G\delta x = 2\kappa(\bm S+\mu\bm N):\delta\dot{\bm\epsilon}
  -\mu~\delta p.
  \]

  When Use adiabatic pressure in fault friction is true, the frictional normal stress is the adiabatic-model pressure evaluated at the constitutive point:

  \[
  \sigma_n^{\rm fric}=p_{\rm ad}(\bm x).
  \]

  The deviatoric normal traction is not included in the friction term in this mode. Thus,

  \[
  F = t-T^{\rm coh}-\mu(V,\Theta)p_{\rm ad}-\eta^dV,
  \]

  \[
  K_V\supset p_{\rm ad}
  \left.\frac{\partial\mu}{\partial V}\right|_\Theta,
  \]

  and

  \[
  \boxed{
  G\delta x = 2\kappa\bm S:\delta\dot{\bm\epsilon}.
  }
  \]

  There is no \(\mu\bm N\) contribution and no \(-\mu\delta p\) term. Require initialized adiabatic conditions when this option is enabled and report a configuration error otherwise.

  ## Surface algebra and ownership

  For each associated particle, define

  \[
  \bm\tau^{\rm trial} =
  2\kappa\dot{\bm\epsilon}
  +\beta\bm\tau_{k-1}
  -2\kappa\upsilon^{\rm hist}\bm S,
  \qquad
  t^{\rm trial}=\bm\tau^{\rm trial}:\bm S.
  \]

  The direct current-slip correction is

  \[
  -2\kappa\chi V_\Gamma\bm S,
  \qquad
  \chi=\frac{h}{I_h}.
  \]

  The weak Q1 residual is

  \[
  R^\Gamma_i =
  \sum_p m_pN_i^p\left[
  t^{\rm trial}_p-\kappa_p\chi_pV_\Gamma(p)-T^{\rm coh}p(V)
  -\mu_p(V,\Theta)\sigma_{n,p}^{\rm fric}-\eta^d_pV_\Gamma(p)
  \right].
  \]

  R_Gamma is represented as a fault-major `std::vector<std::vector<double>>`, with one weak nodal vector per fault. The reconstructed-fault Stokes assembler owns the transient current residual:

  - only locally owned particles contribute;
  - one packed MPI reduction makes every rank’s vector identical;
  - it is neither a generic fault property nor checkpointed;
  - it is invalidated at every new nonlinear linearization.

  The assembler also reports a particle-domain-volume-weighted strong-residual RMS, including per-fault values, without exposing the manager’s projection-matrix factors.

  ### \(K_V\)

  For dynamic pressure,

  \[
  (K_V)_{ij} = \sum_p m_pN_i^pN_j^p\left[
  \kappa_p\chi_p+\frac{\kappa_p}{I_{h,p}}+\sigma_{n,p}
  \left.\frac{\partial\mu}{\partial V}\right|_\Theta
  +\eta^d_p
  \right].
  \]

  For adiabatic pressure, replace (\sigma_{n,p}) in the friction derivative by \(p_{\rm ad}(\bm x_p)\).

  The matrix is block diagonal across faults, with one symmetric tridiagonal block for each ordered Q1 fault. It may be indefinite.

  Represent each fault block as a deal.II sequential sparse matrix with tridiagonal sparsity and factor it using dealii::SparseDirectUMFPACK. The current build provides this tested library implementation, which supports nonsingular indefinite matrices and repeated solves.

  - Do not implement a custom tridiagonal LU or condition estimator.
  - Keep the sparse matrix and UMFPACK factorization private to the assembler.
  - If deal.II lacks UMFPACK, reject activation of the coupled reconstructed-fault solver with a clear configuration diagnostic; unrelated ASPECT configurations remain usable.
  - Treat UMFPACK factorization failure, non-finite solutions, or an unacceptable scaled backward solve residual as explicit singular/ill-conditioned-\(K_V\) diagnostics.
  - Do not introduce a user-facing condition threshold in this stage.

  The solver sees only:
  ```
  void solve_surface_jacobian(const FaultVector &rhs,
                              FaultVector &solution) const;
  ```
  No raw factorization or explicit inverse is exposed.

  ### Non-committing residual evaluation

  The assembler provides:
  ```
  SurfaceResidual
  evaluate_surface_residual(const LinearAlgebra::BlockVector &bulk_state,
                            const FaultVector &slip_rate) const;

  const SurfaceResidual &
  linearize_surface_system(const LinearAlgebra::BlockVector &bulk_state,
                           const FaultVector &slip_rate);

  void
  solve_surface_jacobian(const FaultVector &rhs,
                         FaultVector &solution) const;

  evaluate_surface_residual() is collective but non-committing. It cannot modify:
  ```
  - committed/current/trial manager \(V\);
  - committed (\Theta), cohesive traction, or previous \(I_h\);
  - particle Maxwell stress;
  - the accepted bulk nonlinear iterate.

  linearize_surface_system() builds transient \(R_\Gamma\), \(K_V\), and its factorization for exactly the supplied state.

  ## Stage F — surface residual and \(K_V\)

  Stage F stops after surface residual/Jacobian support. It must not implement \(B\), \(G\), a slip-dependent bulk Stokes residual, condensation, or nonlinear lifecycle changes.

  - Update both authoritative documents with the approved F–I equations, pressure modes, ownership, factorization, active-set semantics, and stage boundaries.
  - Add Use adiabatic pressure in fault friction to PhaseFieldFault, defaulting to false.
  - Register committed Theta only for rate-and-state friction. PhaseFieldFault owns its meaning; the manager provides generic storage.
  - Require rate-state mechanical evaluation to find initialized, finite, positive committed \(\Theta\).
  - Do not construct \(\Theta_0\) generically, invert an RSF law, or add generic `V_init`/`Theta_init` parameters.
  - Implement the pure pointwise surface constitutive response for both pressure modes.
  - Add the reconstructed-fault Stokes assembler in surface-only form: particle/Q1 \(R_\Gamma\), \(K_V\), MPI reduction, UMFPACK factorization, and inverse application.
  - Remove the obsolete Maximum slip rate parameter, Vmax member, and upper clamping. Repository inspection found no separate production use.
  - Require \(V\ge V_{\min}>0\) at the constitutive boundary and evaluate both friction laws and their derivatives at the supplied \(V\), without clipping.
  - Preserve current geometry, particle associations, \(I_h\), cohesive initialization, Maxwell history, and committed-state behavior.

  Stage-F tests:

  - Isolate mechanical, cohesive, friction, radiation, and history terms in \(R_\Gamma\) and \(K_V\).
  - Test rate-state and rate-dependent laws with dynamic pressure.
  - Test adiabatic-pressure residual and \(K_V\) with a controlled adiabatic model.
  - Verify missing/invalid \(\Theta_0\) and unavailable adiabatic pressure fail clearly.
  - Verify one-rank/two-rank replicated \(R_\Gamma\) and \(K_V\) behavior.
  - Test positive-definite and nonsingular-indefinite \(K_V\), plus singular factorization failure.
  - Compare \(-K_V\delta V\) with centered differences of \(R_\Gamma(x,V)\).
  - Re-run all existing reconstructed-fault, Maxwell, cohesive, \(I_h\), and friction tests.

  Commit and review Stage F before Stage G.

  ## Stage G — reconstructed-fault Stokes assembly, \(B\), and \(G\)

  Extend the dedicated assembler; do not put global weak-form actions in the material model.

  - Add manager-owned, constitutively neutral QP-to-fault associations with geometry/mesh cache invalidation. Krylov actions must not perform geometric searches.
  - Add the current slip-dependent bulk stress residual:
    \[
    \bm\tau = \bm\tau^{\rm trial} - 2\kappa\chi V_\Gamma\bm S.
    \]

  - Extend the material point response only with the derivative coefficients required by the assembler.

  - Add assembler operations:
  ```
    void apply_bulk_to_fault(const LinearAlgebra::BlockVector &delta_x,
                             FaultVector &result) const;      // G delta_x

    void apply_fault_to_bulk(const FaultVector &delta_V,
                             LinearAlgebra::BlockVector &result) const; // B delta_V
  ```
  - apply_bulk_to_fault() performs particle/Q1 assembly and its fault-sized MPI reduction.
  - In dynamic-pressure mode it evaluates
    \[
    2\kappa(\bm S+\mu\bm N):
    \delta\dot{\bm\epsilon}
    -\mu,\delta p.
    \]

  - In adiabatic-pressure mode it evaluates only
    \[
    2\kappa\bm S:
    \delta\dot{\bm\epsilon}.
    \]

  - `apply_fault_to_bulk()` interpolates (\delta V) at cached local Stokes QPs and assembles the weak bulk response of
    \[
    \delta\bm\tau = -2\kappa\chi\bm S~\delta V_\Gamma.
    \]
    It returns \(B\delta V\); the derivative of the bulk residual is \(-B\delta V\).

  - Add a simulator-side, non-committing coupled residual evaluator for explicit \((x,V)\). Any temporary ASPECT assembly state must be restored on normal return and exceptions.

  - Do not alter the Krylov operator, condensed right-hand side, or nonlinear solve.

  Stage-G tests:

  - Dynamic-pressure (G): centered differences for velocity-only and pressure-only directions.
  - Adiabatic-pressure (G): centered velocity differences and an explicit zero pressure derivative.
  - Verify the adiabatic action contains neither the (\mu N) term nor the pressure term.
  - Compare (-B\delta V) with centered differences of the bulk residual.
  - Compare (B) and (G) against explicitly assembled small operators, including signs.
  - Verify one-rank/two-rank equivalence and no geometric search during repeated actions.
  - Re-run Stage-F and earlier focused tests.

  Commit and review Stage G before Stage H.

  ## Stage H — active-set-compatible exact condensation

  The condensed operator remains entirely solver-side.

  ### Restricted surface inverse

  The lower-bound treatment must be compatible with condensation. For a current active set (\mathcal A), impose

  \[
  \delta V_i=0,\qquad i\in\mathcal A.
  \]

  Let \(\mathcal F\) contain the free surface nodes. Condensation uses the reduced surface block:

  \[
  A_{\rm eff}^{\mathcal F} =
  A-B_{\mathcal F}K_{\mathcal F\mathcal F}^{-1}G_{\mathcal F}.
  \]

  The recovered update is

  \[
  \delta V_{\mathcal F} =
  K_{\mathcal F\mathcal F}^{-1}\left(
  R_{\Gamma,\mathcal F}+G_{\mathcal F}\delta x
  \right),
  \qquad
  \delta V_{\mathcal A}=0.
  \]

  UMFPACK factors the reduced sparse matrix. This avoids special tridiagonal handling when active nodes split a fault into multiple free intervals.

  ### Solver implementation

  - Add a file-local condensed Krylov operator in the assembled iterative block-AMG path:
      1. apply (A\delta x);
      2. apply (G_{\mathcal F}\delta x);
      3. solve (K_{\mathcal F\mathcal F}z=g);
      4. apply (B_{\mathcal F}z);
      5. return (A\delta x-B_{\mathcal F}z).

  - Add
    [
    B_{\mathcal F}K_{\mathcal F\mathcal F}^{-1}
    R_{\Gamma,\mathcal F}
    ]
    to the condensed right-hand side.

  - Recover the complete (\delta V) with zero active components.
  - Keep the existing Stokes block preconditioner as an approximation to the condensed operator.
  - Do not form (BK^{-1}G) explicitly.
  - Add configuration errors for direct Stokes, matrix-free/GMG, melt transport, unsupported dimensions, and solver schemes without the required assembled Newton
    operator.

  - Stage H provides the restricted-operator mechanism but does not choose the nonlinear active set or change manager state.

  Stage-H tests:

  - Compare unrestricted and restricted condensed actions with explicit matrices.
  - Compare condensed solutions with direct solutions of the complete free-variable block system.
  - Verify the condensed RHS sign and recovered fault row.
  - Verify active components of (\delta V) are exactly zero.
  - Test multiple faults and indefinite but invertible free (K) blocks.
  - Verify unsupported solver modes fail before solving.
  - Re-run Stage-G and earlier focused tests.

  Commit and review Stage H before Stage I.

  ## Stage I — projected Newton lifecycle and lower-bound active set

  ### Initialization semantics

  - At timestep 0, initialize the numerical Newton iterate with
    [
    V^{(0)}=V_{\min}.
    ]

  - The converged timestep-0 solution is the physical committed (V^0).
  - Do not treat a benchmark’s physical V_init as the generic Newton guess.
  - For timestep (k>0), begin from committed (V^{k-1}).
  - Rate-state friction requires an initial-condition path to have supplied positive (\Theta_0). Hold it fixed during the timestep-0 mechanical solve.
  - In later mechanical solves, hold committed (\Theta^{k-1}) fixed. Updating it remains Stage J.

  ### Active-set/projected-direction algorithm

  For each Newton iteration:

  1. Start with no active nodes, except nodes already identified during the current projected solve.
  2. Solve the current unrestricted or restricted condensed system.
  3. For every node at the lower bound within a scale-aware machine tolerance, inspect the recovered Newton direction.
  4. If the direction would decrease \(V\), add that node to the active set and impose \(\delta V_i=0\).
  5. Rebuild/refactor only the reduced surface block and resolve the condensed system.
  6. Repeat until no bound node has an outward direction.
  7. Recompute the active set from the new nonlinear iterate on the next Newton iteration, allowing previously active nodes to be released when the new unconstrained direction points into the admissible region.

  The projected surface residual used for convergence contains free-node residuals and zeros only for active nodes whose unconstrained direction points out of the admissible region. A bound node is therefore not declared converged merely because it lies at \(V_{\min}\).

  After active-set stabilization, apply fraction-to-boundary limiting only to remaining free nodes:

  \[
  \alpha_{\max} = \min\left(
  1,~0.99\min_{\substack{i\in\mathcal F\\delta V_i<0}}
  \frac{V_i-V_{\min}}{-\delta V_i}
  \right).
  \]

  Because outward directions at bound nodes have already been projected out, this step cannot collapse to \(\alpha_{\max}=0\) for that case. There is no upper bound or upper clipping.

  ### Trial and commit lifecycle

  - Form every line-search candidate from the current accepted bulk iterate and current accepted \(V\); rejected step lengths must not accumulate.
  - Evaluate candidates through the non-committing coupled residual interface.
  - # Use separate fixed-scale normalized bulk and projected-surface residuals:
    \[
    \Phi =
    \frac12\left[
    \left(\frac{|R_{\rm bulk}|}{S_{\rm bulk}}\right)^2+
    \left(\frac{|R_\Gamma^{\rm projected}|\Gamma}{S\Gamma}\right)^2
    \right].
    \]

  - Require both normalized blocks to satisfy the existing nonlinear relative tolerance.
  - Accepting a line-search candidate updates only the current Newton \(V\).
  - Mechanical convergence commits manager-owned \(V\) only.
  - Failure or exception restores the previously committed \(V\).
  - (\Theta), cohesive history, previous \(I_h\), and Maxwell stress remain unchanged until Stage J.
  - Until Stage J provides an atomic history update, guard multi-timestep production evolution against silently using stale history.

  Stage-I tests:

  - Timestep-0 start at exactly \(V_{\min}\).
  - Later start from committed \(V^{k-1}\).
  - Required but non-inferred \(\Theta_0\), held fixed during timestep 0.
  - A bound node with a negative unconstrained direction becomes active instead of producing zero line-search length.
  - A bound node with an inward direction remains free.
  - Multiple active-set passes, active-node release on a later Newton iteration, and reduced-system consistency.
  - Fraction-to-boundary limiting for free nodes above \(V_{\min}\).
  - Repeated rejected trials without accumulation.
  - Successful \(V\)-only commit and complete rollback on failure.
  - Separate bulk and projected-surface convergence.
  - Both dynamic- and adiabatic-pressure nonlinear paths.

  ## Finite-difference verification

  Use centered differences for smooth interior states:

  \[
  J\delta y
  \approx
  \frac{R(y+\varepsilon\delta y)-R(y-\varepsilon\delta y)}
  {2\varepsilon}.
  \]

  Use a geometrically decreasing, dimensionally scaled sequence of \(\varepsilon\). Require approximately fourfold error reduction under step halving over at least two pre-roundoff refinements. All centered \(V\) samples remain strictly above \(V_{\min}\).

  Verification by stage:

  - Stage F: (V)-only (R_\Gamma) differences versus (-K_V\delta V), in both pressure modes.
  - Stage G:
      - dynamic mode: velocity-only and pressure-only differences versus the full (G);
      - adiabatic mode: velocity-only differences versus (2\kappa S:\delta\dot\epsilon), plus zero pressure derivative;
      - bulk (V)-only differences versus (-B\delta V).

  - Stage H: complete unrestricted and active-set-restricted block actions, condensed actions, RHS signs, and recovered fault rows.
  - Stage I: complete coupled residual with velocity-only, pressure-only, (V)-only, and mixed perturbations; both friction laws; both pressure modes; nonzero cohesive
    history; and one/two MPI ranks.

  Centered differences are not used across the lower-bound kink. Active-set behavior at (V_{\min}) is tested independently with projected directions and one-sided
  admissible perturbations.

  Every residual sample freezes geometry, phase field, current (I_h), committed particle stress, cohesive history, and committed (\Theta), and commits no state.

  Build each stage with -j4, run its new focused tests and all earlier reconstructed-fault/PhaseFieldFault tests, and report exact commands and results. Do not run the
  complete 1289-test integration suite unless explicitly requested.
