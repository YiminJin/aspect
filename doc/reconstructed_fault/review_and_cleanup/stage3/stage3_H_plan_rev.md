  # Stage H — Exact Solver-Side Condensation

  ## Summary

  Implement the unrestricted condensed system

  \[
  C=A-BK_V^{-1}G,\qquad
  b_C=-R_{\rm bulk}+BK_V^{-1}R_\Gamma,\qquad
  \delta V=K_V^{-1}(R_\Gamma+G\delta x).
  \]

  `ReconstructedFaultCondensedSystem` will reference canonical simulator-owned surface and Stokes-coupling components. It will not construct duplicate instances or own constitutive state. Stage H remains limited to linearization, block actions, condensation, and verification; nonlinear activation, active-set selection, trial updates, and commit/rollback remain Stage I.

  Update current_design.md and specification.tex to replace their current solver-local ownership language with this architecture.

  ## Ownership, Lifetime, and Interfaces

  - When reconstructed-fault coupling is available, Simulator owns exactly one `ReconstructedFaultSurfaceSystem` and one `Assemblers::ReconstructedFaultStokes`. Add narrow `SimulatorAccess` getters. Existing Stage F/G tests will use these canonical instances rather than constructing independent stateful helpers.

  - Add `StokesSolver::ReconstructedFaultCondensedSystem<dim>`. Its constructor obtains and retains references to the canonical components once; it does not repeatedly cast the material model or look up components during operator applications.

  - Represent one coupled linearization with a generation-bound Linearization view returned by:
  ```
  Linearization
  linearize(const LinearAlgebra::BlockSparseMatrix &A,
            const LinearAlgebra::BlockVector &physical_bulk_state,
            const FaultVector &slip_rate,
            const ReconstructedFaultSurfaceLinearSolve<dim> *surface_solve = nullptr);
  ```
  A null solve argument selects the canonical unrestricted (K_V^{-1}).

  - `linearize()` invalidates the prior generation, then builds (B) and (R_\Gamma/K_V/G) from the same physical bulk state and slip rate. It publishes a usable view only
    after both components succeed. Failure leaves no usable old or partial coupled linearization.

  - The returned view, not the persistent helper, defines the lifetime of (A), (B), (G), and (K_V). A new linearization, geometry-cache invalidation, mesh change, or
    destruction of the referenced matrix invalidates the view. The matrix and all frozen constitutive/profile data must remain unchanged while the view is used.

  - Add a narrow semantic interface:

  template <int dim>
  class ReconstructedFaultSurfaceLinearSolve
  {
    public:
      virtual ~ReconstructedFaultSurfaceLinearSolve() = default;

      virtual void
      solve(const FaultVector &rhs,
            FaultVector &solution) const = 0;
  };

  ReconstructedFaultSurfaceSystem provides the default unrestricted implementation. Condensation calls only solve(), apply_G(), and apply_B(); it never accesses (K_V),
  its sparse matrices, or UMFPACK factors. Stage I can therefore supply an active/free-set solve that projects the right-hand side and returns zero active components
  without changing the condensation code.

  - The linearization view exposes overwrite-semantics operations:

  const ReconstructedFaultSurfaceResidual &
  surface_residual() const;

  void
  vmult(LinearAlgebra::BlockVector &result,
        const LinearAlgebra::BlockVector &solver_bulk_direction) const;

  void
  build_condensed_rhs(
    const LinearAlgebra::BlockVector &bulk_newton_rhs,
    LinearAlgebra::BlockVector &condensed_rhs) const;

  void
  recover_slip_rate_increment(
    const LinearAlgebra::BlockVector &solver_bulk_increment,
    FaultVector &slip_rate_increment) const;

  void
  apply_uncondensed_jacobian(
    const LinearAlgebra::BlockVector &solver_bulk_direction,
    const FaultVector &slip_rate_direction,
    LinearAlgebra::BlockVector &bulk_result,
    FaultVector &surface_result) const;

  Here bulk_newton_rhs is already (-R_{\rm bulk}) in solver-row units. Recovery returns a candidate increment and does not change manager-owned (V).

  - Add a semantic surface-Jacobian action to the canonical surface system for verification of (G\delta x-K_V\delta V), without exposing matrix storage or factorization.

  ## Operator and Constraint Semantics

  - vmult() computes (A\delta x), applies (G), invokes the selected semantic surface solve, applies (B), and returns (A\delta x-BK_V^{-1}G\delta x). It never forms the
    Schur correction explicitly.

  - Centralize solver-to-physical conversion in one private helper. It is the only code that converts

  [
  \delta p_{\rm physical}=s_p,\delta\widehat p.
  ]

  It copies velocity unchanged, scales pressure once, sets non-Stokes fields to zero, reconstructs constrained physical values, and updates ghosts before apply_G(). No
  pressure conversion is duplicated in vmult(), recovery, or the uncondensed action.

  - Treat all Krylov vectors as homogeneous bulk perturbations:
      - create a homogeneous Stokes constraint view by copying current constraints and setting every inhomogeneity to zero;
      - zero constrained algebraic entries before applying (A);
      - use the homogeneous constraints to reconstruct hanging-node and periodic values in the full physical vector passed to (G);
      - assemble (B) through homogeneous constraints;
      - return constrained algebraic entries as zero;
      - never inject prescribed boundary values into perturbations.

    Document that nonzero input at constrained degrees of freedom is ignored rather than interpreted as a physical perturbation.

  - Support only the 2-D assembled iterative Stokes path with separate velocity and pressure blocks. Melt transport, the direct Stokes solver, matrix-free/GMG, and
    unsupported dimensions fail before linearization.

  - Retain deal.II SolverFGMRES as the supported outer Krylov method. The condensed operator is generally nonsymmetric because (B) and (G) are not assumed transposes. Do
    not use CG/MINRES or advertise symmetry. The existing block-AMG/Schur preconditioner remains an approximation to the condensed operator.

  ## Verification

  - Extend the coupled fixture with one non-committing residual evaluator and compare centered finite differences against

  [
  \left(A\delta x-B\delta V,;G\delta x-K_V\delta V\right)
  ]

  for velocity-only, pressure-only, slip-only, and mixed directions. Cover both pressure modes, non-unit pressure scaling, nonzero cohesive/profile/Maxwell history, and
  a step-size sweep showing second-order truncation followed by roundoff saturation.

  - Add a small explicit full two-block comparison:
      - materialize small (A), (B), (G), and (K_V) matrices from independent basis actions;
      - solve the full monolithic block system directly;
      - solve the condensed system with SolverFGMRES;
      - recover (\delta V);
      - compare both increments and both block residuals;
      - verify numerically that the chosen condensed matrix is nonsymmetric, ensuring the test exercises the required FGMRES capability.

  - Add a two-fault test with distinct, non-overlapping prescribed faults. Verify independent fault-major (K_V) blocks, combined (B/G) actions, condensed right-hand side
    signs, recovered increments, and agreement with the explicit full-block solve.

  - Add homogeneous-constraint tests covering a constrained velocity degree of freedom and a hanging/periodic relation: inhomogeneous boundary values must not enter a
    perturbation, while homogeneous dependent values are reconstructed for physical (G) evaluation.

  - Run all focused Stage F–H tests in one- and two-rank configurations, the relevant unit tests, standard and Voro-enabled builds with -j4, and git diff --check. Do not
    run the complete ASPECT integration suite.

  ## Stage Boundary and Assumptions

  - This request explicitly supersedes the current authoritative text saying the surface helper is solver-local; both authoritative documents will be updated before
    implementation relies on canonical simulator ownership.

  - Stage H does not register or activate the production coupled nonlinear solve, choose an active set, impose the (V_{\min}) bound, perform a line search, or update/
    commit/rollback (V), (\Theta), cohesive history, (I_h), or Maxwell stress.

  - The default Stage H surface solve is unrestricted. The new semantic solve boundary is introduced specifically so Stage I can replace it with a free-set solve without
    modifying the condensed operator or its signs.
