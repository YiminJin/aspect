  # Stage I — Projected Coupled Newton Lifecycle

  ## Summary

  Connect the existing reconstructed-fault residual, \(K_V\), \(B\), \(G\), and condensed operator to the production Newton solve. Stage I will add:

  - a monotone inner lower-bound active-set solve;
  - exact arrival at \(V_{\min}\);
  - a fixed-merit Armijo line search that fails if no candidate is accepted;
  - separate bulk and free-surface convergence tests;
  - user-supplied (\Theta_0) initialization;
  - manager-owned current/trial/committed \(V\);
  - a separate working bulk iterate that is committed only after convergence.

  The coupled path will activate when reconstructed faults and `PhaseFieldFault` use the existing single Advection, iterated Newton Stokes scheme. No new nonlinear-solver scheme, transformed slip variable, or upper bound on \(V\) will be added.

  Stage I commits only \(V\). Evolution or commit of \(\Theta\), cohesive history, previous \(I_h\), and Maxwell stress remains Stage J. Production advancement beyond timestep zero will remain explicitly unavailable until that history update exists.

  ## Initialization and Required Interfaces

  - Add `PhaseFieldFault::prepare_reconstructed_fault_mechanical_solve()` as the material-owned preparation operation.

    For a fresh, non-restarted timestep-zero model it will:
      - recompute transient current \(I_h\);
      - initialize cohesive history from the prescribed initial \(H\), if absent;
      - project and initialize the user-supplied positive \(\Theta_0\), if rate-and-state friction is selected;
      - validate the complete constitutive state.

    On restart or at \(k>0\), it will recompute only transient current \(I_h\), preserve all committed history, and reject missing, partial, or inadmissible committed state. It will never reconstruct later-time history from initial-condition data.

  - Supply \(\Theta_0\) through the existing compositional-field/particle-property mapping:
      - exactly one field of type generic must use the particles method and map component zero to the reserved particle property phase field fault state;
      - its particle values are projected with the existing Q1 particle-to-fault projection into the registered fault property of the same name;
      - every projected value must be finite and strictly positive;
      - missing, duplicate, wrong-type, wrong-component, or unavailable mappings are input errors;
      - no new particle plugin, scalar initial-state parameter, parsed function, or inferred steady-state value is introduced.

  - For a fresh timestep-zero reconstruction, initialize manager-owned \(V\) exactly to \(V_{\min}\). Restarted or previously committed \(V\) is preserved. Missing committed \(V\) outside fresh initialization is an error.

  - Extend ReconstructedFaultSurfaceSystem with:

    ```
    using FaultActiveSet = std::vector<std::vector<bool>>;

    std::unique_ptr<ReconstructedFaultSurfaceLinearSolve<dim>>
    create_restricted_linear_solve(const FaultActiveSet &) const;
    ```

    For free and active sets \(\mathcal F,\mathcal A\), this solve represents

    \[
    \begin{bmatrix}
    K_{\mathcal F\mathcal F} & 0\\
    0 & I_{\mathcal A\mathcal A}
    \end{bmatrix}.
    \]

    Both active rows and active columns are removed from free equations. The right-hand side is projected onto \(\mathcal F\), and the returned active increments are exactly zero. The implementation will retain UMFPACK, nonsingular-indefinite support, scaled backward-error checks, and generation invalidation.

  - Extend the condensed linearization minimally with:
    ```
    Linearization
    with_surface_solve(
      const ReconstructedFaultSurfaceLinearSolve<dim> &) const;

    LinearAlgebra::BlockVector
    make_physical_bulk_direction(
      const LinearAlgebra::BlockVector &solver_direction) const;
    ```
    Rebinding replaces only the semantic \(K_V^{-1}\) action and reuses the existing \(A\), \(B\), \(G\), \(K_V\), and residual data. The physical-direction operation remains the single implementation of homogeneous constraint handling and solver-pressure conversion.

  ## Projected Newton and Residual Scaling

  Maintain a private working accepted bulk state working_x; do not use the production solution as line-search scratch. Begin manager current \(V\) from its committed value.

  For every outer Newton iteration:

  1. Assemble the complete bulk residual and \(A\) at `working_x`, `current_V`.
  2. Assemble \(R_\Gamma/K_V/G\) and freeze \(B\) at exactly the same state.
  3. Start with an empty active set and solve the unrestricted condensed system.
  4. For each currently free node, classify it as locally at the bound when

     \[
     V_i-V_{\min}
     \le
     100\epsilon_{\rm mach}\max(V_{\min},|V_i|).
     \]

  5. Add such a node to the active set if its recovered direction satisfies \(\delta V_i<0\).
  6. Rebuild only the restricted surface solve, rebind the condensed linearization, and resolve.
  7. Repeat until no node is added.

  Active nodes are only added within this inner solve. The active set is discarded and recomputed from empty at the next Newton iteration, allowing nodes to be released.

  After stabilization, zero active entries of \(R_\Gamma\) and leave free entries unchanged. Use

  \[
  r_b=\frac{|R_{\rm bulk}|^2}{S_b},
  \qquad
  r_\Gamma=
  \frac{|P_{\mathcal F}R_\Gamma|^2}{S_\Gamma}.
  \]

  Both fixed scales are established during the first Newton iteration and remain unchanged for the complete nonlinear solve.

  The existing ASPECT bulk reference is the initial Newton Stokes residual in solver-row units: the velocity/pressure RHS norm evaluated with zero velocity and the current pressure, as used by `compute_initial_newton_residual()`. Reuse that convention non-committingly and define

  \[
  \epsilon_{\rm scale} =
  \max\left(\epsilon_{\rm linear\ Stokes},
  \sqrt{\epsilon_{\rm mach}}\right),
  \]

  \[
  S_b =
  \max\left(
  |R_{\rm bulk}^{(0)}|^2,;
  \epsilon_{\rm scale}S_{b,\rm ASPECT}
  \right).
  \]

  For the surface block, define a physical reference using the complete initial residual and the current Jacobian acting on the local slip scale

  \[
  (V_{\rm char})_i=\max(V_{\min},|V_i^{(0)}|),
  \]

  \[
  S_\Gamma =
  \max\left(
  |P_{\mathcal F}R_\Gamma^{(0)}|^2,;
  \epsilon_{\rm scale}
  \max\left[
  |R_\Gamma^{(0)}|^2,~
  |K_{VV,{\rm char}}|^2
  \right]
  \right).
  \]

  These floors use existing solver accuracy and block-native residual scales; they introduce no dimensional user parameter. If a block and its reference scale are both exactly zero, its normalized residual is defined as zero only while that block remains exactly zero; a later nonzero residual with no reference scale is an explicit numerical failure.

  Convergence is block-wise:

  \[
  r_b<\epsilon_{\rm NL}
  \quad\text{and}\quad
  r_\Gamma<\epsilon_{\rm NL}.
  \]

  The merit function is not used as the convergence criterion.

  ## Exact Bound Contact, Line Search, and Lifecycle

  After the active set stabilizes, compute the maximum admissible step using only free nodes:

  \[
  \alpha_{\max} =
  \min\left(
  1,;
  \min_{\substack{i\in\mathcal F\\ \delta V_i<0}}
  \frac{V_i-V_{\min}}{-\delta V_i}
  \right).
  \]

  There is no 0.99 factor. A free node may arrive exactly at (V_{\min}).

  For each candidate:

  \[
  x^{\rm trial}=x^{(k)}+\alpha\delta x,
  \qquad
  V^{\rm trial}=V^{(k)}+\alpha\delta V.
  \]

  - Start at (\alpha=\alpha_{\max}) and reduce by \(2/3\).
  - Keep the stabilized active set, projected residual definition, and normalization scales fixed for every candidate in that line search.
  - If a candidate component lies just below or above \(V_{\min}\) within the same local bound tolerance, project the nonlinear variable exactly to \(V_{\min}\).
  - A candidate farther below \(V_{\min}\) is a numerical failure and is never passed to the constitutive law.
  - The friction law itself remains unclamped and continues to reject \(V<V_{\min}\).

  Use the fixed joint merit

  \[
  \Phi=\frac12(r_b^2+r_\Gamma^2)
  \]

  and accept only if

  \[
  \Phi_{\rm trial}
  \le
  \left(1-10^{-4}\alpha\right)\Phi_{\rm current}.
  \]

  The initial candidate plus at most Max Newton line search iterations reductions are tested. A value of zero therefore tests the initial admissible candidate once. If no candidate satisfies Armijo, discard the trial and report nonlinear/line-search failure; never accept the last unsuccessful candidate.

  Every trial is formed from the same accepted `working_x` and manager current \(V\). Rejected trials cannot accumulate or modify:

  - `working_x`;
  - production solution;
  - manager current or committed \(V\);
  - \(\Theta\), cohesive history, previous \(I_h\), or Maxwell stress.

  On nonlinear convergence, copy working_x to the production solution and commit manager current \(V\). Any exception, exhausted line search, or Newton-iteration failure to its timestep-committed state, and leaves all deferred history unchanged.

  Use one simulator-side non-committing evaluator for

  \[
  R(x,V)=
  \begin{bmatrix}
  R_{\rm bulk}(x,V)\\
  R_\Gamma(x,V)
  \end{bmatrix}.
  \]

  It must include the complete ordinary and fault bulk residual, including the frozen (V)-independent history term. The canonical `Assemblers::ReconstructedFaultStokes` will provide its genuine cell/QP contribution without creating another stateful coupling object.

  The outer condensed solve will continue to use SolverFGMRES and the existing block-AMG/Schur preconditioners, restart length, cheap/expensive iteration limits, and linear tolerances.

  ## Verification and Stage Boundary

  Update both authoritative design documents with the mapped-(\Theta_0) rule, local bound criterion, principal-free-block solve, exact-bound step, fixed-active-set
  Armijo search, meaningful normalization floors, working-bulk-state lifecycle, and failure semantics.

  Add focused tests for:

  - valid spatially varying (\Theta_0) projection and every invalid mapping/state case;
  - fresh (V=V_{\min}), restart preservation, and begin-from-committed behavior;
  - local bound classification remaining unaffected by a very large slip rate elsewhere;
  - outward and inward bound directions, monotone multi-pass activation, release on the next Newton iteration, mixed/all-active faults, multiple faults, and indefinite
    free (K_V);

  - removal of active columns from free equations;
  - exact arrival at (V_{\min}) with no strict-interior 0.99 behavior;
  - fixed active sets and fixed merit definitions across a line search;
  - exact Armijo acceptance and failure after exhausting permitted reductions;
  - non-accumulating rejected trials and preservation of both production bulk state and committed (V) on failure;
  - finite, meaningful normalization for zero and extremely small initial block residuals;
  - independent bulk and free-surface convergence;
  - end-to-end rate-and-state and rate-dependent solves under dynamic and adiabatic fault pressure;
  - explicit rejection of post-timestep-zero evolution until Stage J supplies atomic history updates.

  Complete the mandatory centered finite-difference sweep of the single coupled residual against the uncondensed block action for velocity-only, pressure-only, slip-
  only, and mixed directions. Cover both pressure modes, nonzero cohesive/profile/Maxwell history, states strictly above but close to (V_{\min}), multiple faults, and
  one- and two-rank execution. Require the expected second-order truncation regime followed by roundoff saturation. Test bound-active states separately with projected
  and one-sided admissible checks.

  Build with -j4, run the focused Stage F–I unit and integration tests in standard and Voro-enabled builds, run the two-rank fixtures, and finish with git diff --check.
  Do not run the complete ASPECT integration suite.

  Stage I will not evolve or commit (\Theta), cohesive history, previous (I_h), or Maxwell stress, and it will not add phase-field evolution. Those operations remain
  Stage J.
