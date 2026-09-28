  # Stage I — Coupled Nonlinear Lifecycle

  ## Summary

  Connect the existing reconstructed-fault residual, \(K_V\), \(B\), \(G\), and condensed operator to the production Newton solve. Stage I will add the lower-bound active-set algorithm, joint line search, separate bulk/surface convergence, user-controlled \(\Theta_0\) initialization, and manager-owned \(V\) commit/rollback.

  The coupled path will activate automatically when `PhaseFieldFault`, reconstructed faults, and the existing `single Advection, iterated Newton Stokes` scheme are selected. No new nonlinear-solver scheme or fault-solver parameters will be introduced. Unsupported schemes remain explicit configuration errors.

  Stage I commits only \(V\). Evolution and post-convergence commit of \(\Theta\), cohesive history, previous \(I_h\), and Maxwell stress remain Stage J. Until then, production advancement beyond timestep zero will fail explicitly rather than reuse stale history.

  ## Constitutive Initialization and Interfaces

  - Add `PhaseFieldFault::prepare_reconstructed_fault_mechanical_solve()` as the single material-owned preparation operation. It will:
      - recompute transient current \(I_h\);
      - initialize cohesive history from the prescribed initial \(H\) when it is not already committed;
      - initialize \(\Theta_0\) when rate-and-state friction is selected;
      - preserve fully initialized checkpoint state;
      - reject partially initialized or inadmissible state;
      - finish by validating all constitutive prerequisites.

  - Supply \(\Theta_0\) through the existing particle/compositional-field mapping infrastructure:
      - exactly one compositional field of type generic must use the particles method and map component zero to the reserved particle property name phase field fault state;
      - its particle values are projected with the existing Q1 particle-to-fault projection and stored in the already registered fault property of the same name;
      - the projected nodal values must all be finite and strictly positive;
      - missing, duplicate, wrong-type, wrong-component, or unavailable mappings are input errors;
      - no new particle-property plugin, scalar \(\Theta_0\) parameter, parsed function, or inferred steady-state value is added.

  - At timestep zero, initialize every newly reconstructed fault with \(V=V_{\min}\). On restart or a later nonlinear invocation, preserve the manager’s committed \(V\) and begin from it.

  - Extend the semantic surface-solve boundary with:
  ```
    using FaultActiveSet = std::vector<std::vector<bool>>;

    std::unique_ptr<ReconstructedFaultSurfaceLinearSolve<dim>>
    create_restricted_linear_solve(const FaultActiveSet &) const;
  ```
  The returned solve will use the current assembled \(K_V\), project its right-hand side onto free nodes, return exactly zero on active nodes, and be invalidated by a new surface linearization.

  - Extend the condensed linearization minimally with:
  ```
    Linearization
    with_surface_solve(
      const ReconstructedFaultSurfaceLinearSolve<dim> &) const;

    LinearAlgebra::BlockVector
    make_physical_bulk_direction(
      const LinearAlgebra::BlockVector &solver_direction) const;
  ```
  The first operation replaces only the semantic \(K_V^{-1}\) action without rebuilding \(A\), \(B\), \(G\), or \(K_V\). The second exposes the already centralized homogeneous- constraint and solver-pressure conversion needed to update the physical bulk iterate.

  ## Projected Newton Algorithm

  For each nonlinear iteration:

  1. Assemble the mathematical bulk residual, \(A\), and the canonical fault bulk contribution at the current accepted \((x,V)\). Linearize the canonical surface system and frozen \(B\) coefficients from that same state.

  2. Start with an empty active set and solve the unrestricted condensed system.

  3. Mark a node active when it is at \(V_{\min}\), within \(100\epsilon_{\rm mach}\max(V_{\min},|V|_\infty)\), and its recovered direction has \(\delta V<0\).

  4. Build a restricted semantic surface solve, rebind the condensed view, and resolve the bulk and surface directions without rebuilding the constitutive linearization.

  5. Repeat additions until no free bound node points below the bound. Active nodes are never removed within one projected solve; the active set is recomputed from empty at the next Newton iteration, allowing release.

  6. Project the surface residual by zeroing precisely the stabilized active entries. Free entries remain unchanged.
  7. Apply fraction-to-boundary only to free nodes:

     \[
     \alpha_{\max} =
     \min\left(
     1,~0.99\min_{\substack{i\in\mathcal F\\ \delta V_i<0}}
     \frac{V_i-V_{\min}}{-\delta V_i}
     \right).
     \]

     There is no upper bound or upper clipping.

  The restricted solve will retain the existing reviewable UMFPACK strategy: materialize each per-fault matrix with identity rows for active nodes and the principal \(K_{\mathcal F\mathcal F}\) block for free nodes. It will support nonsingular indefinite free blocks and retain scaled backward-error diagnostics.

  ## Residual, Line Search, and Lifecycle

  - Implement one simulator-side non-committing coupled residual evaluator for explicit \((x,V)\):

    \[
    R(x,V)=
    \begin{bmatrix}
    R_{\rm bulk}(x,V)\\
    R_\Gamma(x,V)
    \end{bmatrix}.
    \]

    It will include the ordinary Stokes residual and the complete reconstructed-fault contribution, including the frozen \(V\)-independent history term. Evaluation may rebuild computational residual buffers but must restore the accepted bulk iterate and manager trial state on rejection or exception.

  - Invoke the canonical `Assemblers::ReconstructedFaultStokes::execute()` from production Stokes cell assembly when the coupled reconstructed-fault solve is active. Do not create a second stateful assembler or coupling object.

  - Freeze block normalization scales from the first projected residual of the nonlinear solve:

    \[
    r_b=\frac{|R_{\rm bulk}|2}{S_b},
    \qquad
    r\Gamma=
    \frac{|P_{\mathcal F}R_\Gamma|2}{S_\Gamma}.
    \]

    \(S_b\) and \(S_\Gamma\) are their respective initial norms with only a smallest-positive-number guard against exact zero. Convergence requires both \(r_b\) and \(r_\Gamma\) to satisfy the existing nonlinear tolerance.

  - Use the joint merit function

    \[
    \Phi=\frac12(r_b^2+r_\Gamma^2).
    \]

    Start the line search at \(\alpha_{\max}\), use the existing Armijo coefficient \(10^{-4}\), reduction factor \(2/3\), and Max Newton line search iterations. A zero maximum preserves ASPECT’s existing “accept the full admissible step” behavior; exhausting a positive maximum preserves the existing behavior of accepting the last scaled candidate and allowing the outer nonlinear convergence test to decide success.

  - Every candidate is formed from the current accepted bulk iterate and current accepted manager \(V\). Rejected candidates restore both and cannot accumulate.
  - On candidate acceptance, update only the current bulk iterate and manager current \(V\). On nonlinear convergence, copy the bulk iterate to solution and commit manager-owned \(V\).

  - Wrap the complete solve in rollback handling. Any exception or nonlinear failure rolls current/trial \(V\) back to the timestep-committed field and leaves \(\Theta\), cohesive traction, previous \(I_h\), and Maxwell stress untouched.

  - Reuse the assembled block-AMG/Schur preconditioners as an approximation to the nonsymmetric condensed operator and use `SolverFGMRES` with the existing cheap/expensive iteration limits, restart length, and linear tolerance controls. Bulk increments remain homogeneous; pressure is converted to physical units exactly once through the condensed-linearization helper before updating the nonlinear iterate.

  ## Verification and Documentation

  - Update both authoritative design documents with the mapped-generic-field \(\Theta_0\) rule, exact active-set iteration, restricted solve semantics, projected residual norm, line-search constants, automatic solver dispatch, and Stage-I-only \(V\) commit boundary.

  - Add focused tests covering:
      - valid spatially varying \(\Theta_0\) projection and missing, duplicate, wrong-type, wrong-component, nonpositive, and partially initialized state;
      - timestep-zero initialization at exactly \(V_{\min}\), restart preservation, and later begin-from-committed manager semantics;
      - outward and inward directions at the bound, multiple active-set passes, all-active and mixed active/free faults, active-node release on a later iteration, multiple faults, and indefinite free \(K_V\);
      - fraction-to-boundary on free nodes and absence of upper clipping;
      - repeated rejected trials without accumulation, successful \(V\)-only commit, and exception/nonconvergence rollback;
      - separate bulk and projected-surface convergence and joint-merit line-search acceptance;
      - end-to-end coupled solves for rate-and-state and rate-dependent friction, with dynamic and adiabatic fault pressure;
      - the explicit failure on post-timestep-zero evolution while Stage-J history commit remains absent.

  - Complete the mandatory centered finite-difference sweep of the single coupled residual against the uncondensed block action for velocity-only, pressure-only, slip-only, and mixed directions. Cover both pressure modes, nonzero cohesive/profile/Maxwell history, states above but close to \(V_{\min}\), one and two MPI ranks, and demonstrate second-order truncation followed by roundoff saturation. Bound-active behavior will use separate projected/one-sided tests rather than centered differences across the constraint.

  - Build with -j4; run the Stage F–I focused unit and integration tests in the standard and Voro-enabled builds, including the two-rank fixtures, followed by git diff --check. Do not run the complete ASPECT integration suite.
