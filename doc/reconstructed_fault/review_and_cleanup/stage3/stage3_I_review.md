• ## Stage I review outcome

  Stage I implements the coupled bulk/fault nonlinear mechanical solve, lower-bound active-set treatment, non-committing line search, and fresh-state initialization required by the authoritative design.

  The focused tests pass, but I would classify Stage I as implemented with incomplete edge-case verification, rather than fully review-complete. Several tests explicitly requested by the Stage I revision instructions have not yet been added, particularly forced line-search rejection/failure, active-set release, and rollback tests.

  Stage I is also not committed. The current HEAD is the Stage H commit:

  8a87657bf1fbe66263596125e80fb534475b20e8

  No Stage I commit hash exists yet.

  ## Files changed

  ### Design documentation

  - `doc/reconstructed_fault/current_design.md`
    Records the Stage I invariants: local lower-bound classification, principal-free-block solves, exact fraction-to-boundary, fixed-active-set Armijo search, block-wise convergence, fresh-state initialization, and commit/rollback ownership.

  - `doc/reconstructed_fault/specification.tex`
    Adds the corresponding mathematical requirements to the authoritative specification, including the active/free partition, merit function, residual normalization, and Stage I lifecycle boundary.

  ### Public headers

  - `include/aspect/material_model/phase_field_fault.h`
    Adds the material-model operation that prepares a complete, frozen constitutive state before the coupled mechanical solve. Its invariant is that all state needed for pointwise constitutive evaluation is complete before Newton begins.

  - `include/aspect/reconstructed_fault/surface_system.h`
    Adds the fault-vertex active-set representation and a semantic restricted \(K_V^{-1}\) operation. The restricted solve returns zero on active vertices and solves only the principal free block.

  - `include/aspect/simulator.h`
    Adds the internal Stage I solver entry point and an assembly gate controlling when reconstructed-fault Stokes contributions are included.

  - `include/aspect/simulator/solver/reconstructed_fault_condensed_system.h`
    Allows one coupled linearization to be rebound to a different semantic surface inverse without changing the condensation algorithm. It also exposes the already-defined conversion from homogeneous solver coordinates to the physical bulk perturbation required by trial evaluation.

  ### Production sources

  - `source/material_model/phase_field_fault.cc`
    Implements mechanical-state preparation. Fresh timestep-zero runs may initialize missing \(I_h\), cohesive history, (\Theta_0), and \(V\); restart and later-time paths require complete committed state. Partial or invalid state is rejected.

  - `source/reconstructed_fault/manager.cc`
    Rewrites Q1 slip-rate interpolation in an algebraically equivalent form that preserves an exactly constant \(V_{\min}\) profile under floating-point arithmetic.

  - `source/reconstructed_fault/surface_system.cc`
    Implements the restricted semantic surface solve. Active rows and columns are disconnected, active right-hand-side entries are projected out, active increments are exactly zero, and each free block retains the existing UMFPACK factorization and backward-error validation. Surface slip interpolation receives the same constant- preserving adjustment.

  - `source/simulator/assembly.cc`
    Connects the genuine reconstructed-fault Stokes assembler to canonical Stokes assembly only while the coupled residual or Jacobian is being evaluated.

  - `source/simulator/solver.cc`
    Implements the Stage I nonlinear lifecycle: consistent coupled linearization, FGMRES condensed solve, monotone inner active-set iteration, exact fraction-to-boundary, fixed-active-set Armijo search, separate bulk/surface convergence, non-accumulating trials, and success/failure state handling.

  - `source/simulator/solver/reconstructed_fault_condensed_system.cc`
    Implements semantic replacement of unrestricted (K_V^{-1}) by the active/free restricted solve while invalidating the superseded coupled-linearization view.

  - `source/simulator/solver_schemes.cc`
    Dispatches reconstructed-fault mechanics through the coupled solver for single Advection, iterated Newton Stokes. It explicitly rejects later timesteps until Stage J supplies constitutive-history evolution.

  ### Modified test

  - `tests/phase_field_fault_surface_system.cc`
    Adds checks that active entries are exactly zero, active columns do not contaminate free equations, the principal free block is solved, and condensed recovery uses the
    restricted semantic inverse.

  ### New end-to-end tests

  - `tests/phase_field_fault_stage_i.cc`
    Adds a postprocessor that verifies committed \(V\) is finite and admissible, at least one vertex reaches \(V_{\min}\) exactly, the fixture contains a nonzero residual at a bound-active vertex, and reconstructed-fault constitutive state remains valid.

  - `tests/phase_field_fault_stage_i.prm`
    Defines the one-rank, rate-dependent, dynamic-pressure Stage I solve. It uses AT1 with \(\ell=10h\). The low reconstruction threshold is explicitly identified as test bootstrapping; reconstruction behavior is not under test.

  - `tests/phase_field_fault_stage_i/screen-output`
    Reference output for the rate-dependent Stage I solve.

  - `tests/phase_field_fault_stage_i_rate_state.cc`
    Reuses the Stage I verification postprocessor for the rate-and-state fixture.

  - `tests/phase_field_fault_stage_i_rate_state.prm`
    Defines the two-rank rate-and-state case, including a spatially varying particle-advected generic field mapped to phase field fault state[0].

  - `tests/phase_field_fault_stage_i_rate_state/screen-output`
    Reference output for the distributed rate-and-state solve.

  - `tests/phase_field_fault_stage_i_missing_state.cc`
    Reuses the Stage I plugin in the expected-failure fixture.

  - `tests/phase_field_fault_stage_i_missing_state.prm`
    Defines an invalid rate-and-state configuration with no mapped initial \(\Theta\) field.

  - `tests/phase_field_fault_stage_i_missing_state.sh`
    Verifies that the invalid configuration fails with the intended mapping diagnostic.

  - `tests/phase_field_fault_stage_i_missing_state/screen-output`
    Expected diagnostic for the missing-state test.

  The other untracked files under `doc/reconstructed_fault/` predate Stage I and are not included in this change set.

  ## New or modified public interfaces

  The following externally visible C++ interfaces were added or changed:
  ```
  using ReconstructedFaultActiveSet =
    std::vector<std::vector<bool>>;

  std::unique_ptr<ReconstructedFaultSurfaceLinearSolve<dim>>
  ReconstructedFaultSurfaceSystem<dim>::create_restricted_linear_solve(
    const ReconstructedFaultActiveSet &active_set) const;

  void
  MaterialModel::PhaseFieldFault<dim>::
  prepare_reconstructed_fault_mechanical_solve();

  ReconstructedFaultCondensedSystem<dim>::Linearization
  Linearization::with_surface_solve(
    const ReconstructedFaultSurfaceLinearSolve<dim> &new_surface_solve) const;

  LinearAlgebra::BlockVector
  Linearization::make_physical_bulk_direction(
    const LinearAlgebra::BlockVector &solver_direction) const;
  ```
  `make_physical_bulk_direction()` previously existed privately and is now public because non-committing nonlinear trial construction needs the same canonical pressure and constraint conversion used by \(G\).

  The following are simulator-internal interfaces, not general reconstructed-fault APIs:
  ```
  void Simulator<dim>::solve_reconstructed_fault_stokes();
  bool Simulator<dim>::assemble_reconstructed_fault_stokes_terms;
  ```
  No new user-facing parameter was introduced.

  ## Mathematical and design correspondence

   Implementation invariant | Design/specification correspondence
   ---- | ----
   Coupled Newton system uses \(\begin{bmatrix}A&-B\\ G&-K_V\end{bmatrix}\begin{bmatrix}\delta x\\ \delta V\end{bmatrix}=-\begin{bmatrix}R_{\rm bulk}\\ R_\Gamma\end{bmatrix}\) | Existing Stage H coupled-system equations and Stage I nonlinear lifecycle
   Condensed solve remains \((A-BK_V^{-1}G)\delta x=-R_{\rm bulk}+BK_V^{-1}R_\Gamma \) | Condensation and recovery section
   Recovery remains \(\delta V=K_V^{-1}(R_\Gamma+G\delta x)\) | Condensation and recovery section
   Active solve uses \(K_{\mathcal F\mathcal F}^{-1}\) and exact-zero active | `specification.tex`, “Stage-I lower bound and nonlinear lifecycle” increments
   Local classification \(V_i-V_{\min}\le100\epsilon_{\rm mach}\max(V_{\min},\vert V_i\vert)\) | Same Stage I section
   Active set starts empty and grows monotonically within one Newton iteration | Same Stage I section
   Active set is rebuilt from empty after an accepted step | Same Stage I section
   Exact free-node bound step \((V_i-V_{\min})/(-\delta V_i)\), without a 0.99 factor | Same Stage I section
   Roundoff-level contact is projected to exactly \(V_{\min}\), while constitutive evaluation still rejects genuinely inadmissible \(V\) | Same Stage I section   
   Merit function \(\Phi=(r_b^2+r_\Gamma^2)/2\) and Armijo condition \(\Phi_{\rm trial}\le(1-10^{-4}\alpha)\Phi_k\) | Same Stage I section
   Rejected trials are formed from the same accepted \((x,V)\) and never accumulate | Non-committing trial lifecycle
   Bulk and projected-free surface residuals have separate fixed scales and must both converge | Residual-normalization and block-wise convergence requirements
   Fresh timestep-zero state may be initialized; restart/later state may not be reconstructed | Mechanical-preparation requirements
   Only \(V\) is committed |  Stage I boundary; (\Theta), cohesive history, previous (I_h), Maxwell stress, and phase-field evolution remain outside Stage I
   FGMRES is used for the generally nonsymmetric condensed operator | Stage H solver constraint carried into Stage I
   The complete bulk residual includes both \(-BV\) and the frozen \(V\)-independent Maxwell-history contribution | Stage G bulk residual decomposition
   

  ## Assumptions and implementation choices not fully specified

  - The notation \(|P_{\mathcal F}R_\Gamma|_\Gamma\) is implemented as the Euclidean norm of the assembled Q1 nodal residual, not a mass-weighted surface norm.
  - The bulk reference scale is obtained using ASPECT’s existing Stokes residual convention with a zero-velocity reference state and the current pressure.
  - The surface scale floor uses the larger of the current surface residual and the action of \(K_V\) on a characteristic slip-rate vector with entries \(\max(V_{\min},|V_i|)\).
  - A fresh model is interpreted strictly as timestep zero with resume_computation == false.
  - Stage I is restricted to two dimensions, ordered reconstructed-fault polylines, and Q1 fault fields.
  - The fault geometry and quadrature association caches are assumed not to change during one coupled nonlinear solve.
  - The implementation assumes UMFPACK is available for nonsingular, potentially indefinite restricted free blocks.
  - The PhaseFieldFault bulk Maxwell tangent is linear in the current strain rate, so ASPECT’s generic viscosity-derivative scaling is set to zero while the explicit \(B\), \(G\), and \(K_V\) blocks represent reconstructed-fault derivatives.
  - Only the exact single Advection, iterated Newton Stokes production path is redirected to the coupled solve. Existing reconstruction-only/no-Stokes fixtures retain their previous behavior.
  - Existing ASPECT nonlinear tolerance, linear tolerance, line-search iteration limit, restart length, and the established \(2/3\) reduction are reused. No reconstructed- fault-specific solver tuning parameters were added.

  ## Existing behavior that changed

  - A reconstructed-fault model using single Advection, iterated Newton Stokes now solves the coupled ((x,V)) system instead of the ordinary standalone Newton Stokes
    system.

  - Stage I mechanics explicitly fails after timestep zero because the deferred constitutive histories cannot yet be advanced consistently.
  - Fresh rate-and-state mechanics now requires exactly one generic, particle-advected compositional field mapped to phase field fault state[0].
  - Missing, partial, nonfinite, or nonpositive (\Theta) state is now rejected before mechanical iteration.
  - Fresh \(V\) is initialized exactly to (V_{\min}). Successful convergence commits it; nonlinear failure restores the timestep-committed value.
  - Exhausting the Armijo search is now a nonlinear failure. The last unsuccessful candidate is not silently accepted.
  - The production bulk solution is not used as trial scratch. Only velocity and pressure from the accepted working state are copied back after convergence.
  - Q1 interpolation of (V) was algebraically reordered. Constant profiles, especially (V=V_{\min}), are now preserved exactly. Nonconstant interpolation can differ by last-bit floating-point roundoff.

  - initialize_cohesive_state_from_initial_fields() now ensures the current \(I_h\) field is recomputed even when cohesive history is already initialized.
  - The reconstructed-fault Stokes contribution is conditionally included during coupled residual and Jacobian assembly. Ordinary Stokes assembly is unchanged while that gate is false.

  ## Tests and exact results

  ### Builds

  - ASPECT_WITH_VORO=ON, Debug build: succeeded with -j4.
  - ASPECT_WITH_VORO=ON, Release build: succeeded with -j4.
  - Final source state was compiled after the last cleanup edit.
  - git diff --check: passed with no whitespace errors.

  ### Unit tests

  The standard unit-test target completed successfully:

  All tests passed (61429 assertions in 67 test cases)

  ### Focused integration tests

  - phase_field_fault_surface_dynamic_pressure
    Passed, 151.83 s.
    This includes the new restricted-(K_V) checks and confirms that condensed recovery uses the semantic active/free solve.

  - phase_field_fault_stage_i
    Passed, 156.14 s.
    The solve converged in two coupled nonlinear residual evaluations. The final normalized residuals were:

    bulk:  7.076e-08
    fault: 0.000e+00

    The postprocessor verified admissible committed (V), exact contact with (V_{\min}), and a nonzero residual at a bound-active vertex.

  - phase_field_fault_stage_i_rate_state, MPI ranks: 2
    The full simulation completed successfully in 102.05 s. Its first CTest comparison failed only because the expected iteration counts and rounded diagnostics had
    changed. After updating the reference output, the cached CTest comparison passed in 5.85 s. The converged residuals were:

    bulk:  6.282e-08
    fault: 0.000e+00

    The postprocessor verified positive, spatially varying projected (\Theta_0) and admissible committed (V).

  - phase_field_fault_stage_i_missing_state
    Passed, 9.51 s, after the final rebuild.
    It confirmed that missing rate-and-state initialization fails with:

    Rate-and-state reconstructed-fault friction requires exactly one ...

  ### Tests not run

  The complete ASPECT integration suite was not run, following the earlier instruction not to launch the full 1289/1303-test suite unless requested.

  The two long positive Stage I fixtures were not rerun from scratch after the final behavior-neutral cleanup in the material-model state check. The final code compiled,
  and the expected-failure fixture was rerun afterward.

  ## Verification gaps and uncertainties

  These are the principal review findings:

  - There is no dedicated test showing that the local lower-bound tolerance remains unaffected by an extremely large slip rate elsewhere on the same fault.
  - Exact arrival at (V_{\min}) is verified end-to-end, but there is no small isolated test that directly distinguishes the exact fraction-to-boundary formula from the
    removed 0.99 strict-interior rule.

  - Active columns being removed from free equations is tested. Active-set release after rebuilding the set from empty on the next Newton iteration is not directly tested.
  - The active set is fixed by code structure during the line search, but no fixture forces multiple rejected trial lengths and observes this invariant.
  - No test deliberately exhausts the Armijo search and verifies that the last unsuccessful candidate is rejected.
  - No forced-failure integration test verifies simultaneously that the production bulk solution and timestep-committed (V) are restored after one or more previously
    accepted Newton trials.

  - Zero and extremely small initial residual normalization cases are not directly tested.
  - Block-wise convergence is implemented, but there is no fixture in which one residual block meets tolerance while the other intentionally remains unconverged.
  - Restart preservation and the later-timestep rejection are enforced in production code but are not covered by a complete restart integration test.
  - The MPI rate-and-state fixture selects adiabatic pressure but has zero gravity, so it exercises the branch and lifecycle without producing a nonzero adiabatic friction
    pressure. The nonzero dynamic/adiabatic Jacobian behavior remains covered by the earlier Stage F/H finite-difference tests, which were not rerun as part of this Stage
    I pass.

  - The main successful fixtures converge in one accepted Newton update, so they do not exercise actual line-search backtracking or rejected-trial rollback.
  - The surface residual norm is an algebraic nodal (l^2) norm. If the design intended a quadrature- or mass-weighted (\Gamma)-norm, that point needs an explicit decision.
  - Three-dimensional faults and overlapping multiple-fault bulk associations remain outside the supported scope.

  Because the Stage I revision instructions explicitly requested several of the missing edge-case tests above, this is the main outstanding review issue.

  ## Compact diff stat

  Git’s normal diff stat covers the tracked changes:

  14 files changed, 1080 insertions(+), 16 deletions(-)

  The ten new, currently untracked Stage I test files contain another 376 lines. Therefore the complete Stage I working set is:

  24 files changed/new, 1456 insertions, 16 deletions

  Pre-existing untracked reconstructed-fault design and review files are excluded.

