  # Stage H — Exact Solver-Side Condensation

  ## Summary

  Add an unrestricted solver-side representation of

  \[
  C=A-BK_V^{-1}G,
  \qquad
  b_C=-R_{\rm bulk}+BK_V^{-1}R_\Gamma,
  \qquad
  \delta V=K_V^{-1}(R_\Gamma+G\delta x).
  \]

  Stage H will provide the condensed operator, condensed right-hand side, recovery, pressure-scaling adapter, and uncondensed block action needed for verification. It will not activate the coupled nonlinear solve or implement Stage I state management and active sets.

  ## Implementation

  - Add `StokesSolver::ReconstructedFaultCondensedSystem<dim>` under `simulator/solver/`. It owns one solver-local `ReconstructedFaultSurfaceSystem` and one `Assemblers::ReconstructedFaultStokes`; it references the assembled Stokes matrix \(A\), but owns no constitutive or checkpointed state.

  - Support the initial assembled iterative block-AMG path only: 2-D, separate velocity/pressure blocks, no melt transport, no direct or matrix-free/GMG solver. Construction reports an explicit unsupported-configuration error otherwise.

  - `linearize(physical_bulk_state, slip_rate)` will:
      1. invalidate the previous coupled linearization;
      2. assemble/factor \(K_V\) and retain \(R_\Gamma\);
      3. freeze the Stage-G \(B\) coefficients;
      4. publish the coupled linearization only after both components succeed.

  - Krylov directions use ASPECT’s two-block solver convention:
    \[
    \widehat p=p_{\rm physical}/s_p.
    \]
    Before apply_G(), lift the reduced solver vector into a full ghosted physical direction, copy velocity unchanged, set (\delta p_{\rm physical}=s_p\delta\widehat p),
    and zero non-Stokes fields. Thus the dynamic-pressure term is (-\mu s_p\delta\widehat p); adiabatic mode remains pressure-independent.

  - `vmult()` overwrites its result with \(A\delta x-BK_V^{-1}G\delta x\). It uses reusable solver-local scratch vectors, never forms \(BK_V^{-1}G\), and leaves the existing Stokes block preconditioner unchanged.

  - Keep a private direct \(K_V\) action inside ReconstructedFaultSurfaceSystem, accessible only to the condensed-system helper, so the public coupled block action can verify
  \[
  (\delta R_{\rm bulk},\delta R_\Gamma)
  =(A\delta x-B\delta V,;G\delta x-K_V\delta V)
  \]
  without exposing surface matrices or factors.

  - Update the authoritative design documents with the concrete vector conventions, supported solver path, and final Stage H interfaces.

  ## Interfaces

  ReconstructedFaultCondensedSystem<dim> will expose:
  ```
  using FaultVector = std::vector<std::vector<double>>;

  const ReconstructedFaultSurfaceResidual &
  linearize(const LinearAlgebra::BlockVector &physical_bulk_state,
            const FaultVector &slip_rate);

  void vmult(LinearAlgebra::BlockVector &result,
             const LinearAlgebra::BlockVector &solver_bulk_direction) const;

  void build_condensed_rhs(
    const LinearAlgebra::BlockVector &bulk_newton_rhs,
    LinearAlgebra::BlockVector &condensed_rhs) const;

  void recover_slip_rate_increment(
    const LinearAlgebra::BlockVector &solver_bulk_increment,
    FaultVector &slip_rate_increment) const;

  void apply_uncondensed_jacobian(
    const LinearAlgebra::BlockVector &solver_bulk_direction,
    const FaultVector &slip_rate_direction,
    LinearAlgebra::BlockVector &bulk_result,
    FaultVector &surface_result) const;
  ```
  `bulk_newton_rhs` is explicitly \(-R_{\rm bulk}\) in ASPECT solver-row units. All output arguments have overwrite semantics. Recovery returns a candidate \(\delta V\) and does not modify manager state.

  ## Tests

  - Extend the reconstructed-fault coupling fixture with a single non-committing coupled residual evaluator and compare centered differences against the uncondensed block action for velocity-only, pressure-only, slip-only, and mixed directions.
  - Sweep dimensionally scaled finite-difference steps and require the second-order truncation regime followed by roundoff saturation.
  - Test dynamic and adiabatic pressure modes, including the actual non-unit ASPECT pressure scaling conversion.
  - Verify condensed actions independently against explicit \(A-BK_V^{-1}G\) composition.
  - Verify the condensed RHS sign against \(-R_{\rm bulk}+BK_V^{-1}R_\Gamma\).
  - Verify recovered \(\delta V\) satisfies the uncondensed surface row.
  - Retain nonzero cohesive, normalization-profile, and Maxwell-history contributions in the fixtures.
  - Exercise both one-rank and two-rank cases and verify replicated surface results.
  - Add unsupported-configuration tests for direct Stokes, block GMG/matrix-free, melt transport, and 3-D construction.
  - Re-run Stage F/G focused tests, all unit tests, standard and Voro-enabled builds with -j4, and git diff --check. Do not run the complete ASPECT integration suite.

  ## Assumptions and Stage Boundary

  - The assembled Stokes matrix is already expressed for the scaled solver pressure and remains unchanged for the lifetime of one condensed-system object.
  - Geometry, phase field, temperature, compositions, \(I_h\), cohesive state, Maxwell history, friction state, and the Stage-G \(B\) coefficients remain frozen during one linearization.
  - Stage H does not register the fault assembler in the production Stokes assembly, initialize \(V\) or \(\Theta\), select an active set, restrict (K_V), perform a line search, or commit/rollback any state. All lower-bound and nonlinear lifecycle behavior remains exclusively Stage I.
