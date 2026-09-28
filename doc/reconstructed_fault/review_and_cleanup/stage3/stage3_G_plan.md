  # Stage G — Reconstructed-Fault Stokes Coupling

  ## Summary

  Implement the two off-diagonal weak-form couplings without adding condensation or nonlinear lifecycle:

  \[
  R_{\rm bulk}^{\rm fault}(x,V)=-B(x)V,
  \qquad
  \frac{\partial R_\Gamma}{\partial x}\delta x=G\delta x.
  \]

  `Assemblers::ReconstructedFaultStokes` will own the cell/QP bulk contribution and \(B\). `ReconstructedFaultSurfaceSystem` will own the particle-based \(G\) action. `PhaseFieldFault` will expose only the additional pointwise localization quantity needed by the bulk assembler.

  Stage G will not register or activate a coupled Stokes solve. Stage H will consume these actions for condensation, and Stage I will provide the production nonlinear
  state lifecycle.

  ## Implementation Changes

  ### QP association and constitutive data

  - Add a manager-owned Stokes-QP association cache containing, for each relevant local QP:
      - active flag, fault and segment indices;
      - \(\xi\) and both Q1 shape values;
      - signed normal distance;
      - tangent, normal, and geometry version.

  - Build it before cell traversal using the production Stokes velocity quadrature and mapping. Store entries by `CellId`, with one lookup per cell and direct indexing within the cell.

  - Reuse cached entries without geometric searches or MPI communication during residual and \(B\) applications.
  - Invalidate the cache after fault-geometry changes, refinement/repartitioning, restart, and mesh deformation by connecting to existing ASPECT/deal.II signals. Do not add a new `Simulator` lifecycle signal.

  - Expose cache preparation, read-only per-cell association access, and diagnostics containing the locally relevant QP count and rebuild counter.
  - Add one narrow pointwise method directly to PhaseFieldFault:
    ```
    double reconstructed_fault_localization_factor(
      unsigned int fault_index,
      unsigned int segment_index,
      double xi,
      double phase_field) const;
      ```
    `PhaseFieldFault` computes \(\chi=h/I_h\) with the same Q1 surface mixture, bounded-negative phase-field rule, (g>0) singularity handling, and current \(I_h\) used by the Stage-F point response.

  - Obtain \(\kappa\) from the ordinary PhaseFieldFault viscosity material output at each bulk QP. Thus (\kappa) uses the local bulk material state, while \(h\) and \(I_h\) use the profile-uniform surface mixture.

  ### Bulk residual and \(B\)

  - Add `Assemblers::ReconstructedFaultStokes<dim>` as a genuine Assemblers::Interface using ordinary Stokes `Scratch`/`CopyData`.
  - Its local kernel evaluates
    \[
    (Bz)_i =
    \int_\Omega2\kappa\chi z_\Gamma
    \bm S:\dot{\bm\epsilon}(\bm\psi_i)\mathrm d\Omega,
    \qquad
    z_\Gamma=N_0z_j+N_1z_{j+1}.
    \]
    Only velocity test-function rows receive contributions.

  - Add two explicit solver-facing operations to the assembler:
    ```
    void evaluate_slip_dependent_bulk_residual(
      const LinearAlgebra::BlockVector &bulk_state,
      const FaultVector &slip_rate,
      LinearAlgebra::BlockVector &residual) const;

    void apply_B(
      const LinearAlgebra::BlockVector &bulk_state,
      const FaultVector &fault_direction,
      LinearAlgebra::BlockVector &result) const;
    ```

  - Both operations overwrite a caller-initialized full-system distributed vector, assemble with current constraints, and compress it. Non-Stokes and pressure entries remain zero.

  - `evaluate_slip_dependent_bulk_residual()` returns \(-B(x)V\); `apply_B()` returns \(+B(x)\delta V\), matching the block row \(A\delta x-B\delta V\).
  - The virtual `execute()` adds \(+BV\) to the conventional ASPECT Stokes right-hand side, which represents the same residual contribution \(-BV\). Do not add the assembler to `set_stokes_assemblers()` during Stage G because production initialization/trial selection of \(V\) belongs to Stage I.

  - Keep the assembly implementation and `WorkStream` support types file-local where possible. Do not add generic `Scratch`/`CopyData` extensions unless the existing Stokes types prove insufficient.

  ### Particle-based \(G\)

  - Extend ReconstructedFaultSurfaceSystem with:
    ```
    void apply_G(
      const LinearAlgebra::BlockVector &bulk_direction,
      FaultVector &result) const;
    ```

  - Require a successful `linearize_surface_system()` first. The input direction uses the full ASPECT field representation with physical pressure units.

  - During linearization, retain the active-particle point-evaluation cache and only the local immutable coefficients needed by \(G\): Q1 association/weight, \(\kappa\), \(\mu\), \(\bm S\), \(\bm N\), and pressure mode.

  - Apply the exact pointwise action:
    \[
    G\delta x=
    2\kappa(\bm S+\mu\bm N):
    \delta\dot{\bm\epsilon}
    -\mu~\delta p
    \]
    for dynamic pressure, and
    \[
    G\delta x=2\kappa\bm S:\delta\dot{\bm\epsilon}
    \]
    for adiabatic pressure.

  - Assemble with particle-domain volume and surface Q1 shape weights, MPI-sum the fault-sized vector, and return identical replicated results on every rank.

  - Preserve the Stage-F transactional factorization invariant. Failed or replacement linearizations must also discard their stored (G) coefficients and point-evaluation cache.

  ### Documentation and boundaries

  - Update both authoritative documents with the settled cache representation, public actions, vector/sign conventions, and physical-pressure convention.
  - Do not implement \(A-BK_V^{-1}G\), a condensed right-hand side, \(\delta V\) recovery, active sets, line search, state commit/rollback, or post-convergence Maxwell-history updates.

  - Do not add bulk Maxwell-history or cohesive-history forcing beyond the explicitly specified slip-dependent term \(-BV\).

  ## Test Plan

  - Add common Stage-G integration fixtures using normal production reconstruction semantics and the existing resolved AT1/CZM configurations:
      - dynamic-pressure mode on one MPI rank;
      - adiabatic-pressure mode on two MPI ranks.

  - Verify \(B\) by:
      - centered differences of the independently evaluated slip-dependent bulk residual;
      - an independent FE/QP reference integration of \(2\kappa\chi~\delta V_\Gamma\bm S:\epsilon(\psi_i)\);
      - checking the sign \(D_VR_{\rm bulk}\delta V=-B\delta V\);
      - confirming pressure and non-Stokes rows are zero.

  - Verify \(G\) against centered differences of evaluate_surface_residual\(x,V\) for:
      - velocity-only directions;
      - pressure-only directions;
      - mixed velocity/pressure directions.

  - In dynamic-pressure mode, require the \(-\mu\delta p\) and \(2\kappa\mu\bm N:\delta\dot\epsilon\) terms. In adiabatic mode, require both to be absent.

  - Sweep perturbation sizes to demonstrate accurate centered differences over a resolvable range and the expected cancellation/roundoff limit; do not require second- order truncation from these analytically linear cross-block actions.

  - Verify repeated \(B\) applications reuse the QP cache without increasing its rebuild counter, and repeated applications return identical results.

  - Verify \(G\) is identically replicated across MPI ranks and that \(B\) has consistent global norms/owned entries in the two-rank fixture.
  - Rerun all Stage-F surface-system and friction tests, build both build-main and build-pf-cpdi with -j4, and run git diff --check. Do not run the complete ASPECT suite.

  ## Assumptions

  - Stage F, including the refined `ReconstructedFaultSurfaceSystem`, is the implementation baseline.
  - The initial Stage-G implementation remains 2-D, with open non-overlapping fault influence regions; 3-D and overlap remain deferred.
  - Temperature, phase field, compositions, geometry, \(I_h\), and constitutive history are fixed during each cross-block action.
  - Stage H will convert between its internally pressure-scaled Stokes vectors and the physical-pressure convention of `apply_G()`.
  - Stage G exposes complete, independently testable \(B\) and \(G\) actions but deliberately does not activate them in the ordinary Stokes solve before the coupled solver and \(V\)-initialization lifecycle exist.
