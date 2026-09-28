  # Stage G — Reconstructed-Fault Stokes Coupling

  ## Summary

  Implement the complete fault-induced bulk residual, the frozen-linearization (B) action, and the particle-based (G) action. Geometry and constitutive linearization data
  will have distinct cache lifetimes.

  Stage G will expose independently testable actions but will not register the assembler in the production Stokes solve or implement condensation, Newton lifecycle, or
  state commit.

  ## Constitutive Boundary

  Add one narrow pointwise bulk-coupling operation directly to PhaseFieldFault:

  struct ReconstructedFaultBulkPointInputs
  {
    unsigned int fault_index;
    unsigned int segment_index;
    double xi;
    double phase_field;
    double previous_phase_field;
    double temperature;
    std::vector<double> bulk_material_fractions;
  };

  struct ReconstructedFaultBulkPointResponse
  {
    double kappa;
    double localization_factor;
    double history_correction;
  };

  ReconstructedFaultBulkPointResponse
  evaluate_reconstructed_fault_bulk_point(
    const ReconstructedFaultBulkPointInputs &) const;

  The response supplies the scalars needed to form

  [
  2\kappa\chi\boldsymbol S,
  \qquad
  2\kappa\upsilon^{\rm hist}\boldsymbol S.
  ]

  Introduce one private localization/cohesive-input helper that evaluates the surface mixture, (I_h), (h), and

  [
  \chi=\frac{h}{I_h}.
  ]

  Both the existing Stage-F surface point response and the new Stage-G bulk-QP response must call this helper. Neither path may independently reimplement surface-mixture
  interpolation, phase-field handling, (g>0) diagnostics, (h), (I_h), or (\chi).

  The bulk material fractions and temperature remain QP-local for (\kappa). The degradation function and (I_h) continue to use the profile-uniform surface material
  mixture.

  ## Geometry and Linearization Caches

  ### Geometry cache

  Add a manager-owned Stokes-QP geometry cache containing, for every locally owned cell/QP:

  - active/inactive status;
  - fault and segment indices;
  - (\xi), (N_0), and (N_1);
  - signed normal distance;
  - tangent (\boldsymbol s) and normal (\boldsymbol n);
  - physical QP position and fault-geometry version.

  Key entries by CellId, with direct QP indexing inside each cell. It remains valid until fault geometry, mesh topology/partitioning, mapping geometry, or the production
  Stokes quadrature changes.

  Build it with the exact introspection.quadratures.velocities object and the production mapping used to construct Scratch::StokesSystem.

  Store a strict quadrature signature consisting of the ordered reference points and weights. In debug mode, every cache consumer must verify:

  - the cached signature equals the assembler’s current velocity quadrature;
  - the cached QP count equals FEValues::n_quadrature_points;
  - cached entry (q) corresponds to FEValues point (q) in the same order.

  A mismatch is an internal invariant failure, not a cache miss repaired in the hot loop.

  ### (B)-linearization cache

  Assemblers::ReconstructedFaultStokes owns a separate cache containing

  [
  \boldsymbol C_q=2\kappa_q\chi_q\boldsymbol S_q
  ]

  at every active associated QP.

  Expose:

  void linearize_B(
    const LinearAlgebra::BlockVector &bulk_linearization_point);

  linearize_B() evaluates the constitutive coefficient once for the current nonlinear linearization. It constructs a complete candidate cache and publishes it only after
  successful evaluation of all local entries.

  The cache records the geometry-cache generation it depends on. apply_B() rejects an absent or stale linearization. It performs no material-model evaluation,
  reconstruction, point search, or MPI communication.

  Independent residual evaluations do not modify or invalidate the published (B) cache. A line-search residual evaluation therefore cannot accidentally replace the Krylov
  operator’s coefficients.

  ## Bulk Residual and (B)

  Add Assemblers::ReconstructedFaultStokes<dim> as a genuine cell/QP Stokes assembler.

  Expose:

  void evaluate_slip_dependent_bulk_residual(
    const LinearAlgebra::BlockVector &bulk_state,
    const FaultVector &slip_rate,
    LinearAlgebra::BlockVector &result) const;

  void apply_B(
    const FaultVector &fault_direction,
    LinearAlgebra::BlockVector &result) const;

  The standalone operations have overwrite semantics:

  - evaluate_slip_dependent_bulk_residual() sets result = 0 internally, then assembles the complete non-committing fault contribution
    [
    R_{\rm bulk}^{\rm fault}
    -\int_\Omega
    2\kappa
    \left(\chi V_\Gamma+\upsilon^{\rm hist}\right)
    \boldsymbol S:\dot{\boldsymbol\epsilon}(\boldsymbol\psi_i),d\Omega;
    ]

  - apply_B() sets result = 0 internally, then uses the frozen linearization cache to assemble
    [
    (B,\delta V)_i
    \int_\Omega
    \boldsymbol C_q,\delta V_\Gamma:
    \dot{\boldsymbol\epsilon}(\boldsymbol\psi_i),d\Omega.
    ]

  Both operations apply constraints, compress the result, and leave pressure and non-Stokes rows zero.

  The ordinary assembler execute() remains additive: it must not clear CopyData. Using the manager’s active current/trial (V), it adds

  [
  -R_{\rm bulk}^{\rm fault}

  \int_\Omega
  2\kappa\left(\chi V_\Gamma+\upsilon^{\rm hist}\right)
  \boldsymbol S:\dot{\boldsymbol\epsilon}(\boldsymbol\psi_i),d\Omega
  ]

  to the Stokes right-hand side.

  Keep shared cell-integration logic file-local. Do not register this assembler in set_stokes_assemblers() during Stage G.

  ## Particle-Based (G)

  Extend ReconstructedFaultSurfaceSystem with:

  void apply_G(
    const LinearAlgebra::BlockVector &physical_bulk_direction,
    FaultVector &result) const;

  A successful linearize_surface_system() stores the immutable local particle coefficients needed by (G): Q1 association and weight, (\kappa), (\mu), (\boldsymbol S),
  (\boldsymbol N), and pressure mode.

  apply_G() requires that published linearization, overwrites result, evaluates the physical-field direction at the cached particle points, performs the fault-sized MPI
  sum, and returns identical replicated vectors.

  For dynamic pressure:

  [
  G_{\rm phys}\delta x_{\rm phys}

  2\kappa(\boldsymbol S+\mu\boldsymbol N):
  \delta\dot{\boldsymbol\epsilon}
  -\mu,\delta p_{\rm phys}.
  ]

  For adiabatic pressure:

  [
  G_{\rm phys}\delta x_{\rm phys}

  2\kappa\boldsymbol S:
  \delta\dot{\boldsymbol\epsilon}.
  ]

  There is no (\mu\boldsymbol N) or pressure term in adiabatic mode.

  ## Locked Pressure-Scaling Convention

  All explicit bulk_state arguments used for material or residual evaluation contain physical pressure, matching ASPECT’s stored solution and linearization-point vectors.

  The Stage-G standalone apply_G() API also accepts physical pressure directions.

  Define

  [
  s_p=\texttt{get_pressure_scaling()},\qquad
  \delta p_{\rm phys}=s_p,\delta\widehat p,
  ]

  where (\delta\widehat p) is the pressure component of an ASPECT Stokes solver vector.

  Stage H must therefore use

  [
  G_{\rm solver}=G_{\rm phys}D_p,
  \qquad
  D_p=\operatorname{diag}(I_u,s_pI_p).
  ]

  Equivalently, its dynamic-pressure point action is

  [
  G_{\rm solver}\delta\widehat x

  2\kappa(\boldsymbol S+\mu\boldsymbol N):
  \delta\dot{\boldsymbol\epsilon}
  -\mu s_p,\delta\widehat p.
  ]

  The adiabatic-pressure action is unchanged by this conversion. Stage H may implement the conversion by scaling a temporary physical-direction vector or directly scaling
  the pressure term, but it must not reinterpret an ASPECT solver pressure as physical pressure.

  ## Tests

  - Verify the complete standalone bulk residual against an independent FE/QP reference with nonzero (\upsilon^{\rm hist}). Check the (V)-dependent and history
    contributions separately and together.

  - Verify
    [
    D_VR_{\rm bulk}^{\rm fault},\delta V=-B,\delta V
    ]
    by centered differences of the complete residual. The frozen history contribution must cancel from the derivative without being absent from the absolute residual.

  - Verify overwrite behavior by pre-filling standalone result vectors with nonzero values before calling the residual evaluator and apply_B().
  - Verify additive execute() behavior by pre-filling local CopyData and confirming that existing entries are retained.
  - Verify that repeated apply_B() calls:
      - produce identical results;
      - do not call the material model;
      - do not rebuild either cache.

  - Verify that changing an explicit bulk state affects independent residual evaluation but not an already published (B) cache. Re-running linearize_B() must then update
    the action.

  - Verify geometry-cache reuse and invalidation counters.
  - Add a debug-only mismatch test for quadrature count/order and verify the internal invariant diagnostic.
  - Verify (G) by centered differences of the Stage-F surface residual for velocity-only, pressure-only, and mixed directions in both pressure modes.
  - Use a deliberately non-unit pressure scaling factor to verify
    [
    G_{\rm solver}(\delta u,\delta\widehat p)
    G_{\rm phys}(\delta u,s_p\delta\widehat p).
    ]

  - Verify replicated (G) results on two MPI ranks and consistent distributed (B) norms/owned entries.
  - Build build-main and Voro-enabled build-pf-cpdi with -j4; run the focused Stage-F/Stage-G tests and git diff --check. Do not run the complete ASPECT suite.

  ## Boundaries

  - Stage G remains 2-D with non-overlapping fault influence regions.
  - Geometry, phase field, temperature, compositions, (I_h), cohesive history, Maxwell history, and friction state are frozen within one linearization.
  - The (B) cache contains only the derivative coefficient (2\kappa\chi\boldsymbol S); the affine history contribution is evaluated for the residual/RHS and never enters
    (B).

  - Stage G does not implement the condensed operator, condensed right-hand side, (\delta V) recovery, active sets, line search, nonlinear commit/rollback, or history
    updates.
