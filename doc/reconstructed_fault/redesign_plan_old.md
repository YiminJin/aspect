

  # Reconstructed-Fault Surface RSF Architectural Migration

  ## 1. Mathematical workflow

  The restarted 2-D formulation will use one Q1 surface state on each ordered reconstructed-fault polyline:

  \[
  \text{distributed bulk }(u,p,\phi,\tau_{k-1})
  \rightarrow
  \text{profile/fault residual}
  \rightarrow
  (V,\Theta,T^{\rm coh},I_h)_\Gamma
  \rightarrow
  \text{diffuse bulk crack strain/stress}.
  \]

  For each timestep:

  1. Associate locally owned particles and relevant Stokes quadrature points with at most one finite segment-normal strip; reject open-tip extensions and overlapping faults.

  2. Project composition/material data needed by the constitutive law to the fault with the existing particle-domain-volume-weighted Q1 projection.

  3. Evaluate
     \[
     I_h(s)=\int_{T_s}\left(\frac{1}{g(\phi)}-1\right)d\zeta
     \]
     from the actual distributed Q1 phase-field solution.

  4. During mechanical Newton iterations, keep \(\Theta=\Theta_{k-1}\), \(\phi\), \(I_h\), geometry, and history fixed. Compute
     \[
     T^{\rm coh}(V)=\frac{\kappa_k}{I_{h,k}}V
     +\beta\frac{I_{h,k-1}}{I_{h,k}}T^{\rm coh}_{k-1}
     \]
     and
     \[
     \bm\tau=\bm\tau^*-2\kappa_k\frac{h_k}{I_{h,k}}V_\Gamma\bm S.
     \]

  5. Assemble the projection-consistent surface residual and its IMPES Jacobian \(K_V=-\partial R_\Gamma/\partial V\), retaining only the fixed-\(\Theta\) direct-effect derivative.

  6. Solve the exact condensed system
     \[
     (A-BK_V^{-1}G)\delta x
     -R_{\rm bulk}+BK_V^{-1}R_\Gamma,
     \]
     then recover
     \[
     \delta V=K_V^{-1}(R_\Gamma+G\delta x).
     \]

  7. Apply a joint line-search step to \(\bm x=(\bm u,p)\) and \(V\), with fraction-to-boundary limiting so \(V\ge V_{\min}\).

  8. After convergence, update \(\Theta_k\), commit \(T^{\rm coh}_k\), reconstruct the diffuse \(\upsilon_k\), update stress and \(H\), and then evolve \(\phi\).

  ## 2. Existing-code migration table

   Existing class/file | Category | Migration
 -------- | -------- | --------
   `ReconstructedFault<dim>` in `include/aspect/reconstructed_fault.h ` | Keep unchanged | Retain the ordered append-only polyline, implicit Q1 segments, generic vertex-property array, and geometry version. Do not introduce a fault Triangulation or ParticleHandler.
   Geometry reconstruction in `source/reconstructed_fault.cc` | Keep unchanged | Retain prescribed-fault parsing, reference resampling, initial Q1 phase-field ridge reconstruction, and replicated geometry.
   `ReconstructedFaultManager<dim>` | Reuse with modification | Retain ownership, property registry, half-width metadata, and replicated faults. Expose a material-independent immutable particle-association view; add checkpoint support and lifecycle hooks needed by a separate RSF handler.
   `project_to_normal_profiles()` and projection cache | Reuse with modification | Keep finite normal strips, (K=1), open-tip exclusion, closest incident segment, overlap errors, particle IDs/positions/volumes, MPI-reduced mass matrix, and tridiagonal factors. Generalize access so RSF assembly can reuse associations without treating current stress as a projected frozen property.
   `project_particle_properties()` | Keep for static/projected fields | Use for composition fractions, constitutive parameters, and diagnostics. Do not use it to pre-project current trial stress because that would remove its \(u,p\) dependence.
   `ParticleDomainHandler` / `ParticleDomainAccessor` | Keep unchanged | Continue using `get_particle_domain(...).volume()` as \(m_p\). Do not substitute CPDI shape functions for fault Q1 functions.
   `PhaseFieldHandler<dim>` | Reuse with modification | Add narrow const access to the existing length scale and direct distributed Q1 point-evaluation support. Continue using the existing phase-field range and degradation implementation; do not use the core-phase-field extension.
   `RateStateFriction<dim>` | Reuse with modification | Keep it as the sole owner of RSF parameters, friction, fixed-state derivative, state update, timestep limit, \(V_0,V_{\min},D_c\). Add an inverse-friction helper needed to initialize \(\Theta\) for a prescribed target \(\mu_0\), using the same parsed regularized/logarithmic law.
   `PhaseFieldRSF<dim>` | Reuse with substantial modification | Retain EOS, existing material parameters, phase-field model API, and the embedded RSF helper. Replace its currently incomplete mechanical role with narrow reconstructed-fault constitutive accessors for \(\eta,G,\beta,\kappa,\eta^d,\mu_0\), trial stress, and stress-history updates.
   `ElasticOutputs` and ordinary Stokes viscoelastic assembly | Reuse with modification | Reuse the existing viscosity/history-force mechanism to form \(A\) and the bulk trial state. Add the current slip/history stress contribution through a dedicated reconstructed-fault assembler, not a local nonlinear tangent modulus.
   `ImplicitConstitutiveOutputs::nonlinear_tangent_moduli` |          Keep for other models; bypass here | It cannot represent the nonlocal \(BK_V^{-1}G\) operator and must not be used for reconstructed-fault RSF coupling.
   `source/simulator/assembly.cc` and Stokes assemblers | Reuse with modification | Add fault-to-bulk residual assembly and the QP cache ntegration path while preserving ordinary ASPECT Stokes assembly for \(A\).
   internal::StokesBlock and assembled iterative solver in ` source/simulator/solver.cc` | Replace wrapper, retain preconditioner | Replace the matrix-only Krylov operator with a wrapper applying \(A-BK_V^{-1}G\). Continue using the existing Stokes block preconditioner as an approximation to \(A_{\rm eff}\).
   Newton line search and residual handling | Reuse with modification | Trial-update both bulk fields and \(V\), reassemble both residual blocks, use a normalized two-block merit function, and require both blocks to converge independently.
   Direct and matrix-free GMG Stokes solvers | Defer/guard | Reject reconstructed-fault coupled RSF with an early parameter-consistency error in the first implementation. Extend only after the assembled iterative implementation and Jacobian verification pass.
   Reconstructed-fault VTU postprocessor | Keep unchanged | Registered RSF properties will already appear as point data. Add only optional diagnostics if later needed.
   Existing reconstructed-fault unit tests | Reuse and extend | Preserve all geometry, projection, MPI, and output tests; add the surface and coupled tests below.
   Core-phase-field extension and diffuse particle RSF layout | Remove/deprecate from the new path | Do not call `extend_core_phase_field`, interpolate \(d\mu/dV\) to bulk quadrature points, or introduce particle copies of \(V,\Theta,T^{\rm coh}\). The referenced old `phase_field_rsf` particle-property file and `perform_return_mapping()` routine are absent from the current tree and therefore cannot be reused.

  ## 3. Proposed interfaces and data flow

  ### Surface ownership

  Add a material-specific `ReconstructedFaultRSFHandler<dim>`, owned by Simulator only when reconstructed faults and the compatible RSF material model are active. Geometry remains in `ReconstructedFaultManager`.

  The handler registers generic nodal properties for:

  - committed \(V_k\);
  - committed \(\Theta_k\);
  - committed \(T^{\rm coh}_k\);
  - current \(I_{h,k}\);
  - previous \(T^{\rm coh}_{k-1}\);
  - previous \(I_{h,k-1}\);
  - optional projected composition/material components used by the law.

  Newton trial values of \(V\), the pending \(\Theta_k\), and temporary cohesive traction remain handler-owned working vectors until the timestep is committed. Geometric caches contain no nonlinear state.

  Core lifecycle interface:
```
  prepare_timestep();
  assemble_fault_linearization(const LinearAlgebra::BlockVector &x);
  add_current_slip_bulk_residual(LinearAlgebra::BlockVector &rhs) const;

  apply_G(const LinearAlgebra::BlockVector &delta_x,
          std::vector<double> &fault_result) const;

  apply_B(const std::vector<double> &delta_V,
          LinearAlgebra::BlockVector &bulk_result) const;

  apply_K_inverse(const std::vector<double> &src,
                  std::vector<double> &dst) const;

  recover_fault_update(const LinearAlgebra::BlockVector &delta_x);
  set_trial_step(const double alpha);
  commit_mechanical_state();
  commit_timestep_history();
  invalidate_*_cache();
```
  `assemble_fault_linearization()` produces replicated:
```
  struct FaultLinearization
  {
    std::vector<double> residual;       // R_Gamma
    std::vector<double> K_diagonal;
    std::vector<double> K_off_diagonal;
    TridiagonalFactors K_factors;
  };
```
  ### Constitutive material access

  Add a narrow output from the compatible material model:
```
  struct ReconstructedFaultConstitutiveData
  {
    double eta;
    double shear_modulus;
    double beta;
    double kappa;              // eta * (1-beta)
    double radiation_damping;
    double reference_friction;
  };
```
  The material model computes this from existing parameters, composition, temperature, and timestep. No new viscosity, shear modulus, RSF, damping, or phase-field length parameters are introduced.

  The RSF handler uses fault-projected composition fractions for the RSF coefficients, the profile-wise cohesive parameters, and \(g(\phi)\) during \(I_h\) integration. The mixture is held constant along each normal profile, as selected during planning.

  ### Trial stress and residual

  At locally owned particle points, evaluate current FE velocity gradients and pressure using the particle’s surrounding bulk cell and the existing point-evaluation infrastructure:

  \[
  \bm\tau^*=2\kappa_k\dot{\bm\epsilon}(\bm u)+\beta_k\bm\tau_{k-1},
  \]

  \[
  \upsilon^{\rm hist} =
  \frac{\beta_k T^{\rm coh}_{k-1}}{\kappa_k}
  \left[
  h_k\frac{I_{h,k-1}}{I_{h,k}}-h_{k-1}
  \right],
  \]

  \[
  \bm\tau=\bm\tau^*-2\kappa_k\upsilon^{\rm hist}\bm S,
  \quad
  t=\bm\tau^*:S,
  \quad
  \sigma_n=p-\bm\tau^*:\bm N.
  \]

  Then assemble only from locally owned particles:

  \[
  R_i^\Gamma

  \sum_p m_pN_i^p
  \left[
  t_p^*
  -\kappa_p\chi_pV_\Gamma(p)
  -T^{\rm coh}\Gamma(V;p)
  -\mu(V\Gamma,\Theta_{k-1,\Gamma})\sigma_{n,p}
  -\eta_p^dV_\Gamma(p)
  \right].
  \]

  \(K_V\) uses the exact negative derivative of this discrete residual at fixed (\Theta), including mechanical, cohesive, radiation, and fixed-state RSF terms. It remains tridiagonal for the ordered Q1/single-segment association.

  G is not stored as a projected stress. apply_G() reevaluates the linear bulk perturbation at cached particle points and assembles
  \[
  2\kappa(S+\mu N):\delta\dot\epsilon-\mu\delta p
  \]
  into a fault-sized vector followed by MPI reduction.

  B is not a local tangent. `apply_B()` interpolates \(\delta V\) at cached locally owned Stokes quadrature points and assembles the bulk weak-form response of
  \[
  -2\kappa\chi S,\delta V_\Gamma.
  \]
  A remains the existing assembled viscoelastic Stokes Jacobian with current fault slip fixed.

  ### Exact condensed solver

  Add an iterative operator whose `vmult()` performs:

  1. existing Stokes matrix action \(A,\delta x\);
  2. `apply_G(delta_x)`;
  3. replicated tridiagonal solve \(z=K_V^{-1}G\delta x\);
  4. `apply_B(z)`;
  5. return \(A\delta x-Bz\).

  Before solving, add \(BK_V^{-1}R_\Gamma\) to the existing Newton right-hand side. After solving, recover \(\delta V=K_V^{-1}(R_\Gamma+G\delta x)\).

  The first implementation supports only the assembled iterative AMG solver. Direct and matrix-free GMG modes fail early with a precise unsupported-configuration message.

  ### Initialization and nonlinear control

  For an initial fault, set \(T^{\rm coh}_{k-1}=0\) and \(I_{h,k-1}=I_{h,k}\). Initialize the nodal \(V\) by solving the discrete surface force-balance system with \(\mu=\mu_0\), including mechanical-slip, cohesive, and radiation coefficients. Invert the existing RSF law at each nodal projected material mixture to obtain positive \(\Theta\) satisfying \(\mu(V,\Theta)=\mu_0\). Reject a nonpositive force-balance solution or a material law for which no unique positive \(\Theta\) exists.

  Newton line search uses:

  \[
  \alpha
  \le
  0.99\min_{\delta V_i<0}
  \frac{V_i-V_{\min}}{-\delta V_i},
  \]

  followed by ordinary backtracking. Bulk and fault residuals are normalized by their first-iteration block norms for the merit function, and each block must independently meet the configured nonlinear relative tolerance.

  ## 4. Numerical evaluation of \(I_h\)

  ### Fault and profile quadrature

  1. Use `QGauss<1>(3)` on every reconstructed-fault segment. Segment tangents and normals are constant and unambiguous at these points.
  2. At each segment quadrature point, interpolate the fault-projected composition mixture and use it for every \(g(\phi)\) evaluation on that profile.
  3. Integrate independently in \(+\hat{\bm n}\) and \(-\hat{\bm n}\), starting at the fault point.
  4. Use initial outward panels of width \(\ell/2\), where \(\ell\) is the existing phase-field length scale.
  5. For every active panel, evaluate both 4-point and 8-point Gauss rules. Accept or bisect the panel according to
     \[
     |I_8-I_4| \le 10^{-8}\max(|I_8|,\ell).
     \]

  6. A profile side reaches converged support after three consecutive accepted outer panels have monotonically nonincreasing contributions and their sum is at most
     \[
     10^{-8}\max(I_{\rm accumulated},\ell).
     \]
     The ordinary phase-field activation threshold is never used as an integration boundary.

  7. If a proposed panel leaves the physical domain, use batched point-found probes and collective bisection to locate the first boundary intersection, integrate the in-domain partial panel, and terminate that side.

  8. A configurable maximum adaptive-round count is a failure guard, not a physical support cutoff; exceeding it throws with the fault, segment, side, accumulated integral, and last-panel diagnostics.

  The numerical tolerances, profile quadrature orders, required tail-panel count, and maximum rounds live under an \(I_h\) integration subsection. They are numerical controls and introduce no duplicate physical length or phase-field threshold.

  ### Batched distributed evaluation

  Rank zero maintains the replicated profile metadata and builds one packed point list per adaptive round containing every requested Gauss or boundary-probe point for every still-active profile side. Other ranks provide empty request lists but participate in the same collective `Utilities::MPI::RemotePointEvaluation` call against the distributed phase-field `DoFHandler`.

  The owning bulk ranks evaluate the actual Q1 phase-field FE solution. Results return to rank zero, which applies the fixed fault-projected composition mixture, computes \(h=1/g-1\), updates all adaptive profiles, and broadcasts convergence metadata needed for the next collective round.

  After obtaining \(I_h\) at segment quadrature points, rank zero assembles
  \[
  (M_\Gamma){ij}=\int{\Gamma_h}N_iN_jds,
  \qquad
  b_i=\int_{\Gamma_h}N_iI_h^{\rm quad}ds,
  \]
  solves the replicated tridiagonal Q1 mass system \(M_\Gamma I_h=b\), and broadcasts the nodal result and diagnostics. Every rank verifies finite positive values and a checksum/maximum-difference consistency check.

  ## 5. MPI, caches, and lifecycle

  ### MPI semantics

  - Geometry, registered properties, working surface vectors, \(R_\Gamma\), \(K_V\), and its factors are replicated.
  - Only locally owned particles contribute to particle residuals, \(K_V\), G, and generic projections; ghost particles never contribute.
  - Fault-sized local residuals and tridiagonal entries are packed into collective sums. No particle positions or stresses are gathered.
  - Each condensed Krylov `vmult()` imports the locally relevant bulk values needed at particle points, performs a fault-sized \(G\) reduction, solves the identical replicated \(K_V\), and applies B locally.
  - \(B\) contributions use normal distributed-vector compression/addition semantics.
  - Rank-zero coordination is used only for the small adaptive \(I_h\) request metadata and mass solve; distributed FE data remain distributed.
  - One-rank and multi-rank results are compared within floating-point reduction tolerance, not required to be bitwise identical.

  ### Cache construction and invalidation

  - Geometry cache: segment tangents, normals, lengths, shape information, spatial search; invalidate on fault geometry version change.
  - Particle association cache: particle ID, position, domain volume, fault/segment, \(\xi\), signed distance; invalidate on particle advection/migration, ownership/order/volume change, AMR/repartitioning, half-width metadata change, or geometry change.
  - QP association cache: locally owned cell/QP identity, fault/segment, (\xi), signed distance, Q1 values, \(\bm S,\bm N\); invalidate on AMR, repartitioning, quadrature change, or geometry/half-width change.
  - \(I_h\) sampling data: invalidate after any phase-field solution change, projected-composition change, bulk mesh change, geometry change, or integration-setting change.
  - Surface mass matrix/factorization: reusable while fault geometry is unchanged.
  - \(K_V\), \(R_\Gamma\), trial stress, \(\mu\), \(d\mu/dV\), and condensed right-hand-side data: rebuild every Newton linearization.
  - \(B\) and \(G\) geometric locations are cached, but their current constitutive coefficients are refreshed with the Newton/timestep state.
  - Particle/fault/QP caches never store current stress, \(V\), \(\Theta\), \(T^{\rm coh}\), or friction coefficients.

  At timestep commit, move current \(I_h,T^{\rm coh},\Theta,V\) into committed properties atomically. On rejected line-search candidates or failed solves, restore the handler’s saved working vectors. Checkpoint/restart serializes committed geometry, property schema/values, previous history, and geometry version; all caches and matrix factors are rebuilt after restart.

  ## 6. Staged implementation sequence

  1. __Constitutive interfaces and surface storage__
      Add the RSF handler, register committed properties, expose existing material/phase-field quantities through narrow accessors, and implement force-balance/\(\mu_0\) initialization with unit tests.

  2. __Adaptive distributed \(I_h\)__
     Implement segment quadrature, batched remote point evaluation, independent two-sided adaptive support, Q1 surface mass projection, diagnostics, and analytical/MPI tests. Stop for review before mechanical coupling.

  3. __Shared geometric caches__
     Refactor the existing particle association into an immutable reusable view and add the locally owned Stokes-QP association cache. Preserve all current projection behavior and tests.

  4. __Surface residual and \(K_V\)__
     Implement trial/effective stress evaluation, cohesive working update, lagged-\(\Theta\) RSF residual, fixed-state derivative, tridiagonal assembly/factorization, and scalar/local-limit tests.

  5. __Operator actions \(G\) and \(B\)__
     Implement bulk-to-fault and fault-to-bulk actions independently, with hand-assembled small-problem and adjoint/sign diagnostics. Do not alter the Stokes solver yet.

  6. __Exact condensed assembled solver__
     Add \(A-BK_V^{-1}G\), the condensed RHS, \(\delta V\) recovery, AMG-first configuration guards, and linear operator tests.

  7. __Coupled Newton lifecycle__
     Integrate (V) into initialization, residual reporting, fraction-to-boundary line search, rollback, separate block convergence, and failure diagnostics.

  8. __History commit and phase-field feedback__
     Commit \(\Theta,T^{\rm coh},I_h\), reconstruct exact \(\upsilon\), update bulk stress/history, and evaluate \(H\) by interpolating surface cohesive history rather thanvduplicating it on particles.

  9. __Restart/output and fixed-fault benchmarks__
     Serialize committed properties, rebuild caches after restart/AMR, expose useful surface diagnostics in VTU, and run the fixed-phase-field benchmark progression.

  10. __Propagation integration only after review__
     Add initialization of appended fault vertices and geometry/cache rebuilding after the coupled fixed-geometry implementation passes all Jacobian and benchmark tests.

  Each stage is a separate reviewable commit and must leave existing reconstructed-fault geometry/projection tests passing.

  ## 7. Verification plan

  ### Geometry and projection

  - Constant and exact Q1 linear reproduction.
  - Unequal particle-domain volumes.
  - Invariance with normal distance for \(K=1\).
  - Open-tip exclusion and internal-bend segment selection.
  - Explicit multi-fault overlap error.
  - No ghost-particle double counting.
  - One-rank versus two-rank equivalence.

  ### \(I_h\)

  - Analytical stationary-profile integral.
  - 4/8/higher profile-quadrature refinement.
  - Tail-tolerance/support-truncation convergence.
  - Bulk mesh-refinement convergence.
  - Q1 surface projection convergence.
  - One-rank versus two-rank equivalence.
  - Near-domain-boundary profile clipping.
  - Discrete verification of \(\int\upsilon d\zeta=V\).

  ### Constitutive surface system

  - Fixed-profile cancellation of \(\upsilon^{\rm hist}\).
  - Evolving-profile cohesive normalization.
  - Residual sign conventions.
  - Mechanical, cohesive, radiation, and RSF terms separately.
  - Exact regularized fixed-\(\Theta\) derivative.
  - Force-balance/\(\mu_0\) initialization.
  - Agreement with the old scalar local limit where its assumptions apply.

  ### Mandatory finite-difference Jacobian verification

  For a small fixed-geometry model, expose a residual evaluator returning both \((R_{\rm bulk},R_\Gamma)\) without committing history. Compare the block action

  \[
  \begin{bmatrix}
  A & -B \\
  G & -K_V
  \end{bmatrix}
  \begin{bmatrix}
  \delta x \\
  \delta V
  \end{bmatrix}
  \]

  against centered finite differences of the same discrete residual over a decreasing sequence of \(\varepsilon\). Test:

  - velocity-only perturbations;
  - pressure-only perturbations;
  - fault-slip-only perturbations;
  - mixed bulk/fault perturbations;
  - single-rank and two-rank execution;
  - nonzero cohesive history and evolving-profile history;
  - regularized friction away from and near \(V_{\min}\), without crossing the bound.

  Also compare the condensed action directly with \(A-BK_V^{-1}G\) and verify recovered \(\delta V\) satisfies the uncondensed second block row. Coupled BP3-style benchmarks do not begin until these tests demonstrate the expected first-order finite-difference error followed by roundoff saturation.

  ## 8. Conflicts, assumptions, and deferred ambiguities

  - specification.tex describes a runtime codimension-one deal.II triangulation, while current_design.md and the current source require a simple ordered application-owned polyline. The current design explicitly overrides the older specification, and the restarted RSF design also assumes an ordered Q1 polyline; the plan therefore keeps the polyline.

  - The restart document mentions an old perform_return_mapping() and a source/particle/property/phase_field_rsf.cc; neither exists in the current tree. No reuse of those routines can be planned.

  - The current `PhaseFieldRSF` material model parses RSF/viscoelastic parameters but does not implement the described surface return map, surface history, exact bulk-surface coupling, or complete stress-history path. It must be extended rather than treated as an existing implementation of the new method.

  - The existing generic projection can project static particle properties but cannot project current trial stress for the nonlinear solve without destroying the required (u,p) dependence.

  - ASPECT’s existing nonlinear tangent-modulus output is point-local and cannot express the exact nonlocal Schur complement.

  - The existing Newton line search and residual norm cover only bulk Stokes variables; coupled line-search and convergence behavior require explicit modification.

  - pf_rsf.tex has a different friction/sign presentation from the restarted surface design. The restarted residual
    \[
    F=t-T^{\rm coh}-\mu\sigma_n-\eta^dV
    \]
    is authoritative; pf_rsf.tex is used only to confirm \(\beta=\exp(-\Delta tG/\eta)\) and \(\kappa=\eta(1-\beta)\).

  - The first coupled implementation is 2-D, one open ordered fault per influence point, nonnegative \(V\), lagged \(\Theta_{k-1}\), \(K=1\), fixed geometry during Newton, and assembled iterative AMG only.

  - Initial-fault \(T^{\rm coh}_{k-1}=0\) and \(I_{h,k-1}=I_{h,k}\). A state with no positive force-balance \(V\) or no unique positive inverse-friction \(\Theta\) is rejected explicitly.

  - Initialization of cohesive/history variables on vertices appended by later propagation is not specified by the current mathematics. Propagation coupling remains deferred after the fixed-geometry benchmarks and requires a reviewed rule before Stage 10 is implemented.
