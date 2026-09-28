  # Reconstructed-Fault Surface RSF Architectural Migration

  ## 1. Authority and mathematical workflow

  The documents have distinct authority:

  - pf_rsf.tex is authoritative for the continuum constitutive model, including signs, the time-discrete Maxwell law, cohesive law, radiation damping, and crack-driving-
    force formula.

  - reconstructed_fault_surface_rsf_design.md is authoritative for the surface discretization, bulk–fault coupling, MPI model, and software architecture.
  - current_design.md remains authoritative for the generic reconstructed-fault geometry and currently settled implementation decisions.
  - The source tree remains authoritative for existing APIs and reusable infrastructure. Any implementation-critical conflict between these sources must be reported
    before coding.

  The 2-D formulation uses one Q1 kinematic/constitutive state per reconstructed-fault location:

  \[
  \text{distributed }(u,p,\phi,\tau_{k-1})
  \rightarrow
  \text{profile-collapse surface residual}
  \rightarrow
  (V,\Theta,T^{\rm coh},I_h)_\Gamma
  \rightarrow
  \text{diffuse crack strain and bulk stress}.
  \]

  For one timestep:

  1. Reconstruct or reuse the ordered fault geometry and build generic particle/QP associations.
  2. Project composition/material mixtures to the fault.
  3. Evaluate \(I_{h,k}\) from the actual distributed Q1 phase-field solution.
  4. Freeze \(\phi\), \(I_h\), \(\Theta_{k-1}\), \(T^{\rm coh}_{k-1}\), particle stress history, geometry, and association caches during mechanical Newton.
  5. Use the continuum Maxwell law
     \[
     \beta_k=\exp\left(-\frac{\Delta t_kG}{\eta}\right),
     \qquad
     \kappa_k=\eta(1-\beta_k),
     \]
     \[
     \tau_k^{\rm trial}
     2\kappa_k\dot\epsilon_k+\beta_k\tau_{k-1}.
     \]

  6. Evaluate the surface cohesive law
     \[
     T^{\rm coh}_k(V)
     \frac{\kappa_k}{I_{h,k}}V
     +
     \beta_k\frac{I_{h,k-1}}{I_{h,k}}T^{\rm coh}_{k-1},
     \]
     and the exact diffuse crack-strain distribution
     \[
     \upsilon_k
     \upsilon_k
     \frac{h_k}{I_{h,k}}V_\Gamma+\upsilon_k^{\rm hist}.
     \]

  7. Assemble \(R_\Gamma,K_V,B,G\) and solve
     \[
     (A-BK_V^{-1}G)\delta x-R_{\rm bulk}+BK_V^{-1}R_\Gamma.
     \]

  8. Recover
     \[
     \delta V=K_V^{-1}(R_\Gamma+G\delta x),
     \]
     and line-search \(x=(u,p)\) and \(V\) together.

  9. After mechanical convergence, update \(\Theta_k,T^{\rm coh}_k\), particle stress history, and \(H_k\). No particle copies of \(V,\Theta\), or \(T^{\rm coh}\) are
     introduced.

  ## 2. Ownership and code organization

  ### ReconstructedFaultManager<dim>

  Keep the manager material-independent except for the distinguished fault kinematic field \(V\). It owns:

  - reconstructed geometry and ordered Q1 segments;
  - tangents, normals, segment interpolation, and geometry versions;
  - particle-to-fault and QP-to-fault association caches;
  - generic vertex-property registration/storage;
  - the committed, saved, and trial nodal \(V\) vectors;
  - positivity-preserving trial/update/rollback operations for \(V\);
  - generic particle-to-fault projection and fault-to-point interpolation;
  - MPI-consistent replicated geometry and kinematic data.

  Proposed kinematic interface:
```
  const std::vector<double> &
  get_slip_rate(const unsigned int fault_index) const;

  double
  interpolate_slip_rate(const FaultLocation &location,
                        const SlipRateState state) const;

  void
  initialize_slip_rate(const unsigned int fault_index,
                       const std::vector<double> &values);

  void
  begin_slip_rate_trial();

  void
  set_slip_rate_trial(const std::vector<std::vector<double>> &delta_V,
                      const double step_length);

  void
  accept_slip_rate_trial();

  void
  rollback_slip_rate_trial();
```
  ReconstructedFault<dim> remains the geometry/property container; the manager keeps the parallel \(V\) vectors so the geometry class does not gain material-model state.

  ### MaterialModel::PhaseFieldRSF<dim>

  PhaseFieldRSF owns the constitutive state and algorithms:

  - Maxwell coefficients \(\eta,G,\beta,\kappa\);
  - the existing RateStateFriction<dim> helper and all RSF parameters;
  - nodal \(\Theta\);
  - nodal \(T^{\rm coh}\);
  - current and previous \(I_h\);
  - projected fault composition/material mixtures;
  - surface residual \(R_\Gamma\);
  - \(K_V\) assembly and factorization;
  - initialization, commit, rollback, checkpoint, and history updates;
  - the exact diffuse \(\upsilon\) and crack-driving-force evaluation.

  PhaseFieldRSF registers \(\Theta,T^{\rm coh},I_h\), previous history, and diagnostic fields in the manager’s generic property schema for replicated storage/output, but
  it alone interprets and mutates those property slots.

  There is no separately Simulator-owned RSF handler.

  ### Solver-facing coupling interface

  Add a narrow optional material-model interface, implemented by PhaseFieldRSF:
```
  template <int dim>
  class ReconstructedFaultStokesCoupling
  {
  public:
    virtual void
    prepare_fault_linearization(
      const LinearAlgebra::BlockVector &linearization_point,
      const bool assemble_jacobian) = 0;

    virtual const std::vector<double> &
    fault_residual() const = 0;

    virtual void
    apply_G(const LinearAlgebra::BlockVector &src,
            std::vector<double> &dst) const = 0;

    virtual void
    apply_B(const std::vector<double> &src,
            LinearAlgebra::BlockVector &dst) const = 0;

    virtual void
    apply_K_inverse(const std::vector<double> &src,
                    std::vector<double> &dst) const = 0;

    virtual void
    recover_fault_update(
      const LinearAlgebra::BlockVector &delta_x,
      std::vector<std::vector<double>> &delta_V) const = 0;

    virtual double
    fault_residual_norm() const = 0;
  };
```
  The Stokes solver detects this optional interface through the active material model. No new RSF object or general-purpose RSF lifecycle is added to Simulator.

  Existing signals are used for material-owned lifecycle work:

  - post_set_initial_state: register/initialize surface and particle history when geometry is available;
  - start_timestep: save committed surface and particle history and invalidate timestep-dependent coefficients;
  - pre_assemble_stokes_system: lazily recompute \(I_h\) after a changed phase-field solution and prepare the current surface residual;
  - post_nonlinear_solver: commit on success or roll back on failure;
  - existing refinement/checkpoint signals: serialize committed state and invalidate/rebuild caches.

  The only direct solver-scheme changes are those mathematically required to trial-update \(V\), include the surface residual in convergence/line search, and recover
  \(\delta V\).

  ## 3. Existing-code migration

   Existing class/file | Category | Revised treatment
   ---- | ---- | ----
   `ReconstructedFault<dim>` | Keep unchanged | Retain ordered append-only geometry, implicit Q1 segments, generic properties, and geometry version.
   `ReconstructedFaultManager<dim>` | Reuse with modification |  Add distinguished \(V\) ownership and trial/accept/rollback; expose generic immutable particle/QP associations and interpolation. Do not add \(\Theta,T^{\rm coh}\), or RSF formulas.
   Initial reconstruction in `source/reconstructed_fault.cc` | Keep unchanged | Retain prescribed geometry, resampling, Q1 phase-field reconstruction, and replicated geometry.
   `project_to_normal_profiles()` | Reuse with modification | Continue finite normal strips, \(K=1\), open-tip exclusion, closest incident segment, and overlap errors; generalize from particles to reusable point associations.
   Existing particle projection cache | Reuse with modification | Retain IDs, positions, domain volumes, \(\xi\), geometry versions, MPI mass assembly, and tridiagonal factors. Expose it without exposing material state.
   `project_particle_properties()` | Keep for generic/static projection | Use for fault composition/material mixtures and diagnostics. Never use it to freeze current trial stress.
   `ParticleDomainHandler` | Keep unchanged |                        Continue using physical particle-domain volume as \(m_p\).
   Particle interpolator infrastructure | Reuse unchanged | Interpolate frozen particle stress history to Stokes quadrature points where bulk residual assembly requires it.
   New minimal particle stress-history property | Add | Store only independent components of \(\tau_{k-1}\). It contains no \(V,\Theta,T^{\rm coh}\), normal, or slip direction.
   `PhaseFieldHandler<dim>` | Reuse with modification | Expose the existing length scale and distributed Q1 phase-field point evaluation. Do not use core-phase-field extension.
   `RateStateFriction<dim>` | Reuse with modification | Remain the single RSF source of truth. Add only the inverse- friction operation needed for \(\mu(V,\Theta)=\mu_0\) initialization.
   `PhaseFieldRSF<dim>` | Reuse with substantial modification | Own surface constitutive state, Maxwell law, \(I_h\), residual, \(K_V\), history, and coupling interface.
   `MaterialModel::Rheology::Elasticity` | Do not use | Its stress update, rotation, timestep treatment, and state layout are not the pf_rsf.tex model.
   `ElasticOutputs` | Keep for other models; bypass here | Do not use it to implement reconstructed-fault Maxwell history.
   Standard Stokes viscosity assembly | Reuse | Supply \(\kappa=\eta(1-\beta)\) as the effective viscosity defining \(A\).
   New `PhaseFieldRSF` Stokes assembler | Add | Assemble frozen Maxwell history, current fault-slip stress, the bulk residual, and local data needed by \(B\).
   Local tangent-modulus outputs | Keep for other models; bypass here | They cannot represent \(BK_V^{-1}G\).
   `internal::StokesBlock` | Reuse through a wrapper | Wrap it with the exact condensed correction rather than replacing the underlying sparse Stokes matrix/preconditioner.
   Newton line search/residual handling | Reuse with minimal modification | Trial-update manager-owned \(V\), include separately normalized fault convergence, and roll back rejected candidates.
   Direct and matrix-free GMG solvers | Defer/guard | Initially support assembled iterative AMG only.
   Reconstructed-fault VTU output | Reuse with modification | Output distinguished \(V\) plus material-registered generic properties.
   Core-phase-field/local diffuse RSF architecture | Remove from the new path | Do not use core extension, particle RSF state, or interpolated local \(d\mu/dV\) tangents.

  ## 4. Maxwell law, residual, and coupling operators

  ### Frozen particle stress history

  A minimal particle property stores \(\tau_{k-1}\). During Newton:

  - particle positions and \(\tau_{k-1}\) remain fixed;
  - no objective rotation is applied;
  - no intermediate Newton iterate overwrites particle history;
  - QP values of \(\tau_{k-1}\) are obtained with the existing particle interpolator;
  - particle contributions to \(R_\Gamma\) use the particle’s stored tensor directly.

  After mechanical convergence:
  \[
  \bm\tau_k = 2\kappa_k\left(\dot{\bm\epsilon}_k-\upsilon_k\bm S\right) +
  \beta_k\bm\tau_{k-1}
  \]
  is evaluated at locally owned particles and committed atomically. A failed nonlinear solve restores the saved previous history.

  ### Trial and effective stress

  At an associated particle or QP:

  \[
  \bm\tau^*_k =
  2\kappa_k\dot{\bm\epsilon}_k+\beta_k\bm\tau_{k-1},
  \]

  \[
  \upsilon_k^{\rm hist} =
  \frac{\beta_kT^{\rm coh}_{k-1}}{\kappa_k}
  \left[h_k\frac{I_{h,k-1}}{I_{h,k}}-h_{k-1}\right],
  \]

  \[
  \bm\tau_k = \bm\tau_k^* - 2\kappa_k\upsilon_k^{\rm hist}\bm S,
  \]

  \[
  t_k = \bm\tau_k :\bm S,
  \qquad
  \sigma_n=p-\bm\tau_k^*:\bm N.
  \]
  The current stress is
  \[
  \bm\tau_k = \bm\tau_k^* - 2\kappa_k\frac{h_k}{I_{h,k}}V_\Gamma\bm S.
  \]

  ### Surface residual and \(K_V\)

  Only locally owned particles assemble:

  \[
  R_i^\Gamma = 
  \sum_p m_pN_i^p
  \left[
  t_p^*
  -\kappa_p\chi_pV_\Gamma(p)
  -T_\Gamma^{\rm coh}(V;p)
  -\mu(V_\Gamma(p),\Theta_{k-1,\Gamma}(p))\sigma_{n,p}
  -\eta_p^dV_\Gamma(p)
  \right],
  \]

  where \(\chi_p=h_p/I_h(\pi_\Gamma(p))\).

  Use

  \[
  K_V=-\frac{\partial R_\Gamma}{\partial V}
  \]

  with fixed geometry, phase field, particle history, and \(\Theta_{k-1}\). Its entries include:

  \[
  \kappa_p\chi_p
  +
  \frac{\kappa_{\Gamma,p}}{I_{h,p}}
  +
  \sigma_{n,p}
  \left.\frac{\partial\mu}{\partial V}\right|_{\Theta}
  +
  \eta_p^d.
  \]

  The matrix remains replicated and tridiagonal for the current Q1/single-segment formulation.

  ### \(G\)

  apply_G\(\) evaluates bulk perturbations at cached particle locations:

  \[
  \delta(t-\mu\sigma_n) =
  2\kappa(S+\mu N):\delta\dot\epsilon-\mu\delta p.
  \]

  Locally owned particle contributions are assembled into a fault-sized vector and reduced over MPI. Current trial stress is never pre-projected as a frozen field.

  ### \(B\)

  apply_B\(\) interpolates \(\delta V\) through manager-owned QP associations and assembles the weak bulk response to

  \[
  \delta\bm\tau=-2\kappa\chi\bm S\delta V_\Gamma.
  \]

  The ordinary bulk residual assembler uses the same QP association and coefficient evaluation for the current \(V\), preserving discrete consistency between the residual
  and `apply_B()`.

  ### Condensed solve

  The iterative operator applies:

  1. \(A\delta x\) with the existing assembled Stokes block;
  2. \(g=G\delta x\), including its fault-sized MPI reduction;
  3. \(z=K_V^{-1}g\) with the replicated tridiagonal factors;
  4. \(Bz\);
  5. \(A\delta x-Bz\).

  The Newton RHS receives \(BK_V^{-1}R_\Gamma\). After solving, recover

  \[
  \delta V=K_V^{-1}(R_\Gamma+G\delta x).
  \]

  The existing AMG Stokes preconditioner remains an approximation to \(A_{\rm eff}\). Direct and matrix-free GMG modes issue an early unsupported-configuration error in
  the first implementation.

  ## 5. Distributed \(I_h\) evaluation

  ### Profile ownership

  Number all segment quadrature profiles globally and deterministically. Partition these profile IDs into balanced contiguous ranges across MPI ranks. Each profile has
  exactly one owner for adaptive decisions and accumulated integration state.

  For each adaptive round:

  1. Every profile owner packs all required phase-field Gauss points and boundary probes for its active profiles.
  2. All ranks call one collective Utilities::MPI::RemotePointEvaluation with their independently owned request lists.
  3. Bulk-cell owners evaluate the actual Q1 phase field and return results to the requesting profile owner.
  4. Each profile owner updates only its profiles and schedules the next batch.
  5. Once profiles converge, their owners assemble their unique contributions to the fault mass matrix, mass RHS, and diagnostics.
  6. MPI sums produce identical replicated fault matrices/RHS; every rank performs the same tridiagonal Q1 mass solve.

  No coordinator gathers all sampling points or profile values.

  ### Segment and normal quadrature

  - Evaluate \(I_h\) at `QGauss<1>(3)` points on each fault segment, where the segment normal is unambiguous.
  - Integrate independently in \(+\hat{\bm n}\) and \(-\hat{\bm n}\).
  - At the profile origin, distributed point location also returns the minimum diameter of bulk cells containing the point.
  - # Set the initial panel width to
    \[
    \Delta\zeta_0
    \frac{1}{2}\min(\ell,h_{\rm cell}),
    \]
    reusing the existing phase-field length scale.

  - For later panels, cap the proposed width by one half of the minimum cell diameter encountered in the preceding accepted panel and by \(\ell/2\). Bisect further when
    the quadrature estimate requires it.

  - Evaluate 4-point and 8-point Gauss rules for every proposed panel. Accept when
    \[
    |I_8-I_4|
    \le
    \epsilon_{\rm quad}\max(|I_8|,\ell),
    \]
    with default \(\epsilon_{\rm quad}=10^{-8}\).

  ### Adaptive support without monotonicity

  Do not require monotonically decreasing panel contributions.

  For each profile side, maintain an outer rolling window of accepted panels whose combined radial span is at least \(\ell\). Terminate that side when

  \[
  I_{\rm outer\ window}
  \le
  \epsilon_{\rm tail}\max(I_{\rm accumulated},\ell)
  \]

  for two successive, non-overlapping outer windows. Default \(\epsilon_{\rm tail}=10^{-8}\).

  This makes the stopping criterion depend on integrated outer support rather than the number or size of mesh-dependent panels. The phase-field activation threshold is
  never used as the constitutive integration boundary.

  If a panel crosses the domain boundary, collectively bisect the endpoint location until the first connected in-domain boundary intersection is located, integrate the
  remaining partial panel, and terminate that side.

  A maximum adaptive-round count is only a failure guard. Exceeding it reports the owning rank, fault, segment, quadrature point, side, current support length, last
  panel widths/contributions, and accumulated integral.

  ### \(h(\phi)\) diagnostics

  At every sample:

  1. Verify finite \(\phi\).
  2. Evaluate \(g(\phi)\) through the existing phase-field/material interface using the fault-projected mixture held fixed along that profile.
  3. Verify finite \(g\), \(g>0\), and \(g\le1\) within roundoff.
  4. Compute
     \[
     h=\frac{1}{g}-1
     \]
     without clamping.

  5. Reject negative, non-finite, or reciprocal-overflowing \(h\).

  Diagnostics retain, per profile and globally:

  - maximum sampled \(\phi\);
  - minimum positive \(g\);
  - maximum \(h\);
  - location and profile metadata for each extremum;
  - final positive/negative support lengths;
  - accumulated quadrature and tail estimates.

  If \(g\le0\), \(1/g\) overflows, or \(I_h\) becomes non-finite, throw an explicit singular-\(h\) error containing the physical point, \(\phi\), \(g\), fault/segment/profile ID,
  normal side, and \(\zeta\). Do not silently clamp \(\phi\), \(g\), or \(h\).

  ### Projection to fault vertices

  Profile owners assemble:

  \[
  (M_\Gamma)_{ij}\int_{\Gamma_h}N_iN_jds,
  \qquad
  b_i\int_{\Gamma_h}N_iI_h^{\rm quad}ds.
  \]

  After MPI reduction, every rank solves

  \[
  M_\Gamma I_h=b
  \]

  with identical tridiagonal factors. Verify finite positive nodal \(I_h\) and replicated-vector consistency.

  ## 6. Nonlinear lifecycle and caches

  ### Initialization

  For an initial fault:

  - set \(T^{\rm coh}_{k-1}=0\);
  - set \(I_{h,k-1}=I_{h,k}\);
  - assemble the discrete surface force balance with \(\mu=\mu_0\);
  - solve for positive nodal \(V\), including mechanical-slip, cohesive, and radiation terms;
  - invert the existing RSF law to obtain positive \(\Theta\) satisfying \(\mu(V,\Theta)=\mu_0\);
  - reject a nonpositive force-balance solution or a non-unique/nonexistent positive inverse state.

  ### Line search and convergence

  The manager saves committed \(V\) before a Newton step. For \(\delta V_i<0\), limit the initial line-search step by

  \[
  \alpha_{\max} = \min\left(1,0.99\min_i\frac{V_i-V_{\min}}{-\delta V_i}
  \right).
  \]

  Each candidate sets

  \[
  V^{\rm trial}=V^{\rm saved}+\alpha\delta V
  \]

  through the manager, while PhaseFieldRSF reevaluates its surface residual without committing \(\Theta,T^{\rm coh}\), or particle stress.

  Use separately normalized bulk and fault residuals. Both must independently satisfy the configured nonlinear tolerance. Their Euclidean normalized combination is used
  only as the line-search merit function. Rejected candidates roll back manager-owned \(V\) and all material-owned working state.

  ### Cache invalidation

  - Geometry/segment cache: geometry version change.
  - Particle association: geometry or half-width change, particle advection/migration, ID/order/position/domain-volume change, AMR, or repartitioning.
  - QP association: geometry/half-width change, AMR, repartitioning, mapping or Stokes quadrature change.
  - Fault mass matrix: geometry change.
  - \(I_h\) samples/values: phase-field solution, projected mixture, geometry, mesh, mapping, or integration-setting change.
  - Maxwell coefficients: timestep, temperature, composition/material mixture, or parameter change.
  - \(R_\Gamma,K_V\), fixed-state \(\mu,d\mu/dV\), and condensed RHS: every nonlinear linearization.
  - Particle \(\tau_{k-1}\): immutable during Newton; replaced only after successful mechanical convergence.
  - No geometric cache stores stress, \(V,\Theta,T^{\rm coh},I_h\), or friction values.

  Checkpoint/restart stores committed geometry, manager-owned \(V\), generic property schema/values, PhaseFieldRSF history, and particle stress history. All association,
  point-location, matrix-factorization, and nonlinear caches are reconstructed after restart.

  ## 7. Staged implementation

  1. __Manager kinematic field__
     Add distinguished \(V\), interpolation, initialization, trial/accept/rollback, checkpointing, and VTU output. Preserve all generic geometry/property behavior.

  2. __Maxwell stress history__
     Add the minimal particle stress property and implement the pf_rsf.tex Maxwell update with frozen Newton history and no rotation. Test \(\beta,\kappa,\tau^{\rm
     trial}\), particle commit, and rollback independently.

  3. __Distributed adaptive \(I_h\)__
     Implement distributed profile ownership, mesh-aware panels, batched point evaluation, non-monotone rolling-tail support, singular-\(h\) diagnostics, Q1 mass projection, and MPI tests. Stop for review before mechanical coupling.

  4. __`PhaseFieldRSF` surface state__
     Register and own \(\Theta,T^{\rm coh},I_h\) and previous history inside PhaseFieldRSF; implement force-balance/\(\mu_0\) initialization and checkpoint/restart.

  5. __Surface residual and \(K_V\)__
     Implement effective trial stress, cohesive working state, lagged-\(\Theta\) residual, IMPES derivative, tridiagonal factorization, and local-limit tests.

  6. __Bulk residual and association reuse__
     Generalize manager-owned QP associations and add the custom PhaseFieldRSF Stokes residual using \(\kappa\), frozen stress history, \(\upsilon^{\rm hist}\), and current \(V\).

  7. __Independent \(B\) and \(G\) actions__
     Implement and test each action against explicitly assembled small operators and finite differences before changing the solver.

  8. Exact condensed AMG solver
     Add the optional coupling interface, \(A-BK_V^{-1}G\), condensed RHS, \(\delta V\) recovery, and direct/GMG guards.

  9. Coupled Newton lifecycle
     Integrate \(V\) into line search, positivity limiting, separate fault convergence, failure rollback, and nonlinear diagnostics with the minimum solver changes.

  10. History, \(H\), and fixed-fault benchmarks
     Commit \(\Theta,T^{\rm coh}\), particle stress, and \(I_h\); reconstruct exact \(\upsilon\); evaluate \(H\); complete restart/output; run fixed-phase-field benchmarks.

  11. Propagation only after review
     Define and test appended-vertex history initialization before integrating propagation. Do not infer this rule from the initial-fault initialization.

  Each stage is an independently reviewable commit and preserves the tests from earlier stages.

  ## 8. Verification and acceptance

  ### Geometry and associations

  - Constant and exact Q1 field reproduction.
  - Unequal particle-domain volumes.
  - \(K=1\) invariance with normal distance.
  - Open-tip exclusion.
  - Internal-bend selection.
  - Explicit multi-fault overlap error.
  - No ghost-particle double counting.
  - One-rank/two-rank equivalence.
  - Cache invalidation after motion, AMR, and geometry change.

  ### \(I_h\)

  - Analytical stationary-profile integral.
  - Profile quadrature refinement.
  - Segment quadrature refinement.
  - Tail-tolerance/support-truncation convergence.
  - Initial-panel and bulk mesh-refinement convergence.
  - Different distributed profile partitions.
  - One-rank/two-rank equivalence.
  - Boundary-truncated profiles.
  - Non-monotone but integrable synthetic profiles.
  - Exact singular-\(h\), non-finite-\(\phi\), and reciprocal-overflow diagnostics.
  - Q1 surface mass-projection convergence.
  - Discrete verification of \(\int\upsilon,d\zeta=V\).

  ### Maxwell and surface constitutive behavior

  - Exact \(\beta=\exp(-\Delta tG/\eta)\) and \(\kappa=\eta(1-\beta)\).
  - Frozen particle history throughout Newton.
  - No rotation contribution.
  - Particle-to-QP history interpolation.
  - Fixed-profile cancellation of \(\upsilon^{\rm hist}\).
  - Evolving-profile normalization.
  - Mechanical, cohesive, radiation, and friction residual terms separately.
  - Exact regularized fixed-\(\Theta\) derivative.
  - Force-balance/\(\mu_0\) initialization.
  - Commit and failure rollback.

  ### Mandatory coupled finite-difference test

  Expose a non-committing residual evaluator for

  \[
  R(y)=
  \begin{bmatrix}
  R_{\rm bulk}(x,V)\\
  R_\Gamma(x,V)
  \end{bmatrix}.
  \]

  Compare

  \[
  \begin{bmatrix}
  A & -B\\
  G & -K_V
  \end{bmatrix}
  \begin{bmatrix}
  \delta x\\
  \delta V
  \end{bmatrix}
  \]

  against centered finite differences for:

  - velocity-only perturbations;
  - pressure-only perturbations;
  - \(V\)-only perturbations;
  - mixed perturbations;
  - nonzero cohesive history;
  - evolving phase-field profile history;
  - one and two MPI ranks;
  - states near, but not crossing, \(V_{\min}\).

  Also verify:

  - the condensed action equals \(A-BK_V^{-1}G\);
  - the condensed RHS has the correct sign;
  - recovered \(\delta V\) satisfies the uncondensed fault row;
  - finite-difference error decreases at the expected rate before roundoff saturation.

  BP3-style testing begins only after the block and condensed Jacobian tests pass.

  ## 9. Assumptions and remaining deferred decision

  - Current implementation is 2-D, with open ordered Q1 fault polylines.
  - One particle or QP belongs to at most one fault.
  - \(K=1\) inside the finite normal strip.
  - \(V\) is a nonnegative distinguished kinematic field.
  - \(\Theta\) is lagged during Newton; the full \(d\Theta/dV\) feedback is not restored.
  - Particle stress history is frozen during Newton and follows the non-rotational pf_rsf.tex Maxwell law.
  - Assembled iterative AMG is the first supported Stokes solver.
  - \(I_h\) uses the actual FE phase field and fault-projected composition mixture.
  - The corrected sign convention in pf_rsf.tex and the matching residual in the numerical design are authoritative; there is no remaining sign conflict.
  - Initialization of constitutive history on later appended propagation vertices remains scientifically unspecified. Propagation integration stops until that rule is
    reviewed.
