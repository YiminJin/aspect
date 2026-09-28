  # Revised Stage 3 / Stage C — Distributed Adaptive (I_h)

  ## Summary and prerequisite

  Implement the authoritative Stage C algorithm from phase_field_fault_redesign.md: distributed adaptive integration of

  \[
  I_h(s_q)=\int h\left(\phi_h(x_\Gamma(s_q)+\zeta n_q)\right)\mathrm d\zeta,
  \qquad
  h=1/g-1,
  \]

  using the actual distributed Q1 phase field and projecting the quadrature-point integrals onto the fault’s Q1 vertices.

  Before this work begins, complete and review Stage A as a separate commit:

  - Rename PhaseFieldRSF to PhaseFieldFault.
  - Rename/generalize RateStateFriction to FaultFriction.
  - Preserve current rate-state behavior and introduce the friction-law selector.
  - Update the Maxwell property and existing tests to the new material-model name.
  - Do not combine Stage A and adaptive (I_h) in one review packet.

  Stage C then assumes PhaseFieldFault exists and owns common reconstructed phase-field fault mechanics. Stop for review after Stage C; do not begin cohesive state or
  mechanical coupling.

  ## Interfaces and ownership

  - Extend PhaseFieldHandler with two concrete operations used by PhaseFieldFault:
      - Return the existing phase-field length scale.
      - Collectively evaluate the current Q1 phase field at each rank’s independently requested points, returning found, (\phi), and the minimum diameter of all bulk
        cells containing each point.

  - Extend ReconstructedFaultManager with narrow, constitutively neutral operations:
      - Project selected particle-property components and return values as [fault][vertex][requested component], refactoring the existing property-writing projection to
        share the same projection solve.

      - Classify a physical point against the replicated fault influence regions so PhaseFieldFault can detect entry into a different fault without exposing half-width
        containers.

  - Keep all (I_h) semantics in PhaseFieldFault:
      - Private adaptive integration kernel.
      - Private transient current nodal (I_h), profile integrals, projected mixtures, and diagnostics.
      - No generic-property registration, previous (I_h), checkpoint persistence, or constitutive history in Stage C.
      - No public (I_h) getter and no automatic lifecycle/signal connection yet; a future production stage will provide the first runtime caller.

  - Add a narrowly scoped friend PhaseFieldFaultIhTestAccess for unit and integration tests. It may invoke the private computation and inspect private results but
    introduces no runtime API.

  - Add only the actual numerical tolerances as user parameters:
      - Ih quadrature tolerance = 1e-8.
      - Ih tail tolerance = 1e-8.
      - Keep panel depth, boundary bisection, growth factor, and outward-extension guards internal.

  ## Adaptive and distributed algorithm

  - Support only 2-D ordered fault polylines. Reject zero-length segments and unsupported dimensions.
  - Generate QGauss<1>(3) profiles in deterministic fault-major, segment-major, quadrature-point order and distribute balanced contiguous profile-ID ranges over MPI ranks.

  - Project particle-mapped chemical compositions to fault vertices, Q1-interpolate them at each profile origin, and convert them using ASPECT’s existing composition-fraction utility. Hold the resulting mixture fixed along that normal profile. Background-only models use the background fraction directly.

  For each owned profile:

  1. Construct the segment unit normal and sample the origin.
  2. Validate the sampled (\phi) once in release mode: it must be finite and lie in the authoritative PhaseFieldModel::get_phase_field_range().
  3. Set
     \[
     \Delta\zeta_0=\frac12\min(\ell,h_{\rm local}).
     \]

  4. Integrate the positive and negative normal sides independently.
  5. For every candidate panel, batch its boundary probe and 4- and 8-point Gauss samples with all other active profiles.
  6. Accept when
     \[|I_8-I_4|\le\epsilon_{\rm quad}\max(|I_8|,\ell);
     \]
     otherwise bisect.

  7. After acceptance, propose
     \[
     \Delta\zeta_{\rm next}
     \min\left(
     2\Delta\zeta_{\rm accepted},
     \frac{\ell}{2},
     \frac{h_{\rm local}}{2}
     \right),
     \]
     using the minimum containing-cell diameter sampled in the accepted panel.

  Tail termination uses no phase-field activation cutoff and imposes no monotonicity:

  - Form non-overlapping outer windows spanning at least (\ell).
  - Terminate a side only after two consecutive windows satisfy
    \[
    I_{\rm window}
    \le
    \epsilon_{\rm tail}\max(I_{\rm accumulated},\ell).
    \]

  - A numerical-zero phase-field shortcut is unnecessary in the first implementation.

  Fault and boundary handling:

  - A non-monotonic shoulder or secondary peak belonging to the same connected diffuse profile remains part of the integral.
  - If the manager’s geometric classification reports entry into another reconstructed fault’s influence region before tail termination, issue a collective unsupported-overlap failure. Do not integrate the second fault.

  - When leaving the physical domain, bracket the first missing point, collectively bisect to floating-point convergence, integrate the remaining connected in-domain interval, and terminate that side. Never continue into a disconnected re-entry region.

  Validation and safeguards:

  - Every FE phase-field sample receives the release-mode range/finiteness validation.
  - Evaluate \(g\) through the existing phase-field degradation interface with the profile-fixed mixture.
  - Treat ordinary derived \(g/h\) properties as debug invariants, but issue a release-mode constitutive-singularity failure if \(g\le0\), its reciprocal overflows, or usable \(h\) cannot be formed.

  - Require the completed profile and projected nodal \(I_h\) values to be finite and positive.
  - Use fixed emergency guards:
      - Panel refinement depth: 64.
      - Boundary bisections: 64.
      - Accepted outward extensions per side: 256.

  - Guard failures report rank, fault/segment/profile/quadrature identifiers, side, point, \(\zeta\), support length, panel bounds, refinement count, recent estimates, and accumulated integral.

  - Convert owner-local failures into a deterministic collective failure before any rank enters another point-evaluation collective.

  After profile completion:

  - Owners assemble their unique Q1 segment mass-matrix and RHS contributions.
  - MPI-sum the small fault-sized arrays.
  - Solve the replicated tridiagonal Q1 mass systems.
  - Verify finite positive nodal values and one-rank/two-rank consistency.

  ## Test plan

  Use the private friend accessor to drive the exact production state machine with synthetic samplers. Automated unit tests cover:

  - Smooth analytical profile.
  - Compact support.
  - Weak integrable tail.
  - Non-monotonic connected-profile shoulder.
  - Connected separated two-peak profile that must not terminate between peaks.
  - Explicit encounter with a separate reconstructed fault, which must fail as unsupported overlap.
  - Near-singular but finite (g).
  - Exact singularity, reciprocal overflow, non-finite (\phi), and out-of-range (\phi).
  - Panel rejection/bisection and subsequent width regrowth.
  - Domain-boundary clipping.
  - Each fixed emergency guard and its diagnostic fields.
  - Q1 mass-projection accuracy and positivity.

  The optional private diagnostic trace records panel bounds, refinement depth, (I_4), (I_8), accepted/rejected state, contribution, accumulated integral, and tail-
  window state. Unit-test code provides a CSV writer enabled only by a test environment variable; production code gains no visualization setting or output path.

  MPI/integration tests cover:

  - Constant and exact-Q1 distributed phase-field evaluation.
  - Points on partition interfaces and minimum containing-cell diameter.
  - Empty local request lists.
  - Deterministic profile partitioning.
  - Analytical stationary-profile (I_h).
  - Bulk-mesh and fault-surface projection convergence.
  - Background-only and composition-dependent degradation.
  - Boundary-truncated profiles.
  - One-rank/two-rank equivalence of profile integrals, nodal projection, and diagnostics.
  - Collective singularity and separate-fault failures without deadlock.
  - Preservation of all earlier reconstructed-fault and Maxwell tests in debug and release builds.

  ## Stage boundary and documentation

  - Do not register committed (T^{\rm coh}), previous (I_h), or friction state.
  - Do not implement (\chi), (\upsilon), cohesive history, residuals, (K_V), (B), (G), Stokes assembly, or nonlinear lifecycle hooks.
  - Do not expose speculative (I_h) runtime getters or connect pre_assemble_stokes_system; Stage C is exercised through its friend test seam until a later production
    stage supplies a real caller.

  - Update current_design.md to reference phase_field_fault_redesign.md as the architectural authority and record the completed Stage A/Stage C ownership and MPI
    decisions.

  - Deliver Stage A and Stage C as separate independently reviewable commits, stopping after each for review.
