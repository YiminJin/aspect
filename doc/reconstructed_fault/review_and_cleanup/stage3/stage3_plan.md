  # Stage 3 — Distributed Adaptive (I_h)

  ## Summary

  Implement distributed evaluation of

  \[
  I_h=\int_{-\infty}^{+\infty}\left(\frac{1}{g(\phi)}-1\right)\mathrm d\zeta
  \]

  along segment-normal profiles of the reconstructed fault. Profiles are deterministically distributed across MPI ranks, integrated adaptively against the actual distributed Q1 phase field, and projected to fault vertices with a Q1 mass solve.

  Stage 3 will retain the resulting nodal \(I_h\) and diagnostics in a private transient `PhaseFieldRSF` cache with read-only diagnostic access. It will not register persistent \(I_h\) history, add surface constitutive state, or begin mechanical coupling.

  ## Key interfaces and ownership

  - Extend `PhaseFieldHandler` with:
      - `get_length_scale()`.
      - A collective `evaluate_phase_field_at_points()` operation returning, for every locally requested point, whether it was found, the Q1 phase-field value, and the minimum diameter of all containing bulk cells.
      - Use one `RemotePointEvaluation` communication pattern per adaptive batch and reduce duplicate-cell phase values consistently while taking the minimum cell diameter.

  - Extend `ReconstructedFaultManager` with a generic projection operation that returns selected particle-property components as `[fault][vertex][requested component]`.
      - Refactor the existing property-writing projection to reuse this operation.
      - `PhaseFieldRSF` will project the particle properties mapped to chemical compositional fields, interpolate them along each fault segment, and convert them through
        ASPECT’s existing composition-fraction utility.

      - No material-specific mixture data is added to the manager’s persistent property pool.

  - Add private `PhaseFieldRSF` structures for:
      - Maxwell-independent \(I_h\) profile state and the adaptive integrator.
      - Transient nodal \(I_h\), projected mixtures, cache keys, and profile/global diagnostics.
      - Fixed safeguards: 64 panel-refinement levels, 64 boundary bisections, and 256 accepted outward extensions per profile side.
      - A narrowly scoped friend unit-test accessor. No production caller other than PhaseFieldRSF may access the integrator.

  - Add read-only inspection methods for the transient nodal \(I_h\) and replicated diagnostics. These do not permit mutation or constitute persistent Stage 4 history.
  - Add user parameters under Phase field RSF:
      - `Ih` quadrature tolerance = 1e-8.
      - `Ih` tail tolerance = 1e-8.
      - The implementation safeguards remain fixed and are not exposed as parameters.

  ## Distributed algorithm

  - Number profiles fault-major, segment-major, then `QGauss<1>(3)` point. Partition the global profile IDs into balanced contiguous MPI ranges.
  - Use the segment’s unambiguous 2-D unit normal and integrate the positive and negative sides independently. Reject zero-length segments and dimensions other than two.
  - At each profile origin:
      - Interpolate the projected chemical composition and hold the resulting material fractions fixed along the complete normal profile.
      - Sample the actual phase field and minimum containing-cell diameter.
      - Set the initial panel width to (\frac12\min(\ell,h_{\rm cell})).

  - For each proposed panel:
      - Batch its endpoint probe and 4- and 8-point Gauss samples with all other active owner profiles.
      - Evaluate (g(\phi)) using PhaseFieldHandler::energetic_degradation() and the profile’s fixed mixture.
      - Accept when
        \[
        |I_8-I_4|\le\epsilon_{\rm quad}\max(|I_8|,\ell).
        \]

      - On rejection, bisect and increment that panel’s refinement depth.
      - After acceptance, propose twice the accepted width, capped by \(\ell/2\) and half the minimum cell diameter encountered in the accepted panel.

  - Do not require monotone panel contributions. Accumulate non-overlapping outer windows whose span is at least (\ell). Terminate a side only after two consecutive
    windows satisfy
    \[
    I_{\rm window}\le\epsilon_{\rm tail}\max(I_{\rm accumulated},\ell).
    \]

  - If an endpoint or quadrature probe leaves the domain:
      - Bracket the first missing point from the last accepted in-domain front.
      - Collectively bisect the bracket until floating-point convergence, with 64 iterations as the failure safeguard.
      - Adaptively integrate the remaining connected in-domain interval and terminate that side.
      - Do not continue into a disconnected re-entry region.

  - Validate every sample without clamping:
      - Finite \(\phi\).
      - Finite \(g\), \(0<g\le1\).
      - Finite reciprocal and finite nonnegative \(h=1/g-1\).
      - Finite panel, tail, profile, and projected nodal integrals.
      - Singular failures include owner rank, physical point, \(\phi\), \(g\), fault/segment/profile ID, side, and \(\zeta\).
      - Propagate failures collectively so one owner cannot throw while other ranks enter the next collective operation.

  - Profile owners assemble their unique contributions to the fault Q1 mass matrix and RHS. MPI sums replicate them, and every rank solves the same tridiagonal system.
    Require every nodal \(I_h\) to be finite and positive and verify replicated-vector consistency.

  - Connect only to existing signals:
      - Invalidate at `start_timestep` and after mesh refinement.
      - Lazily recompute once from `pre_assemble_stokes_system` after phase-field evolution and fault reconstruction.
      - Reuse the cache during mechanical nonlinear iterations when phase field, geometry, projected mixture, mesh, and settings are unchanged.
      - Add no new `Simulator` lifecycle hook.

  ## Diagnostics and tests

  - Production diagnostics retain extrema and their locations, positive/negative support lengths, accepted-panel counts, maximum refinement depth, accumulated integrals, final tail-window estimates, ownership metadata, and global extrema.

  - The private integrator optionally records a test-only trace containing panel bounds, refinement level, \(I_4\), \(I_8\), acceptance, contribution, accumulated integral, and tail status.

  - Unit-test code provides an optional CSV writer activated only by a test environment variable; production code has no visualization parameter or output path.

  Automated tests will cover:

  - Smooth analytic reference profiles.
  - Compact support and boundary truncation.
  - Weak but integrable tails.
  - Non-monotonic tails.
  - Two separated peaks close enough that the two-window rule must retain the outer peak.
  - Near-singular finite \(g\), exact singular \(g\), reciprocal overflow, and non-finite \(\phi\).
  - Panel refinement, width regrowth, trace consistency, and all fixed failure guards.
  - Constant and exact-Q1 distributed phase-field sampling, including partition interfaces and minimum containing-cell diameter.
  - Analytical stationary-profile \(I_h\) and convergence under bulk and segment refinement.
  - Fault-projected material mixtures.
  - Q1 mass-projection accuracy and positive nodal values.
  - One-rank/two-rank equivalence, different profile partitions, collective diagnostics, and deterministic failures.
  - Preservation of all Stage 1 and Stage 2 tests plus debug and release builds.

  ## Assumptions and stage boundary

  - Stage 3 remains 2-D and requires reconstructed fixed fault geometry and the existing Q1 phase-field variable.
  - Particle-mapped chemical compositions are the source of the profile-fixed material mixture; background-only models require no projected components.
  - The actual stopping decisions are controlled only by the two numerical tolerances. Fixed counts are unreachable safety guards with detailed failures.
  - Do not register persistent \(I_h\), previous \(I_h\), \(\Theta\), or \(T^{\rm coh}\); do not add checkpoint history, surface residuals, \(K_V\), bulk coupling, or solver changes.

  - Update current_design.md with these settled Stage 3 ownership, MPI, numerical, diagnostic, and lifecycle decisions, then stop for review.
