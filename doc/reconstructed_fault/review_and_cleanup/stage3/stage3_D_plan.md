  # Stage D — Common Cohesive State

  ## Summary

  Implement Stage D on the accepted replicated-polyline and generic-property architecture. First reconcile the authoritative documents with that architecture and isolate
  the pending Stage C acceptance changes in their own commit. Then add initial cohesive-state construction, persistent committed history, and mathematically complete
  non-committing evaluations of \(T^{\rm coh}(V)\) and \(\upsilon(V)\).

  Stage D ends without friction residuals, \(K_V\), \(B/G\), Stokes coupling, runtime cohesive-history commits, or timestep/nonlinear-solver signal connections.

  ## Authoritative design and interfaces

  - Update current_design.md and specification.tex so both describe the implemented custom ordered-polyline fault, replicated generic-property storage, current phase-
    field threshold semantics, and the A–K staged sequence. Remove or revise stale runtime Triangulation/DoFHandler and old stage-number requirements.

  - # Promote the approved cohesive contract:
    \[
    T^{\rm coh}_k = 
    \frac{\kappa_k}{I_{h,k}}V_k+
    \beta_k\frac{I_{h,k-1}}{I_{h,k}}T^{\rm coh}_{k-1},
    \]
    \[
    \upsilon_k=
    \frac{h_k}{I_{h,k}}V_k+
    \frac{\beta_kT^{\rm coh}_{k-1}}{\kappa_k}
    \left(h_k\frac{I_{h,k-1}}{I_{h,k}}-h_{k-1}\right).
    \]
    Document that \(h_{k-1}\) remains derived from the previous FE phase-field solution; it is not another fault property.

  - Register two scalar generic properties before fault creation:
      - phase field fault cohesive traction: committed \(T^{\rm coh}_{k-1}\);
      - phase field fault previous I h: committed \(I_{h,k-1}>0\).

  - Add one constitutively neutral manager operation, `project_particle_scalar(...)`, taking finite scalar values keyed by locally owned particle ID and returning
    replicated `[fault][vertex]` Q1 projections plus per-fault residual diagnostics. Refactor the existing named-property projection to share the same cached mass matrix
    and MPI reduction without changing its behavior.

  - Keep cohesive response, initialization, validation, and explicit test-only commit operations private to PhaseFieldFault, exposed only through its narrow internal
    test seam. Add no public material-model or Simulator lifecycle API.

  ## Implementation

  - Add a private immutable cohesive-response operation returning \(T^{\rm coh}\), \(\chi=h_k/I_{h,k}\), the history correction, and \(\upsilon\) from explicit inputs.
    Validate finite state, \(I_{h,k},I_{h,k-1},\kappa_k>0\), \(0\le\beta_k\le1\), \(V\ge0\), and \(h_k,h_{k-1}\ge0\). Permit a signed history correction and signed local \(\upsilon\).

  - Freeze committed \(T^{\rm coh}{k-1}\) and \(I{h,k-1}\) throughout evaluation. The response operation must not mutate generic properties or transient \(I_h\).

  - Implement explicit initial-state construction after initial reconstructed geometry exists:
      1. Compute transient \(I_{h,0}\) from the initial phase field.
      2. At each associated phase-field particle, evaluate
         \[
         q=g(\phi_{\rm eff})\sqrt{2GH},
         \qquad \phi_{\rm eff}=\max(\phi_h,0).
         \]
      3. Reuse the Stage C \(10^{-4}\) empirical undershoot guard, reject upper overshoot, and validate finite \(G>0\), \(H\ge0\), \(g\ge0\), and \(q\ge0\).
      4. Obtain \(H\) from the existing `crack_driving_force` particle property, chemical mixtures from the existing mapped particle properties, \(G\) from the existing shear-modulus data and averaging rule, and \(g\) from PhaseFieldHandler.
      5. Collapse all associated profile particles with the generic volume-weighted consistent Q1 projection to obtain nodal \(T^{\rm coh}_0\).
      6. Atomically initialize the two generic properties with projected \(T^{\rm coh}_0\) and current nodal \(I_{h,0}\). Fully initialized restart state is retained;
         partially initialized or invalid state is rejected.

  - Quantify the profile-uniform-\(q\) assumption using volume-weighted RMS and maximum absolute projection residuals, together with values normalized by the maximum profile \(|q|\). Store and print these diagnostics without imposing a new acceptance threshold or parameter.

  - Provide a private explicit commit helper that validates and writes supplied converged \(T^{\rm coh}k\) and current \(I{h,k}\). Leave it unwired until the later accepted coupled-\(V\) lifecycle stage.

  - Preserve checkpoint behavior through the existing generic-property serialization. Do not introduce cohesive trial/rollback machinery.
  - Before Stage D changes, verify and commit the currently pending Stage C acceptance work separately so the Stage D diff and commit remain reviewable.

  ## Tests and acceptance criteria

  - Unit-test the cohesive response against direct formulas, including zero previous traction, nonzero history, very small relaxation exponents, invalid/singular inputs,
    and the fixed-profile limit.

  - Use different-width analytic Gaussian \(h_k\) and \(h_{k-1}\) profiles so the history correction is pointwise nonzero but integrates to zero. Verify quantitatively that
    \[
    \int\upsilon_k\mathrm d\zeta=V_k,
    \qquad
    \int\upsilon_k^{\rm hist}\mathrm d\zeta=0,
    \qquad
    \frac{\partial\upsilon_k}{\partial V_k}=\frac{h_k}{I_{h,k}},
    \]
    including a centered finite-difference derivative check.

  - Test generic scalar projection for constant and Q1-varying fields, MPI-replicated equality, missing/nonfinite particle values, and known transverse variation with expected RMS/max diagnostics.

  - Add a controlled AT1 initialization test with \(\ell/h\ge10\) that obtains \(H\) from the production particle property, constructs (q), initializes \(T^{\rm coh}_0\) and \(I_{h,0}\), checks finite replicated state, and verifies checkpoint/archive round-trip. Reconstruction accuracy remains outside this fixture’s purpose.

  - Re-run all Stage A–C unit and integration tests, serial and two-rank Stage D tests, debug/release builds, and the Voro-enabled build with -j4.

  - Run the complete ASPECT integration suite to completion before declaring the phase-field API refactor finalized. Report pre-existing or unrelated golden-output failures without changing unrelated expected results.

  - Produce a Stage D review packet and separate Stage D commit, listing changed files, interfaces, equation correspondence, assumptions, uncertainties, exact test results, diff stat, and commit hash.

  ## Assumptions and boundaries

  - Stage D remains two-dimensional and fixed-geometry.
  - The initial \(q\) projection uses every particle associated with the fault influence strip, not only samples above the reconstruction activation threshold.
  - The same bounded-negative rule and empirical \(10^{-4}\) guard apply to the initial \(q\) degradation evaluation; neither is a convergence parameter.
  - \(q\) profile nonuniformity is diagnostic only because no scientifically approved rejection tolerance exists.
  - No caller-supplied initial cohesive-state path is added yet because the current model has the approved \(H\)-based initialization.
  - No Stage E or later constitutive/solver coupling is implemented.
