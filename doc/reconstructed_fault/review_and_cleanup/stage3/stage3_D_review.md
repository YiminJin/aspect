  ## Stage D review disposition

  Stage D is internally consistent with the authoritative design, preserves the manager/material-model ownership boundary, and introduces no Stage E mechanics or lifecycle hooks. I found no blocking correctness issue in the committed implementation.

  The main review limitation is that the initial \(q\rightarrow T^{\rm coh}_0\) production-path test verifies validity and state consistency, but does not compare the projected traction against an independent quantitative reference. The cohesive equations themselves do have quantitative unit coverage.

  Commit: aa87e75738e1315ffce42a79839684dce18058ae
  Subject: Reconstructed fault: add Stage D cohesive state

  ## Implemented invariants

  - `ReconstructedFaultManager` knows how to project a generic scalar, but it does not know what \(q\), \(T^{\rm coh}\), \(H\), \(G\), or \(I_h\) mean.
  - `MaterialModel::PhaseFieldFault` owns the cohesive law and interprets the generic fault properties.
  - Persistent state consists of committed nodal \(T^{\rm coh}_{k-1}\) and \(I_{h,k-1}\).
  - Current \(I_{h,k}\), cohesive responses, and projection diagnostics remain transient material-model state.
  - Cohesive state is either fully uninitialized or fully initialized across every fault vertex. Partial state is rejected.
  - A cohesive commit validates the complete candidate before modifying any stored vertex.
  - Initialization uses the existing particle `crack_driving_force`, existing composition mapping, and existing shear-modulus averaging.
  - Initial phase-field undershoots use \(\phi_{\rm eff}=\max(\phi_h,0)\), with the existing \(10^{-4}\) excessive-undershoot guard. The upper endpoint is not clipped.
  - Particle projection uses stable locally owned particle IDs and the existing distributed, particle-domain-volume-weighted Q1 projection operator.
  - Committed state remains frozen. Stage D does not attach initialization or commit to simulator timestep or nonlinear-solver signals.

  ## Files changed

   File | Stage D change
   ---- | ----
   `doc/reconstructed_fault/current_design.md`:530 | Records the common cohesive equations, state ownership, \(q\)-based initialization, diagnostic policy, and the unwired Stage D lifecycle boundary. Also reconciles the authority statement with `specification.tex`.
   `doc/reconstructed_fault/specification.tex`:607 | Makes the accepted cohesive equations and initialization algorithm authoritative. Aligns older architectural descriptions with the existing replicated ordered-polyline implementation and records the A–K stage sequence.
   `include/aspect/material_model/phase_field_fault.h`:105 | Adds private cohesive response/state operations, generic-property indices, initialization diagnostics, and corresponding internal test access.
   `include/aspect/reconstructed_fault.h`:260 | Adds the constitutively neutral scalar-projection result, residual diagnostics, and projection entry point.
   `source/material_model/phase_field_fault.cc`:115 | Implements the cohesive response law, property registration, validated commit, initial-state construction from \(H\), phase-field sampling, composition-dependent (G), projection, and diagnostics.
   `source/reconstructed_fault.cc`:1505 | Implements distributed scalar projection using the existing association cache, mass operator, MPI reductions, replicated Q1 solve, and residual calculation.
   `tests/phase_field_fault_ih.cc`:33 | Extends the Voro lifecycle fixture to initialize cohesive state and verify committed traction, committed \(I_h\), and diagnostic validity.
   `tests/phase_field_fault_ih.prm`:1 | Documents that the fixture now covers Stage D initialization and that reconstruction accuracy is outside its scope.
   `unit_tests/phase_field_fault_cohesive.cc`:26 | Adds quantitative cohesive-response, conservation, derivative, and invalid-input tests.
   unit_tests/reconstructed_fault.cc:212 | Verifies that the actual cohesive traction and previous-\(I_h\) properties survive checkpoint serialization.
   

  ## Public interface changes

  The following production interfaces were added to ReconstructedFaultManager:
```
  struct ParticleScalarProjectionDiagnostics
  {
    double weighted_rms_residual;
    double maximum_absolute_residual;
    double normalized_weighted_rms_residual;
    double normalized_maximum_absolute_residual;
  };

  struct ParticleScalarProjectionResult
  {
    std::vector<std::vector<double>> nodal_values;
    std::vector<ParticleScalarProjectionDiagnostics> diagnostics;
  };

  ParticleScalarProjectionResult
  project_particle_scalar(
    const std::map<types::particle_index, double> &locally_owned_values);
```
  These are defined at `include/aspect/reconstructed_fault.h`:260.

  Contract:

  - Every active, locally owned associated particle must have one finite input value.
  - Inactive particles may be omitted.
  - Input values are addressed by stable particle ID.
  - The result is replicated and ordered by fault and fault vertex.
  - The operation does not write a registered generic property.
  - Diagnostics are computed per fault.
  - Missing or nonfinite active-particle values are reported collectively before entering later MPI operations.

  No new user parameters, simulator hooks, or public `PhaseFieldFault` methods were introduced.

  The internal `PhaseFieldFaultTestAccess` seam gained wrappers for cohesive response evaluation, explicit initialization, explicit commit, and diagnostic inspection. This
  is test infrastructure, not a supported production interface.

  ## Equation and specification mapping

  ### Cohesive traction

  The implementation at source/material_model/phase_field_fault.cc:145 evaluates

  \[
  T^{\rm coh}_k = \frac{\kappa_k V_k+\beta_k I_{h,k-1}T^{\rm coh}_{k-1}}
  {I_{h,k}}.
  \]

  This corresponds to:

  - `doc/reconstructed_fault/specification.tex`:607
  - `doc/reconstructed_fault/current_design.md`:530

  ### Localization and exact history correction

  The same kernel computes
  \[
  \chi_k=\frac{h_k}{I_{h,k}},
  \]
  and
  \[
  \upsilon_k = \frac{h_k}{I_{h,k}}V_k +
  \frac{\beta_kT^{\rm coh}_{k-1}}{\kappa_k}
  \left(h_k\frac{I_{h,k-1}}{I_{h,k}}-h_{k-1}\right).
  \]
  The implementation preserves the two required identities:
  \[
  \int \upsilon_k\mathrm d\zeta=V_k,
  \qquad
  \frac{\partial\upsilon_k}{\partial V_k}
  =\frac{h_k}{I_{h,k}}.
  \]

  The quantitative checks are at `unit_tests/phase_field_fault_cohesive.cc`:48.

  ### Initial state

  The implementation at `source/material_model/phase_field_fault.cc`:338 constructs
  \[
  q=g(\phi_{\rm eff})\sqrt{2GH},
  \qquad
  \phi_{\rm eff}=\max(\phi_h,0).
  \]

  It then:

  1. Computes current \(I_{h,0}\).
  2. Evaluates the Q1 phase field at locally owned particle positions.
  3. Obtains \(H\) from the existing crack_driving_force property.
  4. Obtains particle composition through the explicit compositional-field mapping.
  5. Computes \(G\) through the material model’s existing averaging rule.
  6. Projects particle \(q\) to nodal \(T^{\rm coh}_0\).
  7. Validates all projected traction and \(I_h\) values.
  8. Commits both fields together.
  9. Reports transverse-profile residual diagnostics.

  This maps to doc/reconstructed_fault/specification.tex:658.

  ### Generic distributed projection

  The projection at source/reconstructed_fault.cc:1505 reuses the Stage C particle association and weighted Q1 mass operator. Only reduced fault-sized right-hand sides and
  residual statistics cross MPI ranks.

  This maps to the consistent weighted least-squares projection contract beginning at doc/reconstructed_fault/specification.tex:737.

  ### Lifecycle boundary

  The commit operation is private and explicitly unwired, matching doc/reconstructed_fault/specification.tex:682. No state is modified during cohesive response evaluation.

  ## Assumptions made

  The implementation relies on the following assumptions:

  - Stage D initialization is currently two-dimensional.
  - Reconstructed fault geometry already exists before cohesive initialization is invoked.
  - Every active fault degree of freedom has sufficient positive particle-domain support.
  - The phase-field-associated particle manager contains `crack_driving_force`.
  - Every chemical compositional field is particle-advected and has an explicit particle-property mapping.
  - The existing `viscosity_averaging` selection is also the authoritative averaging operation for elastic shear modulus.
  - \(V\) is a nonnegative slip-rate magnitude, not a signed tangential velocity.
  - \(T^{\rm coh}\) is stored as a nonnegative traction magnitude.
  - The profile-uniform-\(q\) approximation is applicable. Departures are reported, not rejected.
  - Cohesive response evaluation occurs only for a positive mechanical timestep, so \(\kappa>0\). The zero-timestep case is deliberately rejected by the response kernel.
  - Generic fault properties use ASPECT’s signaling-NaN sentinel. Safe sentinel recognition assumes an IEEE-754 64-bit double.
  - Fully initialized state found after checkpoint restoration is authoritative and must not be recomputed.

  ## Uncertainties and deferred concerns

  No blocking uncertainty was found, but these items remain open by design:

  - The production lifecycle point at which initial state is constructed is not connected. Stage D exposes the operation privately; a later coupling/lifecycle stage must invoke it after the accepted initial phase-field solve and before initial \(H\) ceases to represent the prescribed state.

  - Retrieval of \(h_{k-1}\) from the previous FE phase-field solution is not wired. The pure kernel accepts it explicitly.

  - There is no independent quantitative reference test for the complete particle-\(q\)-to-nodal-\(T^{\rm coh}_0\) initialization path. The lifecycle test establishes finite/
    nonnegative traction, matching committed \(I_h\), and finite diagnostics.

  - No rejection threshold exists for transverse \(q\) variation. This is intentional and follows the specification.

  - Three-dimensional faults and overlapping fault influence regions remain deferred.

  - The complete ASPECT integration suite was not completed, following the explicit instruction to stop it and not rerun it unless requested.

  ## Existing behavior affected

  When reconstructed faults and PhaseFieldFault are enabled:

  - Two additional generic scalar properties are registered:
      - phase field fault cohesive traction
      - phase field fault previous I h

  - These properties become part of reconstructed-fault checkpoint and generic visualization state.
  - Newly created fault vertices initially contain the generic signaling-NaN sentinel until cohesive initialization is explicitly performed.
  - Partially initialized or physically inadmissible cohesive state now produces an explicit failure.
  - The Stage D test postprocessor explicitly initializes cohesive state. Ordinary production runs do not yet call that operation.

  Unchanged behavior:

  - Reconstruction continues to use the activation threshold.
  - Physical phase-field bounds and upper admissibility remain separate from reconstruction activation.
  - The established Stage C (I_h) algorithm and fixed surface material mixture are unchanged.
  - Existing slip-rate trial/update/rollback behavior is unchanged.
  - No friction law or Stokes coupling was introduced.

  ## Tests and exact results

  - Normal DebugRelease build, cmake --build build-main -j4: passed.
  - Voro-enabled DebugRelease build, ASPECT_WITH_VORO=ON, -j4: passed.
  - Normal complete unit executable:
      - 61,390 assertions
      - 63 test cases
      - all passed.

  - Voro-enabled complete unit executable:
      - 61,413 assertions
      - 68 test cases
      - all passed.

  - Focused cohesive tests:
      - 13 assertions
      - 3 test cases
      - all passed.

  - Two-rank adaptive-(I_h) accuracy test:
      - rank-local totals of 2,323 and 59,117 assertions
      - both ranks passed.

  - Voro Stage C/D lifecycle fixture:
      - serial: passed, final run 143.46 s
      - two-rank MPI: passed, final run 55.17 s

  - specification.tex:
      - compiled successfully
      - 19-page PDF generated
      - overfull-box warnings only.

  - git diff --check: passed.

  The attempted complete integration run was terminated on request. Before termination, the Stage D serial and MPI tests passed, while an unrelated
  2d_annulus_pyvista_cookbook test showed a last-digit golden-output difference. The partial run is not counted as complete-suite verification.

  ## Diff stat and repository state

  10 files changed, 1054 insertions(+), 119 deletions(-)
