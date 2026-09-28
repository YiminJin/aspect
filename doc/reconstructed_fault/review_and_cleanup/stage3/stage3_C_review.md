  # Stage C review packet

  Stage C is complete and committed. It implements the private, distributed adaptive \(I_h\) calculation and the subsequent bounded-negative-phase-field correction. No cohesive state, friction coupling, \(K_V\), Stokes coupling, or later-stage lifecycle integration was started.

  ## 1. Commits

  Stage C consists of two commits:

  1. d3ddcfbf84ce3efb0fdb5bb0a46dbffba681a922
     Reconstructed fault: add distributed adaptive I_h

  2. bae6c793e539e118972c88daec43f549a5a866f0
     Reconstructed fault: tolerate bounded I_h undershoot

  The second commit records the reviewed correction:

  \[
  \phi_{\mathrm{eff}}=\max(\phi_h,0)
  \]

  for bounded negative FE undershoots during \(I_h\) evaluation.

  Combined Stage C diff stat:

  22 files changed, 1244 insertions(+), 27 deletions(-)

  ## 2. Outcome and invariants

  - `MaterialModel::PhaseFieldFault` owns the current transient \(I_h\).
  - \(I_h\) remains private, recomputable state and is not checkpointed or connected to the mechanical solve.
  - Each `QGauss<1>(3)` reconstructed-fault quadrature point owns exactly one surface material mixture \(\mathbf f_\Gamma\).
  - That mixture is held fixed over the complete \(+\mathbf n/-\mathbf n\) normal profile.
  - Composition is never resampled along the normal.
  - Bulk phase-field samples are obtained collectively from the distributed Q1 solution.
  - Profile ownership is distributed in balanced, contiguous ranges; there is no rank-zero profile coordinator.
  - Both sides of every profile adapt and terminate independently.
  - The activation threshold does not truncate \(I_h\), and no monotonic-tail condition is required.
  - Completed profile integrals are projected to the replicated Q1 fault vertices using the consistent mass system.
  - Small negative FE undershoots use (\phi_{\mathrm{eff}}=\max(\phi_h,0)), never the activation threshold.
  - The global minimum raw sample is tracked across MPI ranks.
  - Raw undershoots below \(-10^{-4}\) fail with the minimum, tolerance, profile, side, coordinate, and physical-point diagnostics.
  - The upper end is never clipped:
      - raw \(\phi_h>1\) is an invariant failure;
      - \(g=0\), including at \(\phi_h=1\), is an explicit \(I_h\) singularity.

  - The physical range, activation threshold, and fault-model upper admissibility threshold are separate invariants.
  - Reconstruction and refinement continue to use the activation threshold exactly as before.
  - Stage C remains two-dimensional.

  ## 3. Files changed

   File | Stage C change and invariant
   ---- | ----
   `cmake/AspectConfig.cmake.in` | Exports `ASPECT_WITH_VORO` to downstream/test CMake projects so Voro-dependent tests are gated against the actual build configuration.
   `cmake/write_config.cmake` | Reports the Voro feature state in the detailed exported configuration.
   `doc/reconstructed_fault/current_design.md` | Records the final Stage C ownership, fixed-mixture, distributed-profile, adaptive-integration, physical-range, bounded-undershoot, andapplicability invariants.
   `doc/reconstructed_fault/specification.tex` | Makes the final \(I_h\), MPI, fixed-\(\mathbf f_\Gamma\), and \(\phi_{\mathrm{eff}}\) rules authoritative.
   `include/aspect/material_model/phase_field_fault.h` | Declares private transient \(I_h\) state and operations, numerical tolerances, projected-composition property index, raw-minimum tracking, and the narrow internal test seam.
   `include/aspect/phase_field.h` | Separates the physical range from activation and upper-admissibility thresholds and exposes the existing phase-field length scale through `PhaseFieldHandler`.
   `include/aspect/reconstructed_fault.h` | Adds a geometry-only manager query for association with reconstructed-fault normal profiles.
   `source/material_model/phase_field_fault.cc` | Implements fixed surface mixtures, distributed point evaluation, adaptive two-sided integration, domain-boundary handling, overlap diagnostics, mass projection, phase-field singularity checks, and bounded-undershoot handling.
   `source/mesh_refinement/phase_field.cc` | Migrates refinement from the formerly overloaded range accessor to the explicit activation-threshold accessor, preserving its behavior.
   `source/reconstructed_fault.cc` | Migrates reconstruction and prescribed-core validation to the correct threshold accessors and implements the manager-owned normal-profile query.
   `source/simulator/phase_field.cc` | Defines physical range [0,1], generic threshold defaults, the length-scale forwarding accessor, and correct threshold use in legacy slip-rate normalization.
   `tests/CMakeLists.txt` | Adds `ASPECT_WITH_VORO` as a recognized integration-test feature gate.
   `tests/phase_field_fault_ih.cc` | Adds the serial full-path verifier for finite positive nodal \(I_h\) and confirms that the test exercises a bounded negative raw phase-field sample.
   `tests/phase_field_fault_ih.prm` | Defines the heterogeneous particle-composition and reconstructed-fault test; final configuration uses AT1 with \(\ell/h=10\).
   `tests/phase_field_fault_ih.sh` | Restricts comparison output to the Stage C verification result.
   `tests/phase_field_fault_ih.txt` | Supplies the prescribed initial fault and its user-specified core phase-field values.
   `tests/phase_field_fault_ih/screen-output` | Records the expected serial Stage C result.
   `tests/phase_field_fault_ih_mpi.cc` | Reuses the Stage C verifier in the two-rank configuration.
   `tests/phase_field_fault_ih_mpi.prm` | Runs the same physical problem with two MPI ranks.
   tests/phase_field_fault_ih_mpi.sh | Restricts the MPI comparison to the Stage C verification result.
   `tests/phase_field_fault_ih_mpi/screen-output` | Records the expected two-rank Stage C result.
   `unit_tests/phase_field_fault_maxwell.cc` | Tests range/threshold separation, zero-tail behavior, bounded negative clamping, excessive-undershoot rejection, the unclipped upper endpoint, and explicit \(I_h\) singularities.
   

  ## 4. Public interfaces

  ### Modified semantics
```
  MaterialModel::PhaseFieldModel<dim>::get_phase_field_range() const
```
  This existing interface now always represents the physical phase-field range: [0,1]. It no longer doubles as the activation/admissibility interval.

  ### New public interfaces
```
  virtual double
  MaterialModel::PhaseFieldModel<dim>::
  get_phase_field_activation_threshold() const;`
```
  Generic default: 0.01.
```
  virtual double
  MaterialModel::PhaseFieldModel<dim>::
  get_phase_field_upper_admissibility_threshold() const;
```
  Generic default: 0.99.

  `MaterialModel::PhaseFieldFault` overrides both relevant threshold accessors:

  - activation threshold: configured material-model value;
  - upper admissibility threshold: 0.99.
```
  double
  PhaseFieldHandler<dim>::get_length_scale() const;
```
  This forwards the existing GeometricFunction length scale. It does not introduce another (\ell) parameter.
```
  ReconstructedFaultUtilities::NormalProfileProjection
  ReconstructedFaultManager<dim>::
  project_to_normal_profiles(const Point<dim> &position) const;
```
  This exposes only geometry association. It does not expose constitutive state or projection widths.
```
  void
  MaterialModel::PhaseFieldFault<dim>::initialize() override;
```
  The override registers material-owned generic fault storage for projected chemical components when reconstruction is enabled.

  ### New user parameters

  Under Material model / Phase field fault:
```
  I h quadrature tolerance = 1e-8
  I h tail tolerance       = 1e-8
```
  Both must be finite and positive.

  ### Exported build interface

  `ASPECT_WITH_VORO` is now included in AspectConfig.cmake, allowing the separate integration-test project to recognize a Voro-enabled ASPECT build.

  ### Deliberately not public

  The following remain private or test-only:

  - compute_normalization_integrals();
  - current nodal (I_h);
  - minimum raw sampled phase field;
  - effective-phase-field clamping;
  - raw-minimum validation;
  - normalization-integrand validation.

  No Simulator lifecycle hook or solver-facing (I_h) getter was introduced.

  ## 5. Design and equation correspondence

   Implementation invariant | Authoritative design correspondence
   ---- | ----
   \(I_h=\int(1/\bar g-1),d\zeta\) | `current_design.md`, §20; `specification.tex`, Distributed phase-field evaluation
   \(\bar g(\phi,\mathbf f_\Gamma)=\sum_m f_{\Gamma,m}g_m(\phi)\) | Normalization-profile material-mixture subsection
   One fixed \(\mathbf f_\Gamma\) per surface quadrature point |      Stage C fixed-mixture rule; composition is projected once and not sampled along the normal
   \(L_{\rm mat}\gg\ell\) | Explicit applicability condition in both authoritative documents
   Distributed profile ownership | Balanced contiguous deterministic profile-ID ranges required by the MPI design
   Collective distributed Q1 evaluation | Distributed phase-field evaluation requirement; no bulk field or node cloud is gathered
   \(\Delta\zeta_0=\frac12\min(\ell,h_{\rm local})\) | Mesh-aware initial-panel requirement
   Four-/eight-point Gauss comparison | Stage C adaptive quadrature requirement
   Two independent negligible outer windows | Non-monotonic-tail termination rule
   Separate \(+\mathbf n/-\mathbf n\) state | Two-sided normalization-profile requirement
   Consistent Q1 mass projection | Projection of profile-quadrature values to replicated fault vertices
   \(\phi_{\mathrm{eff}}=\max(\phi_h,0)\) | Reviewed bounded-negative-FE-undershoot amendment
   Global raw minimum and \(-10^{-4}\) guard | Reviewed conservative numerical-undershoot invariant
   No upper clipping; \(g=0\) is singular | Explicit separation of physical-range checking from constitutive \(g>0\)
   Physical range versus activation/admissibility thresholds | Existing-interface section of `specification.tex` and range semantics in current_design.md

  No equation for cohesive traction, friction residuals, \(K_V\), \(B\), \(G\), or the condensed Stokes operator is implemented in Stage C.

  ## 6. Assumptions made

  - Stage C remains limited to `dim == 2`, matching current reconstructed-fault geometry support.
  - The reconstructed fault is small and replicated; the bulk mesh, phase field, and particles remain distributed.
  - `QGauss<1>(3)` is the authoritative surface quadrature for current \(I_h\).
  - Every chemical compositional field used by \(I_h\) is particle-advected and explicitly mapped to a particle property.
  - With no chemical compositional fields, the surface mixture is the background-only mixture \((1)\).
  - ASPECT’s existing composition-fraction conversion defines \(\mathbf f_\Gamma\); Stage C does not invent another mixing rule.
  - Material transitions satisfy \(L_{\rm mat}\gg\ell\). Results are not claimed valid for transitions comparable to or narrower than \(\ell\).
  - The upper admissibility threshold remains 0.99; no separate user parameter was introduced.
  - The quadrature and tail tolerances default to \(10^{-8}\).
  - Boundary bisection, panel-refinement depth, and outward-extension limits are internal safety guards rather than user parameters.
  - The \(10^{-4}\) raw undershoot tolerance is internal and dimensionless. It was selected to accept the resolved AT1 result while clearly rejecting the observed under-
    resolved \(O(10^{-3})\) defect.
  - Current \(I_h\) can be discarded and recomputed; persistence and committed/previous-\(I_h\) history belong to later stages.
  - The private test seam is acceptable until a later mechanical stage provides a mathematically necessary production interface.
  - Encountering another reconstructed fault before profile-tail completion remains unsupported and is diagnosed rather than silently mixed.

  ## 7. Uncertainties and review notes

  - The \(10^{-4}\) undershoot tolerance is empirically conservative, not the result of a formal discretization-error bound. The resolved AT1 test produced a global minimum near \(-3.56\times10^{-5}\); the earlier under-resolved case was \(O(10^{-3})\).

  - The full-path test uses a test-local activation threshold of \(10^{-6}\) so reconstruction reaches \(I_h\). With the default 0.1, the test phase solve produced no reconstruction weight.

  - Additional AT1 nonlinear iterations became poorly conditioned under the low test activation threshold. The test therefore retains the normal phase-solver iteration policy.

  - Increasing CZM resolution from \(\ell/h=10\) to \(20\) did not materially reduce its small raw undershoot, indicating that nonlinear/particle-domain discretization effects also contribute.

  - The integration test verifies finite positive \(I_h\), bounded raw undershoot handling, and one-/two-rank completion. It does not yet provide a closed-form numerical \(I_h\) comparison or a systematic convergence study.

  - Three-dimensional reconstructed-fault profiles were not implemented or tested.
  - Multiple nearby faults are rejected; no overlap-combination model has been specified.
  - The complete 1,289-test ASPECT integration suite was not run.
  - The Voro-enabled build emitted existing unused-parameter warnings in the 3-D particle-domain implementation. They were non-fatal and unrelated to Stage C.
  - Out-of-tree callers that previously interpreted `get_phase_field_range()` as [activation, upper admissibility] must migrate to the new accessors.

  ## 8. Existing behavior changed

  - `get_phase_field_range()` now returns [0,1], not [0.01,0.99].
  - Existing in-tree consumers were migrated by invariant:
      - reconstruction and mesh refinement use the activation threshold;
      - prescribed-core validation and legacy slip normalization use activation and upper-admissibility thresholds.

  - Small negative raw phase-field samples no longer immediately invalidate \(I_h\). They are mapped to zero only if the global undershoot remains within \(10^{-4}\).
  - Raw upper overshoots are still rejected without clipping.
  - \(g=0\) is now specifically reported as an \(I_h\) singularity.
  - Enabling reconstructed-fault support with chemical fields registers material-owned generic fault storage for those projected components.
  - Voro-gated tests now depend on the actual exported `ASPECT_WITH_VORO` setting.
  - Reconstruction and refinement activation behavior is otherwise unchanged.
  - No Stokes assembly, constitutive history, friction, or slip-rate behavior is changed by Stage C.

  ## 9. Tests and exact results

  ### Normal build

  cmake --build build-main -j4

  Result: passed for Debug and Release targets.

  ### Voro-enabled build

  Configured with:
```
  cmake -S . -B /tmp/aspect-voro-stage-c \
    -DDEAL_II_DIR=/opt/dealii/9.6-local \
    -DASPECT_WITH_VORO=ON \
    -DVORO_DIR=/home/ein/local/voro++/0.4.6 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DASPECT_WITH_NETCDF=OFF \
    -DASPECT_WITH_PYTHON=OFF
```
  Built with:

  cmake --build /tmp/aspect-voro-stage-c -j4

  Result: passed.

  Without the explicit `VORO_DIR`, configuration reported that Voro was not found and silently disabled `ASPECT_WITH_VORO`.

  ### Unit suite

  cd build-main/unit_tests
  ../aspect --test

  Result:

  All tests passed (2256 assertions in 57 test cases)

  The Stage C unit coverage includes:

  - physical range [0,1];
  - separate activation and upper-admissibility defaults;
  - \(\phi_h=0\);
  - bounded negative clamping to zero;
  - excessive negative-undershoot rejection;
  - unchanged \(\phi_h=1\) upper endpoint;
  - explicit \(g=0\) singularity;
  - upper overshoot rejection.

  ### Existing PhaseFieldFault integration regressions
```
  ctest --test-dir build-main/tests \
    -R '^phase_field_fault_' \
    --output-on-failure
```
  Result:
```
  4/4 passed
  Total test time: 48.06 sec
```
  Covered:

  - Maxwell-stress initialization;
  - empty mapping behavior;
  - incomplete mapping rejection;
  - unsupported friction-law rejection.

  ### Serial Voro \(I_h\)
```
  ctest --test-dir /tmp/aspect-voro-stage-c/tests \
    -R '^phase_field_fault_ih$' -V
```
  Final result:
```
  1/1 passed
```
  The complete numerical execution reached the Stage C verifier successfully. An intermediate CTest result failed only because the expected statistics-table spacing
  differed; after updating the golden output, the final test passed.

  Final AT1 resolution:

  \[
  \ell=0.15625,\qquad h=\frac1{64},\qquad \frac{\ell}{h}=10.
  \]

  Observed global raw minimum:

  \[
  \min\phi_h\approx-3.56\times10^{-5},
  \]

  which is accepted by the \(10^{-4}\) guard and evaluated with \(\phi_{\mathrm{eff}}=0\).

  ### Two-rank Voro (I_h)
```
  ctest --test-dir /tmp/aspect-voro-stage-c/tests \
    -R '^phase_field_fault_ih_mpi$' -V
```
  Result outside the network sandbox:
```
  1/1 passed
  Test time: 35.52 sec
  Total time: 35.54 sec
```
  The first sandboxed attempt could not start MPI because Open MPI was denied access to local network interfaces. That was an environment failure before ASPECT execution, not a test failure.

  ## 10. Stage boundary

  Stage C stops with private, transient nodal \(I_h\) available inside `MaterialModel::PhaseFieldFault`.

  The following remain deferred:

  - previous/committed \(I_h\);
  - cohesive traction or cohesive history;
  - \(\Theta\);
  - friction residuals;
  - \(K_V\);
  - \(B\) and \(G\);
  - condensed Stokes coupling;
  - nonlinear trial/rollback integration;
  - checkpoint persistence for constitutive state;
  - three-dimensional reconstructed faults.
