  ## Proposed Header Layouts

  ### include/aspect/reconstructed_fault.h

  Keep the compact ReconstructedFault container and top-level utility namespaces ungrouped. Organize ReconstructedFaultManager into these ordered Doxygen groups:

  1. Construction and parameter handling
      - Constructors
      - declare_parameters()
      - parse_parameters()

  2. Fault initialization and reconstruction
      - Crack-driving-force initialization
      - Initial fault reconstruction
      - Complete-fault insertion

  3. Generic reconstructed-fault properties
      - PropertyInformation
      - Property registration and lookup

  4. Slip-rate nonlinear state
      - Initialization and access
      - Nonlinear-solve begin/commit/rollback
      - Trial begin/set/accept/rollback
      - Q1 slip-rate interpolation

  5. Particle-to-fault projection
      - Projection request/result/diagnostic types
      - Particle-property projection
      - Fault-property interpolation
      - Scalar projection
      - Cache invalidation and projection query

  6. Fault access and diagnostics
      - Fault accessors
      - Reconstruction and projection diagnostics

  7. Serialization and restart
      - Serialization friendship
      - save(), load(), split-member declaration
      - rebuild_after_deserialization()

  8. Projection-cache implementation
      - Cache-entry and factored-system types
      - Vertex-offset/reduction helpers
      - Cache validity and rebuild helpers
      - Reconstructible cache storage

  9. Persistent manager state
      - Reconstruction parameters and geometry
      - Generic property registry
      - Persistent and transient slip-rate state

  Existing short inline comments will remain as subdivisions inside the final state group; no Doxygen group will be introduced for each individual vector or helper.

  ### include/aspect/material_model/phase_field_fault.h

  Use these ordered Doxygen groups:

  1. Material-model interface
      - evaluate(), initialization, parameter declaration/parsing
      - Phase-field model accessors
      - Compressibility and critical-energy/force accessors

  2. Maxwell constitutive helpers
      - MaxwellCoefficients
      - Coefficient and stress evaluation

  3. Cohesive constitutive helpers
      - CohesiveResponse
      - Non-committing cohesive response evaluation

  4. Initial cohesive-state setup
      - Initial state construction
      - Particle-(q) evaluation
      - Explicit cohesive-state commit
      - Cohesive property indices and projection diagnostics

  5. Normalization-integral evaluation
      - Top-level recomputation
      - Effective-phase-field and singularity utilities
      - Surface-composition projection
      - Profile construction and distributed sampling
      - Q1 projection to the fault
      - Current normalization state and solution evaluator

  6. Adaptive normalization-profile integration
      - Profile/sample types and evaluator aliases
      - Adaptive integrator
      - Quadrature, tail, and undershoot tolerances

  7. Material parameters and state
      - Creep-viscosity helper
      - Equation of state and friction objects
      - Thermal, viscous, elastic, cohesive, and phase-field parameters

  ## Proposed Source Layouts

  ### source/reconstructed_fault.cc

  1. File-local helpers
      - Keep the anonymous-namespace helpers together.
      - Preserve dependency-required order while arranging them in broad geometry/reconstruction/projection clusters without individual banners.

  2. Prescribed-fault parsing and initialization
      - Parsing, closest-point/core-value evaluation, and particle-(H) initialization utilities.

  3. Fault reconstruction
      - Resampling and normal-offset utilities.
      - Manager construction/parameters, crack-driving initialization, reconstruction workflow, and fault insertion.

  4. ReconstructedFault implementation
      - Construction, geometry access, property access, append operations, and geometry version.

  5. Property registration
      - Registration and property lookup.

  6. Slip-rate nonlinear state
      - Initialization/access followed by solve and trial lifecycle operations.

  7. Particle-projection cache
      - Public tridiagonal utility, cache validation/rebuild, factor reuse, offsets, and reduced solves.

  8. Particle-to-fault projection
      - Normal-profile projection utility and manager query.
      - Particle-property projection, fault-property interpolation, scalar projection, and cache invalidation.

  9. Fault access and diagnostics
      - Fault and diagnostic accessors.

  10. Restart reconstruction and serialization support
      - rebuild_after_deserialization().

  11. Explicit instantiations
      - Existing instantiation macro block.

  ### source/material_model/phase_field_fault.cc

  1. File-local helper types
      - Cohesive-history helper and adaptive-normalization state/request types in one section.

  2. Maxwell constitutive law
      - Coefficients followed by stress evaluation.

  3. Material-model interface
      - evaluate() first, then its creep-viscosity helper, accessors, and initialize().

  4. Cohesive constitutive law
      - Non-committing cohesive response.

  5. Initial cohesive-state initialization
      - Top-level initialization first, followed by particle-(q) evaluation and commit helper.

  6. Normalization-integral evaluation
      - Top-level compute_normalization_integrals() workflow.

  7. Normalization phase-field utilities
      - Effective-(\phi), global undershoot validation, and (1/g-1) evaluation.

  8. Adaptive normalization-profile integration
      - integrate_normalization_profiles() as one uninterrupted algorithm.

  9. Surface composition and normalization-profile construction
      - Surface projection, owned-profile construction, and distributed point evaluation.

  10. Projection of normalization integrals
      - Consistent Q1 projection to replicated fault vertices.

  11. Material parameters and parsing
      - Parameter declaration followed by parsing.

  12. Material-model registration
      - Existing registration macro.

  ## Verification

  - Review the diff with moved-code detection and word-level comparison to ensure all executable changes are pure relocation.
  - Run git diff --check.
  - Build debug and release targets with -j4.
  - Run focused *Fault*, *I_h*, *cohesive*, and *Maxwell* unit-test filters.
  - Run the two-rank adaptive-(I_h) unit test.
  - Run the focused PhaseFieldFault integration tests and the Voro-enabled serial/two-rank full path.
  - Do not run the complete 1,289-test suite.

  ## Assumptions

  - Only the four requested production files will be reorganized; tests and documentation will not be edited.
  - Existing function-level Doxygen comments remain unless relocation makes a duplicate section-level navigation comment unnecessary.
  - ReconstructedFault remains ungrouped because it is already compact; the substantial Doxygen hierarchy is reserved for the two large classes.
  - No conflict exists with the authoritative design because this pass changes navigation only, not ownership or behavior.
