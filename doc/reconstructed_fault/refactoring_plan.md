  # Reconstructed-Fault Code-Quality Refactor

  ## Summary

  Refactor the current implementation without changing its mathematics, numerical results, MPI ownership, checkpoint layout, or lifecycle semantics. No conflict was found between `refactoring.md`, `current_design.md`, `specification.tex`, and the implemented architecture.

  The main issues are oversized top-level algorithms, test machinery in a production header, sentinel knowledge leaking into the material model, scattered manager state, repeated projection plumbing, unnamed multi-value returns, and excessive validation of internally derived values.

  ## Refactoring passes

  ### 1. Property abstraction and header cleanup

  - Add one public query to ReconstructedFault:
    ```
    bool
    property_value_is_initialized(const unsigned int vertex_index,
                                  const unsigned int component_index) const;
    ```
    It returns false only for the exact signaling-NaN initialization sentinel. The representation-level comparison moves into ReconstructedFault; signaling-NaN storage and serialization remain unchanged.

  - Replace the IEEE-754 bit inspection in `phase_field_fault.cc` with this query.
  - Keep only the `PhaseFieldFaultTestAccess` forward declaration and friend declaration in the production header.
  - Move the complete test-access implementation to a shared testing-only header under `tests/`, used by the Stage B–D unit tests and the full-path \(I_h\) test.
  - Introduce no other public interfaces and do not expose additional material-model internals.

  ### 2. Reconstructed-fault manager structure

  - Replace the four-element structural-coordinate tuple with a named private StructuralCoordinates result containing distance, signed distance, segment, and \(\xi\).
  - Group private state that changes as a unit:
      - SlipRateState: committed, current, and trial fields; initialization flags; nonlinear/trial activity.
      - ParticleProjectionCacheState: validity and version information, particle associations, factored systems, and diagnostics.

    Continue serializing the existing persistent members individually and in the existing order; neither grouping changes the checkpoint archive.

  - Refactor `reconstruct_initial_faults()` around two private conceptual helpers:
      - `determine_initial_reconstruction_support()` computes the global cell margin, cached stationary-profile supports, reconstruction radii, and interpolated prescribed widths.

      - `reconstruct_initial_fault()` performs the distributed assembly and solve for one prescribed fault and returns named candidate geometry, half-widths, and diagnostics.

    The top-level function will visibly prepare support, reset prior state, reconstruct and commit each fault, then invalidate caches and mark initialization complete.

  - Refactor `rebuild_particle_projection_cache()` into the visible phases:
      - initialize fault-sized systems;
      - associate locally owned particles and assemble local Q1 matrices/support;
      - pack and reduce the fault-sized data;
      - validate support and factor the systems;
      - record cache versions.

  - Share only the genuinely common particle-projection mechanics:
      - one helper computes fault vertex offsets;
      - one helper reduces packed right-hand sides and solves each fault using the cached factors.

    Property projection and caller-computed scalar projection retain separate, straightforward local assembly because their input validation and output responsibilities
    differ.

  - Keep the two collective first-error propagation sites local. A cross-module error framework for two uses would add more complexity than it removes.

  ### 3. Phase-field material-model algorithms

  - Refactor `initialize_cohesive_state_from_initial_fields()` so its top level reads as:
      1. classify persistent cohesive history as wholly initialized or wholly uninitialized;
      2. compute current \(I_h\);
      3. sample local particle \(\phi_p\) and \(H_p\);
      4. interpolate the already projected surface compositions and compute the profile-uniform mixture;
      5. evaluate \(q_p=\bar g\sqrt{2\bar G H_p}\);
      6. project \(q_p\), commit \(T^{\rm coh}_0\) and \(I_{h,0}\), and report residual diagnostics.

  - Add a single private surface-mixture helper around `MaterialUtilities::compute_composition_fractions()`. Use it for both normalization profiles and initial \(q\),
    preserving the invariant that \(\bar g\) and \(\bar G\) use the same surface mixture.

  - Refactor `compute_normalization_integrals()` into named private operations:
      - project chemical composition fields to the fault;
      - construct the rank-owned surface quadrature profiles;
      - evaluate distributed bulk points;
      - run the adaptive two-sided integrations;
      - validate the globally worst raw phase-field undershoot;
      - consistently project profile integrals to fault vertices.

  - Keep `integrate_normalization_profiles()` as one coherent adaptive algorithm. Replace its anonymous state/request structures and nested geometry lambda with named private or file-local types and helpers, and add one introductory algorithm comment. Do not change panel initialization, Gauss rules, tolerances, boundary bisection, tail termination, owner ranges, or collective ordering.

  - Retain the callback seam used by the analytic kernel tests; do not introduce generic metaprogramming.

  ### 4. Assertion and comment audit

  Classify checks according to refactoring.md:

  - Preserve AssertThrow for parameter and file validation, public projection inputs, unsupported dimensions/topologies, overlapping faults, missing particle mappings, checkpoint corruption, insufficient support, singular systems, phase-field bounds, excessive undershoot, \(g\le0\), and inadmissible cohesive state.

  - Convert container-size, cache-version, indexing, state-machine, and helper-postcondition checks to debug `Assert`, `AssertDimension`, or `AssertIndexRange`.

  - Remove or reduce duplicate finiteness checks on values already guaranteed by validated inputs and a checked solver/helper. In particular, avoid revalidating the same projected/interpolated result at consecutive abstraction layers.

  - Retain overflow/non-singularity checks where positive inputs can still produce an unusable numerical result.

  - Replace line-by-line narration with comments explaining Q1 weighting, MPI collective participation, adaptive-profile termination, checkpoint reconstruction, and slip-rate state transitions.

  ## Interfaces and compatibility

  - The sole planned public API addition is ReconstructedFault::property_value_is_initialized(...).
  - Existing reconstruction, property registry, projection, interpolation, slip-rate, and fault-access signatures remain unchanged.
  - Preserve vertex-major property layout, property indices and component offsets, exact checkpoint field order, signaling-NaN initialization, geometry versions, projection weights, MPI packing order, and committed/current/trial slip-rate behavior.
  - Mutable `get_fault()` and public append operations remain as currently specified; topology-safe mutation is still deferred.

  ## Test plan

  Baseline already established:

  - Focused reconstructed-fault/material-model unit selection: 27 cases and 59,274 assertions passed.
  - Two-rank adaptive-\(I_h\) accuracy selection: both ranks passed both cases, with 59,117 and 2,323 assertions respectively.
  - Focused material-model integration tests: 4/4 passed.

  After each coherent pass:

  - Build with `cmake --build build-main -j4`.
  - Re-run the focused unit selection covering reconstruction, property storage, slip-rate lifecycle, checkpoint round-trip, projection, Maxwell behavior, cohesive response, and \(I_h\).

  - Re-run `phase_field_fault_ih_accuracy` with two MPI ranks.
  - Re-run the four focused Maxwell-stress/friction integration tests.
  - In a focused ASPECT_WITH_VORO=ON build, run only `phase_field_fault_ih` and `phase_field_fault_ih_mpi` to cover distributed Q1 sampling, surface mixtures, cohesive initialization, and MPI ownership.

  - Add unit coverage for the new sentinel query before assignment, after assignment, and after checkpoint round-trip.
  - Review the final diff explicitly for changed collective order, packing layout, Q1 weights, reconstruction normalization, adaptive-\(I_h\) decisions, cohesive commit atomicity, slip-rate transitions, property offsets, and serialized field order.

  The complete 1,289-test ASPECT suite will not be run as part of this refactor unless explicitly requested.

  ## Assumptions

  - The numerical behavior at commit 7d04f3fc4 is the reference behavior.
  - No scientific or numerical bug was identified during this review; anything that appears to require a behavioral correction during implementation will be reported before being changed.

  - The refactor remains limited to the four requested production files, their directly affected tests, and the new testing-only access header.
  - No new physics, coupling, propagation, projection scheme, parameter, or optimization is included.

