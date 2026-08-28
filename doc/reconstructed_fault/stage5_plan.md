  # Stage 5: Generic Particle-to-Fault Projection

  ## Summary

  Implement a 2-D consistent weighted least-squares projection from locally owned particle properties to registered reconstructed-fault vertex properties.

  The projection will use:

  \[
  M_{ab}=\sum_p m_pN_a(\xi_p)N_b(\xi_p),
  \qquad
  b_a=\sum_p m_pN_a(\xi_p)z_p,
  \]

  where \(m_p\) is the existing particle-domain volume. There is no distance-dependent kernel: \(K=1\) inside the influence region and zero outside through geometric
  admission.

  Particles belong to the normal-profile strip of at most one fault. Extensions beyond open fault tips are excluded, overlapping influence regions of different faults
  are errors, and intersections/branching/coalescence remain future work.

  ## Mathematical Requirements

  - For each particle and fault segment, compute the unconstrained orthogonal projection coordinate \(\xi\).
  - A segment is eligible only when \(0\le\xi\le1\); this excludes tangent extensions beyond open tips.
  - Interpolate the segment half-width using \(R(\xi)=(1-\xi)R_0+\xi R_1\).
  - Admit the particle when \(|r|\le R(\xi)\), where \(r\) is signed distance along the segment normal.
  - Within one fault, select the eligible segment with minimum \(|r|\), breaking exact ties by lower segment index.
  - If a particle is admitted by more than one fault, throw an explicit unsupported-overlap error.
  - Use Q1 functions \(N_0=1-\xi\), \(N_1=\xi\), and particle-domain volume as the complete weight.
  - Assemble one geometric matrix per fault and reuse its factorization for every scalar component and repeated projection call while the cache is valid.
  - Treat each registered component independently but pack all requested right-hand sides into one MPI reduction.

  ## Interfaces and Data Structures

  ### Public manager interface

  Add to ReconstructedFaultManager<dim>:

  struct ParticlePropertyProjection
  {
    std::string particle_property_name;
    unsigned int first_particle_component;
    std::string fault_property_name;
    unsigned int first_fault_component;
    unsigned int n_components;
  };

  void project_particle_properties(
    const std::vector<ParticlePropertyProjection> &projections);

  void invalidate_particle_projection_cache();

  const std::vector<ParticleProjectionDiagnostics> &
  get_particle_projection_diagnostics() const;

  Each mapping may project a scalar or contiguous component range. Validate source and destination ranges before assembly and reject duplicate destination components.

  The method will use PhaseFieldHandler::get_associated_particle_manager(), ensuring projection uses the same particle set as the phase-field workflow.

  ### Profile-derived influence widths

  During initial reconstruction:

  - Evaluate profile support only for distinct prescribed core values and cache it in a std::map<double,double>.
  - For each prescribed vertex, use the maximum support among its material-specific stationary profiles.
  - Interpolate these endpoint widths along the prescribed polyline when creating uniformly resampled reference vertices.
  - Retain one half-width per reconstructed vertex in manager-owned geometry metadata.
  - Fit geometry normally without changing the width-to-vertex correspondence.
  - Require positive finite widths and require width-vector size to match fault vertex count.

  These widths are geometric projection metadata, not generic fault properties.

  ### Minimal cache record

  Store one record for every locally owned particle, including inactive particles:

  struct ParticleProjectionCacheEntry
  {
    types::particle_index particle_id;
    Point<dim> position;
    double particle_domain_volume;

    bool active;
    unsigned int fault_index;
    unsigned int segment_index;
    double xi;
  };

  Do not cache shape values, signed distance, half-width, or geometry version per entry:

  - shape values are trivial functions of xi;
  - distance and half-width are needed only during association;
  - geometry versions belong to the cache as a whole.

  The cache also stores:

  - fault geometry versions;
  - one factorized projection system per fault;
  - weighted vertex-support diagnostics;
  - contributing-particle counts.

  ### Projection system

  Exploit the 2-D Q1 polyline structure: each fault matrix is symmetric tridiagonal.

  Store:

  - diagonal entries;
  - one off-diagonal;
  - weighted support per vertex;
  - reusable tridiagonal LDLᵀ factors.

  Use a small explicit tridiagonal factorization rather than a dense matrix or distributed sparse matrix. Fail if a support value or factorization pivot is below a
  scale-aware tolerance based on machine epsilon, matrix size, and maximum diagonal magnitude.

  ## Existing Infrastructure and Files

  Primary implementation:

  - include/aspect/reconstructed_fault.h
  - source/reconstructed_fault.cc
  - focused tests under unit_tests/

  Reuse without modifying its architecture:

  - PhaseFieldHandler::get_associated_particle_manager()
  - Particle::Manager::get_particle_handler()
  - Particle::Manager::get_particle_domain_handler()
  - ParticleDomainHandler::get_particle_domain(local_index).volume()
  - ParticlePropertyInformation for source field offsets/component counts
  - Utilities::MPI::sum() for packed matrix, support, and RHS reductions
  - ReconstructedFault::get_properties(vertex) for vertex-major destination storage
  - existing polyline closest-segment calculations, refactored into a shared private 2-D association routine

  Do not use CPDI weighting functions, ghost particles, the old PhaseFieldRSF projection code, or a new fault triangulation.

  ## Cache and MPI Behavior

  ### Cache construction

  1. Iterate ParticleHandler::begin()/end(), which covers locally owned particles only.
  2. Record every particle’s ID, position, and particle-domain volume.
  3. Search all fault segments using the isolated normal-profile association routine.
  4. Assemble local diagonal, off-diagonal, and support arrays from active entries.
  5. Pack all fault systems deterministically and perform one MPI sum.
  6. Factor identical reduced systems on every rank.
  7. Store geometry versions and diagnostics.

  ### Validation and invalidation

  Before reusing the cache, scan locally owned particles and compare:

  - particle count and iteration order;
  - particle IDs;
  - exact positions;
  - particle-domain volumes;
  - fault count and geometry versions;
  - half-width-vector sizes.

  Rebuild if any comparison changes. This detects advection, nonlinear restore/readvance, migration, insertion/removal, AMR, repartitioning, and particle-domain
  regeneration without relying on a missing post-advection signal.

  Particle property-value changes do not invalidate the geometric cache or matrix.

  Fault append operations invalidate through geometry_version(). Appended faults without corresponding half-width metadata produce an explicit unsupported-propagation
  error in Stage 5.

  ### Projection call

  1. Validate all mappings and resolve source/destination component offsets.
  2. Validate or rebuild the geometric cache.
  3. Iterate particles in the same order as cache entries.
  4. Assemble all requested local RHS vectors in component-major packed storage.
  5. Perform one MPI sum for the packed RHS data.
  6. Solve every fault/component using cached LDLᵀ factors.
  7. Write results to fault.get_properties(vertex)[destination_offset].

  No particle positions, meshes, or complete property arrays are gathered.

  ## Failure Cases

  Explicitly reject:

  - calls before reconstructed geometry exists;
  - 3-D projection;
  - missing particle domains;
  - non-finite/non-positive particle-domain volumes;
  - non-finite particle positions or projected source values;
  - zero-length fault segments;
  - invalid property names or component ranges;
  - duplicate destination components;
  - missing or invalid influence-width metadata;
  - one particle admitted by multiple faults;
  - unsupported fault intersections or overlapping influence bands;
  - zero or insufficient weighted support at any fault vertex;
  - singular or numerically non-positive projection systems;
  - geometry growth without corresponding profile-width metadata.

  Particles outside every normal-profile strip simply remain inactive.

  ## Test Plan and Implementation Sequence

  1. Geometry association
      - Refactor the analytic segment projection.
      - Add tests for interior projection, signed distance, varying linear widths, internal-segment selection, and exclusion beyond both open tips.
      - Test multi-fault overlap rejection.

  2. Profile-width metadata
      - Add distinct-core-value support lookup and prescribed-to-resampled width interpolation.
      - Test constant and varying core values, repeated-value cache reuse, and maximum support across materials.

  3. Projection cache and matrix
      - Build synthetic cache entries and tridiagonal systems.
      - Test exact Q1 matrix entries, nonuniform particle-domain volumes, constant/linear reproduction, tip support failures, and singular factorization detection.
      - Test invalidation after position, ID, volume, ownership/order, and geometry-version changes.

  4. Generic property mapping
      - Add explicit source/destination component mappings and packed RHS assembly.
      - Test scalar, vector, multiple source fields, destination offsets, invalid ranges, duplicate destinations, and preservation of unrelated fault components.

  5. MPI and integration
      - Add a two-rank test where each rank owns different contributing particles and verify identical projected values on all ranks.
      - Verify ghost particles do not contribute.
      - Run the reconstructed-fault example and confirm registered projected fields appear through the existing generic VTU output.
      - Update current_design.md with the settled Stage-5 rules.

  ## Current Conflicts and Optional Optimizations

  The older LaTeX specification assumes a fault triangulation and DoFHandler; the authoritative current implementation uses an ordered polyline. Stage 5 should therefore
  assemble directly by fault vertex and segment indices while preserving the same Q1 mathematics.

  The current source has no retained profile half-widths, so manager-owned width metadata is an implementation necessity. It does not alter fault geometry or
  constitutive state.

  Deferred optional optimizations:

  - segment AABB/R-tree acceleration;
  - nonblocking MPI reductions;
  - caching resolved property mappings;
  - generalized kernels;
  - intersections, branching, coalescence, or multi-fault weighting;
  - 3-D closest-surface projection;
  - automatic solver-loop coupling.
