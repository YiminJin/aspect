 # Stage F review disposition

 Stage F is functionally implemented and all focused tests pass. However, I found two items that should be addressed before Stage F is considered fully reviewed:

  1. The assembled surface tests verify Jacobian self-consistency but do not independently verify the absolute mechanical, cohesive, friction, radiation, history, and pressure terms in \(R_\Gamma\). A shared error in both residual and Jacobian could pass.

  2. A failed `linearize_surface_system()` leaves a partially constructed surface_linearization object installed. The numerical exception is correct, but the assembler should publish the new linearization only after every fault block factors successfully.

  No Stage G, H, or I production functionality is present.

  ## Scope and design correspondence

  Implementation | Design equation or responsibility
  ---- | ----
   Non-committing point response | \(F=t-T^{\rm coh}-\mu\sigma_n-\eta^dV\)
   Dynamic-pressure mode | \(\sigma_n=p-\bm\tau:\bm N\)
   Adiabatic-pressure mode | \(\sigma_n=p_{\rm ad}(\bm x)\)
   Surface assembly | \((R_\Gamma)_i=\sum_p m_pN_iF_p\)
   Surface Jacobian | \(K_V=-\partial R_\Gamma/\partial V\)
   Q1 Jacobian coefficient | \(2\kappa\chi\bm S:\bm S+\kappa/I_h+\sigma_n\partial\mu/\partial V+\eta^d\)
   MPI ownership | Locally owned particle contributions followed by one packed sum producing replicated fault vectors
   Factorization | Per-fault sparse tridiagonal matrix factored by `SparseDirectUMFPACK`
   Trial evaluation | Explicit bulk state and fault-major \(V\), without changing committed/current/trial manager state
   Future solver signs | \(\begin{bmatrix}A&-B\\G&-K_V\end{bmatrix}\begin{bmatrix}\delta x\\ \delta V\end{bmatrix}=-\begin{bmatrix}R_{\rm bulk}\\R_\Gamma\end{bmatrix}\)
   Stage boundary | No bulk slip residual, \(B\), \(G\), condensation, active set, line search, or commit lifecycle

  The authoritative descriptions are in:

  - `doc/reconstructed_fault/current_design.md`:644
  - `doc/reconstructed_fault/specification.tex`:1071

  ## Files changed

  ### Authoritative documents

  - `doc/reconstructed_fault/current_design.md`:644
    Adds the F–I coupled-system architecture, Stage-F surface equations, normal-pressure modes, ownership, UMFPACK strategy, signs, and later-stage boundaries.

  - `doc/reconstructed_fault/specification.tex`:1071
    Adds the normative coupled-solver specification and revises the implementation-stage roadmap.

  ### Material and manager interfaces

  - `include/aspect/material_model/reconstructed_fault.h`:21
    Originally introduced the pointwise constitutive types and abstract
    capability. A post-review refactor removed this header and moved the types
    and operations directly into `PhaseFieldFault`.

  - `include/aspect/material_model/phase_field_fault.h`:46
    Makes PhaseFieldFault implement the constitutive capability; adds the pressure-mode setting and Theta property index.

  - `include/aspect/material_model/rheology/fault_friction.h`:132
    Removes the obsolete Vmax member.

  - `include/aspect/reconstructed_fault.h`:384
    Makes the particle/fault association record available to the dedicated assembler and exposes iteration-ordered cached associations.

  ### Material and manager implementations

  - `source/material_model/phase_field_fault.cc`:367
    Implements the non-committing Maxwell/cohesive/friction response, both pressure modes, (R_\Gamma) and (K_V) point coefficients, Theta registration, and constitutive-
    state validation.

  - `source/material_model/rheology/fault_friction.cc`:34
    Removes upper slip-rate clipping and the Maximum slip rate parameter. All friction operations now reject nonfinite or below-bound (V) and otherwise evaluate the
    supplied value.

  - `source/reconstructed_fault.cc`:1580
    Uses the public association type internally and implements the cache accessor without changing association or projection behavior.

  ### Dedicated Stage-F assembler

  - `include/aspect/simulator/assemblers/reconstructed_fault_stokes.h`:21
    Declares the surface residual representation and the three Stage-F assembler operations.

  - `source/simulator/assemblers/reconstructed_fault_stokes.cc`:37
    Implements bulk-field point evaluation, particle/Q1 assembly, MPI reduction, RMS diagnostics, tridiagonal sparse matrices, UMFPACK factorization, and inverse backward-
    error validation.

  ### Existing tests modified

  - `tests/phase_field_fault_friction.cc`:79
    Adds high-slip-rate cases above the removed historical upper clamp.

  - `unit_tests/phase_field_fault_maxwell.cc`:111
    Verifies that Maximum slip rate is absent and adiabatic fault pressure defaults to false.

  ### Shared Stage-F test implementation

  - `tests/phase_field_fault_surface_system.cc`:25
    Exercises rate-state initialization diagnostics, residual/linearization agreement, centered (K_V) finite differences, inverse recovery, MPI residual replication, indefinite coordinate forms, and non-commit behavior.

  ### Dynamic-pressure test

  - `tests/phase_field_fault_surface_dynamic_pressure.cc`
  - `tests/phase_field_fault_surface_dynamic_pressure.prm`
  - `tests/phase_field_fault_surface_dynamic_pressure.sh`
  - `tests/phase_field_fault_surface_dynamic_pressure/screen-output`

  These configure the dynamic-pressure, rate-state, positive-\(K_V\) path.

  ### Adiabatic-pressure MPI test

  - `tests/phase_field_fault_surface_adiabatic_pressure.cc`
  - `tests/phase_field_fault_surface_adiabatic_pressure.prm`
  - `tests/phase_field_fault_surface_adiabatic_pressure.sh`
  - `tests/phase_field_fault_surface_adiabatic_pressure/screen-output`

  These configure a controlled adiabatic pressure and run the Stage-F checks on two MPI ranks.

  ### Rate-dependent indefinite-\(K_V\) test

  - `tests/phase_field_fault_surface_rate_dependent.cc`
  - `tests/phase_field_fault_surface_rate_dependent.prm`
  - `tests/phase_field_fault_surface_rate_dependent.sh`
  - `tests/phase_field_fault_surface_rate_dependent/screen-output`

  These use spatially varying normal stress and velocity-weakening friction to exercise a nonsingular sign-indefinite \(K_V\).

  ### Singular-factorization test

  - `tests/phase_field_fault_surface_singular.cc`
  - `tests/phase_field_fault_surface_singular_system.cc:25`
  - `tests/phase_field_fault_surface_singular.prm`
  - `tests/phase_field_fault_surface_singular.sh`
  - `tests/phase_field_fault_surface_singular/screen-output`

  This is a test-only algebraic fault injection. It disables cached association support, confirms the production factorization diagnostic, and terminates as an expected failure.

  ### Unavailable-adiabatic-state test

  - `tests/phase_field_fault_surface_uninitialized_adiabatic.cc`
  - `tests/phase_field_fault_surface_uninitialized_adiabatic.prm`
  - `tests/phase_field_fault_surface_uninitialized_adiabatic.sh`
  - `tests/phase_field_fault_surface_uninitialized_adiabatic/screen-output`

  This registers a test-only adiabatic model whose `is_initialized()` remains false and verifies the configuration diagnostic.

  ## New or modified public interfaces

  ### Material capability after the post-review refactor
  ```
  template <int dim>
  class PhaseFieldFault
  {
    struct ReconstructedFaultPointInputs;
    struct ReconstructedFaultPointResponse;

    ReconstructedFaultPointResponse
    evaluate_reconstructed_fault_point(
      const ReconstructedFaultPointInputs &) const;

    double minimum_fault_slip_rate() const;

    void validate_reconstructed_fault_constitutive_state() const;
  };
  ```
  The solve-local surface helper checks for `PhaseFieldFault` once at
  construction and retains that concrete reference.

  ### Assembler
  ```
  struct ReconstructedFaultSurfaceResidual
  {
    std::vector<std::vector<double>> values;
    double weighted_rms;
    std::vector<double> per_fault_weighted_rms;
  };

  template <int dim>
  class ReconstructedFaultStokes
  {
    using FaultVector = std::vector<std::vector<double>>;

    ReconstructedFaultSurfaceResidual
    evaluate_surface_residual(const LinearAlgebra::BlockVector &,
                              const FaultVector &) const;

    const ReconstructedFaultSurfaceResidual &
    linearize_surface_system(const LinearAlgebra::BlockVector &,
                             const FaultVector &);

    void
    solve_surface_jacobian(const FaultVector &rhs,
                           FaultVector &solution) const;
  };
  ```
  ### Manager association access
  ```
  struct ParticleFaultAssociation
  {
    types::particle_index particle_id;
    Point<dim> position;
    double particle_domain_volume;
    bool active;
    unsigned int fault_index;
    unsigned int segment_index;
    double xi;
  };

  const std::vector<ParticleFaultAssociation> &
  get_locally_owned_particle_fault_associations();
  ```
  The returned reference is valid only until the projection cache is invalidated or rebuilt.
  
  ### Parameters
  
  Added to Material model/Phase field fault:
  ```
  Use adiabatic pressure in fault friction = false
  ```
  Removed from Fault friction:
  ```
  Maximum slip rate
  ```
  Existing friction method signatures did not change, but their admissibility semantics did.

  ## Invariants enforced

  - Surface assembly is currently 2-D only.
  - Fault geometry and current positive \(I_h\) must exist before evaluation.
  - Cohesive traction and previous \(I_h\) must be initialized.
  - Rate-state friction requires initialized, finite, positive nodal Theta.
  - Adiabatic-pressure mode requires initialized adiabatic conditions.
  - Every supplied fault slip vector must match the replicated fault geometry.
  - \(V\) must be finite and satisfy \(V\ge V_{\min}>0\).
  - Residual and Jacobian coefficients must be finite.
  - The particle-association cache must remain in locally owned particle iteration order.
  - `solve_surface_jacobian()` requires a preceding successful linearization.
  - The factorized solve must have a finite, near-roundoff scaled backward error.
  - Residual-only evaluation must not alter any fault or particle history.

  ## Assumptions not stated completely explicitly

  - Particle-domain volume is the surface projection weight \(m_p\) for both \(R_\Gamma\) and \(K_V\), matching the existing consistent Q1 particle projection.
  - Particle-local material fractions determine bulk Maxwell \(\eta\), \(G\), and \(\kappa\). Projected surface fractions determine \(g\), \(h\), and friction parameters.
  - Current and previous FE fields at a particle position use deal.II’s averaged point-evaluation semantics if a point lies on a cell interface.
  - At timestep zero, the parsed Initial time step is the constitutive Maxwell interval.
  - The manager’s projection-system validation guarantees every production fault has positive global particle support, so RMS denominators are positive.
  - A single particle belongs to at most one fault profile, as guaranteed by the current association model.
  - Replicated geometry ordering and identical packed MPI layouts remain stable for the duration of a surface linearization.
  - UMFPACK is available in configurations that activate the future coupled solver. Without it, surface linearization reports an explicit configuration failure.
  - Theta=2e4 in the rate-state tests is only a test fixture; it is not a proposed initialization rule.
  - The singular test’s mutation of cached associations is deliberately unsupported test-only behavior.

  ## Review findings and uncertainties

  ### 1. Incomplete independent residual oracle

  The tests compare centered finite differences of `evaluate_surface_residual()` with the Jacobian produced by the same constitutive path. This is valuable, but it cannot
  detect an error shared by the residual and derivative.

  In particular, the tests do not independently assert the expected absolute contribution from:

  - trial Maxwell shear traction;
  - cohesive traction;
  - profile-history correction;
  - radiation damping;
  - dynamic pressure;
  - deviatoric normal traction;
  - adiabatic pressure.

  Existing cohesive and friction tests independently verify their lower-level formulas, but the assembled (R_\Gamma) combination is not isolated term by term as requested
  in the Stage-F plan. This is the largest remaining test gap.

  ### 2. Pressure-mode tests are mainly self-consistency tests

  The dynamic and adiabatic tests exercise both branches and their (V)-derivatives, and unavailable adiabatic state is tested. However, neither test compares the assembled residual with a separate numerical reference. An implementation that used the same incorrect pressure in both (R_\Gamma) and (K_V) could potentially pass.

  The Stage-G finite differences of (G) will distinguish the pressure derivatives, but Stage F should still have an independent residual-value check for both modes.

  ### 3. Failed factorization leaves partially installed state

  In source/simulator/assemblers/reconstructed_fault_stokes.cc:318, surface_linearization is replaced before all fault blocks have successfully factored. If a later block fails, the exception is correct, but a caller that catches it can observe a non-null, partially populated linearization.

  The safer invariant is:

  1. assemble and factor into a local temporary;
  2. publish it to surface_linearization only after every block succeeds.

  ### 4. Positive-definite test is not a complete definiteness proof

  The rate-state fixture checks positive coordinate quadratic forms and successful inverse recovery. Positive diagonal coordinate forms alone do not prove that a symmetric tridiagonal matrix is positive definite. The constitutive coefficient is expected to be pointwise positive in this fixture, which mathematically implies the consistent Q1 mass matrix is positive definite under full support, but that property is not directly asserted by the test.

  ### 5. MPI \(K_V\) replication is verified indirectly

  The two-rank test compares replicated residual entries exactly and solves the Jacobian on every rank, recovering the same prescribed direction. It does not directly compare the assembled \(K_V\) coefficients or \(K_V\) actions across ranks because the matrix remains private. This is consistent with encapsulation, but the test claim
  should be understood as indirect.

  ### 6. Backward compatibility of old parameter files changes

  Any external parameter file that still sets Maximum slip rate will now fail parsing. Repository inspection found no remaining production use, and removal was explicitly approved, but this is an intentional input compatibility break.

  Older reconstructed-fault checkpoints also do not contain the newly registered generic Theta property. Backward checkpoint compatibility with pre-Stage-F archives has not been tested.

  ## Existing behavior changed

  - Slip rates greater than the old Vmax are no longer silently clipped.
  - Slip rates below (V_{\min}) are now rejected rather than silently raised to (V_{\min}).
  - compute_time_step() for rate-state friction uses the supplied admissible (V), not an upper-clipped value.
  - Rate-state reconstructed faults gain a generic one-component phase field fault state property.
  - Enabling adiabatic fault pressure changes only the frictional normal pressure; dynamic pressure and deviatoric normal traction are excluded from that friction term.
  - The default remains the previous dynamic-pressure formulation.
  - Calling surface assembly may rebuild and validate the manager’s particle-projection cache, but does not alter constitutive history.

  ## Exact test results

  ### Builds

  - cmake --build build-main -j4 — passed
  - cmake --build build-pf-cpdi -j4 — passed

  ### Stage-F tests

  - phase_field_fault_surface_dynamic_pressure — passed, 81.18 s
  - phase_field_fault_surface_adiabatic_pressure — passed with 2 MPI ranks, 49.91 s
  - phase_field_fault_surface_rate_dependent — passed, 81.68 s
  - phase_field_fault_surface_singular — passed, 80.62 s
  - phase_field_fault_surface_uninitialized_adiabatic — passed, 68.36 s

  The singular and unavailable-adiabatic tests are expected-failure tests that passed by observing their intended diagnostics.

  One initial adiabatic MPI invocation was denied access to local MPI interfaces by the sandbox. The same focused test was rerun with the existing scoped test permission
  and passed.

  ### Existing friction and Maxwell tests

  - phase_field_fault_rate_state — passed
  - phase_field_fault_rate_dependent — passed
  - Combined elapsed time — 19.67 s
  - Maxwell initialization and invalid-configuration group — 5/5 passed, 37.15 s
  - Stage-F parameter unit checks — 6 assertions in 2 cases, passed

  ### Existing cohesive and normalization tests

  - Cohesive unit tests — 14 assertions in 3 cases, passed
  - Adaptive (I_h) accuracy tests — 59,117 assertions in 2 cases, passed
  - phase_field_fault_ih — passed, 249.22 s
  - phase_field_fault_ih_mpi — passed with 2 ranks, 144.48 s
  - phase_field_fault_ih_no_composition — passed, 248.67 s

  The complete 1,289-test ASPECT integration suite was not run, in accordance with the earlier instruction.

  ## Diff stat

  Tracked files:

  10 files changed, 757 insertions(+), 60 deletions(-)

  New Stage-F files:

  25 files, 1,172 lines

  Effective Stage-F working-tree size:

  35 files, approximately 1,929 insertions(+), 60 deletions(-)

  The larger count includes test parameter files, output filters, expected-output files, and the two authoritative-document updates.
