  ## Proposed layout

  include/aspect/
  ├── reconstructed_fault.h                  compatibility umbrella
  └── reconstructed_fault/
      ├── fault.h                            fault container and basic fault data
      ├── initialization.h                   prescribed-fault input utilities
      ├── utilities.h                        reusable numerical utilities
      ├── manager.h                          ReconstructedFaultManager and caches
      └── surface_system.h                   ReconstructedFaultSurfaceSystem

  source/reconstructed_fault/
  ├── fault.cc
  ├── initialization.cc
  ├── utilities.cc
  ├── manager.cc
  └── surface_system.cc

  The genuine bulk assembler will remain where it is:

  include/aspect/simulator/assemblers/reconstructed_fault_stokes.h
  source/simulator/assemblers/reconstructed_fault_stokes.cc

  I propose keeping two compatibility forwarding headers temporarily:

  - include/aspect/reconstructed_fault.h
  - include/aspect/simulator/reconstructed_fault_surface_system.h

  Production code and tests in this repository will switch to the narrow headers. The compatibility headers will prevent avoidable downstream include breakage but will not
  be used by new code.

  ## Header responsibilities

  ### fault.h

  This will contain only:

  - ReconstructedFault
  - the fault’s geometry, segments, distinguished slip-rate field, generic vertex-property storage, and related basic types
  - fault-local versioning and invariants

  It will not depend on SimulatorAccess, PhaseFieldHandler, or surface-system machinery.

  ### initialization.h

  A separate initialization header is justified. The existing prescribed-fault parser, input representation, and closest-point/core-profile initialization form a
  substantial and independently tested concern.

  It will contain:

  - PrescribedInitialFault
  - parse_prescribed_faults(...)
  - closest_point_distance_and_core_phase_field(...)

  It will not contain the simulator-wide crack-driving-force initialization workflow. That workflow requires broad simulator state and therefore belongs to
  ReconstructedFaultManager.

  ### utilities.h

  This will contain genuinely reusable numerical operations whose ownership is not tied to a manager instance:

  - resample_reference_fault(...)
  - solve_normal_offsets(...)
  - NormalProfileProjection
  - project_to_normal_profiles(...)
  - solve_tridiagonal_system(...)

  These routines are currently reused by tests or by code outside the manager. Manager-specific cache building and initialization algorithms will not be moved here merely
  to shorten manager.cc.

  ### manager.h

  This will contain:

  - ReconstructedFaultManager
  - FaultReconstructionDiagnostics
  - manager-owned public association and diagnostic types
  - particle and quadrature-point cache types
  - manager lifecycle, projection, reconstruction, and validation interfaces
  - manager-owned state and serialization

  The existing conceptual organization of cache state will be retained. I will not introduce new state-holder abstractions as part of this relocation.

  ### surface_system.h

  This will contain the existing simulator-side ReconstructedFaultSurfaceSystem:

  - (R_\Gamma) and (K_V) assembly
  - MPI reduction
  - (K_V) factorization and solves
  - pointwise calls into PhaseFieldFault

  It remains a simulator-side subsystem even though its logical filesystem home becomes aspect/reconstructed_fault/. It will not absorb the bulk Stokes assembler.

  ## Correcting the PhaseFieldHandler dependency

  The current code uses PhaseFieldHandler as a path to unrelated simulator objects in several initialization helpers. I will correct that boundary without changing the
  scientific algorithm.

  The public manager APIs will become:

  void initialize_crack_driving_force(
    const std::vector<PrescribedInitialFault> &faults);

  void reconstruct_initial_faults();

  The corresponding overloads taking PhaseFieldHandler & will be removed rather than retained as compatibility wrappers, because retaining them would preserve the
  architectural problem this refactor is intended to fix. All in-tree callers will be updated.

  Responsibility will be divided as follows:

  - PhaseFieldHandler will provide only phase-field-owned information and operations, such as phase-field profiles, stationary crack-driving-force evaluation, and its
    associated particle manager.

  - ReconstructedFaultManager, through its own SimulatorAccess, will obtain the material model, parameters, introspection data, DoF handler, solution, mapping, and MPI
    communicator.

  - A small file-local support-selection helper may receive a PhaseFieldHandler plus the specific DoF/MPI dependencies it requires. It will use the handler only for the
    phase-field profile.

  - The broad initial reconstruction workflow will become a private manager operation because it needs substantial simulator state. I will not express that state through a
    long parameter list or through another subsystem as a proxy.

  - The existing free ReconstructedFaultUtilities::initialize_crack_driving_force(...) will be removed; it is not genuinely generic.

  No repeated dynamic casts or new runtime polymorphic interfaces will be introduced.

  ## Implementation passes

  ### Pass 1: Split the fault, initialization, utilities, and manager units

  - Create the four narrow headers and matching source files.
  - Relocate declarations and definitions according to the responsibilities above.
  - Preserve function bodies and ordering within each semantic section where practical.
  - Replace in-tree umbrella includes with the narrowest sufficient includes.
  - Keep the umbrella forwarding header for compatibility.
  - Update explicit template instantiations without changing their set.
  - Confirm CMake’s recursive source discovery picks up the new directory.

  Verification:

  - Debug/Release compilation with -j4
  - focused reconstructed-fault unit tests, including prescribed initialization, phase-field sampling, and reconstruction/projection utilities
  - git diff --check

  ### Pass 2: Move the surface system

  - Move ReconstructedFaultSurfaceSystem to reconstructed_fault/surface_system.h and .cc.
  - Leave the old simulator header as a forwarding compatibility include.
  - Update all production and test consumers to include the new narrow header.
  - Keep Assemblers::ReconstructedFaultStokes under simulator/assemblers/.
  - Make no surface-system algorithm, matrix, MPI, or factorization changes.

  Verification:

  - Debug/Release compilation with -j4
  - the focused Stage F surface-system tests
  - the Stage G dynamic-pressure, adiabatic-pressure, and rate-dependent tests
  - git diff --check

  ### Pass 3: Remove PhaseFieldHandler proxy use

  - Move broad initialization/reconstruction orchestration into the manager.
  - Change the two manager interfaces described above.
  - Pass isolated unrelated dependencies explicitly only where a small file-local helper needs them.
  - Remove the simulator-dependent free initialization utility.
  - Update signal handlers, solver-scheme callers, unit tests, and integration tests.
  - Audit includes again so that the manager source, rather than public initialization utilities, bears the broad simulator dependencies.

  Verification:

  - Debug/Release compilation with -j4
  - prescribed-fault and phase-field-sampling unit tests
  - reconstructed-fault manager and projection unit tests
  - focused initialization/(I_h)/cohesive-history tests affected by reconstruction startup
  - Stage F/G surface and bulk coupling tests to detect lifecycle regressions
  - git diff --check

  I will inspect the available CTest names before running them and report the exact commands and results. I will not start the complete 1289-test integration suite.

  ## Invariants to preserve

  This refactor will preserve:

  - reconstructed-fault geometry and segment ordering
  - distributed ownership and MPI collective behavior
  - particle and quadrature-point association semantics
  - cache invalidation/version rules
  - generic vertex-property layout and sentinel behavior
  - slip-rate trial/update/rollback semantics
  - prescribed-fault parsing behavior
  - (I_h), cohesive-state, surface-residual, (K_V), and Stage G bulk-coupling algorithms
  - serialization and checkpoint field ordering
  - pressure conventions and constitutive evaluation
  - existing assembler accumulation semantics

  The only intended interface-level behavior change is that callers can no longer pass a PhaseFieldHandler as a generic simulator-state proxy.
