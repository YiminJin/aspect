## Implemented the four Stage 2 cleanup requests and committed them.

  Commit: 655b24c3d105d2776100e1c374f2d1db737add2f
  Reconstructed fault: simplify Maxwell stress lifecycle

  Changes:

  - MaxwellStress::initialize() now explicitly requires a nonempty Mapped particle properties map, preventing ASPECT’s implicit one-to-one fallback.
  - Added a two-rank failure regression test for an empty map.
  - Removed the local property-name constant and hard-coded "maxwell stress" at its two functional uses.
  - Moved MaxwellCoefficients, compute_maxwell_coefficients(), and compute_maxwell_stress() into PhaseFieldRSF as private static members.
  - Removed the entire MaxwellStressUpdateTransaction interface and implementation.
  - Updated doc/reconstructed_fault/current_design.md to specify the simplified lifecycle.

  There is no current lifecycle case that requires the transaction. Particle stress is read-only during Newton and line-search trials, so rejected trials and failed
  nonlinear solves never modify history. Once mechanical convergence is confirmed, one locally owned particle pass can calculate and write the new stress. An exception
  during that final pass is fatal to the timestep rather than a recoverable nonlinear rejection; the simulation does not continue from partially updated in-memory state.
  Consequently, retaining pending maps, commit validation, and rollback state added machinery without supporting a real recoverable path.

  Verification:

  - Debug build: passed.
  - Complete debug unit suite: 1/1 passed.
  - Release build: passed.
  - Release Maxwell property tests: 6 assertions in 1 test case passed.
  - Two-rank initialization test: passed.
  - Incomplete-component-map failure test: passed.
  - New empty-map failure test: passed.
  - The initialization integration test now exercises PhaseFieldRSF::evaluate() with exponent (-10^{-26}) and verifies (\kappa=G\Delta t=10^4), preserving direct coverage
    of the std::expm1 small-exponent behavior.
