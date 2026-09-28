  ## Stage 2

  Commit: 5ad66524926a0c3aca1f616aef631cf1093b477b
  Reconstructed fault: add Stage 2 Maxwell stress history

  Key results:

  - Added the MaxwellStress particle-property plugin, storing exactly one symmetric stress tensor per particle.
  - Initial stress is entirely user-controlled through explicitly mapped particle-advected compositional fields of type stress, evaluated through the initial composition
    model.

  - Missing, duplicate, incorrectly typed, or out-of-range mappings are rejected.
  - Added the non-rotational discrete Maxwell law to MaterialModel::PhaseFieldRSF:
    \[
    \beta=\exp(-\Delta tG/\eta),\qquad
    \kappa=-\eta\operatorname{expm1}(-\Delta tG/\eta),
    \]
    \[
    \tau_k=2\kappa\dot\epsilon_k^b+\beta\tau_{k-1}.
    \]

  - `std::expm1` is used directly for \(\kappa\), avoiding cancellation when (\Delta tG/\eta) is very small.
  - `PhaseFieldRSF` now returns \(\kappa\) as its effective Stokes viscosity.
  - Added material-owned stress-history transactions. Staging never changes committed particles; commit validates all MPI ranks before any rank writes; rollback only
    discards pending data.

  - No objective stress rotation and no MaterialModel::Rheology::Elasticity usage were introduced.
  - No new Simulator lifecycle hook was added. The transaction remains intentionally disconnected until the later stage that supplies the converged slip-corrected strain
    rate.

  - Corrected conversion of the PhaseFieldRSF initial timestep from years to seconds and added finite/positive parameter validation.
  - Updated doc/reconstructed_fault/current_design.md with the settled Stage 2 invariants.

  Verification:

  - Debug build: passed.
  - Complete debug unit suite: 1/1 passed.
  - Focused two-rank Maxwell unit tests: 37 assertions in 5 test cases passed independently on both ranks.
  - Positive two-rank integration test: passed; verified mapped initial-composition values on particles.
  - Negative integration test: passed; verified incomplete tensor mappings are rejected.
  - Release build: passed.
  - Release Maxwell tests: 37 assertions in 5 test cases passed.
  - Isolated commit stat: 12 files changed, 935 insertions, 20 deletions.

  The existing untracked design and review documents were preserved and excluded from the commit.
