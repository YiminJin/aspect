  ## Outcome and invariants

  - `MaterialModel::PhaseFieldRSF` is now `MaterialModel::PhaseFieldFault`.
  - `Rheology::RateStateFriction` is now `Rheology::FaultFriction`.
  - `PhaseFieldFault` continues to own the Maxwell constitutive operations and owns the selected fault-friction object.
  - `ReconstructedFaultManager` remains geometry/kinematics-only.
  - The current rate-and-state equations and Maxwell equations are numerically unchanged.
  - The friction-law selector defaults to rate state.
  - Selecting rate dependent fails explicitly because that law belongs to Stage E; it cannot silently use rate-and-state behavior.
  - No \(I_h\) storage, integration, MPI communication, parameters, or tests were introduced.

  ## Files changed

   File | Change
   ---- | ----
   `cookbooks/reconstructed_fault/reconstructed_fault.prm` | Uses the renamed material plugin and subsection, and explicitly selects the stateful rate-and-state law.
   `doc/reconstructed_fault/current_design.md` | Records the new authority hierarchy and settled PhaseFieldFault/FaultFriction ownership names.
   `doc/reconstructed_fault/specification.tex` |                 Synchronizes the reusable friction interface and distinguishes the deprecated implementation snapshots under tmp/.
   `include/aspect/material_model/phase_field_fault.h` | Renamed from `phase_field_rsf.h`; exposes the generic phase-field-fault material model while keeping Maxwell operations private.
   `include/aspect/material_model/rheology/fault_friction.h` | Renamed from rate_state_friction.h; defines the generic state-aware friction interface and internal law selector.
   `source/material_model/phase_field_fault.cc` | Renamed implementation and plugin registration; preserves the existing Maxwell and material behavior.
   `source/material_model/rheology/fault_friction.cc` | Preserves the existing rate-and-state formulas, parses the selector, and rejects the deferred law.
   `source/particle/property/maxwell_stress.cc` | Enforces that maxwell stress is used with PhaseFieldFault.
   `tests/phase_field_fault_maxwell_stress_initialization.cc` | Renamed Stage B initialization verifier and updated its model diagnostic.
   `tests/phase_field_fault_maxwell_stress_initialization.prm` | Uses the new plugin/subsection and explicitly selects rate state.

   tests/                                   Updated the expected shared-library name.
   phase_field_fault_maxwell_stress_init
   ialization/screen-output
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Renamed the empty-mapping failure test plugin.
   phase_field_fault_maxwell_stress_init
   ialization_empty_map.cc
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Includes the renamed base test.
   phase_field_fault_maxwell_stress_init
   ialization_empty_map.prm
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Updated the expected shared-library name.
   phase_field_fault_maxwell_stress_init
   ialization_empty_map/screen-output
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Renamed the incomplete-mapping failure test plugin.
   phase_field_fault_maxwell_stress_init
   ialization_fail.cc
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Includes the renamed base test.
   phase_field_fault_maxwell_stress_init
   ialization_fail.prm
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Updated the expected shared-library name.
   phase_field_fault_maxwell_stress_init
   ialization_fail/screen-output
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Adds an integration-test plugin for the deferred-law guard.
   phase_field_fault_unsupported_frictio
   n_law.cc
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Verifies that rate dependent is rejected rather than substituted.
   phase_field_fault_unsupported_frictio
   n_law.prm
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Normalizes insignificant trailing blanks in the failure diagnostic.
   phase_field_fault_unsupported_frictio
   n_law.sh
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   tests/                                   Records the expected Stage E deferral diagnostic.
   phase_field_fault_unsupported_frictio
   n_law/screen-output
  ───────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   unit_tests/                              Renamed the Maxwell property unit test and added selector-default/state-ownership checks.
   phase_field_fault_maxwell.cc

  The old phase_field_rsf.* and rate_state_friction.* production paths were removed by these renames. The deprecated snapshots under tmp/ were deliberately left unchanged.

  ## Public-interface changes

  Replaced interfaces:

  - Header:
      - `aspect/material_model/phase_field_rsf.h`
      - → `aspect/material_model/phase_field_fault.h`

  - C++ class:
      - `MaterialModel::PhaseFieldRSF<dim>`
      - → `MaterialModel::PhaseFieldFault<dim>`

  - Material plugin:
      - `phase field rsf`
      - → `phase field fault`

  - Parameter subsection:
      - `Material model / Phase field RSF`
      - → `Material model / Phase field fault`

  - Header:
      - `aspect/material_model/rheology/rate_state_friction.h`
      - → `aspect/material_model/rheology/fault_friction.h`

  - C++ class:
      - MaterialModel::Rheology::RateStateFriction<dim>
      - → MaterialModel::Rheology::FaultFriction<dim>

  - State update:
      - slip_state(V, old_theta, dt)
      - → update_state(V, old_state, dt)

  New public interface:

  bool FaultFriction<dim>::has_state_variable() const;

  New parameter:

  Friction law = rate state | rate dependent

  Its default is rate state. The rate dependent value is syntactically recognized but rejected during parsing until Stage E.

  Preserved under the renamed class:

  - friction_coefficient(...)
  - friction_coefficient_derivative_wrt_slip_rate(...)
  - compute_time_step(...)
  - get_reference_slip_rate()
  - get_minimum_slip_rate()
  - get_characteristic_slip_distance()
  - parameter declaration and parsing for the existing rate-and-state quantities

  The Maxwell coefficient and stress operations remain private to PhaseFieldFault; no solver or Simulator lifecycle interface was added.

  ## Design and equation correspondence

  - `phase_field_fault_redesign.md`, §1 and §2.2:
      - establishes `PhaseFieldFault` as the common fault-mechanics owner;
      - implemented by the material-model rename and private `FaultFriction` member.

  - §3.2:
      - requires generalizing `RateStateFriction` to `FaultFriction`;
      - implemented by the class/file rename and internal `FrictionLaw` enum.

  - §3.3:
      - requires a small state-aware friction API;
      - implemented by `has_state_variable()` and `update_state(...)`.

  - §21, Stage A:
      - all four Stage A requirements are covered: both renames, unchanged rate-state mechanics, selector boundary, and regression tests.

  - `pf_rsf.tex`, equations `eq:rsf` and `eq:aging`:
      - the logarithmic/regularized friction calculations and exact aging-law integration are preserved.

  - `pf_rsf.tex`, equations `eq:maxwell` and `eq:tau`:
      - the private non-rotational Maxwell implementation remains unchanged apart from its owning class name.

  - phase_field_fault_redesign.md, Stage C:
      - intentionally untouched.

  ## Assumptions

  - The requested “Separate Stage A first” review gate meant this commit must stop before Stage C.
  - A full compatibility alias for the old plugin, subsection, C++ classes, or include paths was not required; the authoritative document requests an actual rename.
  - The parameter spellings rate state and rate dependent are the user-facing forms of the enum values rate_state and rate_dependent.
  - Existing friction parameters remain directly within Material model / Phase field fault; adding another subsection would be unnecessary parameter-tree churn.
  - Stage A should establish the selector boundary but must not implement the rate-dependent equation assigned to Stage E.
  - The existing state-shaped friction-coefficient interface remains appropriate while rate-and-state is the only executable law. Stateless overloads belong with the
    actual Stage E implementation.

  ## Existing behavior changed

  - Parameter files using phase field rsf or Phase field RSF must be updated.
  - External C++ code including the old headers or naming the old classes must be updated.
  - External callers of slip_state(...) must use update_state(...).
  - MaxwellStress now diagnoses phase field fault as its required material model.
  - rate dependent is now a recognized configuration choice but deliberately terminates with a specific “reserved for Stage E” diagnostic.
  - Rate-and-state numerical results, Maxwell coefficients, Maxwell stress, particle initialization, and mapping validation are unchanged.

  ## Tests and exact results

  - Debug build:
      - cmake --build build-main --target aspect.exe.debug -j2
      - Passed.

  - Full unit-test executable:
      - cmake --build build-main --target run_unit_tests -j2
      - All tests passed (2245 assertions in 55 test cases).

  - Release build:
      - cmake --build build-main --target aspect.exe.release -j2
      - Passed.

  - Final post-commit Stage A integration set:
      - ctest --test-dir build-main/tests --output-on-failure -R '^phase_field_fault_(maxwell_stress_initialization|unsupported_friction_law)'
      - 4/4 passed.
      - Total time: 22.71 sec.

  The complete 1,287-test integration suite was not run.

  ## Uncertainties and review notes

  There is no known Stage A correctness uncertainty from the tested paths. The remaining review risks are:

  - downstream, out-of-tree consumers may depend on the intentionally removed old names;
  - the full integration suite has not been executed;
  - the future stateless friction API cannot be fully validated until the rate-dependent law is implemented in Stage E.

  The authoritative documents supplied by you—including pf_rsf.tex and phase_field_fault_redesign.md—remain pre-existing untracked workspace files and were not added to
  this commit. They were not modified.

  ## Diff stat

  24 files changed, 173 insertions(+), 80 deletions(-)

  Stage C has not been started.
