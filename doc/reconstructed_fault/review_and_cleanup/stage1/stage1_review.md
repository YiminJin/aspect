  # Stage 1 review packet

  Stage 1 is complete and committed. Stage 2 has not been started.

  Commit:

  9bc8afb2ca4475c0afafc2427f831f05009ec000
  Reconstructed fault: add Stage 1 slip-rate state

  The commit contains only the eight tracked Stage 1 files. The untracked design-source files in doc/reconstructed_fault/ were not included or modified.

  ## 1. Files changed

   File | Stage 1 change and invariant
   ---- | ----
   doc/reconstructed_fault/current_design.md:342 | Records the settled Stage 1 ownership and lifecycle invariants: \(V\) belongs to the manager, geometry remains constitutively neutral, trial values are non-accumulating, checkpoints restore committed state, and physical initialization/positivity remain deferred.
   include/aspect/reconstructed_fault.h:268 | Declares manager-owned \(V\), its initialization/interpolation/trial interface, persistent serialization boundary, and manager-mediated complete-fault insertion. `ReconstructedFault<dim>` remains free of distinguished kinematic or constitutive state.
   source/reconstructed_fault.cc:858 | Enforces the \(V\) invariants: one finite nodal value per vertex, exact fault/vertex layout, Q1 interpolation, non-accumulating trials, explicit accept/rollback, committed-only persistence, and cache invalidation after restored geometry. Reserves slip_rate against generic-property registration.
   include/aspect/postprocess/reconstructed_faults.h:21 | Extends the output view with an optional distinguished slip-rate field while preserving the old constructor call form.
   source/postprocess/reconstructed_faults.cc:22 | Writes slip_rate only when all fault slip rates are initialized. The output is vertex-aligned, finite, and uses committed values rather than an active unaccepted trial.
   source/simulator/checkpoint_restart.cc:644 | Makes the reconstructed-fault manager participate in Simulator checkpoint/restart when fault reconstruction is enabled. No new Simulator lifecycle hook was introduced.
   unit_tests/reconstructed_fault.cc:122 | Adds tests for initialization, validation, Q1 interpolation, non-accumulating trials, accept/rollback, reserved naming, and checkpoint restoration of committed rather than trial \(V\).
   unit_tests/reconstructed_fault_output.cc:11 | Verifies that existing output omits uninitialized \(V\) and initialized output includes the distinguished slip_rate array.


  ## 2. Public interfaces

  ### New `ReconstructedFaultManager<dim>` interfaces
```
  unsigned int
  add_reconstructed_fault(
    const std::vector<Point<dim>> &vertices,
    const std::vector<double> &projection_half_widths);
```
  Invariant: adds one complete fault with one positive finite projection half-width per vertex. The new fault has no implicit \(V\).
```
  bool
  slip_rates_are_initialized() const;
```
  Invariant: returns true only when geometry exists and every reconstructed fault has exactly one initialized slip-rate value per vertex.
```
  void
  initialize_slip_rate(
    const unsigned int fault_index,
    const std::vector<double> &values);
```
  Invariant: initialization is one-time per fault, must occur outside an active trial, and requires one finite value per vertex.
```
  const std::vector<double> &
  get_slip_rate(const unsigned int fault_index) const;
```
  Invariant: exposes the current value—committed outside a trial, candidate during a trial—and rejects uninitialized access.
```
  const std::vector<std::vector<double>> &
  get_slip_rates() const;
```
  Invariant: exposes all current values only after every fault is initialized.
```
  const std::vector<std::vector<double>> &
  get_committed_slip_rates() const;
```
  Invariant: exposes the last accepted values, including while a different trial candidate is active. This is the checkpoint/output view.
```
  double
  interpolate_slip_rate(
    const unsigned int fault_index,
    const unsigned int segment_index,
    const double xi) const;
```
  Invariant: evaluates the current field on a valid segment for \(0\leq\xi\leq1\).
```
  void
  begin_slip_rate_trial();
```
  Invariant: all faults must be initialized and no trial may already be active. The current candidate is reset to the committed base.
```
  void
  set_slip_rate_trial(
    const std::vector<std::vector<double>> &delta_V,
    const double step_length);
```
  Invariant: every candidate is recomputed from the saved base:
  \[
  V^{\mathrm{trial}} = V^{\mathrm{saved}}+\alpha\delta V.
  \]
  Repeated line-search evaluations therefore do not accumulate rejected candidates.
```
  void
  accept_slip_rate_trial();
```
  Invariant: the current candidate becomes committed and the trial ends.
```
  void
  rollback_slip_rate_trial();
```
  Invariant: the saved base is restored and the trial ends.

  ### Modified public contracts

  `ReconstructedFaultManager::register_property()` retains its signature, but `slip_rate` is now a reserved, case-sensitive name. It cannot be registered as a generic property.

  `Postprocess::internal::ReconstructedFaultOutput` now has:
```
  explicit ReconstructedFaultOutput(
    const std::vector<ReconstructedFault<dim>> &faults,
    const std::vector<
      typename ReconstructedFaultManager<dim>::PropertyInformation>
      &property_information,
    const std::vector<std::vector<double>> *slip_rates = nullptr);
```
  The default preserves existing two-argument callers. A supplied field must contain one finite value per output vertex.

  `ReconstructedFaultManager::PropertyInformation` gained Boost serialization support. This does not change its scientific meaning, indices, or property layout.

  ### Interfaces deliberately unchanged

  - `ReconstructedFault<dim>` has no new public state or kinematic methods.
  - No material-model interface was added.
  - No RSF-specific Simulator object was added.
  - No new Simulator lifecycle callback was added.
  - No solver or Stokes coupling interface was added.

  ## 3. Design and equation correspondence

   Implementationinvariant | Authoritative design correspondence
   ---- | ----
   Manager owns distinguished \(V\); ReconstructedFault remains geometry/property-only | redesign_plan.md, §2 “Ownership and code organization”, `ReconstructedFaultManager<dim>`; reconstructed_fault_surface_rsf_design.md, §4.2 “Fault-surface fields”; current_design.md, §4 and new §18.
   One nodal \(V\) value per ordered fault vertex | reconstructed_fault_surface_rsf_design.md, §2, which defines the ordered Q1 fault \(\Gamma_h=\bigcup_e\Gamma_e\); §4.2 identifies \(V\) as a fault-surface field.
   Q1 interpolation | reconstructed_fault_surface_rsf_design.md, §2: \(N_0(\xi)=1-\xi\), \(N_1(\xi)=\xi\). The implemented invariant is \[V_h(\xi)=(1-\xi)V_i+\xi V_{i+1}.\]
   Saved-base trial updates | redesign_plan.md, §6 “Nonlinear lifecycle and caches,” line-search equation \[V^{\rm trial}=V^{\rm saved}+\alpha\delta V.\]
   Explicit accept and rollback | redesign_plan.md, §2 proposed manager interface and §6 rejected-candidate rollback rule; §7 Stage 1 explicitly requests trial/accept/rollback.
   Optional initialization without inventing a physical value | redesign_plan.md, §7 separates Stage 1 storage/initialization mechanics from Stage 4 force-balance/\(\mu_0\) constitutive initialization.
   Committed-only checkpoint persistence; reconstructible caches omitted | redesign_plan.md, §6 checkpoint paragraph; specification.tex, “Checkpoint/restart,” which distinguishes persistent state from reconstructible caches.
   slip_rate VTU output | redesign_plan.md, §3 migration table and §7 Stage 1; current_design.md, §15 and §18.
   Replicated manager state, no bulk gathering | redesign_plan.md, §2 manager ownership; reconstructed_fault_surface_rsf_design.md, §17 MPI rules. Stage 1 adds no MPI gather or root-owned state.
   Geometry/property behavior remains generic | redesign_plan.md, §7 Stage 1; current_design.md, §§14–17. Existing property storage and projection semantics are unchanged.
   \(V\) is not cached in geometric association data | reconstructed_fault_surface_rsf_design.md, §18 cache rules. \(V\) remains separate from the existing particle/geometry cache.

  No equation from pf_rsf.tex is implemented in this commit. Maxwell stress, friction, \(\Theta\), \(T^{\rm coh}\), \(I_h\), surface residuals, and Stokes coupling are outside Stage 1.

  ## 4. Assumptions made

  These choices were required to make Stage 1 executable but were not all stated explicitly:

  1. __An empty manager is not “initialized.”__
     `slip_rates_are_initialized()` returns false when no reconstructed geometry exists.

  2. __Partial per-fault initialization is allowed.__
     Individual faults can be initialized separately, but interpolation across an initialized fault is the only per-fault operation permitted before all faults are initialized. Trial operations and aggregate access require complete initialization.

  3. __Initialization is one-time per fault.__
     Reinitialization is rejected. Subsequent changes must use the trial/accept lifecycle.

  4. __“Committed” means the last accepted manager value.__
     Stage 1 has one acceptance layer. There is not yet a second distinction between an accepted nonlinear iterate and a timestep-committed value.

  5. __Finite values are required, but positivity is not imposed.__
     No (V_{\min}) was introduced because that quantity belongs to the later physical line-search rule and must not be duplicated in the generic manager.

  6. __Interpolation addresses explicit fault/segment coordinates.__
     The planned pseudocode referenced FaultLocation and SlipRateState, but neither is an existing source-tree type. Stage 1 therefore uses `fault_index`, `segment_index`, and `xi`, and interpolates the current view.

  7. __Output and checkpoints use committed values.__
     An active unaccepted candidate is intentionally excluded from persistent and diagnostic output.

  8. __Slip rate remains optional.__
     Existing reconstructed-fault workflows continue to operate without initializing (V); they simply do not emit slip_rate.

  9. __The reserved name is exactly slip_rate.__
     The check is case-sensitive, matching current property-name behavior.

  10. __Complete-fault insertion requires usable projection geometry.__
     A fault must have at least two vertices and one positive finite half-width per vertex.

  11. __Replicated consistency is a caller invariant.__
     Stage 1 does not add an MPI broadcast or checksum around initialization or trials. Every rank is expected to invoke the same operations with identical small fault-
     sized inputs.

  12. __Checkpoint format compatibility was not versioned.__
     The implementation assumes checkpoints are read by the same Stage 1-capable code layout.

  ## 5. Uncertainties and review points

  1. __Positivity ownership has a design tension.__
     The manager ownership section says “positivity-preserving trial/update/rollback,” but the staged plan places positivity limiting in Stage 9, where \(V_{\min}\) and the coupled line search exist. Stage 1 consequently guarantees only finite, correctly shaped values. Negative \(V\) is currently representable.

  2. __A future nonlinear lifecycle may need two commit levels.__
     If “accept line-search candidate” and “commit successful timestep” must be distinct, the current committed/saved/current model will need an additional saved timestep
     state or a clarified naming contract.

  3. __Existing mutable geometry access remains capable of bypassing the manager.__
     `get_fault()` still returns a mutable ReconstructedFault, and `ReconstructedFault::append_*()` remains public to preserve existing generic behavior. A caller that appends directly after initializing \(V\) can break the one-value-per-vertex invariant. Propagation and appended-vertex state are explicitly deferred, so Stage 1 does not define or silently invent that state.

  4. __The public aggregate accessors exceed the pseudocode’s minimal interface.__
     `get_slip_rates()` supports future solver-wide operations, while `get_committed_slip_rates()` gives checkpoint/output an explicit non-trial view. Review may prefer a narrower output-specific view later.

  5. __The interpolation signature differs from the proposed pseudocode.__
     It does not yet accept a generic FaultLocation or selectable SlipRateState. Introducing those types before their association users exist was judged premature.

  6. __Old reconstructed-fault checkpoints are not backward-compatible.__
     With fault reconstruction enabled, pre-Stage-1 checkpoint archives do not contain the newly serialized manager record. No archive version/migration path was specified.

  7. __Generic property persistence may need a later ownership audit.__
     Stage 1 checkpoints all generic property values. A later material-owned trial/committed distinction for \(\Theta\), \(T^{\rm coh}\), or \(I_h\) must ensure only committed generic state reaches a checkpoint.

  8. __No end-to-end Simulator restart test was added.__
     The manager’s binary archive round trip is tested, including active-trial exclusion, and the Simulator archive includes the manager. A full filesystem checkpoint/resume integration test was not added in this stage.

  ## 6. Tests added or modified

  ### New test cases

  - ReconstructedFaultManager slip-rate lifecycle and interpolation
      - Complete initialization state.
      - Q1 reproduction.
      - Trial candidates formed from the saved base.
      - Committed values remain unchanged during a trial.
      - Accept and rollback semantics.
      - Invalid lifecycle and interpolation coordinate rejection.

  - ReconstructedFaultManager validates slip-rate initialization
      - Wrong initialization size.
      - NaN initialization.
      - Trial before initialization.
      - Wrong fault count in an update.
      - Wrong vertex count in an update.
      - Infinite update value.

  - ReconstructedFaultManager checkpoint restores committed slip rate
      - Geometry, generic schema/value, and (V) round trip.
      - An active trial is not persisted.
      - Restored current (V) equals committed (V).
      - Restored Q1 interpolation remains valid.

  - Reconstructed fault output writes distinguished slip rate
      - Built-in slip_rate array exists.
      - Values remain vertex-aligned.

  ### Existing tests modified

  - Generic property registration now verifies that slip_rate is rejected.
  - Existing geometry/output test verifies that uninitialized (V) does not add a slip_rate array.

  ## 7. Exact verification results

  ### Full debug unit suite

  Command:

  make -C build-main run_unit_tests -j2

  Result:

  All tests passed (2228 assertions in 53 test cases)
  Built target run_unit_tests

  ### Focused two-rank MPI suite

  Command, from build-main/unit_tests:

  mpirun -np 2 ../aspect --test \
    'ReconstructedFault*,Reconstructed fault*,Fault projection*'

  Result on each rank:

  All tests passed (82 assertions in 11 test cases)

  ### Release build

  Command:

  cmake --build build-main --target aspect.exe.release -j2

  Result:

  [100%] Built target aspect.exe.release

  The release build emitted only pre-existing unrelated warnings from particle_domain.cc.

  ### Diff validation

  git diff --check

  Result: no errors.

  ### Additional attempted verification

  The complete unit suite was also invoked with two MPI ranks. Reconstructed-fault tests passed, but one rank failed unrelated random CPO expectations and an unrelated
  missing-file exception-message test: 58 failures across three pre-existing test cases. The focused reconstructed-fault MPI run above was therefore used as the relevant
  clean result.

  ## 8. Existing behavior that may have changed

  - Registering a generic property named slip_rate now throws.
  - Reconstructed-fault checkpoints now contain manager state when reconstruction is enabled.
  - Checkpoints written before this serialization change are not expected to restart with the new archive layout.
  - VTU files gain a scalar slip_rate point-data array once all faults initialize (V). Files remain unchanged while (V) is uninitialized.
  - VTU/checkpoint output during an active trial observes committed rather than candidate (V).
  - Initial fault reconstruction now routes complete fault creation through the manager-owned insertion invariant. Geometry and generic property results remain the same.
  - Manager geometry insertion now explicitly rejects degenerate faults, mismatched half-width layouts, and nonpositive/nonfinite half-widths.
  - Invalid slip-rate layouts are runtime errors in release builds rather than debug-only assertions.
  - Existing callers of the two-argument ReconstructedFaultOutput constructor remain source-compatible.
  - Existing geometry-only and generic-property workflows do not need to initialize (V).

  ## 9. Compact diff stat

   doc/reconstructed_fault/current_design.md         |  27 ++-
   include/aspect/postprocess/reconstructed_faults.h |   4 +-
   include/aspect/reconstructed_fault.h              | 101 ++++++++
   source/postprocess/reconstructed_faults.cc        |  29 ++-
   source/reconstructed_fault.cc                     | 277 +++++++++++++++++++++-
   source/simulator/checkpoint_restart.cc            |   4 +
   unit_tests/reconstructed_fault.cc                 |  96 ++++++++
   unit_tests/reconstructed_fault_output.cc          |  22 ++
   8 files changed, 552 insertions(+), 8 deletions(-)

  Stage 2 remains untouched pending review of this packet.
