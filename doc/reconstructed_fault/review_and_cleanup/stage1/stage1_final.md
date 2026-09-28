  ## 1. Commit hash

  21ad77933ac7029cda10be08ddd42afdabe209b3
  Reconstructed fault: clarify Stage 1 slip-rate lifecycle

  ## 2. Files changed

  - doc/reconstructed_fault/current_design.md:300 — documents the three-state lifecycle, \(V\ge0\), committed-only generic properties, checkpoint compatibility, mutable-geometry limitation, and deferred full restart test.

  - include/aspect/reconstructed_fault.h:291 — exposes the corrected nonlinear-solve lifecycle and removes speculative aggregate accessors.
  - source/reconstructed_fault.cc:959 — enforces separate timestep/current/trial states and nonnegative finite \(V\).
  - source/postprocess/reconstructed_faults.cc:206 — obtains timestep-committed \(V\) per fault for persistent VTU output.
  - unit_tests/reconstructed_fault.cc:126 — verifies the corrected lifecycle, nonnegativity, rollback, commit, and checkpoint behavior.

  ## 3. Final (V) lifecycle
```
                           line-search trial
                           Vtrial = V(n) + α δV
                          /                    \
                         / reject               \ accept
                        ▼                        ▼
  Vk ── begin solve ──► V(n) ◄──────────────── Vtrial
  │                     │
  │ failed solve        │ successful convergence
  ◄─────────────────────┘
                        │
                        ▼
                   commit Vk = V(n)

  checkpoint / persistent VTU output ──► Vk only
```
  All three states enforce finite \(V\ge0\). The manager does not know about \(V_{\min}\).

  4. Public interfaces added, removed, or renamed

  Added:
```
  void begin_slip_rate_nonlinear_solve();
  void commit_slip_rate_nonlinear_solve();
  void rollback_slip_rate_nonlinear_solve();

  const std::vector<double> &
  get_timestep_committed_slip_rate(unsigned int fault_index) const;
```
  Removed:
```
  const std::vector<std::vector<double>> &get_slip_rates() const;
  const std::vector<std::vector<double>> &get_committed_slip_rates() const;
```
  Effectively renamed and narrowed:
```
  get_committed_slip_rates()
      → get_timestep_committed_slip_rate(fault_index)
```
  Semantics clarified without a signature change:
```
  get_slip_rate(fault_index)
```
  It returns the active trial candidate when a trial exists; otherwise it returns the current accepted Newton iterate.

  ## 5. Remaining uncertainty

  - Mutable `get_fault()` and public `append_*()` can still bypass manager-owned nodal-field alignment. This is documented and must be resolved before propagation or topology changes.
  - Checkpoints predating serialized manager state are not guaranteed to be compatible; no migration mechanism was added.
  - A full Simulator filesystem checkpoint/restart test remains mandatory before restart support is considered complete. The manager archive round-trip remains covered.
  - No Stage 2 work was started.

  ## 6. Test results

  - Full debug suite:

    All tests passed (2237 assertions in 53 test cases)

  - Focused two-rank reconstructed-fault suite, on each rank:

    All tests passed (91 assertions in 11 test cases)

  - Release build:

    [100%] Built target aspect.exe.release

  - git diff --check: passed.

  The release build emitted only the pre-existing optimizer warnings in checkpoint_restart.cc.