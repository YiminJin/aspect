# Approved K2.2 temporal completion

The Cartesian rejection and exact lookup-reuse optimization is closed.
Recoverable baseline: ../performance/accepted-cartesian-baseline/manifest.json,
source.tar.gz, working-tree.patch, executable and plugin. Source archive SHA256:
2adcaac8bf9af0b7d019eed8d04bbf37b260d9e50fa671651c5d66a6cd11825a.
No production source changes are planned in this completion task.

The saved t=1-s reversal has a measured cancellation mechanism:
../domain-convergence/temporal-cancellation.json. Friction and radiation
anomaly differences individually shrink; their nearly opposite contributions
leave a larger remainder in the finer pair. Slip admits an exact decomposition
into shrinking endpoint-solution/time-load differences, which also cancel.
Cohesive accounting driven by accepted V follows the same trend; its projection
remainder remains separately reported, not treated as a mechanical reference.
This permits finishing the existing time sequence; it is not a convergence pass.

Only the already-approved dt=.25 and .125 cases are replayed to 2 s. They have
no saved executable checkpoints. Output-path-only wrappers preserve the saved
accepted states and allow prefix comparison. The completed spatial studies and
dt=.5 trajectory are reused. No per-vertex scalar solve serves as a reference.

Prospective costs remain conservatively within the previously approved
4000--6500 / 7500--11000 seconds estimates and 9000 / 14400 seconds hard caps.
The accepted performance change should shorten these, but no unmeasured
whole-trajectory speedup is assumed. Budget up to 6 GiB per process initially;
23 GiB RAM and 69 GiB disk were available before launch. Run on distinct
physical cores and inspect actual placement to avoid the prior CPU-0 collision.
ASPECT_FAULT_PERFORMANCE=1 records opt-in subsystem timings/cache work; the
existing run_case.py records peak RSS, elapsed time, exit status and hashes.

Stop for a solver/invariant or allowance failure, an unexplained remaining
plateau, or a required production/criterion change. No K2.3 run is authorized.

## Completed outcome

Both cases completed through 2 s: 1750.999/2908.272 s and
5784664/5786604 KiB peak RSS. All 155 saved-prefix exports match byte-for-byte.
Final nonlinear/fresh-linear, fixed-profile, independent Theta update,
weak-balance and both 1e-4 allowance checks pass. The complete three-level
assessment reuses the dt=.5 and spatial results. Mean-removed traction nearly
plateaus at 2 s (.001056798 -> .001035198 Pa adjacent RMS differences), even
though total fields improve. Signed cancellation is measured, not a proof
of an asymptotically resolved finest reference. No production changes or
additional cases were made; all runs have ended. Stop for review before K2.3.
See doc/reconstructed_fault/benchmarking/stage_K2_2_temporal_completion.md.
