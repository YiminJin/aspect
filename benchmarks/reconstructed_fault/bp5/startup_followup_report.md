# BP5-friction startup: 75 s, adaptive restriction, and restart

This bounded follow-up uses the unchanged Release executable/plugin from
`small_startup_report.md`. No physical, solver, bound, mesh, initial-state or
history-update algorithm changes are made. The artificial Maxwell initialization
interval remains **4e6 s**, separate from physical time.

## Checks selected before execution

1. Four fresh 75-s physical steps to 300 s. Compare only at 300 s against the
   completed 300-s (one step) and 150-s (two steps) runs. Retain their predictor
   setting 0.1 so this is a genuine 75-s comparison, not a 0.02-limited run.
2. Four fresh adaptive physical steps with the intended long-run maximum step
   **4e6 s** and weighted logarithmic state-change limit **0.02**. Keep the
   convection and reconstructed-fault timestep models selected. An independent
   calculation on the identical initial state predicts **66.9642717091 s** for
   the first step. The end-time setting cannot clip these four steps; the
   accepted-step guard terminates the run.
3. Copy that run's ordinary accepted-step-2 checkpoint into an isolated branch
   and resume through steps 3 and 4. Compare against its uninterrupted steps,
   including the first resumed history update and subsequent controller
   proposals. Do not reconstruct or reload initial state.

All three use four MPI ranks, one thread per rank, 1200-s individual hard caps
and a 3600-s aggregate simulation limit. No automatic retries. Files are under
`weakening30-dc010-ell100/startup-followup/{75,adaptive,resume}`. Each launch
manifest hashes the existing executable/plugin and all inputs. No server
package or long run is enabled by these tests.

## Verification definitions

- Require actual bulk/surface nonlinear convergence and fresh linear residual
  acceptance, not only exit zero or lifecycle messages.
- Check exact fresh initial-state equality with the prior 300-s run; timestep
  zero must retain initialized Theta and stress.
- Reuse native work/QP reconstruction and the independent exact-aging/once-only
  slip-update checks. For restart, the first incoming state is the checkpoint's
  evolved state, not an initial field.
- For temporal differences, report weakening (0–30 km), transition (28–35 km)
  and deep controls, with maximum and native-work-weighted RMS norms. A
  contraction factor is `difference(300,150)/difference(150,75)` at 300 s.
  These are pairwise differences, not errors against an exact solution.
- Verify the compiled controller actually restricts the 4e6-s ceiling; evaluate
  its predictor independently and compare its proposal to the next accepted dt.
  Report realized accepted-state changes separately from frozen-rate predictions.
- Restart comparison retains the existing 1e-8 per-component coefficient for
  bulk, particle, fault and raw constitutive data. Stable IDs, fixed coordinates
  and source associations must match. No tolerance change is made to obtain a
  passing comparison. Check that no weak-initialization or initial-mesh export
  is produced on resume, and that only steps 3–4 are mechanically solved.

## 75-s comparison result

Four real steps reach 300 s with all convergence/lifecycle/native-work checks
passing, unchanged initial state, zero lower-active nodes and full Newton steps.
Wall time is **561.93 s**, maximum child RSS **2,784,468 KiB** (not sum over ranks).

At the common time 300 s in the 0–30 km weakening region:

| Quantity | max difference, 300 vs 150 s | max difference, 150 vs 75 s | contraction |
|---|---:|---:|---:|
| V | 4.38548e-11 m/s | 2.11076e-11 m/s | 2.07768 |
| Accumulated slip | 6.57822e-9 m | 3.24858e-9 m | 2.02495 |
| Committed Theta | 0.00166408 s | 0.000820505 s | 2.02811 |

The respective native-work-weighted RMS contractions are **2.07730, 2.02486,
2.02801**. Transition-region RMS contractions are similarly 2.07574, 2.02437,
2.02563. Deep differences are negligible (V at most 3.55e-18 m/s), and also
decrease. Full regional measures are in `temporal_300s.json`.

This supports ordinary first-order timestep convergence over the tested startup
window. It does not claim zero temporal error: the 150/75-s V and slip
differences are still **2.2368%** and **1.1145%** of the respective fine-region
maxima. The tiny committed-state difference does not imply equally tiny
mechanical error, because mechanics uses the incoming split state.

## Actual adaptive-controller result

The unchanged compiled predictor's limiting/bisection branch is now exercised
against a **4e6-s ceiling**, not a preselected small maximum timestep. Its first
proposal is `66.964271709050308` s, differing from the independent initial-state
calculation by about 3.4e-12 s (the safe bisection bracket has this resolution).

| Accepted step | Physical time (s) | dt used (s) | realized weighted state change |
|---:|---:|---:|---:|
| 1 | 66.9642717091 | 66.9642717091 | 0.0199999975264 |
| 2 | 134.1074064313 | 67.1431347223 | 0.0200000998465 |
| 3 | 201.4295371744 | 67.3221307431 | 0.0200000981237 |
| 4 | 268.9311461613 | 67.5016089868 | 0.0200000964303 |

Every predictor measure is `0.019999999999999386`; each proposal is the actual
next accepted timestep, demonstrating that no other selected ceiling is
responsible for the small dt. The final (not executed) next proposal is
**67.6815706799 s**, computed from the fourth evolved state.

The realized measure exceeds 0.02 by at most **9.98465e-8**, about five parts
per million of the limit. This is not hidden by a relaxed assertion: the
approved rule bounds the **frozen-rate predictor**, whereas the exact accepted
aging update uses the newly solved rate. No post-solve rejection/cutback rule
has been introduced. The initial predictor restriction and every independent
history check pass; all nodes remain free and all Newton alphas are one.

Wall time is **572.46 s** and maximum child RSS **2,778,648 KiB** (not aggregate
MPI memory). Results are in `adaptive_checks.json`, the raw controller CSV and
the accepted-state/native-QP exports. Ordinary checkpoint `adaptive/restart/01`
is labeled accepted step 2 at 134.10740643133786 s.

## Checkpoint/resume result

The ordinary step-2 checkpoint is copied into `resume/restart/01`, preserving
its file hashes in `checkpoint_source.json`. The new process solves **only
steps 3 and 4**. It produces neither `weak_initialization.csv` nor initial-mesh
exports. Its first incoming Theta is exactly the checkpoint's evolved Theta;
the independent aging and fused slip-accumulation checks verify one update per
accepted step, with no reset or duplicate history publication.

All **96 measured field/controller comparisons have zero absolute difference**
against the uninterrupted reference, exceeding the unchanged 1e-8 comparison
requirement. This includes:

- every exported bulk component and stable-ID particle position/property;
- incoming/outgoing Theta, V, accumulated slip, weak traction/friction loads;
- current constitutive pressure/stress, phase, Ih and localization;
- source QP identities, coordinates/associations, work weights and background
  shear loads; existing mature-zero-C and endpoint/geometry assertions pass;
- the checkpoint's next timestep, both resumed accepted times, and controller
  measures/proposals rebuilt after each resumed history update.

The first resumed dt is exactly **67.322130743101980 s**. The last next-dt
proposal (not executed) is exactly the reference **67.68157067992345 s**.
Final resumed normalized bulk/surface residuals are respectively
`1.049099e-11 / 1.218895e-9` at step 3 and
`1.085089e-11 / 1.218918e-9` at step 4; fresh linear checks pass.

Resume wall time is **227.54 s**, maximum child RSS **2,840,692 KiB**. Total
simulation time for the three authorized runs is **1361.93 s (22.70 min)**.
No process timed out or was retried. This qualifies the short restart/controller
cycle on the same four MPI ranks, not cross-rank restart or an earthquake cycle.

## Files changed and scope

- Added `run_startup_followup.py` and `analyze_startup_followup.py` for the three
  isolated runs, reuse of previous temporal evidence, and exact-clock/lifecycle
  comparisons. Syntax checks and all three analysis commands pass.
- `check_startup_30km.py` accepts an optional preceding state and a separate
  initial-reference path, allowing its existing checks to cover the first
  resumed step. Default behavior is unchanged; the fresh 75-s and adaptive
  runs exercise it too.
- Added this report and a README link. Run inputs, manifests, full results,
  the ordinary checkpoint and comparison JSON files are preserved separately
  from previous studies.

**No C++ production or plugin code changed; no rebuild was needed.** The tests
use the already verified executable/plugin, with hashes recorded at every
launch. No solver/physical acceptance tolerance was changed, no further
timestep level was added, and the blocked server package was not enabled.
The observed temporal contraction supports the startup discretization; the
0.02 heuristic is now operationally/restart tested but does not certify
longer-time or event-scale temporal accuracy.

Recoverable source/input snapshot: `startup-followup/tested-followup.tar.gz`,
SHA256 `13de170e9b5ca7c5366e40ffcd4276b9b7252ecfc2f2b31b7e70eadc6fd28dcf`.
It includes the prior tested source/plugin snapshot, new scripts/checker and
all three parameter files/launch manifests. Full simulation outputs remain in
their original result directories; the checkpoint is not replaced by the archive.

## Reproduction

```sh
python3 benchmarks/reconstructed_fault/bp5/run_startup_followup.py prepare 75
python3 benchmarks/reconstructed_fault/bp5/run_startup_followup.py run 75
python3 benchmarks/reconstructed_fault/bp5/analyze_startup_followup.py temporal
python3 benchmarks/reconstructed_fault/bp5/run_startup_followup.py prepare adaptive
python3 benchmarks/reconstructed_fault/bp5/run_startup_followup.py run adaptive
python3 benchmarks/reconstructed_fault/bp5/analyze_startup_followup.py adaptive
python3 benchmarks/reconstructed_fault/bp5/run_startup_followup.py prepare resume
python3 benchmarks/reconstructed_fault/bp5/run_startup_followup.py run resume
python3 benchmarks/reconstructed_fault/bp5/analyze_startup_followup.py restart
```

Completed labels refuse overwrite. The restart uses checkpointed benchmark
metadata and the accepted cumulative-slip prefix; those output files are not
substitutes for serialized constitutive/controller state. The new branch's
predictor CSV is append-only and has no header in the unchanged tested plugin;
the offline reader uses its explicit existing schema.
