# BP3 job 3610007: diagnosis and bounded replay

Version control retains analysis/replay sources and compact results. Original
server logs, raw state/load exports and local build/run logs remain uncommitted;
the reduction script requires that supplied packet at the documented local paths.

[Findings and measured limitations](../../../../../doc/reconstructed_fault/particle_free/BP3_SERVER_3610007_DIAGNOSIS.md).
Original supplied state files are unchanged in `../server_3610007/`.
The original logs remain in `../../bp3_dg_first_event/`. This directory contains
independent reductions and a prepared diagnostic replay; it is not evidence that
the server failure has been repaired. `strict_reference_match_passed=false`.

- `analysis.json`: original input hashes, trajectory/load checks, demonstrated
  frame mismatch and explicitly labeled surrogate/cancellation estimates.
- `analyze.py`: recomputes those results using NumPy, without launching ASPECT.
  Run from the repository root with
  `python3 benchmarks/reconstructed_fault/particle_free/evidence/server_3610007_analysis/analyze.py`.
- `work-consistency-trace.patch`: one-file observational patch against the
  pre-diagnostic source; `diagnostic-source.json` gives before/after hashes.
- `local-equivalence.json`: twelve byte-identical output comparisons against the
  earlier two-rank adaptive control. `local/` retains commands, native logs,
  expected negative-test rejection/CSV and parser validation. Negative-test
  CMake uses workstation paths; it must never be used for the server plugin.
- `replay-3610007.prm`: complete supplied resolved input with comments removed
  (necessary for the shared-library pre-parser) and exactly five output/restart
  overrides, enumerated in `replay-parameter-delta.json`.

## Install the diagnostic plugin

Replace **only** the maintained source file
`benchmarks/reconstructed_fault/bp3_native/server/plugin/work_audit.cc` in the
server ASPECT source tree with the updated repository version. Alternatively,
copy `work-consistency-trace.patch` there and run from that source root:

```sh
git apply --check work-consistency-trace.patch
git apply work-consistency-trace.patch
```

Apply the patch once, against the matching source identified by the before hash.
Rebuild the ordinary plugin using the same modules, compiler and floating-point
flags as the failed run. No core rebuild or new production parameter is needed:

```sh
cmake --build "$SCRATCH/bp3_dg/plugin/build" -j 4
```

The diagnostic macro `ASPECT_BP3_TEST_WORK_CONSISTENCY_FAILURE` must **not** be
defined. It is used only for the local negative test. The original observer guard
remains active and can still stop this replay.

## Preserve the failed run and branch from its valid checkpoint

Leave the failed `output/` and original logs untouched. Copy this directory's
`replay-3610007.prm` to `$SCRATCH/bp3_dg/`. Inspect the original marker and label:

```sh
cd "$SCRATCH/bp3_dg"
cat output/restart/last_good_checkpoint.txt
cat output/restart/03/bp3_accepted_state.txt
```

They must identify slot **3**, accepted step **809**, time
**5172072493.8510532** for this packet. If they differ, preserve the files and
resolve the discrepancy before replaying. Use a fresh branch directory; do not
copy the newer live CSVs into it. Copy the **entire** selected checkpoint,
including its metadata and all distributed state, not just the label:

```sh
mkdir work_replay_3610007
mkdir work_replay_3610007/restart
cp -a output/restart/03 work_replay_3610007/restart/
cp -a output/restart/last_good_checkpoint.txt work_replay_3610007/restart/
source ./environment.sh
./plugin/aspect-release --validate replay-3610007.prm
```

The new branch's lightweight series can start at step 810. Do not transplant
post-809 summaries or create indices referring to absent profile payloads.
Retain the parent run as the source of earlier evidence. Reserve disk for the
copied checkpoint and unchanged terminal/dense outputs.

## Launch directly in an allocation

Use the same 160 ranks, rank layout and dependency stack as job 3610007 where
possible. No server job was launched or submitted during this diagnosis.

```sh
cd "$SCRATCH/bp3_dg"
source ./environment.sh
ibrun ./plugin/aspect-release replay-3610007.prm
```

The replay preserves the mesh, physical fields, all adaptive timestep controls,
quadrature, filtering and numerical tolerances. It stops by accepted step **816**
or the **7200 s** graceful wall request, checkpoints every accepted step, and
retains the original physical end/event caps. Do not force a timestep sequence.
Compare the replayed adaptive steps with the original 810–815 records. As before,
the graceful request is checked at accepted states: scheduler margin must cover
one additional solve, terminal output and checkpoint writing.

Return the replay stdout/stderr, its `accepted_steps.csv`, and, if the guard
fails, `work_replay_3610007/work_consistency_failure_STEP.csv`. The root log
prints `Work consistency trace:` even when detailed diagnostics are disabled.
These small files distinguish the first-segment-frame error from a remaining
matching-frame publication discrepancy. Keep the replay checkpoints on the
server for a further failing-cell analysis if needed; no large checkpoint
transfer is requested now. If no failure recurs, retain all traces and report
the stop step rather than treating the original failure as resolved.
