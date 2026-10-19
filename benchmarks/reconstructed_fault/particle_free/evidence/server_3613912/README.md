# Server replay 3613912

Version control retains the prepared inputs, analysis/repair sources and selected
compact results. Raw server logs, full resolved parameter output, failure CSVs,
per-run local records and checkpoints remain local. The packet reductions require
those supplied raw files; the committed compact summaries preserve the measured
findings without adding the raw output to Git.

## Verified observer repair

The supplied packet was extracted from `server_3613912.tar.gz`; its initial
archive hash and the unchanged extracted data hashes are recorded in evidence.
`verify_packet.py` independently verifies the signed errors, effective parameters,
all six replayed acceptance records and checkpoint **815 / slot 03**.
[Cause, correction and measured tests](../../../../../doc/reconstructed_fault/particle_free/BP3_DG_WORK_OBSERVER_FIX.md).
`packet-verification.json` contains the nodal results; `repair-verification.json`
summarizes the bounded local qualification. Runtime source changes are confined
to **one file**, `benchmarks/reconstructed_fault/bp3_native/server/plugin/work_audit.cc`.
All original data, errors and old source archives remain. Strict matching is false.

### Update the server plugin

Replace the existing server source with the updated repository file at:

```text
$WORK/build/aspect/pf-rsf-dg/source/benchmarks/reconstructed_fault/bp3_native/server/plugin/work_audit.cc
```

Alternatively apply `work-observer-frame-repair.patch` from that source root
using `git apply --check` then `git apply`. This patch is against the diagnostic
version used in replay 3613912; `repair-source.json` gives its before/after hashes.
Apply once. No core rebuild is required. Rebuild using the same dependency stack:

```sh
cmake --build "$SCRATCH/bp3_dg/plugin/build" -j 4
```

Ordinary builds must not define `ASPECT_BP3_TEST_WORK_CONSISTENCY_FAILURE`.
The repaired observer checks the actual local mechanical frame. Root trace
fields `original_*` retain the old first-segment projection for diagnosis;
`matching_*` are the checked loads, with `guard_uses_matching_frame=1`.
An old-frame value above the bound can still be printed. The actual guard
remains `consistency<1e-5`, using the corrected native loads. Independent DG
publication is unchanged, and a genuinely inconsistent corrected load still fails.

### Verify in a fresh checkpoint branch

Copy **both** `verify-3613912.prm` and `continue-3613912.prm` from this directory
to `$SCRATCH/bp3_dg/`. Their full parameter sets preserve the actual server physics,
mesh, native adaptive controls, Gauss6/source3, Helmholtz20 and tolerances.
They are native inputs, with no Python/shell run controller.

The input uses the **replay's step-815 checkpoint**, not the parent step-809
checkpoint in `output/`. Preserve the failed outputs and both original/replay
restart directories. Verify the marker and label before copying:

```sh
cd "$SCRATCH/bp3_dg"
cat work_replay_3610007/restart/last_good_checkpoint.txt
cat work_replay_3610007/restart/03/bp3_accepted_state.txt
```

They must read `3` and `815 5172072493.8982706`. Use an empty, fresh branch:

```sh
mkdir work_repair_3613912
mkdir work_repair_3613912/restart
cp -a work_replay_3610007/restart/03 work_repair_3613912/restart/
cp -a work_replay_3610007/restart/last_good_checkpoint.txt work_repair_3613912/restart/
source ./environment.sh
./plugin/aspect-release --validate verify-3613912.prm
./plugin/aspect-release --validate continue-3613912.prm
```

Copy the **entire** selected checkpoint, including all distributed state and
metadata. Do not copy the parent/replay CSVs into the new branch. Existing output
prefix guards remain active. Allow disk for the checkpoint and terminal output.
In an allocation with the same stack and preferably the same 160-rank layout:

```sh
ibrun ./plugin/aspect-release verify-3613912.prm
```

This prepared server check starts after 815 and stops by accepted step **818**,
with checkpoint every accepted step and a **7200 s** accepted-state graceful wall
request. Physical end/event conditions remain. The first/max timestep caps remain
4e6/4e7 s, max log-state change .2 and native growth/cutback are unchanged; there
is no fixed 50000 s cap. Leave scheduler margin for an additional solve, terminal
output and checkpoint. This server check has **not** been launched locally.

Verify that 816–818 complete (or inspect any earlier physical/wall stop), both
`matching_*` errors remain below 1e-5 Pa, independent publication passes and
accepted counts remain one. Inspect the last-good marker and accepted checkpoint
label rather than assuming a slot. Preserve the bounded logs, summaries and a
valid checkpoint before continuing.

### Manual first-event continuation

After successful bounded verification, the continuation input resumes the same
**work_repair_3613912** branch. It removes the step-818 stop, restores checkpoint
every 10 and the packaged first-event resume wall request of 82800 s. Physical
end/event caps and every physical/adaptive setting stay the same:

```sh
ibrun ./plugin/aspect-release continue-3613912.prm
```

Use the same scheduler margin principle. This remains the selected unqualified
first-event experiment; the observer repair does not resolve spatial or endpoint
temporal findings. Do not use an input pointing back to the failed `output/`.
No server job was submitted or run by this repair task.

### Local verification evidence

- `native-equivalence.json`: nine physical output files unchanged byte for byte.
- `native-rejection.json`: eleven outputs match after rejection/identical-dt retry;
  publication remains once on both ranks.
- `native-restart.json`: fourteen final outputs match uninterrupted advancement
  byte for byte, with the same accepted timestep sequence and publication counts.
- `native-negative.json`: an inconsistent corrected load is rejected at the
  unchanged guard, with collective failure export and bounded termination.
- `repair-local/`: exact local commands, inputs, artifact hashes, logs and
  ordinary/test compile definitions. Both server inputs pass parser validation
  with rebuilt ordinary plugins. These are local tests, not full-grid advancement.

Run `analyze_logs.py` and `verify_packet.py` from the repository root to recompute
the independent supplied-data reductions. No ASPECT job is launched by either.

---

## Earlier replay-log analysis (before packet extraction)

19 October 2026. Raw logs remain unchanged in
`../../bp3_dg_first_event/BP3.o3613912` and `BP3.e3613912`.
`analyze_logs.py` extracts all seven work-consistency traces, records hashes,
checks the original accepted time/dt sequence for 810–815, and compares the
mechanical solve/publication log lines with job 3610007. These logged quantities
match exactly; this is not a bitwise comparison of every native state vector.
Run from the repository root:

```sh
python3 benchmarks/reconstructed_fault/particle_free/evidence/server_3613912/analyze_logs.py
```

The original frame check again fails at 816, normal component, node 4918:
1.001357615619641e-5 Pa versus the unchanged 1e-5 Pa bound. Directly integrated
frame contribution is 9.931623371162834e-6 Pa at the same node. The native-frame
shadow passes at all seven steps; at 816 its maximum normal/shear errors are
2.0867824647698363e-7 / 5.0678682972525474e-7 Pa. Thus changing only the observer's
projection frame removes the reported guard violation in the replayed state.
Signed nodal reconciliation and a tested production repair are pending; no
runtime source was changed in this log-analysis turn. All prior spatial and
endpoint temporal findings and `strict_reference_match_passed=false` remain.

Please place these small replay artifacts in this directory, keeping them
separate from the original server packet:

1. `work_consistency_failure_816.csv` — required signed nodal loads, direct
   frame contributions and original/matching-frame errors.
2. `accepted_steps.csv` — completed replay observations through 815.
3. `parameters.prm` — effective resolved replay configuration.
4. `last_good_checkpoint.txt` — copied from the replay's `restart/` directory.
5. `bp3_accepted_state.txt` — from the selected replay checkpoint. Stdout shows
   the most recent completed checkpoint at step 815, slot 03; verify the marker
   and selected label rather than inferring validity from modification time.

The server paths share prefix `$SCRATCH/bp3_dg/work_replay_3610007/`.
Keep the complete original and replay checkpoints on the server; no large
checkpoint, particle dump or visualization transfer is needed at this stage.
Do not overwrite the parent step-809 checkpoint or failure outputs. The intended
repair is to use the same local association frame as mechanics throughout the
observer, retaining the existing guard and independent publication checks.
