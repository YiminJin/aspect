# Restored 150 × 50 km BP3

The maintained runtime is [plugin/](plugin/), built as
`libbp3_restore_150x50.release.so`. Its [README](plugin/README.md) describes
the model, supported controls, outputs and restart procedure.
The parent `bp3.cc` / `libbp3` belongs to historical experiments.

- Prepared physical inputs: `fixtures/bp3_150x50/`.
- Detailed startup comparisons: `bp3_150x50_raw.prm`,
  `bp3_150x50_filter20.prm`, `bp3_150x50_filter40.prm`.
- Production continuation: `bp3_150x50_first_event.prm`, after reviewing
  the filter20 startup checks. Resolve its plugin/input/output paths for the server.

The production continuation writes full-fault profiles at 0.1 m maximum
nodal change of cumulative signed slip or 31557600 s, with initial/final/event
forcing and profiles on heavy-output steps. Accepted summaries and stations
remain every step. It no longer writes a duplicate every-step cumulative-slip
table. Slip/Theta/stress evolution and checkpoint histories are unaffected by
output cadence. Plot the saved profiles using their instantaneous V; an offline
legacy export contains saved states only.

`Write detailed diagnostics` and `Audit full state every step` are false for
production; the dedicated startup PRMs retain their explicit diagnostics.
The first-step Maxwell, native weak traction, inert H, fixed geometry/Ih,
boundary/compression and accepted-Theta checks remain active.

Version-5 checkpoints remain readable. Use `branch_output.sh PARENT ID NEW_DIR`
when restarting an older selected checkpoint; both metadata and referenced
profile payloads must be restored. New PRM output intervals apply to the saved
last-written references. The parent remains unchanged.

See [the cleanup report](output-cleanup-evidence/REPORT.md) for the environment
audit, exact before/after comparisons, retained compatibility checks, output
sizes and validation limits. See [server AMG/GMG comparison](amg_gmg_server_comparison.md)
for the separate solver-performance investigation.
