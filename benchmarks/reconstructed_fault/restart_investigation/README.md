# Cohesive restart investigation

This is a separate read-only investigation after R3a, not a correction or R3b.
No production source, original fixture, parameter, solver tolerance, history
assertion or checkpoint is changed. See the detailed cause and proposed one-line
correction in [the rolling review](../../../doc/reconstructed_fault/refactor_review.md).

The preserved R3a Release executable reproduces SIGSEGV on one rank from a
one-rank checkpoint. Both the original and the optional-observer-disconnected
variants fail identically. All original restored-history assertions pass in the
same-rank original case. GDB locates the crash at manager.cc:1077, reached via
solver.cc:1126 and :1161. Its JSON snapshots show valid 1-by-8 committed/current/
trial/candidate V (all 1e-12), active solve and trial, but zero prescribed-rate
maps for one fault. The empty container is already present at deserialization
exit, before the observer runs. Fresh construction creates one map per fault;
restart rebuild omits this transient layout. BP3 explicitly recreates the maps
in its setup and therefore masks this defect.

Generated inputs, outputs, copied checkpoint branches and full evidence are
ignored and retained locally. Original R3a evidence stays untouched. The
initial two-to-one-rank attempt stopped earlier in the observer's entry-2
fingerprint assertion; it was not used to claim the crash reproduction. Creating
one-rank checkpoint data with unchanged physical settings avoids that additional
cross-rank observer issue. The original fixture still fails to converge at step
two after saving step one's checkpoint. No convergence repair is attempted.

`build_symbols.py` recompiles only the manager/solver unity groups with `-g`/
`-g1`, retaining the Release optimization flags, all original source and all other
R3a objects. It drops PCH for those diagnostic objects and relinks into
`build-restart-investigation/aspect-symbols`, without rebuilding the reference.
`inspect_manager.gdb` records load entry/exit and pre-call vector contents/sizes,
status flags, and source-located backtraces. The first debugger run used a source
line breakpoint for load completion that optimization skipped; the second uses
a function-entry/finish breakpoint and captures both states. GDB stops the
inferior at SIGSEGV and then quits, so its enclosing MPI return code is not the
raw inferior's 139; the stopped backtrace is the evidence.

`plugin/no_restore_observer.cc` is a diagnostic copy of the existing restart
observer. Only the include location and its start-timestep callback registration
change. All history assertions remain byte-for-byte present, including the
ordinary Stage-I/Stage-J assertions. This copy is used only for the diagnostic
comparison; the original plugin/case remains available.

From the worktree root, with the staged inputs/checkpoints present:

```sh
python3 benchmarks/reconstructed_fault/restart_investigation/build_symbols.py
bash benchmarks/reconstructed_fault/restart_investigation/run_case.sh original-native-one
bash benchmarks/reconstructed_fault/restart_investigation/run_case.sh no-observer-native-one
bash benchmarks/reconstructed_fault/restart_investigation/run_gdb.sh debug-native-one
bash benchmarks/reconstructed_fault/restart_investigation/run_gdb.sh debug-no-observer-native-one
python3 benchmarks/reconstructed_fault/restart_investigation/verify_evidence.py
```

The runner refuses to overwrite logs. Reproduction requires new case/output
names or a fresh directory; exact commands/environment/outcomes are recorded
in `evidence/*.json`. `manager-states.json`, `investigation-verification.json`
and `*-gdb.log` contain the key evidence. The suggested source correction and
archive-round-trip regression have not been applied or tested.
