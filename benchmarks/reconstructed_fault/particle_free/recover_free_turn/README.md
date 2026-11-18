# Recover the first free-turn interval

Selected task: `CODEX_RECOVER_FREE_TURN_STEP16.txt`. Results and limitations:
`doc/reconstructed_fault/particle_free/RECOVER_FREE_TURN_STEP16_REPORT.md`.
Each runner invocation selects one window and refuses an existing output label.
It uses four bound cores, one thread/rank, and the unchanged protected checker.
Do not run analysis/compression alongside the four-rank solve.

The diagnostic source and executable have their own identities in
`../evidence/recover_free_turn_step16/functional-trace-tested-state.json`.
The physical recovery uses the original protected executable, not that build.
The command manifests distinguish current source hashes from executable identity.

Diagnostic replay (deliberate unpublished stop before iteration40):

```sh
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/run.py recover-free-turn-trace-complete --executable build-particle-free/first-free-turn/recovery-functional-trace/aspect-release --resume-from build-particle-free/bulk-tip/runs/first-turn-positive-fourcore-end/output --snapshot-id 1 --trace --seconds 7200
```

Selected half-step fallback from the same accepted .075 s parent:

```sh
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/run.py recover-free-turn-positive-half --executable build-particle-free/first-free-turn/tested/aspect-release --resume-from build-particle-free/bulk-tip/runs/first-turn-positive-fourcore-end/output --snapshot-id 1 --dt .0025 --seconds 7200
```

Both commands are executed through the existing `refactoring_boundary/run_logged.py`
wrapper; exact launch environments and closed-process times are retained. No
historical run directory is reused. The pending clock in a restored parent audit
records its original .005 interval before the benchmark restart-dt hook changes
the new interval; the physical timestep log records the actual .0025 schedule.

Physically reflected history from initialization, then matching half-steps
(verify slot01 is the accepted step15 in the completed prefix before using it):

```sh
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/run.py recover-free-turn-reflected-prefix --executable build-particle-free/first-free-turn/tested/aspect-release --reflected --end-step 15 --point-stride 15 --seconds 10800
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/run.py recover-free-turn-reflected-half --executable build-particle-free/first-free-turn/tested/aspect-release --reflected --resume-from build-particle-free/bulk-tip/runs/recover-free-turn-reflected-prefix/output --snapshot-id 1 --dt .0025 --seconds 7200
```

The prefix exports native phase and section/plane data at every accepted step,
and full point fields at15. The resumed windows export full point fields at16/17
and audit the restored parent. Sparse point-export cadence changes observations
only; the complete physical histories remain in the checkpoints.

After the process and lossless export compression finish:

```sh
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/analyze_trace.py recover-free-turn-trace-complete
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/visualize_trace.py
```

After both signs finish, the matched endpoint and full-history accounting use:

```sh
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/analyze_endpoints.py recover-free-turn-positive-half --step 17 --reflected recover-free-turn-reflected-half
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/summarize_history.py --reflected
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/analyze_signed_sections.py recover-free-turn-positive-half --step 17
python3 benchmarks/reconstructed_fault/particle_free/recover_free_turn/analyze_signed_sections.py recover-free-turn-reflected-half --step 17
```

`analyze_endpoints.py` reduces accepted endpoints with native Bernstein phase,
native integration measures, section/plane histories and physical-work accounting.
If both signs reach a common endpoint it compares a physically reflected history,
not an invented mirrored result. Its figures deliberately omit any historical
control selected merely by step number: timestep halving changes that numbering.
The tiny-probe scalar subtraction estimate is not the full assembly-error bound.
Closed sparse row data, raw strip data, functional probes and failed logs remain
available for independent examination.
