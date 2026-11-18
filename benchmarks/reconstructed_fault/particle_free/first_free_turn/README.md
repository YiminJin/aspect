# First native free-turn experiment

Current reviewed result: the positive history accepts through step15, .075 s.
The final .080 s interval fails phase KKT at the unchanged 80-iteration limit.
The reflected case has not run. See the [result report](../../../../doc/reconstructed_fault/particle_free/FIRST_FREE_TURN_REPORT.md).

Opt-in material-area/reference-arclength RSF with fixed vertical material strips.
The coupled native phase VI determines the path. See
[`FIRST_FREE_TURN_FORMULATION.md`](../../../../doc/reconstructed_fault/particle_free/FIRST_FREE_TURN_FORMULATION.md)
for equations, membership, conjugate selection and limitations.

`positive.prm` and `reflected.prm` are the selected peak-.60, 128×128 inputs.
The whole load history is reflected across y=2. At t>.04 a traceless normal
strain rate changes the homogeneous instantaneous preferred plane by 20 degrees.
The original horizontal material remains structurally horizontal.

The runner uses the frozen `build-particle-free/first-free-turn/tested` pair,
midpoint dt=.005 and one thread per rank. It refuses an existing run label.
The user selected four physical cores after the positive prefix. With `--ranks 4`,
MPI binds ranks to distinct cores; verified lossless compression runs only after
MPI exits. Do not run a numerical analysis worker concurrently with four ranks.
For a fresh reproduction, use new labels and run these windows sequentially:

```sh
python3 benchmarks/reconstructed_fault/particle_free/first_free_turn/run.py positive-prefix --label repeat-positive-prefix --ranks 4
python3 benchmarks/reconstructed_fault/particle_free/first_free_turn/run.py positive-end --label repeat-positive-end --ranks 4 --resume-from build-particle-free/bulk-tip/runs/repeat-positive-prefix/output
python3 benchmarks/reconstructed_fault/particle_free/first_free_turn/run.py reflected-prefix --label repeat-reflected-prefix --ranks 4
python3 benchmarks/reconstructed_fault/particle_free/first_free_turn/run.py reflected-end --label repeat-reflected-end --ranks 4 --resume-from build-particle-free/bulk-tip/runs/repeat-reflected-prefix/output
```

The prefix stops after accepted step12 with the physical end time still .08;
continuation uses that same law, state and terminal clock. No straight checkpoint
is converted. `--rejection --rejection-step N` selects the existing deterministic
rollback checker, and `--snapshot-id 3` explicitly recovers slot3 when needed.
Restart staging copies only the selected snapshot and leaves the source tree
intact. A nonuniform short two-to-four-rank restore and first-append comparison
is recorded in `four-rank-recovery-verification.json`.
`strip-append` is a 32×16 lifecycle control, not a spatially qualified turn.
`smoke-baseline`, `smoke-native`, `smoke-strips` compare protected/default/opt-in
straight behavior. The protected executable/checker are never overwritten.

`verify_recovery.py` checks every common exported physical-history field,
phase/solution equality, section and plane state, restored parent14 and final16.
`analyze.py LABEL...` uses the native Bernstein polynomial and physical weights;
`--pair-step N` additionally compares reflected fields and draws the common
absolute phi=.05 contour on cell-resolved physical axes. Contour subdivision
16/32 is an observation check, not physical mesh refinement.

The original positive history is split across `first-turn-positive-prefix`
(steps1–12, two ranks), `first-turn-positive-end` (13–14, two ranks) and
`first-turn-positive-fourcore-end` (accepted15, failed16; four ranks). The unaccepted two-rank
trial15 was deliberately stopped to change rank count; preserve its log and cost.
`summarize.py` combines the completed accepted windows, checks that no interval
is omitted or counted twice, and includes that interrupted work in execution cost.
Use `--seed-label first-turn-positive-prefix` when analyzing the four-core window.

To reproduce the delivered analysis without another simulation:

```sh
python3 benchmarks/reconstructed_fault/particle_free/first_free_turn/analyze.py first-turn-positive-prefix first-turn-positive-end
python3 benchmarks/reconstructed_fault/particle_free/first_free_turn/analyze.py first-turn-positive-fourcore-end --seed-label first-turn-positive-prefix --restore-parent first-turn-positive-end --parent-step 14
python3 benchmarks/reconstructed_fault/particle_free/first_free_turn/summarize.py
python3 benchmarks/reconstructed_fault/particle_free/first_free_turn/visualize_completed.py
```

`summarize.py` defaults to the available positive history; `--include-reflected`
requires actual reflected data. It validates contiguous accepted steps without
asserting that the target was reached. `visualize_completed.py` explicitly checks
the accepted15/failed16 outcome and writes four figures (PNG and PDF, including
the evolution figure from `summarize.py`), CSV plot tables and compact JSON.
It cannot turn an unaccepted trial or a reflected copy into a physical endpoint.
The phi=.05 front has 16/32/64/128 contour extraction checks and a local native
Q2-root refinement. These improve observation accuracy, not physical resolution.

Raw run outputs/checkpoints remain under the ignored build tree. Compact evidence,
inputs, logs, source/binary identities and figures reside in
`../evidence/first_free_turn/`. The original small-phase registration and failed
unconditional-bound-polish experiment are retained. Never interpret tiny positive
formation as resolved contour extension or delete it by a positive cutoff.
