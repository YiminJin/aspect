# Bounded replay and adaptive continuation

**Current status: restart qualification is blocked.** The fresh 0–4 prefix
matches the cleaned reference exactly, but cold Ih recomputation on resume
differs by roundoff and trips the mature law's bitwise fixed-profile invariant.
No resumed state was accepted. Continuation is therefore **prepare-only**;
the launcher rejects execution until this core restart issue is resolved and
qualified. Commands below document the intended bounded mode, not permission
to start a longer run. See `stage_K5_research_restart_report.md`.

The 100-km reference fixture is `bp3_modified_fully_frictional.prm`:
fully frictional continuous Q1 velocity/state/slip, mature C=0, work-weighted
mechanics, paired boundary completion/continuation, fixed initial background,
and the unchanged split history cycle. This is **modified BP3**.

Run commands below from the repository root. All runs use the same Release
ASPECT executable, four MPI ranks, explicit sparse B/G and pivoted tridiagonal
surface solve. No comparison/profiling or retired investigation switches are
inherited from the environment. Every fresh output directory must be new.

## Seven-step regression / qualification

```
python3 benchmarks/reconstructed_fault/bp3/run_research.py --configuration frictional --velocity-preconditioner amg --mode replay --output /path/to/new-replay
python3 benchmarks/reconstructed_fault/bp3/run_research.py --configuration frictional --velocity-preconditioner amg --mode restart-qualification --output /path/to/new-restart-test
```

These commands explicitly retain the original 100-km AMG restart-audit fixture;
the launcher now defaults to the 200-km wide research model with GMG. The qualification
runs 0–4, checkpoints and exits, then starts a separate MPI
process from that ordinary checkpoint for steps 5–7. It retains the complete
clock in both invocations and preserves the step-4 checkpoint before rolling
checkpoint replacement. ASPECT checkpoints **after advancing its clock**:
this contains the physical state committed after step 4, with the step-5
time/dt already selected. This is not a step-5 committed state.

The version-4 benchmark record preserves diagnostic accumulated slip, preceding
Theta, last accepted step, event/output state, the two formulation selectors,
and the original H/geometry/Ih audit baseline. Physical particle/fault histories
and background coefficients remain owned by the ordinary simulator/manager
checkpoint. Work-measure version-2/3 checkpoints are deliberately not converted.
Completion inputs must remain present and hash-identical. The background table
is **not** reloaded/recalibrated on resume; its serialized coefficients are used.

## Ordinary adaptive continuation (not launched by this task)

For example, to continue an already qualified output directory, stopping no
later than absolute physical time 10 yr, accepted step 20, or a 2400-s process
budget:

```
python3 benchmarks/reconstructed_fault/bp3/run_research.py --configuration frictional --velocity-preconditioner amg --mode continuation --resume \
  --output /path/to/qualified-output --end-time 315576000 --end-step 20 --wall-seconds 2400 --prepare-only
```

Without `--resume`, the same command starts fresh in a new directory. The time
and step limits are **absolute**, not durations since restart. Supply limits
beyond the checkpoint's last accepted state. `--prepare-only` creates inputs
and provenance without execution (in a new disposable preflight directory for
fresh validation); it does not authorize or launch a trajectory.

The continuation selects only the normal convection and reconstructed-fault
time-step models: no saved clock and no imposed comparison times. The first
resumed timestep is already chosen in the checkpoint; subsequent choices use
those ordinary controllers. It preserves physical acceptance, aging, inert-H,
Maxwell, geometry and endpoint checks. End-time, end-step and wall-time criteria
all request a final checkpoint. Checkpointing also occurs every accepted step.

The wall criterion is checked only between accepted solves, at 60 s less than
the process cap. An expensive solve can overrun that soft limit: `timeout`
sends TERM at the hard budget and KILL after at most 15 s grace. If interrupted
mid-step, recover only the last complete checkpoint identified by
`restart/last_good_checkpoint.txt`; do not regard an incomplete trial/output as
accepted. There is no automatic retry. Protect or copy evidence before another
continuation invocation; existing `continuation.*` launch records are not
silently overwritten.

Same-four-rank restart is the qualification scope. Cross-rank restart and
long-time/adaptive-trajectory accuracy are not established by this short test.
