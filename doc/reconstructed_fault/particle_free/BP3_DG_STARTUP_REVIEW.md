# BP3 DG source-package review and bounded startup

17 October 2026. Worktree `aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved
incoming work. This task repairs stopping/output participation, supplies a
bounded server startup and measures local resource controls. No server job is
submitted or launched. The previous source archive and all scientific evidence
remain intact. `strict_reference_match_passed=false`.

## Collective decision and source audit

In `bp3_native/server/plugin/output.cc`, each rank computes
`local_stop = end_time_due || accepted_step_due || local_wall_due || event_due`.
`long_run_stop = MPI_max(unsigned(local_stop))` now precedes every dependent
output decision. The full request is reduced; reducing only a later termination
criterion would leave the earlier output collectives unsafe.

The heavy request (including the existing maximum-slip increment reduction) is
also reduced before `allow_native_output`. The profile request is reduced before
`write_long_profile` and its root-write/broadcast. `BP3 output complete` consumes
the same uniform heavy flag after native writers, commits the schedule on every
rank and clears only that flag. It does not clear `long_run_stop`. The scheduled
observer depends on `BP3 output complete`, uses the same persistent stop flag,
and reduces its field request before all dense exports/index collectives.
`BP3 long run complete` returns that same collective flag. Configuration and
writer names are common across ranks. No rank-local elapsed-time condition
remains around a collective writer in this path.

The dedicated regression CMake target alone defines
`ASPECT_BP3_TEST_NONROOT_WALL_STOP`: rank 1 requests a wall stop at positive
accepted step 2, with per-rank request/schedule traces. Ordinary server builds
have neither this definition nor a new test parameter/environment switch.
The physical clock/recurrence, acceptance/publication checks and tolerances are
unchanged. Resource-only timers wrap dense stress/coefficient exports; one native
`Resume snapshot` scope measures checkpoint read/restoration. Checkpoint storage,
rotation and contents are unchanged.

## Completed local qualification

The regression uses the retained 2000×1000 m strengthening mature fixture,
unlimited homogeneous domain-free DGQ2, Gauss6/source3, Helmholtz20 and stationary
points. It has 4136 cells, 149538 native solution DoFs, **111672 additional DG
history DoFs**, and 66176 fixed particles (33088 on each of two ranks).
The test timestep sequence is explicitly 200000 s; it isolates output/restart
and is not an adaptive-first-event accuracy test. Each native test has a 240 s
process-group timeout. Production startup/event inputs retain native adaptivity.

- Only rank 1 requests wall stop at step 2; rank 0 requests none. Both observe
  stop/heavy/profile=1. Other stop conditions and ordinary heavy/profile output
  are all false there. Initial and terminal field/native output indices are
  exactly steps 0 and 2; the run exits successfully and labels its checkpoint
  with accepted step 2, t=400000 s.
- Continuation reaches step 5, t=1000000 s, with the identical accepted timestep
  sequence. All compared DG coefficients, physical FE fields, phase/I/Z,
  rate/state/slip, reconstructed full tensor and weak loads have **zero**
  difference against the uninterrupted control. Full bulk/particle rank audit
  files are byte identical. Existing lifecycle checks assert the cumulative
  publication count equals the accepted positive step; each accepted solve's
  observer callback is seen once.
- The separate scheduled-output split/restart regression passes with zero
  compared difference: uninterrupted snapshots 0/3/5, restart snapshot 5 only,
  and no spurious step-4 output (`qualification/regression.json`).
- Rebuilt native unit suite: **124 cases / 118034 assertions pass**. All four
  standalone parameter files pass native `--validate` using the ordinary BP3
  wrapper and relative paths. An independent parameter-map comparison permits
  only resume mode, stop step, wall budget and checkpoint cadence to differ.
  Parser validation does not construct the full mesh or qualify server runtime.

Replay sources are in `bp3_native/tests/collective_stop/`. Current-build native
outputs, inputs, commands, hashes, request traces, RSS and complete timer tables
are retained under `build-particle-free/bp3-startup-review/verified/`; compact
results are included in the source distribution's `qualification/`. Preliminary
successful pre-read-timer controls are also retained. The initial harness attempt
failed before native launch because `/usr/bin/time` is absent locally; the
harness now uses native-child `getrusage` without that executable. Original
failed logs are retained; no physical failure was overwritten.

## Measured local costs (two ranks)

Native-child peak RSS covers the entire launch including checkpoint write/read;
these are per-rank high-water values, not isolated checkpoint-window peaks or
measured node totals. Timings are rounded native timer entries; nested sections
must not be summed as disjoint costs. Concurrent local tests and filesystem cache
state affect cost. Full raw values/commands are in `qualification/costs.json`.

| Local trajectory | Harness elapsed s | Peak RSS rank 0 / 1 MiB | Condensed solve s | Checkpoint write / read-restore s | Dense stress / coefficients s |
|---|---:|---:|---:|---:|---:|
| Five-step uninterrupted |100.55|427.12 / 414.73|38.8 (10 calls)|.636 / —|1.44 / .153 (2 snapshots)|
| Wall stop at step 2 |56.72|421.72 / 408.27|18.9 (4 calls)|.640 / —|1.47 / .148 (2 snapshots)|
| Wall checkpoint → step 5 |47.77|426.16 / 421.48|19.2 (6 calls)|.437 / .222|.410 / .0369 (1 snapshot)|

The step-2 checkpoint directory is **14.235 MiB** including terminal metadata;
the uninterrupted step-5 directory is **14.238 MiB**. All-rank dense files total
6.806 MiB at initialization, 13.208 MiB at step 2 and 13.198 MiB at step 5.
These include authoritative tensors, DG/physical-FE coefficients and normal cuts,
not per-step profiles/audits or checkpoints. Local whole-node peak memory was not
sampled; summing rank high-water values would only give an upper estimate.
The native full-grid startup must collect the missing node and checkpoint-window
memory evidence as specified below.

## Prepared server startup and resource gate

[Manual package procedure](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/README.md),
[resource collection](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/RESOURCES.md).

`startup.prm` retains the full event geometry, fixed graded mesh and physical
inputs; it stops at accepted step 5, checkpoints every accepted step and uses a
7200 s graceful budget. `startup-resume.prm` resumes it to accepted step 8 with
the same settings. `parameters.prm` and `resume.prm` remain the first-event inputs.
All four retain the 47336400000 s end cap, first/max timestep caps 4e6/4e7 s,
log-state bound .2, growth/cutback rules, initial constitutive interval 4e6 s,
Gauss6/source3 and Helmholtz20. No fixed 50000 s timestep cap is introduced.
Unlimited DGQ2, frozen phase/geometry/mesh, stationary diagnostic particles,
normalization/derivatives, constitutive laws and all acceptance tolerances remain.

The first server startup must record actual cells/native and DG DoFs/particles,
per-rank and per-node peak RSS around checkpoint write/read, mechanical and
checkpoint/dense timings and all-rank logical/allocated file sizes. Native
`Create snapshot`, new `Resume snapshot`, dense-export and mechanical timers
supply durations; scheduler/process/node accounting supplies memory. VmPeak is
not RSS, and scheduler MaxRSS is not node-total memory. No full-grid server
memory/cost measurement is claimed here.

The current serializer gathers 39 doubles per 2D cell globally on every rank,
then merges global maps and packs vectors with overlapping lifetimes. For the
estimated 552084 cells, one numerical copy is 172250208 bytes (164.27 MiB) per
rank, excluding multiple copies, map/CellId overhead and all other state. This
replicated term is not divided by the MPI rank count. Storage is not redesigned
and particles are not removed.

A graceful wall budget is checked only at accepted states. Scheduler margin must
cover another solve, terminal output and a complete checkpoint. Preserve startup
and resume evidence before subsequent rotation. After startup succeeds, direct
`ibrun ./plugin/aspect-release resume.prm` continues into the unqualified event.
After abrupt termination, preserve the failed output and branch from the complete
`restart/` directory/last-good marker in a fresh output directory; never delete
newer diagnostics or weaken the consistency guard to force an in-place resume.
The build requirements now explicitly include deal.II with SUNDIALS support.

## Retained limitations and next action

The full-grid startup is prepared, not executed. The first-event run is an
experiment, not a qualified accuracy result. Fixed-FF refinement changes localized
tensor stress by 6.598 kPa. The absolute endpoint 1% shear/filter temporal screen
still fails (1.150/.523/1.047% reported controls), despite the smaller uncertainty
in the signed spatial contrast. The 49.704 Pa Q1 surface-state interpolation
imbalance remains. Neither particles nor DG is an accuracy reference. No limiter,
filter, constitutive, propagation, mesh-sweep or coupled-evolution work follows.
The next action is the user's bounded server startup and resource assessment,
then a separate decision about continuing the first-event experiment.
