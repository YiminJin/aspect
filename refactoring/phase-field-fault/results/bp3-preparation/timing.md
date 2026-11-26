# Wall time through the third slip event

Two 160-rank Stampede3 jobs on the same 60° donor trajectory provide elapsed
time for the loading and accelerating regimes. The run starts from the
prescribed initial state in job 3610007. The resumed job 3616642 published
steps 816–4237 and reached its 48-hour scheduler limit during attempt 4238.
The last indexed profile, at 4182, still has a fault-wide maximum slip speed
of 1.37553 m/s. No duration to recovery or interval to a second onset has
been measured on this trajectory.

| Measured interval | Accepted steps | Simulated time advanced | Elapsed wall time | Mean wall s/step | 100-step blocks, wall s/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warm interseismic loading, 100→700 | 600 | 119.4205 years | 37,896 s | 63.16 | 60.33–69.99 |
| Late loading, acceleration and threshold crossing, 700→800 | 100 | 0.203121 years | 6,965 s | 69.65 | One mixed block |
| Resumed first slip episode, 900→4200 | 3,300 | 6.26825 seconds | 165,734 s | 50.22 | 48.38–58.08 |
| Latest timed part of that episode, 3900→4200 | 300 | 0.40984 seconds | 15,840 s | 52.80 | 49.96–58.08 |

Interseismic throughput was 11.34 simulated years per wall hour in the warm
window. All accepted fault maxima in that window were below 10⁻⁸ m/s.
The resumed event advanced only 0.1362 simulated seconds per wall hour.
Its latest 300 timed steps advanced 0.0931 seconds per wall hour. Saved fault
profiles sample the high-rate interval intermittently. Step counts and
physical clocks come from all accepted states. Directly extrapolating years
per hour through rupture would miss the adaptive collapse of the timestep
to milliseconds.

The first upward crossing of 10⁻³ m/s was step 774 at 163.893087 simulated
years. There were 3,464 accepted states from that crossing through 4237
inclusive, before a measured return to interseismic sliding. The wall timer
places the crossing between 12.68 and 14.61 hours from startup. Assuming a
constant cost per accepted step within the one mixed 100-step block gives
14.11 hours to onset. The last cumulative wall timer at 4200 reads 62.24
hours. Neither the trailing 37 accepted states nor scheduler downtime is
included in that number.

## Allocation scenarios

For a fresh run, if later loading intervals each cost the same 14.11 hours as
the first, three loading intervals plus two complete events reach the third
onset. Three complete events are needed to observe the third return. With
the measured event cost of 50.22 wall seconds/step the sensitivities are:

| Assumed advances after each onset through recovery | Wall hours to third onset | Wall hours through third recovery | Equivalent 48 h allocations through recovery |
| ---: | ---: | ---: | ---: |
| 4,000 | 153.94 | 209.74 (8.74 days) | At least 5 |
| 8,000 | 265.54 | 377.15 (15.71 days) | At least 8 |

If the server's logged checkpoint after accepted step 4229 can be recovered
and replayed, it already contains the first loading interval and 3,455
post-onset advances. From that parent the same scenarios cost **91.63 or
203.23 additional wall hours to third onset**, and **147.43 or 314.84
additional hours through third recovery**. These figures count the first
eight replay advances once. Restoring/checking the parent takes additional time.

These are budget scenarios for the measured donor mesh, hardware, MPI layout,
output and checkpoint cadence. The loading period includes the step that
crosses the onset threshold. In the first scenario just 537 further
steps remain for the observed episode, including its unobserved peak and
recovery. The two step totals give the cost of changing that assumption,
rather than a confidence interval. More steps, longer recurrence, time
spent writing a final safe checkpoint, queue delays or different solver costs
can exceed these allocations. The existing timing windows already include
the scheduled output and checkpoint overhead they encountered.

The first verified recovery will supply an event step count. The second
verified onset supplies the first recurrence interval. At each durable
checkpoint, update the forecast with those phase-specific counts and the
measured wall seconds per accepted step on the actual continuation. When
using a checkpoint, count only steps still to run. Until a matched
`pf-rsf` physical case is restartable and timed on the same layout,
the branch has no measured time to the third event.

## Reproduce the measurements

Run these commands from the current worktree root. They read the donor's
archived timing parser at an immutable Git revision, both original execution
logs and job 3610007's accepted-state table. Only the fresh scratch
directory is written. In the recorded run it was `build-bp3-prep/timing`.

```sh
D=../aspect-pf-rsf-particle-free
P=$D/benchmarks/reconstructed_fault/particle_free
T=$(mktemp -d)
analysis=refactoring/phase-field-fault/results/bp3-preparation/timing.py
git -C "$D" show 69e8a10a93a08718d31af961197899e320657c04:benchmarks/reconstructed_fault/particle_free/plot_bp3_computing_statistics.py > "$T/parse.py"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
python3 "$T/parse.py" "$P/bp3_dg_first_event/output_start/log.txt" \
  "$P/bp3_dg_first_event/output_resumed/log.txt" --output "$T/server-counters"
python3 "$analysis" "$T/server-counters" "$P/evidence/server_3610007/accepted_steps.csv" --resume-step 4229
```

The parser pairs the timestep header with the explicit accepted-state marker.
It excludes the interrupted attempts. In each invocation, subtract two
successive ASPECT integer-second `including restarts` timer readings and
count accepted rows between them. This cancels the restored wall-clock
offset. The first timer in each invocation lacks a preceding reading.
The warm-window mean excludes those start/resume blocks. The 700 and 800
cumulative readings used to bracket onset both come from the fresh run.
Producer code is in `source/simulator/helper_functions.cc` under
`Simulator::maybe_write_timing_output` in the donor. The generated
`server-counters/summary.json` records input hashes and ranks, and
`server-counters/timing_windows.csv` lists every pair used.
