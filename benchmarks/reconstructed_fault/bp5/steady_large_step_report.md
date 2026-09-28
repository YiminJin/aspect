# Steady BP5: physical timestep qualification

**Decision:** the requested 4e6-s ceiling passes nonlinear convergence but not
the local temporal-accuracy comparison. Use **125000 s** for the regenerated
steady-initialized server package. This passes the stated 5% work-weighted
screen; endpoint/max-norm limitations below remain explicit. No physics,
state-change safeguard or solver tolerance was changed.

## Scope and reproducibility

This is a separate bounded test of `bp5_steady_initialization`, not a continuation
of the old eight-day inverse-state run. Physics, mesh, native weak prestress,
friction, frozen phase, endpoint treatment and solver tolerances are unchanged.
The artificial initialization interval remains 4e6 s and does not age state.
All times below are seconds.

`weakening30-dc010-ell100/steady-large-step/` preserves immutable run directories,
parameters, source snapshots/hashes, logs, native work/state CSVs and checkpoints.
The sandbox MPI launch failed before ASPECT started; its log is separately
preserved. No failed numerical case was retried.

Only diagnostic instrumentation was added to `startup_time_step.cc`, gated by
`ASPECT_BP5_TIMESTEP_AUDIT`. It calls the existing read-only convection and fault
controllers; `determine_reaction` records the actual manager-selected timestep
and returns the unchanged default reaction. No controller formula was changed.
The new timestep run's initialized `state_work_0.csv` agrees bitwise with all
columns of the previously qualified 300-s startup, including native weak loads.

Commands (four MPI ranks, Release, one thread/rank):

```sh
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_steady_initialization -j4
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py prepare startup
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py run startup
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_large_step.py startup
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py prepare half4m
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py run half4m
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_large_step.py half4m
```

## Resolved clock and restrictions

Fresh startup uses maximum dt=4e6, end=1.6e7, last accepted real step=4,
graceful wall=1100 s and hard process cap=1200 s. The unchanged model list is
`convection time step, reconstructed fault time step, BP5 state startup` with
weighted logarithmic state-change limit 0.02. Resolved minimum dt=0,
first-step cap=5.69e300, growth cap=previous dt times 1.91. There is no function
clock, environment override or inherited 300-s restriction. Full-state dumps
are off; native weak/state, convergence and ordinary selected outputs remain.

| Real step | Convection proposal | Fault proposal | Predictor proposal | Ceiling | Growth cap | Selected |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.2206775794e10 | 6.6699428744e6 | 4e6 | 4e6 | inactive | 4e6 |
| 2 | 1.2206775794e10 | 6.6699428744e6 | 4e6 | 4e6 | 7.64e6 | 4e6 |
| 3 | 1.2237018777e10 | 6.6705137197e6 | 4e6 | 4e6 | 7.64e6 | 4e6 |
| 4 | 1.2236405712e10 | 6.6715612624e6 | 4e6 | 4e6 | 7.64e6 | 4e6 |

The ceiling controls all four steps; termination does not shorten any. Audit
rows are indexed by the accepted state proposing the **next** timestep; the
row after step 4 does not mean a fifth step was run.

All steps genuinely converge, including fresh linear checks. All 1156 nodes
remain free, minimum accepted alpha=1. Initialization has two accepted Newton
updates; real steps have 0,1,1,1 updates and 31,59,57,57 Krylov iterations.
Zero updates on step 1 means the initial iterate passed the residual tests,
not that its history was omitted. The exact aging audit passes to 2.22e-16.

| Real step | Predicted weighted log change | Realized weighted log change |
|---|---:|---:|
| 1 | 0.000303629 | 0.000303629 |
| 2 | 0.000291724 | 0.000652929 |
| 3 | 0.000627330 | 0.000998212 |
| 4 | 0.000959081 | 0.001329411 |

The predictor uses the preceding accepted rate; the realized measure uses the
new accepted rate. Both are below 0.02, but this is not a temporal-error bound.

## Same-checkpoint comparison

`startup/restart/01` is the accepted step-1 checkpoint at t=4e6. The one-step
control is the already computed step 2. `half4m` restores a copy and takes
two steps of 2e6 to the identical t=8e6. The existing checked archive utility
changes only the two pending-clock doubles (time,dt), at offset 237; preceding
dt, step index, all histories, FE state and prestress bytes remain identical.
All other checkpoint files are hash-checked. There is no reinitialization.

Native weak tractions below are **production row loads divided by their work
weights**, not particle-center tractions. RMS uses these same work weights.
The evolving-change denominator is the difference from the common accepted
step-1 state. It does not use the 50 MPa background or initial total shear.

| Quantity | Maximum difference | Work-weighted RMS difference | RMS difference / evolving change |
|---|---:|---:|---:|
| V | 4.76674e-13 m/s | 1.36604e-13 m/s | 199.8% |
| Accumulated slip | 1.81628e-6 m | 6.92572e-7 m | 0.01732% |
| Committed Theta | 1781.41 s | 677.497 s | 41.34% |
| Weak shear | 92.7025 Pa | 71.1192 Pa | 124.73% |
| Weak normal compression | 91.9094 Pa | 4.72754 Pa | 62.21% |

The common incoming Theta is bitwise identical. The final half-step correctly
consumes its midpoint update: it differs from the full-step incoming Theta
by at most 3882.11 s. That is the intended split timing, not restart mismatch.
Weak friction uses the actual incoming state at each solve, not newly committed
state. The full-step maximum rate error is 28.26% of the maximum evolving rate
change (the RMS comparison is larger because errors and changes have different
spatial distributions).

**4e6 is not qualified by this comparison**, despite successful nonlinear
convergence and very small errors normalized by total/background fields.
The authorized next comparison reduces the interval/ceiling to 1e6 from the
same t=4e6 checkpoint; it does not repeat the loading prefix.

Detailed result files are `startup/checks.json`, `half4m/checks.json`,
`comparison-half4m.json`, `timestep_selection.csv`, `state_work_*.csv`,
`work_weak_*.csv`, `accepted_steps.csv` and branch `checkpoint_source.json`.

### Reduced interval: 1e6 versus two 5e5 steps

From the same t=4e6 checkpoint to t=5e6 (`one1m`, `half1m`):

| Quantity | Maximum difference | Work-weighted RMS difference | RMS difference / evolving change |
|---|---:|---:|---:|
| V | 9.80696e-14 m/s | 4.10987e-14 m/s | 20.75% |
| Accumulated slip | 1.21983e-7 m | 4.74641e-8 m | 0.004747% |
| Committed Theta | 121.321 s | 47.2131 s | 20.50% |
| Weak shear | 20.4075 Pa | 18.1414 Pa | 15.10% |
| Weak normal compression | 6.19847 Pa | 0.916601 Pa | 13.77% |

This contracts materially but is still not adequate for ceiling qualification.
All three solves converge, remain entirely free, and accept alpha=1; the
full step uses 1 accepted Newton update/40 Krylov iterations (see accepted CSV
for exact counters), while midpoint and final incoming state are correctly
distinct. The realized state-change measure is at most 0.000137866.

The next authorized repeat is 250000 versus two 125000-s steps from the same
checkpoint. Before seeing those results, use a conservative **5% work-weighted
error/evolving-change screen** for V, committed Theta and both weak tractions;
also report slip and maxima. This is a benchmark ceiling-selection screen,
not a change to production solver tolerances or physical acceptance criteria,
and not a proof of full-cycle temporal convergence.

### Reduced interval: 250000 versus two 125000 steps

From t=4e6 to t=4.25e6 (`one250k`, `half250k`):

| Quantity | Maximum difference | Work-weighted RMS difference | RMS difference / evolving change |
|---|---:|---:|---:|
| V | 5.36605e-14 m/s | 1.14745e-14 m/s | 4.635% |
| Accumulated slip | 1.03417e-8 m | 3.12806e-9 m | 0.001251% |
| Committed Theta | 10.3290 s | 3.12408 s | 6.677% |
| Weak shear | 5.36088 Pa | 4.88712 Pa | 3.582% |
| Weak normal compression | 5.25929 Pa | 0.262520 Pa | 3.272% |

The state increment still fails the predeclared screen, so 250000 is not
qualified. The next comparison reuses the **first** 125000-s step of `half250k`
as its full-step control, and adds only `half125k` (two 62500-s steps to
t=4.125e6). There is no new startup or redundant full-step run. Checkpoint
history remains the identical accepted t=4e6 state in every comparison.

These are local step-doubling tests over progressively shorter intervals,
**not** a multi-resolution trajectory convergence study over one common final
time. Each individual pair has an identical beginning and end time.

### Selected interval: 125000 versus two 62500 steps

Both branches start at the identical accepted t=4e6 checkpoint and end at
t=4.125e6. The full-step result is reused from `half250k/state_work_2.csv`.

| Quantity | Maximum difference | Work-weighted RMS difference | RMS difference / evolving change |
|---|---:|---:|---:|
| V | 7.24379e-14 m/s | 6.53280e-15 m/s | 2.554% |
| Accumulated slip | 4.99272e-9 m | 8.24012e-10 m | 0.0006593% |
| Committed Theta | 4.99106 s | 0.823518 s | 3.660% |
| Weak shear | 3.86510 Pa | 2.67827 Pa | 1.924% |
| Weak normal compression | 6.46488 Pa | 0.277371 Pa | 3.349% |

This passes the predeclared RMS screen. The maximum V and normal-traction
errors occur at the top endpoint (xd=0), and are 9.24% and 8.82% of the
respective *global peak evolving changes*. The maximum shear difference is
at the bottom endpoint (xd=115470.05 m). The absolute normal-traction RMS
error increased slightly from 0.26252 to 0.27737 Pa between the last two
intervals. Thus **pointwise/endpoint temporal convergence is not established**;
do not present this bounded RMS qualification as full trajectory convergence.
The V maximum error is also 7.24e-5 Vp, but that total-rate normalization is
not what determined the ceiling.

The initial Theta in both branches agrees bitwise. The final half-step's
incoming Theta includes its single correct midpoint update (maximum difference
108.890 s from the full-step incoming state). Predicted/realized state changes:

| Solve | Predicted | Realized | Accepted Newton updates | Krylov iterations | Minimum alpha | Lower-active |
|---|---:|---:|---:|---:|---:|---:|
| Full 125000 | 9.29409e-6 | 1.63976e-5 | 1 | 33 | 1 | 0 |
| First 62500 | 4.64850e-6 | 8.16645e-6 | 1 | 33 | 1 | 0 |
| Second 62500 | 8.16135e-6 | 7.85679e-6 | 1 | 33 | 1 | 0 |

All new solves satisfy the unchanged nonlinear 1e-8 criteria and fresh linear
checks. Every branch checks the exact aging update, incoming/committed handoff,
once-only accumulated slip update, and frozen native background. Restart never
reruns steady initialization. The previously qualified full-state restart
comparison is reused, not repeated.

Commands for the reductions:

```sh
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py prepare one1m --ceiling 1e6
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py prepare half1m --ceiling 1e6
# run both, then:
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_large_step.py one1m
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_large_step.py half1m --one one1m
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py prepare one250k --ceiling 250000
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py prepare half250k --ceiling 250000
# run both, then:
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_large_step.py one250k
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_large_step.py half250k --one one250k
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py prepare half125k --ceiling 125000
python3 benchmarks/reconstructed_fault/bp5/run_steady_large_step.py run half125k
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_large_step.py half125k --one half250k
python3 benchmarks/reconstructed_fault/bp5/package_steady.py half250k half125k
```

## Server package

`server-30km-steady/` is separate from every old package. It uses only
`libbp5_steady_initialization.release.so`, an empty captured-prestress filename,
physical ceiling 125000 s, artificial interval 4e6 s and state predictor 0.02.
Fresh-start output goes to `output-30km-steady`; old eight-day checkpoints are
incompatible and must not be resumed. The 4-step/1.6e7-s test limits are removed;
first-event-through-decay stopping, 1500-year safety end, sparse coordinated
output, lightweight diagnostics and hourly/on-termination checkpoints return.
Expensive per-step full-state exports and opt-in timestep/probe callbacks are off.

The server's first-step ceiling is inferred from the same initialized state
and monotone controller restriction; a new four-step run at 125000 s was
**not** added. The requested reduction tests only the same loaded checkpoint
interval. No long trajectory or broader convergence campaign was launched.

## Execution cost and scope

Fresh four-step startup: 511.28 s; two-half-step replay: 189.66 s.
Peak child RSS values are 2,754,152 and 2,903,016 KiB respectively; these are
maximum individual child-process measurements, not summed four-rank memory.
All runs retain the 1100/1200-s soft/hard limits. No long run was launched.
The 1e6 full/half runs cost 101.00/171.58 s, 250000 full/half cost
90.25/159.25 s, and the final two 62500 steps cost 154.33 s. Total simulation
execution was 1377.34 s (23 minutes), excluding build/analysis and the pre-ASPECT
sandbox launch failure. No repeated full startup or automatic numerical retry.

Changes in this task are benchmark-local: opt-in timestep audit, immutable
run/comparison scripts, package generator/template, report and maintained README.
No production ASPECT equation/controller/solver change. The package guard was
also tested to reject the 250000-s result before creating any package. Python
syntax compilation passed; relevant numerical verification is the seven bounded
four-rank runs above. No full ASPECT suite was run.

Post-packaging checks also pass: manifest verification, `bash -n` for the
environment/batch scripts, an independent Release `-j4` build of the packaged
standalone plugin, and ASPECT `--validate` for both fresh and resume parameters
under the clean packaged environment. Validation changes only the local plugin
and output paths into `/tmp`, not scientific inputs; it is parameter validation,
not another simulation or a server ABI test. Results are recorded in
`steady-large-step/server_package_check.json`. The package evidence/provenance
snapshot predates these post-packaging checks and remains immutable.
