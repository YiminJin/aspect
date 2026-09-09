# K1 correctness baseline and completed bounded convergence cross

2026-09-09. **Gate K1 passes with the reviewed provisional K1-only 1e-4
containment allowance.** The actual slip-normalization requirement remains
1e-4; full I_h, support, initialization, physical data and configured solver
tolerances are unchanged. This verifies the finite-step finite-width equations,
not time resolution of the initial transient at dt=0.5 s or a sharp-fault limit.

## Provenance and algebraic correction

The diagnosis and accepted safeguards are detailed in
`stage_K1_residual_floor_review.md`. The local cellwise constant-velocity
subtraction removes avoidable strain-evaluation cancellation. The remaining
represented-iterate floor uses the independently computed matrix/state
precision scale documented in current_design.md/specification.tex, fixed for
both convergence and merit. It does not fit a threshold to a stalled residual.
The original pressure compatibility cap and fresh linear checks are retained.

All evidence below is under
`benchmarks/reconstructed_fault/uniform_shear/residual-floor/`. Each run's
resources.json records exact command, binary/plugin hash, parameter hash,
wall time and peak RSS. Git HEAD is f4032b1824ef4892020af8c58ef981aca03dc0ec
plus the preserved working tree. `correctness-baseline.patch` saves tracked
changes; `correctness-baseline.sha256` identifies the tested executables/plugin.
`correctness-source.tar.gz` includes the exact production headers/sources,
Stage-I unit file and benchmark plugin, including untracked source additions.
No commit or unrelated cleanup was requested. Production code is frozen for K2.1.

## Spatial errors, dt=0.5 s, fixed ell=0.15625 m

All trajectories reach 6 s. Each scalar reference starts **once** from its
own retained initial C0, Theta0=200 s, stress0=1500 Pa and independently
integrated complete Q1 profile. Later ASPECT histories never reset it.

| Bulk mesh | max V error (m/s) | max Theta error (s) | max C error (Pa) | max slip error (m) | raw initial stress RMS / max (Pa) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16x64 | 3.33924e-7 | .0243501 | .00334268 | 3.76374e-7 | 2.80838 / 9.03456 |
| 32x128 | 1.06211e-7 | .00872712 | .00119343 | 1.34284e-7 | .697651 / 2.29051 |
| 64x256 | 5.81167e-8 | .00560269 | .000767293 | 8.63094e-8 | .174299 / .578196 |

Raw stress is the complete unsmoothed QP Maxwell response, including the
frozen old-stress term. Its initial error decreases approximately fourfold
per refinement. The prior cell-ID/weighted-moment audit remains valid;
neither extrema nor transverse oscillations are filtered. The finest mesh
passes every existing resolved conditional check (0.2% relative plus 1e-5
of the fixed physical diagnostic scale), including raw point stress and
velocity profiles. The coarser meshes do not pass all resolved checks;
they establish convergence, not three independently resolved solutions.
Slower trajectory-error reduction toward the retained-support floor is
reported rather than hidden by renormalizing I_h or extending support.

Initialization changes are separate from mechanical errors:

| Mesh | center phi | independent I_h (m) | retained C0 (Pa) | omitted fraction | max actual slip-normalization error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16x64 | .600582656 | 108.072238204 | 313.511760147 | 5.64918e-5 | 5.86985e-5 |
| 32x128 | .599910323 | 108.098095072 | 318.747655768 | 5.83027e-5 | 5.23649e-5 |
| 64x256 | .599764017 | 108.144833844 | 317.513762685 | 5.91396e-5 | 6.00643e-5 |

C0 is nonmonotone with resolution. Comparing trajectories initialized with
different C0 as if that difference were purely a mechanical error would be
incorrect. The normal-profile, geometry, initial projection and particle-domain
checks are rerun at every resolution. The measured half-width remains
.3088215939070757 m. Finest initialization converges in 14 nonlinear updates
to relative phase residual 1.210e-9 (requested 1e-8).

## Timestep convergence on 64x256

The three independent references share the same retained initial data and
are advanced at the **actual accepted** loading/timestep sequences. No CFL
shortening occurs. The continuous-time diagnostic is an independent scalar
dt=1/512 s trajectory, checked against dt=1/256 s; its difference is recorded
in each errors.json. It is not a new ASPECT reference fitted to output.

| dt (s) | max V error versus matching finite-step reference (m/s) | V temporal error at t=2 (m/s) | Theta temporal error at t=2 (s) | C temporal error at t=2 (Pa) | slip temporal error at t=6 (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.59237e-8 | 2.21581e-4 | 5.23161 | -.457407 | 1.06630e-5 |
| 1 | 3.88957e-8 | 6.69969e-5 | 1.73919 | -.152234 | 5.72788e-6 |
| .5 | 5.81167e-8 | 1.94830e-5 | .496875 | -.0422812 | 2.84522e-6 |

All finest-mesh timestep cases pass the unchanged finite-step reference
criteria. The early V transient still has about 19% temporal error at .5 s;
this is not claimed as a continuously time-resolved trajectory. Common-time
errors decrease overall; not every signed diagnostic is monotone (e.g.
accumulated slip at t=4 for dt=2 versus dt=1). The complete t=2,4,6 table
is retained, including those entries. The accepted K1 gate asks convergence
and finite-step reference agreement, not a newly invented continuous-time
accuracy target. No full parameter product or additional accuracy threshold
was introduced.

## MPI, restart and failure safeguards

- The unchanged .5 s coarse trajectory completes on one and two ranks. All
  50 linear calls per run pass fresh residual and compatibility checks.
  t=4.5 s bulk residuals are 4.44915e-13 / 4.51076e-13; both genuinely
  converge and continue through 6 s. Separate velocity/continuity and
  dimensional/normalized surface values are in one-/two-residuals.json.
- `rank-comparison.json`: max velocity difference 1.36e-18 m/s (ux),
  pressure 9.41e-11 Pa, V 1.77e-17 m/s, Theta 3.37e-12 s. Each run also
  passes its own measured support and actual normalization checks.
- K1 restart from t=4 s gives bit-for-bit identical exported bulk, surface,
  H and particle stress through 6 s. Frozen original phi/H snapshots survive
  restart; they are not resampled from a later solution.
- All twelve selected integration regressions pass, including changed
  loading, pressure gauge, condensed dynamic/prescribed modes, exhaustion,
  rollback on both ranks and Stage-J restart. The initial concurrent Stage-J
  timeout/stale-checkpoint failure is retained and explained; the unchanged
  create/resume pair passes in isolation (374.52/128.16 s).
- Stage-I tests pass 52 assertions in 10 cases on one rank and on each of
  two ranks. A residual 100 times the independent precision scale still
  fails convergence and exhausts Armijo. Fifteen Python reference/analysis
  tests pass. Build uses -j4; no complete integration suite was run.

Representative precision allowance: the coarse late-step smooth Q2 shear
mode calibration is 6.94e-16 m/s, versus a 1e-9 m/s physical absolute error
allowance. This is a mode calibration, not a bound on the inverse of every
coupled operator. Actual remaining velocity directions are also recorded
(at t=4.5: 1.30e-16 m/s; at t=6: 1.79e-19 m/s). Surface convergence remains
independent and strict; stagnation is not an acceptance condition.

## Reproduction and cost

For each case in {space16_dt05, space32_dt05, space64_dt05, space64_dt1,
space64_dt2}, use the checked-in output-only override under residual-floor/
convergence with `convergence/run_case.py --configuration Release --timeout
7200`. Each override includes the original approved fixture. Analysis commands:

```sh
python3 analyze.py OUTPUT --provisional-k1-containment > CASE-analysis.json
python3 analyze_convergence.py OUTPUT CASE-analysis.json > CASE-errors.json
python3 summarize_residual_floor.py CASE.log --nx NX > CASE-residuals.json
python3 summarize_convergence.py residual-floor/convergence
```

Commands are relative to the uniform_shear benchmark directory; runner commands
and exact absolute paths are saved per run. Wall seconds / peak RSS KiB:
coarse 138.14 / 260968; middle 171.29 / 397912; fine .5 972.57 / 933312;
fine 1 630.84 / 933748; fine 2 410.92 / 940604. Coarse MPI 227.25 s /
260968 KiB (maximum child, not summed memory); K1 restart 61.65 s / 260804
KiB. Some runs shared the machine with focused tests; these are observed
costs, not scaling benchmarks.

Artifacts: `convergence/case_summary.csv`, `common_time_comparison.csv`,
`summary.json`, `initialization_differences.json`, `spatial_errors.png`,
`initial_profiles.png`, `histories.png`; per-case raw QPs and sampled
transverse profiles in `CASE-errors/`. Historical failure artifacts remain
in their original directories. K2.1 may now use this frozen correctness
baseline; **the K1 containment allowance does not transfer to K2**.
