# BP3 AMG startup timestep audit — September 26, 2026

The active limiter in `output-test-amg/` is **BP5 state startup**, specifically
its bound on predicted logarithmic state change. Increasing the first-step cap
and global maximum does not override this independently selected controller.
No simulation, Python script, source edit or parameter change was used for
this audit. Supplied outputs were read in place.

## Direct evidence

The executed [original.prm](output-test-amg/original.prm) and resolved
[parameters.prm](output-test-amg/parameters.prm) select:

- Maximum first time step: `4e6` s.
- Maximum time step: `4e7` s.
- Maximum logarithmic state change: `0.1`.
- Controllers: convection time step, reconstructed fault time step,
  BP5 state startup.
- Fresh start, seconds, AMG, Helmholtz normal filtering at 20 m.

The state bound also changed relative to the earlier filter-startup runs
(`0.02`). This is new executed-run evidence of relaxation, superseding the
recovery note's earlier statement that only the older policy was evidenced.
It does not edit the maintained candidate PRMs or establish temporal accuracy.

The [selection audit](output-test-amg/timestep_selection.csv) records:

| Accepted state | Time (s) | State proposal = selected next dt (s) | Fault-law cap (s) | Convection cap (s) | Growth cap (s) |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 552.083566 | 2,666,352.925 | 1,951,653,287.724 | Unrestricted |
| 1 | 552.083566 | 590.185962 | 2,630,057.023 | 1,950,994,176.163 | 1,054.479612 |
| 2 | 1,142.269528 | 630.854254 | 2,905,893.499 | 1,942,836,628.011 | 1,127.255187 |
| 3 | 1,773.123782 | 674.325550 | 3,210,682.257 | 1,931,985,274.021 | 1,204.931624 |

The ceiling is `4e7` throughout, the first cap is `4e6` at state zero only,
and termination reduction is false in all rows. The duplicated state-zero
record is not an additional physical step. An AWK comparison confirmed that
all five recorded selections equal the state predictor's proposal exactly.
The [predictor audit](output-test-amg/state_startup_predictor.csv) has four
unique states: each reported measure equals its `0.1` limit to within `1e-12`.

The [accepted-step record](output-test-amg/accepted_steps.csv) establishes
completed intervals of 552.084, 590.186 and 630.854 s. The copied
[log](output-test-amg/log.txt) reaches the beginning of timestep 4 with
dt = 674.326 s; completion of that step is not recorded in this upload.
The log identifies OPTIMIZED mode and 48 MPI ranks. Accepted rows report
passing fresh linear checks; no timestep cutback is needed to explain the
selected intervals.

## Why the bound gives approximately 550 seconds

[The predictor used for this run](gmg-bp5-cleanup-evidence/retired_state_startup.cc)
(subsequently retired from the BP3 build at the user's request) computes, from committed fault
state and slip rate, the largest interval satisfying

\[
\max_i \frac{b_i}{a_i}
\left|\log\frac{\Theta_{i,\mathrm{pred}}(\Delta t)}{\Theta_i}\right|
\leq 0.1.
\]

It uses a noncommitting constant-rate prediction with the production aging law
in [fault_friction.cc](../../../source/material_model/rheology/fault_friction.cc):

\[
\Theta_{\mathrm{pred}}=
\Theta e^{-V\Delta t/D_c}
+\frac{D_c}{V}(1-e^{-V\Delta t/D_c}).
\]

The [initial profile](output-test-amg/profiles/fault_0.csv) has shallow
Theta approximately 8,000 s and V approximately `1e-9` m/s. With `Dc = 0.008` m,
the steady state `Dc/V` is approximately `8e6` s, much larger than the initial
Theta. Thus the shallow state initially ages at nearly one second per second,
even though slip is slow: `dTheta/dt = 1 - V Theta/Dc`, approximately `0.999`.
Here `b/a = 0.015/0.010 = 1.5`, giving the useful approximation

\[
\Delta t \simeq 8000\,[\exp(0.1/1.5)-1]
=551.512846\ \mathrm{s}.
\]

An AWK evaluation of the exact frozen-rate formula over shallow exported
vertices gives `552.083566235` s, matching the controller. One vertex producing
that minimum is node 2846 at down-dip distance 840 m, with
`V = 9.99310818122e-10` m/s and `Theta = 8000` s. This independently explains
the magnitude from the exported physical state; the controller audit, not
this partial-domain reconstruction, establishes the global selected limit.

The bound permits approximately `exp(0.1/1.5)-1 = 6.89%` predicted shallow
state growth per step. As Theta increases, a larger absolute interval fits
the same fractional bound, explaining the gradual growth of dt. `0.1` is a
weighted logarithmic state-change limit, not simply a 10% Theta change or a
Newton convergence tolerance.

## Consequence and next bounded task

The larger first/global maxima are effective upper bounds, but are inactive
here. To obtain substantially larger startup steps one would have to relax or
remove the explicitly selected state predictor, which changes the temporal
accuracy policy. Without it, the current state's fault-law proposal is about
`2.666e6` s; that is a controller comparison, not an accuracy endorsement or
a prediction of the subsequent trajectory. The material's separate
`Initial time step = 4e6` is the artificial initialization Maxwell interval,
not the first physical timestep.

No additional data is needed to identify this limiter. If larger startup
steps are desired, the next bounded task is a matched-physical-time temporal
accuracy comparison for a proposed relaxation before adopting it in a long
run. No such comparison was launched or newly qualified by this audit.
