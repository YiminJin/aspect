# K4.1b — Matched finite-width differences under timestep refinement

## Decision

Matched width differences are substantially more stable than the individual
trajectories, and contract on the extra authorized scalar level. Nevertheless,
the **complete transient of all K4 observables is not yet temporally resolved
to the existing accuracy level**. No K4.2 run is started. The original K4.1
criterion and its failed decision remain unchanged. A separate width-specific
criterion is proposed below for review, not silently substituted or passed.

## Bounded execution

First read the existing .25/.125-s trajectories at all 25 common times,
t=0,.25,...,6 s. For each width define e_w=Q_w^dt-Q_w^(dt/2). Compute

\[
 \Delta_\ell Q=Q_{\ell/2}-Q_\ell,\qquad
 E_\Delta=\|e_{\ell/2}-e_\ell\|,\qquad
 E_{\rm common}=\|\tfrac12(e_{\ell/2}+e_\ell)\|.
\]

Retain individual changes ||e_ell|| and ||e_half|| separately. Scalars retain
the signed width difference before taking its change's absolute value.
Velocity profiles are subtracted pointwise on the common physical y grid
before maximum/L2 norms; subtracting two norms is not the same diagnostic.

For V/q/C/Theta/slip the maximum matched change was 1.2--2.1% of the sum of
the individual maximum changes. For velocity maximum/L2 it was 20.75%/11.94%.
This justified the one additional .0625-s scalar level for both widths.
The run-eligibility check (ratios below .25) is not an acceptance criterion.

Reused the same 4096-cell independent initialization, full I_h, fixed H/phi,
K1 loading, support and retained initial histories. No phase solve, production
solver, geometry, tolerance or initialization change was made. Each new
trajectory contains timestep zero plus 96 real steps to 6 s. Both scalar runs
together took **.055 s** inside Python. There were no retries or further levels.

## Matched changes and contraction

Contraction factors compare maxima on the **same 25 times** shared by all
three levels, not on different sampling grids. Values above one improve.
The final column separately retains the new pair's maximum over **all 49**
common times, including .125 s, which was absent from the original comparison.

| Observable | E_delta(.25,.125) | E_delta(.125,.0625), same times | Contraction | New pair, all times |
| --- | ---: | ---: | ---: | ---: |
| V (m/s) | 1.39219e-5 | 8.66464e-6 | 1.607 | 1.03758e-5 |
| q (Pa) | 1.55752 | .966455 | 1.612 | 1.13230 |
| C (Pa) | .0156799 | .00974272 | 1.609 | .0113481 |
| Theta (s) | .142267 | .0799203 | 1.780 | .129537 |
| Accumulated slip (m) | 1.56262e-6 | 9.69521e-7 | 1.612 | 1.13377e-6 |
| Full crack-strain integral (m/s) | 1.39219e-5 | 8.66464e-6 | 1.607 | 1.03758e-5 |
| Supported crack-strain integral (m/s) | 1.39354e-5 | 8.67301e-6 | 1.607 | 1.03857e-5 |
| Velocity-profile maximum (m/s) | 5.04041e-5 | 3.12259e-5 | 1.614 | 3.70392e-5 |
| Velocity-profile L2 (m/s) | 1.88019e-5 | 1.16511e-5 | 1.614 | 1.38280e-5 |

I_h, initialized H/phi, profile width, calibration and localization moments
are frozen inputs; their matched differences have exactly zero timestep
change. The history-localization integral is identically zero. Their nonzero
between-width differences remain as recorded in K4.1, not reinitialized away.

## Unresolved common-mode transient

The new pair, over all 49 common times, gives:

| Observable | Individual ell0 change | Individual half-width change | Common-mode change | Matched change / sum of individual maxima |
| --- | ---: | ---: | ---: | ---: |
| V (m/s) | 2.35277e-4 | 2.45652e-4 | 2.40465e-4 | 2.157% |
| q (Pa) | 28.5768 | 29.7092 | 29.1430 | 1.943% |
| C (Pa) | .264090 | .275438 | .269764 | 2.103% |
| Theta (s) | 4.39116 | 4.52070 | 4.45593 | 1.454% |
| Slip (m) | 2.85933e-5 | 2.97271e-5 | 2.91602e-5 | 1.944% |
| Velocity maximum (m/s) | 7.90381e-5 | 9.92213e-5 | 8.74471e-5 | 20.778% |
| Velocity L2 (m/s) | 5.27289e-5 | 6.29538e-5 | 5.76537e-5 | 11.953% |

The unchanged original K4.1 error/quarter-allowance maxima are **567.96 and
578.24**, still far above one. Common-mode cancellation is real and useful,
but does not qualify those absolute trajectories. Differences between two
timesteps are empirical uncertainty indicators, not proven asymptotic error
bounds; the observed factors do not justify unqualified Richardson extrapolation.

## Width-effect accuracy: early and late behavior

At .125 s, the new fine matched signals/change estimates are respectively:
V = 6.36024e-5 / 1.03758e-5 m/s, q = -10.37379 / 1.13230 Pa,
Theta = -1.56395 / .129537 s, and velocity maximum = 2.72696e-4 /
3.70392e-5 m/s. These remain material uncertainties in the width effects.
At .5 s the V width signal is only 4.74052e-6 m/s while its timestep change
is 2.94559e-6 m/s. Early samples cannot be dropped to claim a resolved
full-trajectory result.

At 6 s, width differences are much more stable:

| Observable | Delta_ell Q at dt=.0625 | Change from dt=.125 |
| --- | ---: | ---: |
| V (m/s) | -4.27048e-9 | 2.74141e-11 |
| q (Pa) | -16.664096 | .000178504 |
| C (Pa) | -16.447403 | .0000170767 |
| Theta (s) | -1.116814 | .000514065 |
| Slip (m) | +1.76670e-5 | 6.15481e-9 |
| Velocity maximum (m/s) | 1.61411e-5 | 3.56780e-10 |

All-time signed values, changes and common-mode contributions are retained
in `k41b/all-times.csv`; fieldwise differences are saved separately in NPZ.

## Proposed criterion refinement — review required

For a width-effect study it is appropriate to assess E_delta directly,
separately from absolute-trajectory accuracy. A conservative, explicit proposal
is to retain the K1 physical scales and .002 relative / 1e-5 absolute
coefficients, but define a **separate** temporal width budget

\[
 E_\Delta(t)\le\tfrac14\left(.002\|\Delta_\ell Q(t)\|
                                      +10^{-5}S_Q\right).
\]

This changes the quantity to which the accuracy budget applies; it is not the
currently approved K4.1 criterion. Preserve original trajectory-failure
reporting, require contraction on identical times, and retain support and
normalization as independent gates. Use the fixed absolute term around zero
width signals, not a denominator fitted to the observed residual/change.

The script reports this as `proposed_criterion_only=true`. Even this direct
width test does **not** qualify all observables over the full interval: new
pair maximum ratios are V 1124.2, q 126.7, C .907, Theta 101.0, slip 169.5,
velocity max/L2 588.6/594.4. C qualifies in this diagnostic and the late-time
width effects are substantially better resolved, but the full transient does
not. This is evidence for reporting width uncertainty separately, not for
silently approving K4.2 or relaxing the coefficients until the run passes.

**Review point:** retain the improved but unresolved early width transient,
the robust late-time contrast and the original failed absolute-trajectory
criterion. No additional timestep or restricted-time K4.2 is assumed authorized.

## Reproduction and checks

From the repository root:

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/finite-width/temporal_width.py
python3 benchmarks/reconstructed_fault/uniform_shear/finite-width/temporal_width.py --add-level
python3 benchmarks/reconstructed_fault/uniform_shear/finite-width/test_temporal_width.py
```

`--add-level` refuses to overwrite existing .0625 trajectories. Artifacts are
`finite-width/k41b/{saved-pair.json,comparison.json,all-times.csv}`, the matched
velocity NPZs and the two new `k41/*-4096/dt0.0625.{json,npz}` trajectories.
The original `k41/decision.json` remains unchanged and unqualified.

Three focused checks pass in .052 s: all new states cross-check against the
original independent K1 scalar recurrence including timestep-zero retention;
profile subtraction occurs before the norm; and contraction uses identical
physical times without readiness override. Roots, support/normalization and
velocity boundary checks pass. No ASPECT, production edit, new phase solve,
spatial refinement or K4.2 action occurred.
