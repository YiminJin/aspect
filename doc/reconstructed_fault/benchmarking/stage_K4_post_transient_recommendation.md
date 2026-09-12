# K4: persistent post-transient temporal qualification

## Recommendation

**Proceed, after review, with clearly labeled post-transient K4.2 comparisons
on [4,6] s.** This is an informative two-second interval, not a passed
all-time K4.1 reference. No K4.2 case was run or prepared by this offline check.
Retain the unresolved 0--4 s transient and its influence on the later state.

Use the saved dt=.125/.0625 pair and the previously reported *width-specific*
budget, without changing any coefficient or fixed scale:

\[
 \|\Delta_\ell Q_{.125}-\Delta_\ell Q_{.0625}\|
 \le\tfrac14(.002\|\Delta_\ell Q_{.0625}\|+10^{-5}S_Q).
\]

This budget was explicitly proposed in K4.1b; this report evaluates persistent
qualification under it, not a silent rewrite of the original K4.1 criterion.
As an additional check, the two **individual** trajectories meet the original
quarter-allowance criterion on this interval as well. Only the restriction of
the assessed time interval needs to be recognized as a scope change.

## Earliest persistent times

These are the earliest **sampled** times at which every later common sample
through 6 s passes. Samples are spaced by .125 s. No interpolation or claim
of continuous-time certification between samples is made. A positive onset
is bracketed by its preceding failed sample and the reported passing sample;
an isolated initial/final pass is not used as evidence of persistence.

| Observable | Last failed sample (s) | Earliest persistent t_qual (s) |
| --- | ---: | ---: |
| V | 3.875 | **4.000** |
| q | 1.750 | 1.875 |
| Cohesive traction C | None | 0 |
| Theta | 1.000 | 1.125 |
| Accumulated slip | 2.250 | 2.375 |
| Velocity profile, maximum norm | 3.250 | 3.375 |
| Velocity profile, L2 norm | 3.250 | 3.375 |
| Full crack-strain integral | 3.875 | **4.000** |
| Supported crack-strain integral | 3.875 | **4.000** |
| History-localization integral | None | 0 (identically zero) |
| I_h, H, phi, support/profile moments | None | 0 (frozen inputs) |

The common interval [4,6] contains **17 checked samples**, including all
intermediate samples after the loading ramp finishes at 4 s. Its onset is
limited by V and the full/supported integrals. The data bracket the onset by
(3.875,4] s rather than determining a more precise crossing time.

The older .25/.125 pair does not supply a common qualifying interval through
6 s: accumulated-slip width uncertainty still fails at the endpoint. Thus
dt=.25 is not a supported cheaper substitute for this recommendation.

## Remaining signal and temporal uncertainty on [4,6] s

Ranges below are the signed half-width-minus-full-width signal from dt=.0625,
except for the unsigned profile norms. Changes are maximum .125-to-.0625
matched differences over the whole interval, not just at 6 s.

| Observable | Width-signal range | Maximum temporal change | Largest fraction of width temporal budget |
| --- | ---: | ---: | ---: |
| V (m/s) | -4.2705e-9 to -3.5425e-9 | 2.3476e-10 | .9325 |
| q (Pa) | -17.0088 to -16.6641 | 1.7850e-4 | .0148 |
| C (Pa) | -16.7863 to -16.4474 | 1.7314e-5 | .00143 |
| Theta (s) | -1.40153 to -1.11681 | 6.5353e-4 | .5498 |
| Slip (m) | 1.76670e-5 to 1.76749e-5 | 6.2550e-9 | .6051 |
| Velocity maximum (m/s) | 1.59112e-5 to 1.61411e-5 | 6.3723e-9 | .7695 |
| Velocity L2 (m/s) | 5.63395e-6 to 5.71532e-6 | 2.2503e-9 | .7274 |
| Full crack-strain integral (m/s) | Same as V | 2.3476e-10 | .9325 |
| Supported integral (m/s) | 5.9595e-10 to 1.2546e-9 | 2.3510e-10 | .9380 |
| I_h (m) | -.344277429, fixed | 0 | No new temporal tolerance assigned |

The traction, velocity-profile, Theta and slip signals remain useful for a
width-versus-spatial-discretization test. In particular, traction changes are
about 16--17 Pa and the profile maximum changes by about 1.6e-5 m/s, despite
the very small change in the integrated slip rate V. The small V and supported
integral contrasts should be reported, but should not drive an expensive
resolution campaign: they are susceptible to discretization/support errors
comparable to the signals. Supported versus full integral differences still
include the measured fixed-support truncation; no tail is renormalized away.

The model-required C0/H0/calibration changes remain part of the width family.
The late traction difference is not claimed to isolate only the diffuse
mechanical kernel independently of initialization.

## Absolute-trajectory and early-transient status

For the individual dt=.125/.0625 trajectories, the maximum ratio to the
**original** quarter K1 allowance is .98520/.98572 on [4,6] s, versus
567.96/578.24 over the full 0--6 s trajectory. There is little spare margin
in the post-transient individual-trajectory check; do not round these into a
stronger convergence claim or extrapolate to an untested timestep.

The interval maximum common-mode timestep changes are about .08239 Pa in q,
.005043 Pa in C, .04026 s in Theta, and 3.6142e-7 m in accumulated slip.
They are much larger than the matched changes, although they satisfy the
absolute-trajectory budget on this restricted interval. Early common-mode
and matched uncertainties remain as reported in K4.1b, with no data removed.
No finite-step convergence bound at uncomputed times is asserted.

## Meaning of the recommended restricted K4.2

If approved, retain the planned width/mesh controls and use dt=.125 s,
with the existing dt=.0625 independent reference as the temporal comparator.
Every production case must run from the same prescribed initialization at
t=0 through 6 s. Do not start at t_qual, load reference histories at 4 s,
reset accumulated slip, or subtract initial error to erase its physical
influence. Keep all convergence, history, geometry, support and normalization
guards active over the **entire** run; restrict only the finite-width accuracy
assessment to [4,6] s. Compare matching-width differences against both saved
time-discrete references so the temporal allowance remains explicit.

This can separate width effects from production spatial error in q/C, velocity
profiles, Theta and accumulated slip while retaining initial-state differences.
It cannot certify the early transient, evolving-profile behavior, universal
thin-fault accuracy, or a complete K4.1/K3/K2 gate. No numerical or physical
parameters, tolerances, acceptance coefficients or support policy change is
part of this recommendation.

## Offline evidence

`finite-width/post_transient.py` reads the saved comparisons and finds the
first all-passing suffix. It also checks individual trajectories against the
unchanged original budget. No reference dynamics are re-evaluated.
Results are in `finite-width/k41b/post-transient.json`; the complete per-time
input remains `k41b/all-times.csv`. Two focused tests cover an isolated pass
followed by failure, an all-passing interval, failure at the final sample,
and every actual sample after each reported onset. No simulation was run.
