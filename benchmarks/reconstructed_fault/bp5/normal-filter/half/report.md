# BP5 normal-input filter: completed half-clock comparison

## Decision

**Numerical feasibility passes, but the filter is not qualified as a cure for
the stress bands.** Both lengths smooth the normal input to friction, with modest
whole-fault velocity sensitivity. However, the shallow velocity chord variation
increases and the raw stress/history pattern remains. Keep raw as the production
default. If filtering is pursued, retain 100 m as an experimental candidate and
first assess whether its introduced shallow velocity structure is a transient
adjustment or persists; do not select 200 m merely because it looks smoother.
This task runs no additional trajectories and authorizes no long continuation.

## Completion and time levels

All three 32-rank Release runs accepted steps **5613–5622**, with bitwise-identical
actual intervals and matched reported times. The summed constitutive duration is
**0.013942844071142269 s**. The reported final time is
5310111071.5773554 s; subtracting large absolute timestamps is not the interval
metric. This is ten half-sized intervals, not twenty steps spanning original R.

The exported restored fault table and all 32 rank-local restored particle tables
are byte-identical across branches. Exported phase norm, H moments, fault
coordinates and previous I_h remain unchanged. Every logged fresh linear
residual is below its requested target. No rejected line-search candidates were
reported; every accepted alpha is 1. Theta audits are at 2.22e-16 relative error.
The largest realized weighted logarithmic state change is 0.01019, below 0.02.

| Quantity | R | F1: 100 m | F2: 200 m |
| --- | ---: | ---: | ---: |
| Newton updates, first / subsequent steps | 2 / 2 | 4 / 2 | 4 / 2 |
| Total reported Krylov iterations | 1016 | 1084 | 1084 |
| Lower-active nodes, every step | 110 | 110 | 110 |
| Maximum reported normalized nonlinear residual | 2.05e-11 | 8.43e-11 | 3.19e-9 |
| Maximum surface RMS residual (Pa) | 8.85e-6 | 1.45e-4 | 5.50e-3 |
| New-run wall time (s) | 414 | 433 | 433 |

These are new-run timer totals, not the approximately 80000 s including restart
history. Filtering adds about 4.6% total time here, including its extra initial
Newton updates. The copied logs do not isolate filter application/factorization
time or peak memory; no separate performance run was made.

## What is actually smoothed

The offline comparison uses the **first resumed Newton base**, with checkpoint
histories and the reduced pending dt. It is not the accepted checkpoint stress.
All modes use the same production M/K, weights and frozen mechanical field.
Whole-fault weighted mean is 50.5006383 MPa and is preserved to roundoff.

Native work-point mean-removed normal-input RMS at that frozen state (MPa):

| Window | Raw pointwise | Q1 projection only | 100 m | 200 m |
| --- | ---: | ---: | ---: | ---: |
| 22–35 km | 0.32543 | 0.04620 | 0.04456 | 0.04337 |
| 70–71 km | 0.75086 | 0.05539 | 0.02538 | 0.01272 |
| 79–80 km | 0.79577 | 0.07124 | 0.04820 | 0.03182 |

Much of the apparent suppression is therefore projection, not Helmholtz
smoothing. Native weak friction is not simply a projected stress plot: at this
same frozen state its whole-fault load-vector relative change is 2.02e-6 for
projection alone, 6.50e-4 for 100 m, and 8.24e-4 for 200 m. Spatially varying mu
prevents local friction loads from being preserved by preserving the stress mean.

At the last accepted state, raw constitutive normal stress remains approximately
**41.2507–53.1010 MPa** in every branch. Actual filtered inputs are
**49.0811–52.7937 MPa** (F1) and **49.4921–52.7663 MPa** (F2); no tensile samples
or normal-stress clipping occur. The filtered versus raw full-fault work mean
differs by at most 1.70e-8 Pa. Window means are not artificially recentered.

The endpoint effect is not negligible locally: the bottom-end friction input
changes by several MPa, and the bottom 2-km velocity difference has an 11–12%
weighted relative norm. Its maximum absolute change is only 2.02e-10 / 2.26e-10
m/s for F1/F2. The top remains lower-bound active with no rate change. The broad
top stress trend persists; 200 m is not asserted to preserve every endpoint detail.

## Mechanical response and roughness

| Final-state quantity | R | F1 | F2 |
| --- | ---: | ---: | ---: |
| Maximum V (m/s) | 0.0960637 | 0.0967175 | 0.0971453 |
| Work-row-weighted relative velocity difference from R | — | 0.839% | 1.148% |
| Maximum velocity difference from R (m/s) | — | 0.0014542 | 0.0021522 |
| Maximum accumulated-slip difference from R (micrometres) | — | 22.32 | 32.98 |
| Shallow V chord RMS (m/s), 22–35 km | 2.257e-4 | 6.222e-4 | 7.081e-4 |

F2–F1's final velocity difference is 0.407% in the same weighted norm.
The shallow chord metric increases **2.76x / 3.14x**. It includes physical
curvature, not only numerical error. The plotted branch differences nevertheless
show the added alternating component. The effect is already present on the first
filtered solve: chord RMS is 2.206e-4 / 7.165e-4 / 8.217e-4 m/s for R/F1/F2.
It relaxes somewhat over the following nine filtered steps, but remains above R.
Most of the branch separation is an immediate response to changing friction input
at an already loaded checkpoint, not demonstrated long-term instability.

Deep and 79–80-km velocity chord measures decrease by about 35% and 14–16%,
respectively. Their absolute rate differences are only about 1e-11 m/s.
Figures integrate accepted dt*(V_F−V_R) for tiny deep slip differences to avoid
subtracting metres of old cumulative slip to resolve 1e-13-m increments.
The shallow maximum branch rate changes are comparable to R's approximately
0.00114 m/s change in maximum rate over this very short interval; the modest
percentage of total V must not be mistaken for negligible transient influence.

## Friction accounting and retained history

All mechanics use incoming Theta, not the newly committed value. At step 5613,
the maps and weights match exactly, so the native friction difference separates
into mu_R*(sigma_F−sigma_R) and sigma_F*(mu_F−mu_R). Summed over the exported
windows, these are +0.500 / −4.640 MN for F1 and −0.214 / −8.363 MN for F2,
with pointwise closure below 7.46e-9 Pa. These signed sums can cancel spatially;
they are not force norms or separate causal solves.

For example, in 70–71 km the two contributions divided by total native work
measure are +874 / −1817 Pa (F1) and +2541 / −3484 Pa (F2). In 79–80 km they are
+4366 / −2686 Pa and +6380 / −4697 Pa. Thus smooth stress does not imply an
unchanged friction load; the solved rate changes the friction coefficient too.

The existing strict attribution check skips later steps because weights are not
bitwise identical. A final-step audit found identical cell/QP identities and
coordinates, but maximum weight differences of 4.44e-16. No interpolation or
silent relaxation of the common-weight criterion was used to attribute those
later steps. Nodal profiles remain directly comparable on the identical fault.

Raw mechanical Q1 stress differs from R by at most 3.82 / 4.19 Pa at the end,
despite the much larger change in filtered input. Raw full-fault Q1 normal RMS
grows from 469.057 to 469.103 kPa in all branches. Working-history normal RMS
grows by approximately 0.0814%, 0.0101%, and 0.0122% in the shallow/deep/feature
windows; retained-particle RMS grows by 0.0857%, 0.0420%, and 0.0418%. These
growth rates are essentially unchanged by filtering. Particle RMS uses equal
real-particle weights; working-FE RMS uses native work weights. Their absolute
values must not be compared as the same sampling measure.

**The inherited pattern and its small continued growth persist.** The experiment
is consistent with filtering altering frictional response without repairing the
underlying history pattern. It does not establish recurrence, nucleation, event
accuracy, or long-term stability from 0.014 s of evolution.

## Artifacts and reproduction

- `summary.json`: combined qualification, offline and native-evolution numbers.
- `offline.png`: frozen field, projection/filter lengths and endpoint behavior.
- `evolution.png`: final matched-window stresses, velocity and slip differences.
- `R/output`, `F1/output`, `F2/output`: original server artifacts, unchanged.

From the source root (analysis only):

```sh
python3 benchmarks/reconstructed_fault/bp5/normal-filter/analyze.py benchmarks/reconstructed_fault/bp5/normal-filter/half
python3 benchmarks/reconstructed_fault/bp5/normal-filter/analyze.py benchmarks/reconstructed_fault/bp5/normal-filter/half --evolution
python3 benchmarks/reconstructed_fault/bp5/normal-filter/assess_half.py benchmarks/reconstructed_fault/bp5/normal-filter/half --plot
```

Only analysis code and derived reports/figures were added; production source,
parameters and copied server outputs were not modified. No new solve was run.
