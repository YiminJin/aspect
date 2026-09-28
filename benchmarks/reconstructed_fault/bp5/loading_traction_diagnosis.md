# Loading startup: saved-state traction and sawtooth diagnosis

## Decision

The saved outputs identify a **spatially fixed mechanical pattern**, already
present at initialization, with additional amplification when weak loads are
displayed as consistent Q1 coefficients. Timestep subdivision changes the
mean/evolving response substantially but barely changes the shear sawtooth.
It is not created by sorting, substituting newly committed stress history,
or using outgoing state in the mechanical balance.

No new ASPECT solve, trajectory, history update, tolerance change or production
edit was made. The conditional tighter solve is not warranted by the present
evidence: the pattern is already supported independently by fixed source/work
moments, its persistence across the two histories, and a large separation from
the accepted residual scale. This is not a mathematical bound on every possible
solver-error contribution. The remaining uncertainty is which part of the
spatial representation (bulk FE accommodation versus localization/work
quadrature) dominates; the present data do not uniquely separate those two.

## Data and definitions

Use `weakening30-dc010-ell100/loading-startup/`:

- startup accepted states 0, 1, 2, 3;
- matched half-run accepted state 3;
- `work_qp_*_rank*.csv`, `work_weak_*.csv`, `state_work_*.csv`,
  `profiles/fault_*.csv`, initialized background and resolved parameters.

The matched final time is **2086054.244063706 s**. Full-step mechanics uses
dt=1086054.244063706 s; final-half mechanics uses dt=543027.122031853 s.
Their incoming states/histories differ legitimately because the latter has
accepted an intermediate half-step. Plots show `Theta_in`, not `Theta_out`.
This is a temporal comparison, not an identical-incoming-history A/B solve.

For every owned bulk QP use exactly the native measure
`W_q=JxW_q*chi_q`, source coordinate/basis `N_i(q)`, and row mass
`m_i=sum_q W_q N_i`. The pressure and signed deviatoric contributions are

\[
 P_i={\sum_q W_qN_i\Delta p_q\over m_i},\qquad
 D_i=-{\sum_q W_qN_i(\Delta\tau_q:N)\over m_i},\qquad
 \Delta\sigma_{n,i}=P_i+D_i.
\]

The fixed 50-MPa background is excluded from these plots, not from mechanics.
It is added back to obtain total normal traction.

For this frozen mature case the shear perturbation is decomposed as

\[
 \Delta q=\underbrace{2\kappa\dot\epsilon:S}_{\text{strain}}
 +\underbrace{\beta\tau^{\rm old}_{\rm FE}:S}_{\text{incoming history}}
 -\underbrace{2\kappa\chi V\,S:S}_{\text{current slip}},\quad S:S=1/2.
\]

The resolved material has uniform eta=1e26 Pa s, G=32038120320 Pa and zero
thermal viscosity exponent. Kappa uses the actual interval and `expm1`, with
the artificial 1e6-s interval at step zero. The old-history contribution is
**recovered algebraically** from the saved current tensor minus its current
strain/slip contributions. It is the frozen **working FE** history consumed by
mechanics, not newly committed particle stress and not a second independent
measurement of the transfer. Recovered history is below 1e-6 Pa at states 0/1,
as required by retained-zero initialization. Both off-diagonal tensor entries
are included in contractions; `S:N=0` excludes a direct slip normal term.

Complete raw-QP coverage is checked against native row masses before using
decomposed rows. Raw exports cover the transition and selected control/tip
windows, not the entire fault; unavailable decomposed entries are NaN, never
zero. Whole-fault normal plots use the saved complete native pressure/normal
loads. The detailed common-time comparisons focus on 27–36 km.

## Native loads versus plotted representations

Two distinct output quantities must not be conflated:

1. The previous `loading-whole.png` / `loading-transition.png` used native
   `L_i/m_i` averages.
2. `profiles/fault_*.csv` columns `q_weak_Pa` and `sigma_n_weak_Pa` are
   **consistent Q1 coefficients**, obtained from `M z=L`. Despite their names,
   they are not `L_i/m_i`.

Reassembled pressure and deviatoric loads match saved native averages within
4.6e-13 Pa. The decomposed sum agrees within 3.15e-7 Pa, including cancellation
when removing the 50-MPa background. Multiplying saved reconstructed Q1 fields
by the actual QP mass matrix reproduces their native loads within **1.63e-7 Pa**.
All checks retain the existing 1e-5-Pa allowance. IDs/coordinates match and the
stored fault ordering is strictly monotone; plotting simply orders by physical
down-dip distance without resampling or smoothing.

In the 27–29.8-km window the native shear sawtooth RMS is 14.13 Pa, while its
consistent Q1 representation is 41.42 Pa: **2.93x amplification**. This is
consistent with the mass operator: for alternating nodal signs, measured
`(M z)_i/(m_i z_i)=0.333003–0.333650`. Its inverse amplifies near-grid-scale
load structure by approximately three. It does not create the original teeth;
they were already present in native averages, including the previous plots.

## Normal-stress cancellation

Work-weighted means over 27–29.8 km (Pa, all perturbations):

| Contribution | Full step | Two half-steps |
|---|---:|---:|
| Pressure | +86.4691 | +91.7704 |
| Negative deviatoric normal traction | -82.6190 | -89.4299 |
| Sum | **+3.85006** | **+2.34053** |

Thus pressure alone exaggerates both the actual normal perturbation and its
temporal difference. The full-step sum ranges 3.5898–4.1396 Pa; the half-step
sum ranges 1.9815–2.6599 Pa. Their spatial means differ, but their neighboring
tooth pattern remains strongly correlated. These perturbations are not tensile
total tractions: total normal compression remains approximately 50 MPa.

## Shear cancellation and retained history

Over the same window, signed work-weighted means are:

| Contribution (Pa) | Full step | Two half-steps |
|---|---:|---:|
| Strain-rate shear | +233458.031 | +115528.726 |
| Current-slip shear | -233922.747 | -115744.896 |
| Incoming-history shear | +553.214 | +364.949 |
| Sum, excluding fixed background | **+88.497** | **+148.779** |

The current terms shrink roughly with dt through kappa; history carries the
intermediate accepted stress in the half-step case. Comparing either large
current term alone would therefore be misleading.

For tooth amplitude define the explicitly diagnostic chord departure
`c_i(z)=z_i-(z_(i-1)+z_(i+1))/2` on the uniform local 100-m fault grid. Original
curves remain untouched. This statistic also responds to smooth curvature,
so flat-material windows are reported separately from the transition.

Signed projections of each component's chord onto the **total shear chord**
(these add to one; they are not fractions of mean traction):

| Component | Full step | Two half-steps |
|---|---:|---:|
| Strain rate | -0.3550 | -0.1409 |
| Current slip | +0.7921 | +0.3483 |
| Incoming history | +0.5630 | +0.7926 |

The strain contribution partially cancels the other contributions. History
retains and reinforces the pattern, but cannot be its original cause:
initialization already has a 9.054-Pa native shear chord RMS with zero history.
Initial projected state/background were balanced with the native weak measure
to 1.53e-7 Pa at target Vp, so this is not the old inconsistent inverse-state
prestress pulse. Small projected-mixture tails near 30 km are retained rather
than assumed absent or clipped away in this diagnosis.

## Spatial persistence at the common time

RMS chord amplitudes, identical native weights and physical nodes:

| Window / quantity | Full | Half | Half/full | Correlation |
|---|---:|---:|---:|---:|
| 27–29.8 km: native shear (Pa) | 14.1292 | 14.1013 | 0.9980 | 0.99864 |
| 27–29.8 km: Q1 shear (Pa) | 41.4243 | 41.5892 | 1.0040 | 0.99967 |
| 27–29.8 km: normal sum (Pa) | 0.28778 | 0.30289 | 1.0525 | 0.99024 |
| 27–29.8 km: V/Vp | 2.12693e-4 | 2.16450e-4 | 1.0177 | 0.99977 |
| 30.2–32.8 km: native shear (Pa) | 24.7394 | 25.1902 | 1.0182 | 0.99597 |
| 33.2–36 km: native shear (Pa) | 24.3609 | 23.8993 | 0.9810 | 0.999995 |

All 29 shallow-window nodes keep their shear, normal-sum and velocity chord
signs. For example, the shear chord at 27.6 km is -20.526/-20.495 Pa and at
28.5 km -19.838/-19.842 Pa. At 29.6 km it is -23.917/-25.991 Pa, where smooth
transition curvature increasingly contributes. The detailed JSON preserves
pressure/deviatoric component statistics and sign changes separately.

The bulk cells contributing active QPs in 27–36 km are 24.4140625 m; local
fault spacing is 100 m. The purely geometric/current-profile work moment
`<chi>_i=sum(JxW*chi^2*N_i)/m_i` has 0.02533% peak-to-peak variation and is
unchanged between full and half results to roundoff. Its chord correlation
with native shear is -0.9631 already at initialization, -0.9643 in the full
result, and -0.9610 in the half result. Velocity correlations are about -0.9645.
This identifies a fixed localization/work-quadrature imprint strongly coupled
to the stress/rate pattern, not evidence that the I_h algorithm or support
policy itself is incorrect. No such algorithm was reopened or changed.

Accepted full/half surface residual RMS is approximately 0.001527/0.000167 Pa,
far below the 14-Pa shear pattern. All fresh-linear checks passed. In addition
to this scale separation, the geometry correlation, initialization-time
presence and persistent node locations supply independent evidence against
nonlinear stopping error as the dominant source. Residual size alone would
not establish that conclusion for an ill-conditioned coupled system.

## Deliverables and next action

All new output is in `loading-startup/traction-audit/`:

- `whole-fault-normal.png`: complete native pressure, negative deviatoric
  normal traction, and their sum;
- `components-transition.png`, `components-VW_closeup.png`: signed current
  strain/slip/history cancellation and native versus Q1 shear;
- `matched-transition.png`, `matched-VW_closeup.png`: changes from initialization,
  matched rates and correctly timed incoming state;
- `component-chords.png`, `neighbor-chords.png`: unsmoothed neighboring-node
  diagnostics with signed component contributions;
- `native-versus-Q1.png`: both shear and normal representations, identical axes;
- five decomposed CSVs and `summary.json`, including recomposition tests.

Reproduce offline:

```sh
MPLCONFIGDIR=/tmp/mpl-bp5-loading python3 \
  benchmarks/reconstructed_fault/bp5/analyze_loading_tractions.py
```

Only this analysis script and report are new; original outputs, server inputs,
source, initialization and numerical settings are unchanged. Recommendation:
**do not tighten production tolerances to address these teeth**. Treat them as
the present spatial representation's small-scale mechanical response, with
additional Q1 visualization amplification. If their size matters to subsequent
interpretation, the discriminating next test is a frozen spatial comparison
of localization/work moments and bulk accommodation, not another trajectory
or state-initialization change. That test was not launched here.
