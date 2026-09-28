# Transient friction versus spatial coupling: saved 32-step disturbance

## Decision

The measured amplification is predominantly a transient state/rate-friction
response about the evolving creep front, not evidence of a large additional
bulk/normal-stress amplifier. The distinction between absolute disturbance and
relative disturbance matters substantially:

- Full-feedback absolute state-difference norm gain is **1.005202**.
- Relative state-difference norm gain is **1.077315**. Keeping the *initial*
  reference state as denominator instead gives **0.836215**; changing to the
  evolving denominator contributes a factor **1.288323**. Their product is
  1.077315. This is an exact accounting identity for the saved fields, not a
  counterfactual trajectory.
- Actual absolute growth occurs in the shallow part of the perturbation,
  peaking at **3.085% at 15.9 km**. At 16.2 km, relative amplification of
  **31.86%** accompanies **27.05% absolute decay**.
- Given the measured incoming state disturbance, a production-QP/Q1
  friction-only linear response differs from the full velocity disturbance by
  **0.234% initially and 3.308% finally** in the 15--18 km work-weighted norm.
  Normal-feedback removal changes the final velocity disturbance by only
  0.404%, as established in the preceding analysis.

Spatial representation is nevertheless quantitatively relevant. In particular,
independent pointwise friction is not the implemented weak Q1 friction law. The
pointwise heuristic overpredicts relative amplification around 16.2 km, while
the actual weak friction response closely explains the measured velocity.
These data do not qualify the two-node wavelength spatially or establish a
continuum eigenmode. No new trajectories or production changes were made.

## Inputs and timing

Use only `reference32`, `plus32`, `state32`, and `normal32` from
`benchmarks/reconstructed_fault/bp3/first_long_run/state-disturbance/`.
The interval is the existing one year after accepted step 11, with 32 steps
of 986175 s (steps 12--43). Every mechanics evaluation uses the saved
**incoming** state; aging maps it to the separately saved outgoing state.
Successive outgoing/incoming arrays were checked for exact equality.

The exact map is the existing `FaultFriction::update_state`:

\[
 F(T,V)=T e^{-x}-\frac{D_c}{V}\operatorname{expm1}(-x),
 \qquad x=V\Delta t/D_c,\quad D_c=0.008\ {m m}.
\]

The mature constitutive law and split update agree with `current_design.md`
(mature specialization/history cycle) and `specification.tex`. The analysis
does not change their equations, work measure, or timing. The state control
uses its own incoming state but the reference accepted V in this map; the
normal control uses its own V and state, with reference normal traction only
in the mechanical friction term.

Raw QP records were streamed from the checksum-verified September 17 archive,
without restoring complete simulation outputs. The reduced cache retains
actual quadrature points in 14--19 km. Every reported 15--18 km test-function
support is complete: integrated row measures match the native mass row sums
to 1e-15 relative. QP velocity/state interpolate the recorded nodal V and
**incoming** Theta to 2.22e-16 relative. Nothing is reconstructed by sorting
physical coordinates into a new surface topology.

## 1. Disturbance and reference denominator

For each location define

\[
 d_k=\Theta^P_k-\Theta^R_k,\qquad r_k=d_k/\Theta^R_k.
 \qquad
 \frac{r_k}{r_0}=\frac{d_k}{d_0}
                   \frac{\Theta^R_0}{\Theta^R_k}.
\]

The gains below use signed differences; none of these selected locations
changes the sign of its state difference. Thus a gain greater than one also
means growth of its absolute magnitude. Gains at taper endpoints with zero
initial perturbation are undefined and are not interpreted as amplification.

| Down dip (km) | Initial d (s) | Final d (s) | Absolute gain | Denominator factor | Relative gain |
|---|---:|---:|---:|---:|---:|
| 15.5 | +26528.77 | +26601.69 | 1.002749 | 0.975735 | 0.978417 |
| 15.9 | +25905.25 | +26704.37 | 1.030848 | 1.064552 | 1.097391 |
| 16.0 | -16319.79 | -16453.32 | 1.008182 | 1.196351 | 1.206140 |
| 16.2 | -6624.66 | -4832.86 | 0.729525 | 1.807522 | 1.318633 |
| 16.5 | +4318.34 | +3442.22 | 0.797117 | 1.393190 | 1.110535 |
| 17.0 | -3015.15 | -2466.85 | 0.818150 | 1.025445 | 0.838967 |
| 17.5 | +944.05 | +721.81 | 0.764594 | 0.979751 | 0.749111 |

The whole-fault norm identity uses the consistent production mass M. Specifically,
the denominator-frozen gain is
`||d_final/Theta_ref_initial||_M / ||d_initial/Theta_ref_initial||_M`,
and the denominator-only factor compares the two denominators using the *same*
final difference field. It is not obtained by subtracting 0.52% from 7.73%:
absolute and relative norms weight this very nonuniform reference state
differently. Ratios here use d/Theta; the preceding report used log(Theta_P/Theta_R).
At this perturbation amplitude the quoted gains agree to the displayed digits.

Velocity also depends on the changing reference mobility. Its first nonzero
response is step 12, not the unchanged checkpoint V:

| Down dip (km) | First delta V (m/s) | Final delta V (m/s) | Absolute delta V gain | Gain of delta V/V_ref |
|---|---:|---:|---:|---:|
| 15.5 | -4.39662e-17 | -5.10241e-17 | 1.1605 | 0.9976 |
| 16.0 | +5.78166e-15 | +8.93636e-15 | 1.5456 | 1.2719 |
| 16.2 | +1.75379e-14 | +2.85515e-14 | 1.6280 | 1.1289 |
| 16.5 | -2.33945e-14 | -2.89261e-14 | 1.2365 | 1.1049 |
| 17.0 | +1.17457e-14 | +9.91509e-15 | 0.8441 | 0.8370 |
| 17.5 | -3.41683e-15 | -2.54546e-15 | 0.7450 | 0.7468 |

For example, the 16.2-km reference V grows by a factor 1.4421. This combines
with 1.1289 growth of delta V/V_ref to give the 1.6280 absolute gain. Globally,
absolute velocity-disturbance norm gain is 1.269890, while the norm of
delta V/V_ref gains 1.066489. These are observations about the evolving
reference, not removal of its physical influence.

## 2. Local transient heuristic and the actual weak balance

For clarity, the approximate local growth coefficients are derived explicitly.
At fixed shear/normal traction, the regularized RSF law gives the exact
partial-derivative ratio

\[
 \frac{\mu_\Theta}{\mu_V}=\frac ba\frac V\Theta,
 \qquad \frac{\delta V}{V}\simeq-\frac ba\frac{\delta\Theta}{\Theta}.
\]

Ignoring spatial coupling and damping, linearizing continuous aging gives

\[
 R=V\Theta/D_c,\qquad
 \lambda_{\rm abs}=\left(\frac ba-1\right)\frac V{D_c},
 \qquad
 \lambda_{\rm rel}=\frac ba\frac V{D_c}-\frac1\Theta
                  =\frac1\Theta\left(\frac ba R-1\right).
\]

Thus relative growth need not imply absolute growth or steady-state
rate weakening: an initially over-aged/slipping-down reference with R>1
can have positive lambda_rel even where b<a. Conversely, with very small R,
reference aging can reduce a relative disturbance while its absolute
difference grows slightly.

Evaluate the law at reference QPs with incoming Theta. The actual effective a
is recovered from the exported production mu and checked by reproducing mu;
it is not substituted by a nominal sharp-fault/nodal depth formula. Here b=0.015.
The following values are test-function/work-weighted averages over each node's
actual QPs. The corresponding nodal R is recorded separately in the CSV.

| km | Effective a (approx.) | R, first → last | lambda_abs, first → last (1/yr) | lambda_rel, first → last (1/yr) |
|---|---:|---:|---:|---:|
| 15.5 | .012506 | .1663 → .1983 | .00097 → .00113 | -.02396 → -.02227 |
| 15.9 | .014498 | 1.7740 → 1.9835 | .00444 → .00529 | .07173 → .09601 |
| 16.0 | .014999 | 2.2329 → 2.3199 | -.00127 → -.00171 | .18259 → .23217 |
| 16.2 | .016002 | 2.2763 → 1.8989 | -.06026 → -.08528 | .46619 → .53376 |
| 16.5 | .017506 | 1.5504 → 1.2576 | -.16022 → -.18045 | .23724 → .07975 |
| 17.0 | .020000 | 1.0429 → 1.0252 | -.20448 → -.20623 | -.17088 → -.18592 |
| 17.5 | .022506 | .9668 → .9838 | -.26968 → -.26906 | -.29746 → -.28236 |

Averages of products are used: substituting the displayed average a and R
back into the formula does not recover the averaged coefficient exactly.
In particular, the near-zero absolute coefficient at 16.0 km is sensitive
to variation across its support.

The heuristic's spatial pattern agrees with relative decay near 15.5 and
17--17.5 km and amplification around 16--16.5 km. Exponentiating the sum of
the averaged lambda_rel*dt gives final gains 0.9771, 1.0870, 1.2298, 1.6631,
1.1837, 0.8362, 0.7487 at the seven tabulated sites, respectively. This is a
**continuous-time local heuristic**, not a separate numerical reference:
finite-step split aging and Q1 weak interpolation remain in the actual
experiment. The discrepancy at 16.2 km is substantial, so pointwise friction
must not be claimed to predict all the observed amplitudes.

### Production-weighted friction-only instantaneous comparison

To avoid replacing the weak balance by nodal collocation, assemble directly
from saved reference QPs

\[
 D_{ij}=\sum_q w_qN_iN_j\sigma^R_q\mu_{V,q},\quad
 T_{ij}=\sum_q w_qN_iN_j\sigma^R_q\mu_{\Theta,q},\quad
 D\delta V_f=-T\delta\Theta_{\rm in},\quad w_q=JxW_q\chi_q.
\]

This is an offline small linear response, **not a new trajectory**, and is
conditioned on the actual measured incoming state difference at each step.
It leaves out changes in driving/normal traction and the negligible damping
and mixture contributions; those are retained in the separate exact force
budget. The two outer 14--19 km window values are fixed to measured increments.
Setting them to zero instead changes the 15--18 km result by at most 9.2e-12
relative, excluding a window-boundary explanation.

Full-feedback discrepancy `||delta V_f-delta V||/||delta V||` grows from
0.002338 to 0.033082 over the year. For the normal control it ends at 0.030606;
for the state-update control at 0.048076. These norms use the consistent work
mass and the same 15--18 km nodal mask. The linearized weak rate+state budget
differs from its exact nonlinear difference by at most 9.11e-5 relative to
the state-load norm, well below these percentages.

At 16.2 km the Q1 weak friction prediction is only 84.2% initially and 74.7%
finally of the *pointwise* estimate using the work-averaged a. At 17.5 km it
is 100.3--100.5%. This identifies meaningful spatial state/rate interpolation
effects near the sharp reference-state variation, separately from the much
smaller additional bulk/normal response. It does not claim the averaged-a
pointwise estimate is another authoritative discretization.

For example, final signed **weak-row differences**, divided by each row's
work measure, are:

| km | Shear (Pa) | Normal friction (Pa) | Incoming-state friction (Pa) | Rate friction (Pa) |
|---|---:|---:|---:|---:|
| 16.0 | +.030354 | +.019986 | +23.623798 | -23.674138 |
| 16.2 | -.753819 | -.075874 | +24.339897 | -23.510204 |
| 16.5 | +.701433 | +.048775 | -27.596789 | +26.846581 |
| 17.0 | -.266555 | -.007873 | +15.987038 | -15.712610 |
| 17.5 | +.068652 | +.002812 | -4.661165 | +4.589701 |

These are not point tractions. The exact telescoping decomposition retains
the small mixture/damping terms in the CSV; summed rows reproduce native
unreplaced residual differences. The bulk/normal terms mostly oppose the
state-driven rate response in this region; their size is assessed against
the friction disturbance, not the tiny final residual or 50-MPa background.

## 3. Exact inherited-state and velocity-feedback accounting

With P denoting a branch and R its evolving reference, use the identity

\[
 d_k=E^R_kd_{k-1}+b_k,\qquad E^R_k=e^{-V^R_k\Delta t/D_c},
 \qquad b_k=F(\Theta^P_{k-1},V^P_k)-F(\Theta^P_{k-1},V^R_k).
\]

The first term always damps an inherited difference. The second is the exact
contribution of changing the update velocity at the *same incoming perturbed
state*. Use V_ref for both arguments in the state control: b_k=0 exactly.
No state is advanced twice and no branch is reconstructed from reference
history. Long-double diagnostic subtraction avoids loss of precision in b_k;
production still uses its unchanged double-precision map.

Propagating this identity gives an initial inherited contribution plus all
velocity contributions, each damped by subsequent reference-rate factors:

| km | Initial d (s) | Surviving initial contribution (s) | Accumulated velocity contribution (s) | Final d (s) |
|---|---:|---:|---:|---:|
| 15.5 | +26528.77 | +26401.48 | +200.21 | +26601.69 |
| 15.9 | +25905.25 | +22415.47 | +4288.90 | +26704.37 |
| 16.0 | -16319.79 | -11643.95 | -4809.37 | -16453.32 |
| 16.2 | -6624.66 | -2045.81 | -2787.05 | -4832.86 |
| 16.5 | +4318.34 | +1288.77 | +2153.45 | +3442.22 |
| 17.0 | -3015.15 | -1326.81 | -1140.04 | -2466.85 |
| 17.5 | +944.05 | +421.76 | +300.05 | +721.81 |

Among initially perturbed sampled nodes, 15.1--15.9 km grows in absolute
magnitude in all 32 steps. The 16.0-km difference initially decays, then grows
from step 17 onward (27/32 updates), ending 0.818% above its initial magnitude.
Every update at 16.1--17.9 km **still decays absolutely**: feedback slows that
decay but never reverses it. Zero-initial-amplitude taper nodes are excluded
from gain statements.

For a concrete final-step comparison:

- 15.9 km: inherited decay changes d by **-131.12 s**, while velocity feedback
  contributes **+164.44 s**: genuine absolute growth of about 33.32 s.
- 16.2 km: inherited decay changes the negative d by **+209.95 s** toward zero;
  feedback contributes **-140.89 s**. The difference still moves 69.05 s toward
  zero even while its *relative* amplitude grows.
- 17.5 km: inherited decay **-18.10 s**, feedback **+11.94 s**: absolute decay
  of about 6.16 s.

Removing state feedback gives final absolute gains 0.71349, 0.30882, 0.29844,
0.44005 and 0.44676 at 16.0, 16.2, 16.5, 17.0 and 17.5 km. Full-feedback
values are 1.00818, 0.72953, 0.79712, 0.81815 and 0.76459. Thus aging feedback
is substantial in *preserving* disturbances, even where there is no absolute
growth. Normal-feedback removal leaves these gains almost unchanged.

## Verification, artifacts, and remaining uncertainty

`analyze_disturbance_transient.py` performs this analysis only. Run from the
repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/aspect-bp3-slip-plot timeout 120 \
python3 benchmarks/reconstructed_fault/bp3/analyze_disturbance_transient.py
```

The first pass streamed and hash-checked 512 archived binary files (four
branches, 32 accepted states, four ranks); subsequent passes reuse the small
window cache. No ASPECT, MPI simulation, loading prefix, or production edit.
All offline runs completed within the short execution budget.

Maximum check errors:

- Production aging map versus independently recomputed committed state:
  2.89e-7 s over all nodes/states (large absolute background states).
- Exact per-update difference split: 4.25e-7 s; accumulated split: 2.03e-6 s.
  These are negligible relative to the tabulated disturbance changes.
- Exact QP friction-budget telescoping: 8.13e-9 Pa.
- Native weak-row residual-difference reproduction: 6.76e-10 Pa.
- QP V/incoming-Theta interpolation: 2.22e-16 relative.
- Complete test-function work-measure reproduction: 9.99e-16 relative.
- Linearized versus exact weak friction disturbance: 9.11e-5 relative.
- Window-boundary influence: 9.20e-12 relative.

Outputs under `first_long_run/state-disturbance/transient-accounting/`:

- `node_updates.csv`: every step/location, absolute and relative differences,
  denominator factors, exact inherited/velocity split, R, and growth coefficients.
- `selected_locations.csv`: the seven tabulated sites through all 32 steps and
  all three branches.
- `weak_friction_rows.csv`: correctly timed production-weighted force terms.
- `norms_and_instantaneous_closure.csv`: global denominator accounting and
  the conditional weak-friction response comparison.
- `checks.json`, `qp_provenance.json`, `window_qp.npz`: verification and the
  reduced exact-QP evidence, without re-expanding full simulation output.
- `state_and_denominator.png`, `velocity_controls.png`, `spatial_growth.png`.

The correct interpretation is **transient frictional response with important
reference-denominator effects and localized Q1 spatial mixing**, not a new
large bulk-mechanical instability established by this experiment. The remaining
uncertainty is spatial qualification of the short pattern and the local
state/rate representation. The conditional friction-only responses are not an
independently evolved alternative trajectory, so their 3.3% agreement does not
bound every possible cumulative spatial effect. The existing two-clock
uncertainty is retained; no mesh, timestep, physics, or solver change is proposed
by this offline task.
