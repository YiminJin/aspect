# K5 revised work-measure replay: real-timestep halving

## Decision

**Address temporal accuracy next.** The one fresh four-rank run completed all
20 real half-steps in **1369.655 s (22.83 min)**, below the 40-minute cap, and
stopped at precisely the baseline step-10 comparison time. The accepted-clock
guard prevented a remainder solve. No equations, initialization, spatial
discretization, boundary treatment, history algorithm or tolerance changed.

At 29.24190894 yr the 39.95-km rate decreases 3.885% in absolute rate, but that
means a **20.30% deeper deficit relative to V_p**, not a negligible change in
the notch. The 39.90–39.95-km rate contrast increases **37.99%**, and the final
50-m element's slip gradient increases **11.91%**. All RSF nodes remain free.
The notch persists and deepens; it is not removed by temporal refinement.
Two levels establish material time-discretization sensitivity, not a temporal
limit or proof that mixed free/prescribed spatial coupling is harmless.

## Controlled comparison

This is one fresh four-rank run with a 2400-s hard cap. The reference is the
accepted revised-work trajectory through **step 10**, not its roundoff-sized
step-11 remainder. Each of the ten real baseline durations is divided into two
equal intervals, giving 20 real steps and the same final physical time
922804465.59751701 s (29.24190894 yr). The artificial initialization interval
remains **4e6 s**; it is neither halved nor counted as elapsed physical time.

The run includes `work-replay-50-local4/run.prm` and changes only its output
directory and selection of `BP3 replay complete`. The 20-interval clock is the
only changed temporal input. A resolved-parameter comparison verifies that all
other parsed settings match. The existing strict replay-cap mode checks that
production controllers admit each requested half-step; it does not override
CFL, RSF or other timestep restrictions. No restart, second case or automatic
retry is used.

The mesh, 50-m local surface discretization, both completed boundary treatments,
fixed background, mature friction, frozen phase, full I_h, source quadrature,
pressure treatment, split nodal aging update, Maxwell history transfer and
nonlinear/linear tolerances are unchanged. Initial V, Theta, slip, q and
friction are compared with the baseline. All real histories are evolved from
that initialization, not reset at shared comparison times.

Offline localization comparison requires bitwise-equal physical QP coordinates,
quadrature weights and phase values. It permits only `16 eps max(abs(chi))`
absolute roundoff in chi (about 9.79e-18 1/m), since projection of a mixture
whose two degradation laws are identical can still change the final arithmetic
bits. An initial bitwise-chi diagnostic stopped on a 1.30e-18 difference;
the diagnostic was corrected, not the simulation or any production tolerance.
The maximum measured difference at each shared time is retained in
`comparison/summary.json`.

## Observations and friction timing

At every shared accepted time, the analysis compares V, Theta and accumulated
slip at 39.90/39.95 km (with 15/18/25/40 km controls), the exact Q1 last-element
slip gradient, and current constitutive stress at the identical physical bulk
quadrature points around 40 km. The 15–18-km raw stress and weak traction
windows are secondary observations. Current constitutive stress is evaluated
by the existing accepted-state exporter using frozen **old working FE history**;
published compositional tau fields are not substituted for current stress.

The friction diagnostic holds accepted V and sigma_n fixed and compares

\[
 \Delta f_q=\sigma_{n,q}\left[
 \mu(V_q,\widehat\Theta_{k,q})-
 \mu(V_q,\widehat\Theta_{k-1,q})\right],\qquad
 \Delta L_i=\sum_q J_q\chi_q N_i(q)\Delta f_q.
\]

Both states are interpolated from their own stored nodal arrays at the actual
production QP coordinate. Theta_k is the **already committed production update**,
not a new pointwise aging update or a projected alternative state. Mechanics
actually used Theta_{k-1}; changing it to Theta_k here is only an observational
counterfactual. The implied unreplaced residual change is `-Delta L_i`.

Near 40 km the projected surface mixture is fully velocity strengthening, so
the exact configured regularized law is

\[
 \mu(V,\Theta)=0.025\,\operatorname{asinh}\!\left[
 \frac{V}{2\,10^{-6}}
 \exp\!\left(\frac{0.6+0.015\log(\Theta\,10^{-6}/0.008)}{0.025}\right)
 \right].
\]

The independently reconstructed lagged weak load is checked against the recorded
production friction load (undoing its consistent Q1 mass solve), not merely
against a pointwise approximate law. The reported weak traction changes divide
by the same mechanical row mass. No full-versus-half difference is divided by
the tiny converged nonlinear residual, and no counterfactual friction is fed
back into mechanics.

## Shared-time result

Rates below are divided by V_p=1e-9 m/s. The gradient is
`(slip(40 km)-slip(39.95 km))/50 m`, dimensionless. The complete shared-time
Theta/slip values are in `comparison/nodes.csv`, with full nodal profiles and
all element gradients in `profiles_*.csv` and `slip_gradient_*.csv`.

| Time (yr) | V39.90 baseline | V39.90 half | V39.95 baseline | V39.95 half | Last gradient baseline | Last gradient half |
|---:|---:|---:|---:|---:|---:|---:|
| 0.07881506 | 0.99999623 | 0.99984242 | 0.99999489 | 0.99978393 | 2.543090e-10 | 5.439305e-9 |
| 0.15749434 | 0.99968089 | 0.99948548 | 0.99955961 | 0.99928884 | 2.212369e-8 | 3.435085e-8 |
| 0.30777175 | 0.99904336 | 0.99874221 | 0.99865733 | 0.99823303 | 1.494735e-7 | 1.741663e-7 |
| 0.59480161 | 0.99772464 | 0.99714288 | 0.99670512 | 0.99585514 | 7.463731e-7 | 8.017129e-7 |
| 1.14302864 | 0.99508206 | 0.99386739 | 0.99246814 | 0.99053439 | 3.352507e-6 | 3.542133e-6 |
| 2.19014227 | 0.99025119 | 0.98797449 | 0.98360348 | 0.97923877 | 1.418877e-5 | 1.511437e-5 |
| 4.19012931 | 0.98258415 | 0.97920074 | 0.96630240 | 0.95707764 | 5.672511e-5 | 6.144148e-5 |
| 8.01010455 | 0.97231110 | 0.96900180 | 0.93553996 | 0.91777062 | 2.121373e-4 | 2.338568e-4 |
| 15.30625726 | 0.96204182 | 0.96299395 | 0.88951558 | 0.86093298 | 7.209160e-4 | 8.058678e-4 |
| 29.24190894 | 0.95730307 | 0.96949577 | 0.83936855 | 0.80675812 | 2.133752e-3 | 2.387892e-3 |

Final state:

| Quantity | Baseline | Half real timesteps | Change |
|---|---:|---:|---:|
| Theta at 39.90 km (s) | 8,356,810.146 | 8,251,712.142 | -1.258% |
| Theta at 39.95 km (s) | 9,530,974.203 | 9,916,231.144 | +4.042% |
| Slip at 39.90 km (m) | 0.890417308 | 0.894371218 | +0.444% |
| Slip at 39.95 km (m) | 0.816116854 | 0.803409854 | -1.557% |
| Slip at prescribed 40 km (m) | 0.922804466 | 0.922804466 | identical |
| 1-V39.95/V_p | 0.160631450 | 0.193241880 | +20.301% |
| (V39.90-V39.95)/V_p | 0.117934519 | 0.162737647 | +37.990% |

`final_junction.png` shows the same spatial notch location but larger adjacent
rate/Theta/slip contrasts; no smoothing is applied. The prescribed side retains
V_p exactly and Theta=8e6 s. The changes are not lower-bound activation.

### Current constitutive stress

Identical physical bulk QPs are compared, using the accepted strain/pressure
and the working **preceding FE Maxwell history actually consumed by mechanics**.
The perturbation tensor is not the published old-history compositional tensor.
Normal traction is `50 MPa + p - tau:N`; comparing pressure alone would miss
its cancellation with deviatoric traction. Raw sample extrema over 39–41 km:

| Quantity (MPa) | Baseline min / max | Half-step min / max | Peak-to-peak change |
|---|---:|---:|---:|
| p | -2.624986 / 2.598243 | -2.771667 / 2.743468 | +5.589% |
| -tau:N | -1.419445 / 1.263749 | -1.493632 / 1.344506 | +5.775% |
| sigma_n | 47.208131 / 52.673123 | 47.094388 / 52.754054 | +3.562% |

These are transverse raw extrema, not the traction acting on a single surface
row. The actual `sum(J chi N_i sigma)/sum(J chi N_i)` means over the same window
range from **49.997150 to 50.004567 MPa** in the baseline and **49.996500 to
50.005782 MPa** with half steps: the weak peak-to-peak variation increases from
**7.416 to 9.282 kPa (+25.16%)**, remaining small compared with 50 MPa.
`final_weak_stress.png` separates p and -tau:N rather than inferring a
normal-traction change from either component alone. `current_stress_*.csv`
preserves the paired raw tensors/tractions at every shared time;
`raw_stress.csv` and `weak_stress.csv` keep the representations distinct.

### Secondary 15–18 km observations

At 15 km final V changes from **3.24664e-14 to 1.65569e-14 m/s (-49.00%)**;
slip from **0.002506730 to 0.001253551 m (-49.99%)**; Theta increases only
0.1142%. This is substantially startup timing: the first real step deposits
0.002487106 m in the baseline versus 0.001243593 m with half steps before the
next mechanics consumes the aged state. The same supplied initial Theta there
is 8000 s; it is not changed or evolved through the artificial dt0.

At 18 km final V changes **2.92663e-10 -> 2.88666e-10 m/s (-1.366%)**,
Theta **2.7335221e7 -> 2.7713663e7 s (+1.384%)**, and slip **0.331841205 ->
0.321843421 m (-3.013%)**. Thus small final-rate differences do not imply
identical accumulated histories. `final_transitions.png` retains the profiles.

Raw normal-traction peak-to-peak variation changes **2.343637 -> 2.500603 MPa**
in the 13–16.5-km window and **2.849226 -> 2.804063 MPa** in 16.5–20 km.
The corresponding native weak variations are **84.444 -> 84.761 kPa** and
**47.212 -> 49.158 kPa**. These observations are secondary, not evidence that
the 15/18-km constitutive transition or the pressure treatment should change.

## Lagged versus updated Theta diagnostic

At the final accepted state, the change in row-mass-normalized weak friction
when replacing only the lagged state by its committed update is:

| Row (km) | Baseline change (kPa) | Half-step change (kPa) |
|---:|---:|---:|
| 39.90 | +12.1538 | +2.1589 |
| 39.95 | +30.4684 | +16.1771 |
| 40.00, prescribed | +7.8104 | +4.5005 |

At 39.95 km the nodal diagnostic mu changes **0.528300986 -> 0.529171393**
in the baseline and **0.528285851 -> 0.528775133** with half steps. The weak
friction increment decreases by about 47%, but remains nonzero. It is about
0.115%/0.061% of the respective approximately 26.46-MPa weak friction load,
not compared against the tiny converged force residual.

The weak changes are integrated over the basis support. In particular, Theta
at the prescribed 40-km vertex is unchanged, but its weak row overlaps the
updating free element. Likewise the half-step nodal mu at 39.90 km decreases
while its integrated row friction increases due to neighbouring state changes.
These are not inconsistencies between point and weak diagnostics.

The final aging exponents V*dt/D_c at 39.95 km are **46.14** and **22.17**.
An exact aging update for a frozen rate does not by itself make a long
operator-split mechanical step time-accurate. However, this diagnostic is not a
causal isolation of all temporal error: halving dt also changes Maxwell updates,
particle advection/transfer and the entire preceding mechanical trajectory.
No updated Theta was used in either production mechanical residual.

## Verification and cost

- Exactly initialization plus **20 real steps**; all ten requested shared times
  agree within roundoff, and each real duration is the corresponding baseline
  duration divided by two. End marker: `audited accepted step=20,
  time=922804465.59751701`; no `Timestep 21`.
- **112 fresh linear checks**, all within their existing requested tolerances;
  2387 total Krylov iterations and 91 accepted Newton updates. Worst fresh/target
  ratio 0.996391. Minimum accepted alpha 0.133971; no Armijo retuning.
- Every accepted state satisfies the unchanged nonlinear criteria. Final
  relative bulk/surface residuals **2.45034e-13 / 1.51598e-9**; largest over the
  accepted sequence **8.19611e-10 / 9.834e-9** (target 1e-8).
- **440 free / 0 lower-active** throughout; prescribed V_p exact. Cohesion is
  identically zero under the retained mature model.
- Theta reference maximum relative discrepancy **4.44089e-16** (unchanged
  1e-12 audit). Slip recurrence checked at every step. The first Maxwell commit
  discrepancy is **2.68567e-8 Pa** on a 2640.04-Pa stress scale, with retained
  old working FE stress exactly zero. Stable-ID inert H check covers **385920**
  particles. Geometry and I_h are unchanged throughout.
- Raw accepted-state observer versus production weak traction: maximum
  **2.23270e-6 Pa**, below its existing 1e-5-Pa check. Independent lagged weak
  friction reproduction: maximum **1.41681e-6 Pa** across both trajectories.
- Runtime **1369.655 s**. Python child-process peak-RSS accounting reports
  **1,535,144 KiB (1.464 GiB)**; this is a child high-water figure, **not summed
  four-rank memory**. No retry or second simulation.

The analysis command below passes its full assertions. Python compilation
checks also pass. No C++ or plugin change was needed, so no build or broad unit
suite was rerun; the preceding task's tested guard is exercised by this run.

The recommended next decision is a further controlled temporal-accuracy check
before interpreting the notch as purely spatial or changing the mixed
free/prescribed formulation. Neither a new timestep level nor a history-scheme
change is launched here. The remaining spatial question is retained, not closed.

## Reproducibility

```sh
python3 benchmarks/reconstructed_fault/bp3/run_work_replay.py prepare --half-timesteps
python3 benchmarks/reconstructed_fault/bp3/run_work_replay.py run --half-timesteps
MPLCONFIGDIR=/tmp/aspect-work-replay-mpl python3 benchmarks/reconstructed_fault/bp3/analyze_work_halfdt.py
```

The fresh output is `benchmarks/reconstructed_fault/bp3/work-replay-halfdt-50-local4/`.
`provenance.json` records command, flags and executable/plugin/input hashes;
`clock.csv` and `shared_times.json` define the exact intended comparison.
`comparison/` contains nodal, current raw-stress, native weak-stress and friction
timing CSVs, with no smoothing or subtraction of initialization errors.

Only the benchmark runner and offline analysis are added/changed for this task;
no production or plugin constitutive code is modified. The guard tested in the
preceding task is reused. Two timestep levels can establish material sensitivity
but cannot prove temporal convergence or distinguish every split-history source
of time error.
