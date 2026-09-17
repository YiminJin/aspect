# Fully frictional corrected-work replay: seven-step comparison

## Scope and initialization

The authorized fixture removes **all** deep prescribed-V rows, including the
bottom endpoint. Velocity, state and accumulated slip use the ordinary
continuous Q1 representation through 40 km; no split-trace selector is active.
This is a deliberate departure from the official BP3 prescribed-deep-slip
condition, not a new default. The mature law, work measure, fixed mesh/phase,
full Ih, support, both boundary completions/source extensions, fixed background,
outer loading, true normal stress and solver/history rules are unchanged.

No saved fully frictional run matching these settings was found. The new
fresh run is `benchmarks/reconstructed_fault/bp3/fully-frictional-seven-local4/`;
the comparison is saved `work-replay-50-local4`, not the split-trace run.
Use exactly its seven accepted real timesteps through 132230424.76671731 s,
4.1901293117 yr. The artificial initialization interval remains 4e6 s.

The deep region already has a=.025, b=.015 and supplied
Theta0=Dc/Vinit=8e6 s. At Vinit=Vp=1e-9 m/s and sigma0=50 MPa, the initial
reference balance is

\[
 \tau_0=\mu(V_{\rm init},\Theta_0)\sigma_0+\eta^d V_{\rm init}
       =26.5461223651\ {m MPa}.
\]

The retained mature numerical background includes its original small
discretization corrections: deep nodal effective values differ from tau0
by -871.092 to +49.850 Pa. These are not refitted. The converged initial deep
rates are 0.999530830–1.000427424 Vinit; the bottom endpoint is
0.999531421 Vinit. Theta0 is retained (roundoff from 8e6 s <=2.24e-8 s), and
initial particle stress remains zero. Initial nodal Theta, particle IDs/H/stress,
phase and completed Ih match the constrained baseline.

Realized lateral constraints are exactly left=(2.5e-10,4.330127018922193e-10)
m/s and right its negative, with speed Vp/2. The existing top/bottom zero
perturbation tractions and pressure treatment are unchanged. The continued
bottom source now uses the solved endpoint rate; it is not secretly pinned
to Vp. No production numerical algorithm was changed.

## Commands and evidence definitions

```
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_trace_replay.py --fully-frictional
python3 benchmarks/reconstructed_fault/bp3/analyze_fully_frictional.py
```

One four-rank process, 2400 s hard cap, no automatic retry. Strict saved-clock
checks retain the other production timestep-controller restrictions. The
benchmark termination guard stops only after the audited final accepted state.
`provenance.json`, `source.patch`, `execution.json`, `clock.csv` and `run.log`
preserve the source/input/executable identities and execution evidence.

Stress below means accepted current constitutive stress from u,p,V and the
frozen working FE old history, **not** the newly committed particle stress
array. Raw comparisons match physical bulk QPs. Native weak traction uses
the unchanged work weights JxW*chi*N; the nodal support/mass is identical in
the two continuous-Q1 runs. Report both raw extrema and weak averages.

Shallow deficit is Vp*t minus accumulated slip, not merely small current V.
Loading is measured by the increase in native weak shear traction relative
to each run's accepted initial state, together with unchanged realized outer
velocities. Regional means of nodal quantities use geometric half-element
length weights for reporting only. The weak equation itself is not changed.

## Results

The single fresh four-rank run accepted initialization and exactly seven real
steps, ending at 4.1901293117 yr. The former 40-km local minimum disappears
without introducing a slip discontinuity. Shallow slip deficit and tectonic
loading remain, deep creep stays close to Vp, and the top/bottom stress ranges
remain close to the constrained control. This supports the requested model
change for this bounded interval, not long-term or spatial convergence.

### Former 40-km junction

| Final quantity | Prescribed-deep control | Fully frictional |
|---|---:|---:|
| V(39.90 km)/Vp | 0.982584152 | 0.969043404 |
| V(39.95 km)/Vp | 0.966302401 | 0.969139462 |
| V(40.00 km)/Vp | 1 | 0.969253857 |
| V(40.05 km)/Vp | 1 | 0.969345357 |
| 39.95-km depression below neighboring chord / Vp | 0.0249896748 | 0.00000916909 |
| Depression below both neighboring rates / Vp | 0.0162817507 | 0 |
| Slip gradient on 39.95–40.00 km | 5.67251e-5 | 1.92443e-7 |
| Raw sigma_n range, 39.8–40.2 km (kPa) | 170.784 | 7.47465 |
| Work-weighted mean-removed raw sigma_n RMS (kPa) | 27.4398 | 1.29953 |

The rate rises monotonically through the former notch. The chord defect is
99.9633% smaller and the last-element slip gradient is 99.6607% smaller.
Accumulated slip remains continuous: the smooth 2.54533-mm deficit relative
to Vp*t at 40 km is **not** a junction jump. Committed Theta there is
8.25370e6 s, with its own ordinary aging update rather than a prescribed state.

In the same raw-QP window, sigma_n is 49.9947443–50.0022190 MPa. The
pressure range is 14.7622 kPa versus 171.610 kPa in the control, and the
minus-tau:N range is 18.2320 versus 74.4099 kPa. Thus the improvement is in
actual normal traction, not merely one stress component. Its range is
95.6233% smaller. In the wider 37–43 km window the remaining sigma_n range
is 10.6704 kPa; small residual discretization features are not declared solved.

### Shallow loading and deep creep

At 0–15 km the final length-weighted mean rate is 1.80155e-6 Vp, the mean
accumulated slip deficit is **129.7418 mm**, and mean native weak shear
traction has increased **196.382 kPa** from the accepted initial state.
The constrained control increases 196.720 kPa. In the 0–13 km core the
increase is 167.838 kPa, positive at every reported node. At 10 km the
deficit is 129.7418 mm and shear-traction increase is 202.269 kPa.
The imposed outer velocities remain exact. The shallow fault therefore
remains loaded and slip-deficient; removing deep kinematic rows has not
removed tectonic forcing. None of these rates is lower-bound active.

At 45–100 km the rate range is **0.978115–0.997343 Vp**, with mean
0.989374 Vp. The mean slip deficit is only 0.88455 mm. Representative
rates are 0.987134 Vp near 60 km and 0.997343 Vp near 100 km. The solved
bottom endpoint is 0.997874383 Vp, with Theta=8.01704e6 s and accumulated
deficit 0.21112 mm. There are no prescribed nodes anywhere in this run.

### Top, bottom, and remaining shallow features

Raw current normal traction, at matching physical bulk QPs:

| Window | Control range (kPa) | Fully frictional range (kPa) | Fully frictional min/max (MPa) |
|---|---:|---:|---:|
| Top, xd=-2 to 2.5 km | 79.8779 | 79.6351 | 50.1569165 / 50.2365516 |
| Bottom, xd=L-2.5 to L+2 km | 98.9447 | 97.8795 | 49.9391140 / 50.0369935 |

Here L=115.470053838 km. Work-weighted mean normal traction is 50.1974858
MPa in the top window and 49.9932206 MPa in the bottom window. Native nodal
weak normal traction at the top/bottom endpoint is respectively 50.2319482
and 49.9947002 MPa. These different sampling/averaging definitions are not
interchanged. Existing boundary corrections survive the released bottom;
the residual boundary errors have not vanished or been spatially qualified.

The 15–18 km transition features remain largely unchanged. For example,
the saved 15-km window raw normal-stress range is 213.769 kPa versus
213.881 kPa in the control. The fully frictional 18-km window still spans
49.4322832–50.5627362 MPa. No claim of curing those separate features follows
from releasing the 40-km constraint. `raw_stress.csv` also records pressure,
individual deviatoric components, shear and strain mismatch at all times.

## Correctness and cost

- The plugin build with `-j4` passed. No core solver implementation was
  changed in this task and no broad test suite was rerun.
- All eight accepted states pass the saved timestep, geometry, phase/Ih,
  history, continuous-state and raw-stress identity checks. Initial particle
  IDs/H/zero stress and nodal Theta match the saved control. Matching bulk
  QPs retain their weights and association mask; localization agrees to the
  tight comparison tolerance. Exported unassociated positive-phase tail
  points are counted separately, not silently treated as new support loss.
- There are 1236 free, zero prescribed and zero lower-active nodes at every
  accepted state. Every real step passes the exact aging and accumulated-slip
  update audit, with one update per accepted step.
- All **71 fresh linear checks** passed; the run used **1360 Krylov
  iterations**. Final normalized bulk/surface residuals are
  1.78427e-13 / 2.00497e-13. Unnormalized norms are 1.0039044e-4 and
  8.5884258e-7 in their respective discrete norm conventions; these have
  different units/scales and should not be compared to one another.
- The first-step surface tangent finite differences cover the former
  junction (node 796) and newly free bottom endpoint (node 0). At h=1e-16
  the relative errors are 3.26920e-8 and 5.87517e-8; at h=1e-15 they are
  3.36911e-7 and 4.18011e-7. The full-fault pressure G check gives 9.56218e-12.
  Existing tolerances were not relaxed.
- Independent reconstruction of the physical weak rows at the bottom and
  junction agrees within **7.11822e-6 Pa m** over all saved states. The first
  Maxwell publication check has error 5.37134e-8 Pa on a 5274.322 Pa scale,
  retaining the approved timestep-zero and split-history semantics.
- Wall time is **840.126 s (14.00 min)**, below the 2400-s cap. Reported
  peak child RSS is **1,524,044 KiB (1.453 GiB)**, a per-process high-water
  measurement, not aggregate four-rank memory. There was no retry.

The offline checker was rerun successfully after final diagnostic labeling.
No additional ASPECT trajectory was launched for report completeness.

## Artifacts and decision

Under `benchmarks/reconstructed_fault/bp3/fully-frictional-seven-local4/`:

- `run.log`, `execution.json`: convergence, first-update/derivative checks,
  termination and cost;
- `run.prm`, `clock.csv`, `provenance.json`, `source.patch`: exact bounded
  replay and recoverable tested source/input identities;
- `analysis/comparison.png`: whole-fault rate/deficit/loading, junction
  close-up and top/bottom controls;
- `analysis/junction.csv`, `nodes.csv`, `regions.csv`, `raw_stress.csv`:
  initialization and each accepted physical time, not just final extrema;
- `analysis/weak_balances.csv`, `checks.csv`, `matching_differences.csv`,
  `summary.json`: weak-row reconstruction and matching-input checks.

The requested seven-step test passes: a fully frictional, continuous fault
removes the 40-km neighboring undershoot and sharply reduces its normal-stress
concentration while retaining shallow loading and near-plate-rate deep creep.
This is an explicitly selected **modified BP3 model**, not a repair proven
equivalent to official prescribed deep slip. Longer-time behavior, timestep
and spatial convergence, and the remaining 15–18 km features are untested
by this bounded comparison. Stop here; no automatic continuation is needed.
