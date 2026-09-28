# Eight-panel short loading qualification

**Passed the bounded startup, checkpoint/history and full/half-step screens.**
Retained weakening 0–30 km, transition 30–33 km, eight surface quadrature
panels, ell=100 m, Dc=0.1 m, true normal-stress feedback, fully frictional
mature fault, fixed mesh/phase/support, paired endpoint corrections, and the
existing split history update. No production C++ change or new solver build
was needed in this task; the preceding qualified eight-panel binary/plugin
were used unchanged. No long run was launched.

## Startup

The previous initialization-only run had no checkpoint, so a fresh startup
was necessary. Its accepted initial velocity, incoming/outgoing state and slip
were bitwise identical to `surface-quadrature/panels8-initial`. Initial state is
8e7 s in the weakening plateau and 1e8 s in strengthening, using the actual
projected mixture through the transition. The native weak prestress error was
1.52597e-7 Pa, below the unchanged 1e-5-Pa initialization allowance. Initial
committed Maxwell stress and slip remained zero.

Artificial Maxwell initialization remains 1e6 s, not elapsed physical time.
Actual adaptive selection:

| Real step | Convection proposal (s) | Fault proposal (s) | State proposal (s) | Selected dt (s) | Limiter |
|---:|---:|---:|---:|---:|---|
| 1 | 1.220714e10 | 6667933.96 | 1072902.49 | 1000000 | First-step ceiling |
| 2 | 1.220714e10 | 6667934.11 | 1086454.61 | 1086454.61 | State predictor |
| 3 | 1.220030e10 | 6780100.49 | 1022994.90 | 1022994.90 | State predictor |

Global ceiling 1e7 s and growth factor 1.91 were not controlling. No inherited
300-s restriction is present. Final time is **3109449.5059839734 s (35.989 days)**.
Predicted/realized maximum weighted log-state changes were respectively
0.018649486/0.018649484, 0.020000000/0.021521078 and
0.020000000/0.021615090. The predictor is a heuristic; the slight realized
exceedance is retained, not hidden or used to relax its 0.02 setting.

Final region measurements use native work-weighted tractions and changes
relative to this run's own accepted initialization, not the 50-MPa background.

| Region | V/Vp range | Maximum slip deficit (m) | Max abs shear change (Pa) | Max abs normal change (Pa) |
|---|---:|---:|---:|---:|
| Shallow interior, 2–28 km | 0.960376–0.961735 | 6.10969e-5 | 259.21 | 44.04 |
| Transition window, 27–36 km | 0.961362–0.999923 | 5.97265e-5 | 1077.61 | 1.76 |
| Deep interior, 35–110 km | 0.999912–0.999973 | 1.61415e-7 | 134.16 | 6.30 |
| Top, 0–2 km | 0.959806–0.960510 | 6.19702e-5 | 24.28 | 130.16 |
| Bottom, xd>113 km | 0.999919–0.999985 | 1.58291e-7 | 58.68 | 42.98 |

The intended early shallow aging/slowdown develops while deep creep remains
near plate rate. This is not locking, nucleation or event evidence. Whole-fault
and transition plots are `loading-whole.png` and `loading-transition.png`;
incoming state is dotted, whereas plotted R uses committed state. No curves
were smoothed.

## Checkpoint and half-step comparison

Copy the ordinary accepted-step-1 checkpoint at t=1e6 s. The existing byte-checked
branch tool changes only the pending time/dt in the copied serialized clock.
All other uncompressed checkpoint bytes, mesh, particle history and benchmark
metadata are identical. The full second interval 1086454.605589005 s is compared
with two halves of approximately 543227.3027945025 s, ending at the same
**2086454.605589005 s** (the final half uses the exactly represented remainder).

| Existing startup screen | Maximum observed | Limit | Result |
|---|---:|---:|---|
| abs(log(V_full/V_half)), V>=0.1Vp | 0.01084604 | 0.02 | Pass |
| abs(log(Theta_full/Theta_half)) | 5.67634e-5 | 1e-3 | Pass |
| abs(slip_full-slip_half)/Dc | 5.69517e-5 | 1e-4 | Pass |

| Quantity | Maximum absolute difference | Work-weighted RMS difference | RMS difference / evolving-change RMS |
|---|---:|---:|---:|
| V (m/s) | 1.05824e-11 | 5.30924e-12 | 36.11% |
| Committed Theta (s) | 4565.58 | 2303.25 | 1.896% |
| Slip (m) | 5.69517e-6 | 2.87281e-6 | 0.2661% |
| Native weak shear (Pa) | 167.384 | 23.333 | 43.04% |
| Native weak normal (Pa) | 16.643 | 1.274 | 28.18% |

These pass the predeclared exploratory startup screens, **not a temporal
convergence criterion**. Mechanical increments remain materially timestep
dependent. No extra cap-halving experiment or physical-state adjustment was
made to improve the comparison. Incoming state differs in the final full and
half solves because the second half correctly uses its preceding accepted
update; both branches start from exactly the same incoming state.

The initializer did not rerun on restart. Cold normalization agreed with the
persisted frozen field to 2.22045e-16 relative. Raw QP geometry/phase/source
associations matched; Ih was unchanged and localization differed by at most
5.2042e-18. Stored background/native masses were preserved to existing checks.
Every incoming Theta matched the preceding committed state exactly; stable
aging checks were within 2.22e-16 relative, and slip matched exactly one fused
dt*V accumulation per accepted step. The first Maxwell publication independently
agreed within 7.5755e-8 Pa on a 6664.33-Pa stress scale, with zero incoming FE
history and 1,034,856 stable H identities. Later Maxwell updates reuse the
unchanged production implementation; this is not an independent reconstruction
of every later particle update or a cross-rank restart qualification.

## Convergence, resources and provenance

Every accepted solve passed the unchanged bulk/surface criteria and fresh
linear checks. All **1156 nodes stayed free**, no lower-bound contacts occurred,
and every accepted alpha was 1. Startup accepted Newton updates were 1/1/2/2,
with 41/42/62/61 Krylov iterations. Both resumed half-steps used two Newton
updates and 54/55 Krylov iterations. Final startup relative bulk/surface
residuals were 7.587113e-12 / 1.385041e-9; final half-branch values were
3.298291e-12 / 9.25536e-11.

Four-rank Release wall times: **470.54 s startup**, **310.58 s half branch**,
both below the existing 1100/1200-s graceful/hard caps. Maximum child RSS was
6,042,408 / 6,093,404 KiB (5.76 / 5.81 GiB), not measured aggregate RSS.
The startup Ih timer was 85.9 s over five calls with one cold integration;
cached preparations remained active. No fine-grained performance campaign was
added. For server use allow at least 32 GiB for a four-rank job; 48 GiB provides
more practical headroom. Multi-event peak memory is not qualified here.

Commands:

```sh
python3 benchmarks/reconstructed_fault/bp5/run_loading_surface.py prepare startup
python3 benchmarks/reconstructed_fault/bp5/run_loading_surface.py run startup
python3 benchmarks/reconstructed_fault/bp5/run_loading_surface.py check startup
python3 benchmarks/reconstructed_fault/bp5/run_loading_surface.py prepare half
python3 benchmarks/reconstructed_fault/bp5/run_loading_surface.py run half
python3 benchmarks/reconstructed_fault/bp5/run_loading_surface.py check half
python3 benchmarks/reconstructed_fault/bp5/run_loading_surface.py plot
```

The two `checks.json` files and `comparison.json` retain quantitative results.
`launch.json`, `source-tested/`, logs, source/binary/input hashes and
`half/checkpoint_source.json` retain the tested state and branching proof.
No ASPECT/plugin runtime sources changed during this check. Earlier focused
normalization and lifecycle unit-test evidence is reused; no full suite was run.

The separate `server-30km-loading-surface8/` package retains the exploratory
1e7-s ceiling, first-event-through-decay termination, 1500-year safety end,
hourly/on-termination restart, lightweight histories and sparse heavy output.
It includes the new 27,720-profile completion file and removes local step/time
limits. **Start fresh, and do not mix the old three-point completion file or
old trajectories with this configuration.** Source inventory is
`bp5/first_event_source_files.md`, copied into the package as `SOURCE_FILES.md`.
The previous server package and all older evidence remain untouched.
