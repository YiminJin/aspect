# BP5 loading-driven initialization: bounded qualification

## Decision

The single authorized candidate, `R_VW=0.8`, passes initialization, three
adaptive steps and the checkpoint-based two-half-step comparison. Prepare an
**exploratory**, fresh-start `server-30km-loading` package. This is not temporal
convergence through a first event: weak shear/normal traction differences are
still 42.8%/28.5% of their small evolving changes in work-weighted RMS.
No additional timestep, initial-state or physical-parameter search was run.

## Implementation and preserved problem

The existing `bp5_steady_initialization` implementation now declares
`Postprocess/BP3/Weakening initial state ratio`, default 1, requiring `0<R<=1`.
After ordinary surface chemical projection it sets
`Theta_i=(Dc/Vinit)*R^(1-f_i)` using the production material fractions. For the
configured arithmetic two-material law, the VS fraction is exactly
`(a_i-a_VW)/(a_VS-a_VW)`. Particle initial inputs use the corresponding
horizontal-depth profile; the final projected-mixture nodal state is
authoritative. Default 1 retains the uniform state and old restart identity.

The existing native weak initializer evaluates **Q1-interpolated state and
projected mixture**, with the actual bulk-QP work weights and both endpoint
continuations. It solves the same mass equation for friction plus damping,
excluding the private zero-strain probe's crack-induced shear. Captured shear
and its correction are replaced together; background is never recalibrated
after mechanics. The ratio/profile version identifies loading checkpoints;
different ratios, steady, inverse-state and unmarked seeded checkpoints are
not interchangeable. Restart restores histories and selectors, not initial data.

No generic ASPECT source/header, physical equation, work measure, aging map,
solver tolerance, mesh or support rule changed in this task. The 300x100-km,
60-degree, fully frictional mature-C=0 fixture retains Dc=0.1 m, ell=100 m,
the 30–33-km transition, AMG/FGMRES, sparse B/G and tridiagonal surface inverse.
There are 114984 cells, 4130126 DoFs, 1034856 particles and 1156 free fault nodes.

Changed benchmark files: `bp3/bp3_model.h`, `bp3/bp3.cc`,
`bp5/steady_initialization.h`, `bp5/startup_time_step.cc`,
`bp5/test_steady_initialization.cc`. New launcher, checker/plotter, package
builder and README are `run_loading_startup.py`, `analyze_loading_startup.py`,
`package_loading.py`, `loading_server_README.md`. The two authoritative design
documents record this explicitly selected initial-data variant.
`Record timestep selection` is an output-only parameter; its read-only
controller audit does not select or modify timesteps.

## Initialization

| Quantity | Observed |
|---|---:|
| VW / VS initial state | 8e7 / 1e8 s |
| VW / VS target initial ratio | 0.8 / 1 |
| VW background shear | 38.64536654033 MPa |
| VS background shear | 26.54612236514 MPa |
| Native weak prestress maximum error | 1.52606e-7 Pa (allowance 1e-5 Pa) |
| Independent initialization QP friction audit, 27–36 km | 1.00137e-7 Pa |
| Initial maximum abs(V/Vp-1) | 3.07502e-4 (screen 0.005) |
| Initial normal-traction range | 49.996596–50.002431 MPa |
| Initial committed Maxwell stress / slip | exactly zero / zero |
| Initial free / lower-active nodes | 1156 / 0 |

The realized shallow `V*Theta/Dc` is about 0.799754–0.799915, reflecting the
small normally solved mechanical departure from Vp. The raw Q1 chemical
coefficients range from -0.000490071 to 1.000485905. The **existing production
composition-fraction utility** converts these to valid material fractions;
there is no new clipping of projected chemistry, state or background shear.
The initializer checks its resulting fraction and only permits roundoff at
its bounds. CSV output preserves both raw chemistry and production fractions.

The independent QP audit is restricted to initialization, where the archived
initial projected mixture is correct. An initial checker mistakenly reused
that mixture after particle advection, yielding a 0.0247-Pa step-1 mismatch.
That diagnostic error was corrected without rerunning or changing mechanics.
Subsequent force budgets use the native production weak loads and the correct
incoming state, not a reconstruction from initial chemistry or committed state.

## Timesteps and early response

Artificial Maxwell initialization is 1e6 s, not elapsed physical time.
The first physical cap is 1e6 s, global ceiling 1e7 s, CFL 0.5, growth limit
91%, state-predictor bound 0.02. Generated parameters confirm no inherited
300-s or 125000-s cap.

| Next real step | Convection proposal (s) | Fault proposal (s) | State proposal (s) | First/growth cap (s) | Selected (s) |
|---|---:|---:|---:|---:|---:|
| 1 | 1.2207074e10 | 6667305.45 | 1072507.10 | 1000000 | 1000000 |
| 2 | 1.2207074e10 | 6667305.45 | 1086054.24 | 1910000 | 1086054.24 |
| 3 | 1.2200282e10 | 6779232.73 | 1022670.37 | 2074363.61 | 1022670.37 |

Thus the first cap controls step 1 and the state predictor controls steps 2–3.
The final time is 3108724.613695916 s (35.9806 days). The maximum weighted
predicted/realized log-state changes are respectively:

| Step | Predicted | Realized |
|---|---:|---:|
| 1 | 0.018656315 | 0.018656312 |
| 2 | 0.020000000 | 0.021519877 |
| 3 | 0.020000000 | 0.021616252 |

The predictor remains a heuristic, not a guarantee for the newly solved rate.
Its slight realized exceedance is reported, not hidden or used to change it.
The shallow interior (2–28 km) ends at V/Vp=0.960206–0.961793 and
R=0.774514–0.775774, with maximum slip deficit 6.15011e-5 m. The deep interior
(35–110 km) remains near Vp: 0.999878–1.000015. These are early aging/slowdown,
not locking or event/nucleation evidence. Small resolved-grid ripples remain;
no smoothing or extra spatial qualification is claimed.

Native weak mechanical changes are relative to accepted initialization,
not normalized by the large background:

| Final region | Max abs shear change (Pa) | Max abs normal change (Pa) | Max slip deficit (m) |
|---|---:|---:|---:|
| Shallow interior, 2–28 km | 259.26 | 43.99 | 6.1501e-5 |
| Transition window, 27–36 km | 1083.20 | 1.779 | 5.9936e-5 |
| Deep interior, 35–110 km | 150.97 | 6.319 | 2.3296e-7 |
| Top, 0–2 km | 29.31 | 118.03 | 6.2193e-5 |
| Bottom, last 2.47 km | 77.54 | 27.37 | 2.0294e-7 |

Whole-fault and 27–36-km plots are `loading-startup/loading-whole.png` and
`loading-transition.png`. Incoming state is dotted; the plotted R explicitly
uses committed state. Force comparisons instead use the mechanics' incoming state.

## Checkpoint subdivision and accuracy

Copy the ordinary accepted-step-1 checkpoint at t=1e6 s; change only its pending
clock through the existing byte-checked branching tool. Mesh/history/background
and all other uncompressed bytes remain identical. One full interval
1086054.244063706 s is compared with two intervals of 543027.122031853 s,
ending at exactly 2086054.244063706 s. Live controllers allow both half-steps.
No initialization is rerun, and the first resumed incoming nodal state is
bitwise identical to the saved committed state. The second half-step uses its
own updated state; comparing its incoming state to the full step's incoming
state is a timing difference, not a restart error.

| Prescribed startup screen | Maximum | Work-weighted RMS | Limit | Result |
|---|---:|---:|---:|---|
| abs(log(V_full/V_half)), V>=0.1Vp | 0.01084360 | 0.00543621 | 0.02 | Pass |
| abs(log(Theta_full/Theta_half)) | 5.67539e-5 | 2.86111e-5 | 1e-3 | Pass |
| abs(slip_full-slip_half)/Dc | 5.69622e-5 | 2.87071e-5 | 1e-4 | Pass |

All nodes exceed 0.1Vp here. Absolute differences and evolving-change scales:

| Quantity | Max absolute difference | Weighted RMS difference | RMS difference / RMS evolving change |
|---|---:|---:|---:|
| V (m/s) | 1.05791e-11 | 5.30731e-12 | 36.10% |
| Committed Theta (s) | 4564.74 | 2301.57 | 1.896% |
| Slip (m) | 5.69622e-6 | 2.87071e-6 | 0.2660% |
| Native weak shear (Pa) | 167.451 | 23.3222 | 42.81% |
| Native weak normal (Pa) | 16.4967 | 1.27243 | 28.48% |

The latter percentages prevent a false claim of converged mechanical increments.
They do not replace the task's predeclared screens. No additional cap-halving
campaign was run; long-time accuracy and low-rate active-set behavior remain
open qualifications. The 1e7 ceiling is expressly exploratory.

## Lifecycle, convergence and cost

Every accepted mechanics solve passes the original bulk/surface relative
criteria and fresh linear checks. Startup Newton accepted updates are 1/1/2/2
for states 0/1/2/3, with 42/42/62/61 Krylov iterations. The two resumed steps
use 2/2 updates and 54/55 Krylov iterations. Every accepted alpha is 1;
all nodes remain free. Final surface RMS is at most 0.002636 Pa, consistent
with the unchanged relative criterion (not the much tighter initializer check).

The state audit checks the independent stable aging formula at every node
(largest reported relative error 2.22e-16). Slip exactly matches one fused
dt*V accumulation per accepted step. Initial state is retained without aging.
The first finite Maxwell publication independently matches the zero-history
update within 7.58e-8 Pa against a 6679.73-Pa stress scale; prior FE history is
exactly zero. Later Maxwell publication uses unchanged production code; no
second initializer/update is added. Restart byte checks preserve incoming
particle stress/history, and the continued stable-ID H/geometry checks pass.
This is not a second independent all-particle Maxwell reconstruction at every
later state, nor a cross-rank restart test.

At exported matching bulk QPs, phase, geometry and source association are
unchanged; I_h is unchanged and localization differs by at most roundoff
(about 5.2e-18 during startup). Restart cold normalization differs relatively
by 3.33e-16 before restoring the validated frozen value. Both paired endpoint
corrections remain enabled and participate in the same force/history path.

Four-rank Release startup: **406.82 s**; restarted half comparison: **231.99 s**.
Both finish below the 1100/1200-s soft/hard limits. Largest child-process peak
RSS is 2.68/2.72 GiB respectively; this is **not measured aggregate RSS**.
Four times the latter is about 10.9 GiB as a conservative rank-only estimate.
No additional ASPECT trajectory or server submission was performed.

## Reproduction and provenance

Base HEAD: `359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`, with existing working
changes retained. `bp5/loading-provenance-before/` contains the pre-task diff,
affected benchmark sources and prior plugins. Each run has `launch.json`,
source snapshots/hashes, logs and `execution.json`; the half branch also has
the checkpoint byte/hash audit. Tested binary SHA256:
`00091b029bd2b0ecde7f9554b0f987ec35145842f66e1c57966884802a442372`.
Tested plugin SHA256:
`2dd8df3ce805c6853043c8d50fbca0f8c32d80ffb3e0c5ef947045122ab22df5`.

Commands (immutable case directories; do not repeat over existing evidence):

```sh
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_steady_initialization -j4
python3 benchmarks/reconstructed_fault/bp5/run_loading_startup.py prepare startup
python3 benchmarks/reconstructed_fault/bp5/run_loading_startup.py run startup
python3 benchmarks/reconstructed_fault/bp5/analyze_loading_startup.py startup
python3 benchmarks/reconstructed_fault/bp5/run_loading_startup.py prepare half
python3 benchmarks/reconstructed_fault/bp5/run_loading_startup.py run half
python3 benchmarks/reconstructed_fault/bp5/analyze_loading_startup.py half
python3 benchmarks/reconstructed_fault/bp5/analyze_loading_startup.py plot
python3 benchmarks/reconstructed_fault/bp5/package_loading.py
```

The legacy inverse-state plugin also compiles. The standalone initializer test
passes six default-state locations, four steady-aging intervals, five loading
mixtures, distinct ratio identities and the scalar predictor. The existing
50-digit aging test passes ten stable-reference cases, deliberately incorrect
state rejections and zero interval (compile with deal.II's bundled Boost include).
The loading parameters pass ASPECT validation; Python tools compile, and diff
whitespace checks pass. No complete ASPECT suite or old trajectory was rerun.

The independent scalar screen, **not a coupled prediction**, gives an initial
allowance 1073835.18 s and 1.54120 years to V/Vp=0.1 under constant traction.
The actual coupled first proposal is slightly smaller because solved initial
rates differ from Vp. The future first-event outcome remains unknown.
