# Uniform steady state and native weak variable prestress

## Scope and initial-data change

Explicitly select the new `bp5_steady_initialization` plugin, rather than
`bp5_initialization`. The latter remains the inverse-state, uniform-background
historical configuration. Do not load both plugins in one process.

The new choice supplies `Theta0 = Dc/Vinit = 1e8 s` everywhere, including
initial particle composition and the nodal state. After ordinary surface
chemical projection it constructs the frozen background using the production
surface evaluator:

\[
M\tau_{\rm bg}=\int N_i\,[\mu(V_{\rm init},\Theta_0,\mathbf f_\Gamma)
  \sigma_{\rm bg}+\eta^dV_{\rm init}]\,d\nu,\qquad\sigma_{\rm bg}=50\ {\rm MPa}.
\]

The work measure, source association, projected mixture, consistent Q1 mass,
MPI ownership/reduction and both endpoint continuations are the production
ones. No particle-center substitute or nodal friction inversion is used.
The private constitutive probe retains the constrained phase/temperature/
chemical fields and zeros its velocity and pressure. Its crack-induced shear
is **excluded** from prestress; the normal perturbation is verified zero.
Only its friction and damping weak loads define the background. A second
production residual evaluation verifies the newly added weak background.

Both old prestress channels are replaced: shear is the new Q1 field, normal
is 50 MPa, and the old correction coefficients are exactly `(0,0,1)`.
A nonempty `Mature prestress file` is rejected in this plugin. It never calls
the captured-prestress importer or the projected nodal/weak inverse-state
initializer. Initialization remains a stress-perturbation problem; particle
Maxwell stress is retained zero at timestep zero.

The benchmark checkpoint adds `BP5 initial condition = steady state/native
weak prestress v1` alongside the unchanged accepted-history archive. It is
required by the new plugin and rejected by the old variant. Restored history
prevents reinitialization even for a timestep-zero checkpoint. The manager's
serialized background/state is authoritative; only runtime selectors are
reattached on resume.

## Unchanged model and bounded tests

Keep the 300x100-km box, 60-degree fully frictional geometry, 0–30-km weakening
and 30–33-km transition, `a=0.004/0.04`, `b=0.03`, `Dc=0.1 m`, frozen AT1
`ell=100 m`, zero cohesive law, true normal stress, outer plate loading and
paired endpoint corrections. Mesh, particles, fault grid, support, full Ih,
AMG/sparse-B/G/pivoted surface solver and all tolerances are unchanged.

Artificial initialization uses 4e6 s without physical history evolution.
The short verification uses three real 300-s steps; the single half-step
comparison uses six 150-s steps through the same 900 s. The 0.02 weighted
log-state-change predictor and ordinary convection/RSF timestep models remain
enabled. An ordinary accepted-step-2 checkpoint at 600 s is resumed to 900 s
on the same four ranks and binary/plugin. This is not a long-run or temporal
convergence qualification.

Results are isolated in `weakening30-dc010-ell100/steady-startup/`.
Each launch records source revision, binary/input hashes and a recoverable
source copy plus working-tree patch. A sandbox MPI failure before ASPECT
entry is retained in `startup/launch-sandbox.*`; the actual simulation runs
outside that restriction. No failed numerical case was retried.

## Initialization and startup results

- Uniform initial particle prescription and nodal state: `1e8 s`; nodal
  timestep-zero retention is exact.
- Analytic plateau background: approximately 38.980082 MPa weakening and
  26.546122 MPa strengthening. The projected field's full nodal range is
  26.543146–38.983079 MPa (small transition projection overshoots retained).
- Maximum native weak background imbalance: `1.57358e-7 Pa`.
- Independent exported-QP reintegration: `1.08063e-7 Pa` over 339 completely
  covered rows. The maximum **pointwise** background/target discrepancy is
  2991.06 Pa: weak balance is not a claim of exact pointwise interpolation.
- Initial accepted `V/Vp`: 0.99896753–0.99997206. Ordinary mechanical
  discretization/initial velocity adjustment is not absorbed into background.
- At 900 s: `V/Vp=0.9999996935–1.0000000597`; state range
  `99999999.99997976–100000000.00017333 s`.
- All 1156 nodes remain free; all accepted alphas are one. Exact aging,
  first Maxwell publication, frozen H/geometry/Ih and fresh-linear checks pass.
- Native surface residual checks and the independent initialization check
  retain the existing `1e-5 Pa` verification allowance, not a new tolerance.
- Maximum predictor measure: `2.32306e-8`, below 0.02; the selected 300-s
  verification ceiling controls startup rather than the predictor here.
- Startup wall time: 439.53 s; maximum child RSS: 2,753,964 KiB (per-process
  high-water mark, not total simultaneous rank memory).

## One half-step comparison

Six 150-s steps also pass through 900 s, with no lower-active nodes. Fresh
initialization is identical to the 300-s case, including background and state.
The differences below use common physical times and current native weak
tractions (load divided by its work weight), not just a 50-MPa normalization.

| Time (s) | max relative V difference | max slip difference (m) | max state difference (s) | max shear difference (Pa) | max normal difference (Pa) |
|---:|---:|---:|---:|---:|---:|
| 300 | 3.69815e-8 | 1.25597e-14 | 1.25766e-5 | 0.00610825 | 0.00274645 |
| 600 | 3.70315e-8 | 2.54349e-14 | 2.54512e-5 | 0.00614413 | 0.00205467 |
| 900 | 3.81815e-8 | 3.85967e-14 | 3.86238e-5 | 0.00620740 | 0.00147744 |

These are small startup differences; two levels do not establish an order of
convergence or long-time accuracy. The state differences are also reported in
seconds, because the 1e8-s background can obscure the tiny evolving increment.
The 0.02 predictor remains enabled (maximum measure 1.16153e-8 in this case).
Wall time: 643.62 s; maximum child RSS: 2,726,148 KiB.

## Restart

The resumed step from the ordinary accepted-step-2 (600-s) checkpoint passes
at 900 s. No initialization, mesh regeneration or prestress import is repeated.
The restored pending timestep and newly proposed timestep are both 300 s,
with identical predictor data and unchanged 0.02 limit.

All 44 scalar/component comparison differences are **zero**: velocity, incoming and committed state,
slip, bulk FE components, stable-ID particle properties (including stress/H),
current constitutive stress and traction, phase, Ih, chi, source association,
work background and controller output. Incoming/committed state is additionally
required bitwise identical; its actual nonzero increment exceeds 1e-5 s, so
this check cannot pass just because a 1e8-s reference hides a reset.
The standard per-component comparison allowance remains 1e-8; no tolerance
was changed to obtain restart agreement. This qualifies the same four ranks
and binary, not a cross-rank restart or conversion of an old initial condition.

Resumed solve: final relative bulk/surface residuals
`4.516219e-10 / 1.988798e-16`, with fresh linear checks passing. Wall time:
104.63 s; maximum child RSS: 2,866,632 KiB. Total simulation wall time across
the three cases: 1187.77 s (19.80 minutes).

## Verification and remaining limits

- Release plugin built with `-j4`; no production solver rebuild/change.
- Standalone C++ regression passes six positions and four aging intervals.
  Its mock configured law deliberately has no inverse-state API, so that
  initialization cannot accidentally call the old inverse.
- Startup, half-step, independent native-QP integration, stable-ID history
  checks and full-state restart comparison all pass. See each `checks.json`
  and `steady-startup/comparisons.json` for numerical details.
- Fresh/restart code rejects incompatible initialization metadata; an old
  checkpoint rejection and a timestep-zero restart were not separately run.
- The 300/150-s ceilings deliberately bound this startup test. The unchanged
  0.02 safeguard is evaluated but nonbinding near steady sliding. These tests
  do not qualify a 4e6-s physical step or a first-event trajectory.
- No server package was regenerated. Old `server-30km-adaptive/` still uses
  the inverse-state/uniform-background initial condition; do not restart its
  eight-day trajectory with the new initial-data plugin.

## Reproduction

Tested base revision: `359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`, with the
preserved pre-existing working changes and this benchmark patch. Recoverable
source snapshots are in each case's `source-tested/`; exact inputs and hashes
are in `launch.json`.

- ASPECT SHA-256: `00091b029bd2b0ecde7f9554b0f987ec35145842f66e1c57966884802a442372`
- New plugin SHA-256: `62f4012a235303c33c06d4de2226ea74968bce288b0acf5483fc36f8ba653540`

Modified implementation files: `bp3/bp3.cc` (selection/order/restart metadata),
`bp3/bp3_model.h` (selected constant initial state), `bp5/CMakeLists.txt`
(separate plugin), and new `bp5/steady_initialization.h` (native weak prestress).
New bounded launch/check scripts and the standalone test are under `bp5/`.
The BP5 README, current design and specification document this explicit
initial-data variant. The package collector also includes the new header so
its mirrored CMake targets remain buildable; existing packages are untouched.
No files under production `source/` or `include/` were changed by this task.

```
cmake -S benchmarks/reconstructed_fault/bp5 -B benchmarks/reconstructed_fault/bp5/build
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_steady_initialization -j4
c++ -std=c++17 -DASPECT_BP5_STEADY_INITIALIZATION \
  benchmarks/reconstructed_fault/bp5/test_steady_initialization.cc -o /tmp/test_bp5_steady_initialization
/tmp/test_bp5_steady_initialization
python3 benchmarks/reconstructed_fault/bp5/run_steady_startup.py prepare startup
python3 benchmarks/reconstructed_fault/bp5/run_steady_startup.py run startup
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_startup.py startup
# Repeat prepare/run/check for half and resume, in that order, then:
python3 benchmarks/reconstructed_fault/bp5/analyze_steady_startup.py compare
```

Existing output directories are not overwritten. No server package or
first-event run is launched by this task. The old adaptive server package
still has its old initialization and is not the new steady-state fixture.
