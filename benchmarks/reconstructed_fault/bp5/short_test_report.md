# BP5-friction / Dc=0.1 / ell=100 m: bounded short-test report

## Decision

**Stop at the newly exposed constitutive lower-bound invariant failure.**
The requested four frozen mechanical responses pass, as do candidate nonlinear
initialization and its first real step. The second real step fails during a
line-search trial, before acceptance, on the guard requiring finite
`V >= V_min`. No retry, solver/physics adjustment, finer coupled trajectory,
restart branch or half-step branch was run afterward.

This is a **2-D modified research case**, not official 3-D BP5 and not a
production-qualified earthquake-cycle configuration. The 1% profile-width
qualification failure remains recorded. Only this task's approved approximately
2% diagnostic exception was used.

## Configuration and implementation

Base source revision: `359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`, with the
pre-existing uncommitted changes preserved in
`dc010-ell100/source-before/working.patch` and `status.txt`.

The separately named entry is [bp5_friction_dc010_ell100.prm](bp5_friction_dc010_ell100.prm).
Exact generated inputs are under `dc010-ell100/fixtures/{candidate,reference}/`;
their manifests and each run's `launch.json` record input and binary SHA-256
hashes. Both prestress files are byte-identical to the maintained physical
background correction (`a5de2127a03cb99a820d3eb2b1d15ece0c2a6164006fe4bf6d43ab5af9cdd96e`).
In particular, its stored denominator was not replaced by the new I_h.

| Quantity | Value |
|---|---:|
| Shallow/deep a | 0.004 / 0.04 |
| b; Dc | 0.03; 0.1 m |
| ell; prescribed core phase | 100 m; 0.6 |
| Box; dip | 300 × 100 km; 60° |
| G; background normal stress | 32.03812032 GPa; 50 MPa |
| Loading / intended initial V | 1e-9 m/s |
| Minimum V | 1e-20 m/s, unchanged |
| Artificial initialization interval | 4e6 s; not physical elapsed time |
| Real-step ceiling | 4e6 s; ordinary controllers remain active |
| Fault | 1156 vertices, 1155 elements; no prescribed nodes |
| Fault spacing | 99.9603362–100.0000000 m; exactly 100 m in the probe region |

The horizontal 15–18 km strengthening extension and initial-composition
particle property are unchanged. With `a=.004+.036*S`, the nominal `a=b`
crossing is 17.166667 km. Mature C=0, split exact aging, true normal feedback,
the work measure, both endpoint treatments, fixed phase, viscosity, damping,
AMG and all nonlinear/linear tolerances are retained.

Minimal parameter plumbing:

- `FaultFriction::initial_state_for_friction_coefficient()` inverts the live
  configured law. It mixes parameters before inversion and evaluates the
  regularized inverse in logarithmic form using `expm1`, avoiding overflowing
  `sinh`. The constitutive law and production state update are unchanged.
- `PhaseFieldFault::get_fault_friction()` exposes that configured law read-only
  to benchmark initialization. Particle, nodal and timestep-zero audit states
  now use it; old hard-coded BP3 analytical helpers remain historical tests.
- Benchmark/test-only selectors add the two requested modes, the wider probe
  patch, raw current-history diagnostics, and the bounded checkpoint schedule.
  No BP5 parameters became global defaults.

The live configured-law checks give:

| Down dip | a | Initial Theta (s) |
|---|---:|---:|
| Shallow / 15 km | .004 | 25,118.8643152 |
| 16.5 km | .022 | 1,584,893.19247 |
| 17.166667 km | .03 | 10,000,000.00004 |
| Deep / 18 km and below | .04 | 100,000,000.0000002 |

Nominal shear prestress is **26,546,122.3651393 Pa**. The inverse-law and live
derivative checks pass at the sampled material mixtures; nodal and particle
initial-state errors are zero at exported precision. The production QP
response is also checked against the new a/b/Dc and the same projected
surface mixture and interpolated state. Independent configured aging checks
pass; the old-parameter regression errors are at most 2.22e-16.
These checks do not equate nominal Vinit with the realized coupled initial V.

## A. Frozen spatial-response comparison — PASS

Both modes use the same cosine-squared taper centered at 16.5 km with
half-width 6.25 km and amplitude 1e-12 m/s. Thus the 3125-m input spans four
wavelengths, rather than being squeezed into the old 3-km window.

The reference halves bulk spacing in 7–26 km and the first/last 5 km,
including the existing normal halo and mesh grading. Artificial tangential
patch edges are 5, 7, 26 and 110.470 km. No special 40-km refinement is added.

| Mesh | Cells | DoFs | Particles | Finest h |
|---|---:|---:|---:|---:|
| Candidate | 114,984 | 4,130,126 | 1,034,856 | 24.4140625 m |
| Local reference | 191,958 | 6,805,716 | 1,727,622 | 12.20703125 m |

Elastic stiffness below is the native work-conjugate rate stiffness multiplied
by G/kappa. Both cases use the identical initialization kappa. Continuum
predictions use the **actual tapered Q1 input spectrum**, not a pure sinusoid.

| Nominal wavelength | Candidate (MPa/m) | Fine (MPa/m) | Coarse/fine − 1 | Actual-Q1 continuum (MPa/m) |
|---|---:|---:|---:|---:|
| 3125 m | 52.858675 | 52.970971 | −0.2120% | 53.020019 |
| 200 m | 101.046236 | 103.916780 | −2.7623% | 104.967396 |

Both satisfy the <5% refinement target. Fine/continuum discrepancies are
−0.0925% and −1.0009%; agreement between meshes is not claimed to establish
sharp-fault accuracy. The signed direct-minus-bulk-relaxation decomposition
reproduces each measured stiffness. Work-pair errors are at most 7.62e-15,
action errors at most 7.06e-15, and fresh relative linear residuals at most
8.71e-11 against the unchanged 1e-10 target. All full/frozen-normal finite
difference derivative checks and noncommit checks pass.

The homogeneous, steady-state consistent-Q1 screen over **200–1500 m** has
minimum stiffness 91.009 MPa/m; its alternating endpoint is 104.942 MPa/m.
Compare Kc=13 and 15.6 MPa/m at 50 and 60 MPa. This is not a proof of transient
stability and does not require intended long-wave nucleation modes to be stable.
Pure-mode band/sharp ratios at 3750/3125 m are 0.852296/0.826090.

### Profile and normalization

Continuum profile: full support 395.291640 m, RMS width 44.223682 m,
I_h=13053.839449 m. Production profile-table agreement is checked against the
independent stationary calculation. Realized quantities below sample columns
every 125 m over 13–20 km; they are not extrema certified over the entire fault.

| Quantity | Candidate | Local fine |
|---|---:|---:|
| Realized peak phi range | .587960–.599554 | .596955–.599701 |
| Realized positive-Q1 support range (m) | 422.864–451.055 | 408.769–422.864 |
| Realized chi RMS width (m) | 45.04469–45.07493 | 44.43541–44.43625 |
| Maximum RMS-width error | 1.92486% | .480654% |
| Maximum column J / projected I_h − 1, absolute | 5.65244e-4 | 3.48713e-6 |
| Maximum completed endpoint-column relative error | 1.33444e-7 | 2.05903e-8 |

The existing length-study column-normalization diagnostic (1%) and endpoint
completion checks pass without changing integration tolerances. The candidate
does **not** pass the original 1% RMS-width target. Its wider positive-Q1
support is a representation effect, not a widened production association policy.

Artifacts: `probe_comparison.json`, `coefficients.csv`,
`profile_columns.csv`, `*_spectrum.csv`, `short_wave_screen.csv`,
[probe comparison plot](dc010-ell100/probe_comparison.png), and the two
`probe-*/` directories. Their nonzero process exit is the existing deliberate
noncommitting stop **after** `MECHANICAL MODES VERIFIED`, not nonlinear
convergence or an accepted trajectory.

## B. Coupled candidate — initialization and step 1 PASS; step 2 DISCREPANCY

| Accepted state | Physical time | Max Vdt/Dc | Final normalized bulk / surface residual |
|---|---:|---:|---:|
| 0 | 0 | 0 | 9.5131e-15 / 1.1392e-11 |
| 1 | 4e6 s | .0421159 | 6.4737e-10 / 5.0095e-9 |

Initialization takes three Newton updates; step 1 needs zero updates from its
initial iterate. Every returned linear direction passes its fresh residual
check. Both accepted states have 1156 free, zero lower-active and zero prescribed
nodes. V ranges from **0.760663 to 1.052898 Vp**; the minimum is at 14.9 km
and maximum at 18.1 km. This already-present structure is visible in the plots,
not attributed to a new evolved instability or declared spatially converged.

Timestep zero retains supplied Theta and zero particle Maxwell history.
The first real exact aging update passes at 2.22e-16 relative error. At 10 km,
Theta changes from 25,118.864 to 3,945,383.392 s, a factor of **157.069**.
The first Maxwell publication differs from its independent incoming-zero-history
evaluation by at most **1.843e-7 Pa** on an 89,711.77-Pa stress scale. Inert H,
fault geometry and completed I_h remain unchanged.

The current stress diagnostics use the accepted velocity/pressure and the
**incoming working FE history**, not the newly committed particle stress.
Current stress can therefore be nonzero at t=0 while retained particle stress
is zero. Native weak friction uses incoming Theta; separate `state_work_*.csv`
files report incoming and outgoing state explicitly.

Both boundary continuations are exercised: **204 source QPs at each end**.
Endpoint test-mass closure is 3.109e-15 and endpoint weak-traction reproduction
error is 1.526e-7 Pa. The positive-FE tail omitted outside the unchanged nominal
association radius is 2.128e-5 of the localization measure in the exported
windows (not a whole-fault maximum claim). No missing admitted wedge is found.

Recombining the separately summed large weak forces loses at most 1.66e-7 Pa
after dividing by the actual row measure. The analysis initially compared that
quantity in integrated-force units; it was corrected to the existing observer's
Pa units. No production tolerance or stopping test was changed.

Artifacts: `candidate-six/accepted_steps.csv`, `state_work_{0,1}.csv`,
`work_weak_{0,1}.csv`, `work_qp_*_rank*.csv`, `checks.json`,
`first_update_maxwell.csv`, initial heavy output, and
[whole fault](dc010-ell100/accepted_whole.png),
[transition](dc010-ell100/accepted_transition.png),
[top](dc010-ell100/accepted_top.png),
[bottom](dc010-ell100/accepted_bottom.png) plots.
There is no accepted step 2 and no coherent step-2 checkpoint.

### Stopping failure and bounded read-only audit

At physical target time 8e6 s, step 2 reached Newton iteration 6. The last
reported normalized residuals were **0.818607 / 0.481069**, not converged.
All its returned linear directions passed fresh checks. The next line-search
trial raised the unchanged constitutive guard at
`source/material_model/phase_field_fault.cc:577`, called from
`ReconstructedFaultSurfaceSystem::assemble_bulk_work_system()`.

The nodal trial builder validates finite absolute values at or above Vmin.
The work-path QP evaluation subsequently uses
`shape_0*V_left + shape_1*V_right` (surface_system.cc:834).
An offline counterexample using saved production coordinates shows that two
exactly admissible endpoint values, both 1e-20, can yield
**9.999999999999998e-21**: one ULP, or 1.504633e-36 m/s, below the bound.
Both ordinary and possible fused multiply-add evaluation orders have examples
that violate the bound. The saved rank-3 sample set contains 16,440 such
ordinary-sum cases out of 156,873 tested interior-coordinate samples.

This demonstrates an interpolation vulnerability, **not the identity of the
failed trial's actual QP**: the failing nodal pair/value was not exported.
See `audit_bound.py`, `bound_arithmetic.json` and the untouched failed `run.log`.
As a separate physical scale check, holding initial traction fixed while applying
the observed shallow aging increase would predict V≈3.38e-26 m/s from the
inverse law. Thus contact with the retained 1e-20 bound is unsurprising; valid
contact should not itself cause an exception. That scalar estimate is not a
replacement for the coupled mechanical result.

**Recommended next action:** capture/qualify bound-preserving Q1 slip-rate
interpolation at the work-measure QPs, with equal-bound and near-bound
regressions, before resuming this short test. Do not lower Vmin, change friction,
or reduce the timestep merely to avoid exercising the invariant. No production
interpolation correction or further solve has been made in this task.

## Requested check ledger

| Check | Result |
|---|---|
| New parameter plumbing, inverse initialization, production-QP friction | PASS |
| Both regenerated meshes, profiles and endpoint completion files | PASS |
| Four frozen responses, work/Jacobian checks, <5% refinement change | PASS |
| Candidate profile width | PASS only under approved ~2% diagnostic exception; original 1% fails |
| Nonlinear candidate initialization and first history update | PASS |
| Six-step candidate | DISCREPANCY: step-2 constitutive bound invariant |
| Fine coupled initialization/first two physical intervals | UNRUN after stopping failure |
| Coupled spatial-profile comparison and localized refinement-error assessment | UNRUN |
| Step-2 checkpoint / restart intervals 3–4 | UNRUN; prerequisite checkpoint does not exist |
| Half-step comparison of intervals 3–4 | UNRUN |
| Full-cycle / ell=50 evolution / production promotion | NOT AUTHORIZED; not run |

## Cost, reproducibility and implementation limits

| Simulation | Wall seconds | Largest child peak RSS |
|---|---:|---:|
| Candidate two probes | 105.50 | 2.44 GiB |
| Reference two probes | 190.53 | 3.62 GiB |
| Candidate initialization + step 1 + failed step 2 | 583.78 | 2.62 GiB |
| **Aggregate** | **879.81 s = 14.66 min** | — |

RSS is the largest child/rank high-water mark, **not summed job memory**.
The task stopped on failure well before the 3600-second simulation budget.
Mesh/completion generation took 43.10 s separately. Build log intervals were
about 127 s for the core, 108 s spanning the two plugin build passes, and 25 s
for the probe/mesh targets; these are setup intervals, not simulation charges.
All builds used `-j4`. The focused `FaultFriction declares*` test passed all
5 assertions; the full test suite and historical trajectory campaign were not run.

Exact executed commands and environment are in each `launch.json`; the entry
sequence is in [README](README.md). `analyze_probes.py`, `analyze_coupled.py`,
`plot_accepted.py` and `audit_bound.py` reproduce the offline reductions.
`accepted_prefix.log` is only a generated analysis view of genuinely accepted
states; it neither overwrites the failed log nor qualifies the failed run.

Files touched by this task, beyond generated evidence:

- `include/aspect/material_model/{phase_field_fault.h,rheology/fault_friction.h}`
  and `source/material_model/rheology/fault_friction.cc`: configured-law access
  and stable inverse initialization; no mechanical update change.
- `benchmarks/reconstructed_fault/bp3/{bp3_model.h,bp3.cc,work_replay.h}`:
  configured initial states and opt-in BP5 diagnostics/checkpoint selection.
- `tests/{reconstructed_fault_mechanical_modes.cc,bp3_length_scale_checks.h,bp3_length_scale_mesh.cc}`:
  bounded modes, parameter checks and widened local-reference patch.
- The new BP5 PRM, launch/analysis scripts, README and this report.

**Unqualified working-tree paths:** `prepare_comparison.py` and the opt-in
post-step-2 checkpoint disabling branch are prepared but unexercised in this
case. No BP5 restart/half-step equivalence is claimed. The new generic inverse
has been exercised for the regularized BP5-friction material mixtures used here.
Old-parameter friction/aging checks pass, but an old-parameter initialization
replay and other selectable law modes were not covered by this run.
Unrelated working changes were preserved; no commit was requested or made.
