# K5: BP3 stress-perturbation initialization and bounded smoke

2026-09-12. This supersedes the rejected Airy production initialization, not
its preserved audit. The current plugin uses zero initial Maxwell perturbation
history and fixed surface background tractions. No solver tolerance, phase
profile, support width, deep constraint, or constitutive evolution equation
was changed. The coarse initialization mechanics checks and first real step
pass. The smoke hit its 900-s cap in the second real step, still far from
nonlinear convergence; the planned three-real-step trajectory is incomplete.
Neither result is a converged official BP3 solution.

## Discrete initial balance and ownership

The existing generic two-component manager property holds positive shear and
compressive normal background traction. PhaseFieldFault adds these only to
surface traction. Bulk assembly and Maxwell history contain no background:

\[
 R_\Gamma=\tau_{bg}+\Delta\tau:S-C_{eval}
 -\mu(V,\Theta)(50\,\mathrm{MPa}+\Delta p-\Delta\tau:N)-\eta^d V.
\]

The sign is positive for background shear. The exact timestep-zero cohesive
quantity is **evaluated**, not stored, traction:

\[
 C_{eval,0}=\kappa_\Gamma V_{init}/I_{h,0}+\beta_\Gamma C_0.
\]

The benchmark calls the existing noncommitting surface evaluation, taking its
cohesive/friction/damping weak terms at Vinit, supplied nodal Theta0 and
sigma_bg=50 MPa. It solves the existing consistent Q1 mass system for tau_bg.
The shear from the zero-rate probe is deliberately excluded: zero strain is
not zero constitutive stress in the presence of slip. S:N=0 ensures that this
probe's normal stress is sigma_bg. No probe vector is published.

This gives an exact **weak root at zero stress perturbation**, with maximum
relative mass-equation error 2.303e-16. It does not force a discrete bulk
velocity to realize zero stress. The background differs from
tau0_BP3 + projected(C_eval,0) by -551.75 to +4048.03 Pa, reflecting the
represented surface composition and nonlinear Q1 state. This correction is
exported, not hidden in Theta0. Both background components remain frozen
after initialization; later C is not added to them.

At fixed bulk unknowns, K_V uses sigma_total*dmu/dV. Backgrounds have no
bulk-increment derivative. G still differentiates Delta p and Delta tau only.
True-pressure mode is retained; there is no pressure-gauge shift or adiabatic
replacement. Top/bottom impose zero perturbation traction. Side velocities
are the official rotated rigid translations.

## Initialization result (one rank, Release)

Case: `benchmarks/reconstructed_fault/bp3/perturbation_initialization.prm`.
Same 36106-cell smoke mesh, 1155 fault elements, ell=400 m, dt0=4e6 s.
Distance-weighted particle interpolation is explicit, matching the immediately
preceding interpolation check. Vp=Vinit=1e-9 m/s.

| Check | Result |
|---|---:|
| Realized left velocity (m/s) | (+2.5e-10, +4.330127018922193e-10) |
| Realized right velocity | exactly the opposite |
| Constraint-vector error, 96 samples/side | exactly zero |
| beta0 | 0.9999999987184752 |
| kappa0 (Pa s) | 1.2815248119788472e17 |
| max / RMS bulk speed divided by Vp/2 | 1.004015 / 0.997789 |
| RMS bulk-minus-piecewise-rigid speed / (Vp/2), whole box | 0.041444 |
| Same RMS outside 3 ell | 0.00016921 |
| Same maximum outside 3 ell | 0.0051083 |
| max / RMS dt0*(u-u_rigid) (m) | 0.00210995 / 8.28879e-5 |
| Initial retained particle and transferred Maxwell history | zero |
| Bulk Delta p range (MPa, saved FE samples) | -0.127964 to +0.135923 |
| Free V/Vinit | 0.987149 to 1.016008 |
| Official nodal Theta0 relative error | zero |
| Actual constitutive sigma_total range (MPa) | 49.829053 to 50.073433 |
| Actual constitutive sigma_total weighted mean (MPa) | 49.999993 |
| Q1 Delta shear range (Pa) | -42262.76 to +9623.34 |
| Deep V absolute error | zero |
| Free / lower-active / deep prescribed nodes | 400 / 0 / 756 |
| Final bulk / surface relative residual | 4.75072e-15 / 2.09857e-11 |
| Final absolute bulk / surface residual | 6.80709e-6 / 2.29601e-5 Pa (surface) |
| Fresh linear residuals / targets | 0.943940/1.42748; 9.73085e-7/1.97429e-6; 3.83673e-11/4.92644e-11 |
| Runtime / peak RSS | 243.33 s / 3.950 GiB |

The discontinuous piecewise-rigid comparator necessarily differs from the
diffuse velocity across the fault. Its whole-box maximum is not a far-field
displacement error. Outside the diffuse zone, the small correction and
plate-scale velocity contrast with the rejected Airy initialization. Neither
the small perturbation pressure nor the 1.6% free-V departure is claimed zero.
VTU boundary errors (~7e-18 m/s) reflect Float32 output; physical constraints
were checked in the working double-precision vector.

### Coarse normalization limitation, not repaired

83 independently integrated saved-Q1 normal columns include the official
stations, regular RSF samples, and both tips. Physical-box boundary truncation
is retained; an absent half-space is not supplied artificially.

- Current half-width: 790.5832804 m (unchanged stationary-profile support).
- Maximum omitted h fraction: 1.49449e-4.
- Maximum supported column normalization error: 6.52605%, at an open tip.
  At RSF samples >=500 m from the top: maximum 3.51336e-4.
- Full-profile representation error at tips is also ~6.529%; this is not
  predominantly omitted tail. No background calibration renormalizes I_h.
- Saved-field replay at the production QGauss(3) bulk points gives total
  supported integral 1.1549249034e-4 m^2/s versus surface integral of V
  1.1547500184e-4 m^2/s: relative difference 1.51448e-4. QGauss(8), only an
  offline integration diagnostic, gives 2.07420e-4. History localization is
  zero at initialization, up to floating-point cancellation.

These are measured coarse-fixture limitations. They do not certify the 1e-4
criteria of earlier benchmark families; those criteria have not been silently
relaxed or transferred to this BP3 feasibility smoke. The initialization
mechanics checks support the user-authorized short dynamics test, not a
high-accuracy normalization or full BP3 validation claim.

## Focused verification and reproducibility

Build (no core source rebuild was needed: the existing background facility is
reused):

```
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release bp3_coupling_checks.release -j4
python3 benchmarks/reconstructed_fault/bp3/run.py benchmarks/reconstructed_fault/bp3/perturbation_initialization.prm --cap 900
python3 benchmarks/reconstructed_fault/bp3/analyze_perturbation.py benchmarks/reconstructed_fault/bp3/perturbation_initialization
python3 benchmarks/reconstructed_fault/bp3/run.py benchmarks/reconstructed_fault/bp3/perturbation_smoke.prm --cap 900
```

All output directories/logs are unique and retained. Run wrappers refuse to
overwrite evidence. `.resources.json` files record executable/plugin/source/
parameter hashes, wall time, peak memory, and exit status. Positive results
also require actual nonlinear convergence and every fresh linear check.

Background tests (run with the same runner, cap 120 s):

| Wrapper | Ranks | Result | Wall time |
|---|---:|---|---:|
| perturbation_dynamic.prm | 1 | surface K_V/G/B actions and background checks pass | 12.59 s |
| perturbation_rate.prm | 1 | includes nonsingular indefinite K_V; pass | 12.14 s |
| perturbation_dynamic_mpi.prm | 2 | ownership/reduction/actions pass | 6.63 s |
| perturbation_inclined.prm | 1 | adds 60-degree fixed-bulk derivative; pass | 12.30 s |
| perturbation_inclined_mpi.prm | 2 | same final regression; pass | 9.34 s |
| perturbation_restart_guard.prm | 1 | expected explicit fresh-start-only rejection before mesh/checkpoint | 1.67 s |

The initial `perturbation_dynamic_two` attempt failed before ASPECT in sandbox
MPI interface initialization (0.16 s); preserved separately. The approved
outside-sandbox MPI runs above passed. Tests distinguish residual offset sign,
total-normal coefficient in K_V, zero background G derivative, disabling the
optional facility, actual assembled finite differences, and S:N=0 at 60 degrees.
No full test suite, mesh campaign, or long-event simulation was run.

## Real-step outcome: incomplete, no retry

`perturbation_smoke.prm` repeated initialization with additional output/history
checks, then used the actual timestep controller. Its t=0 fields reproduce
the separate initialization exactly. Only steps 0 and 1 are accepted.

| Accepted-state check | Step 1 |
|---|---:|
| Time / timestep (s) | 2624650.718478746 / 2624650.718478746 |
| Free V/Vinit | 1.000144--1.018971 |
| Actual total sigma_n range (MPa) | 49.886348--50.048185 |
| Actual total sigma_n mean (MPa) | 49.999994 |
| Total shear at surface station (MPa) | 27.039976 |
| Delta shear at surface station (Pa) | -27817.65 |
| Surface-station total sigma_n (MPa, consistent Q1 field) | 49.956455 |
| Surface-station V / accumulated slip | 1.004466e-9 m/s / 0.00263637 m |
| Free / lower-active / deep prescribed nodes | 400 / 0 / 756 |
| Deep V error | zero |
| Split aging-law relative error (independent long-double evaluation) | 2.22045e-16 |
| Newly committed particle stress component maximum (Pa) | 237601.69 |
| Bulk / surface relative residual | 5.06921e-13 / 9.18401e-11 |
| Absolute bulk / surface residual | 4.69822e-6 / 9.77653e-5 Pa (surface) |
| Production-Q3 supported instantaneous integral (m^2/s) | 1.1566721574e-4 |
| Signed history integral (m^2/s) | -1.69e-21 (roundoff) |
| Integral of V along the surface (m^2/s) | 1.1564965755e-4 |
| Relative total supported normalization difference | 1.51822e-4 |

The phase field is bitwise unchanged in saved VTU output. Reconstructed
coordinates and both background fields are bitwise unchanged in the fault
exports. The first real mechanics consumed retained zero stress from t=0:
bulk stress-history VTU arrays remain zero at step 1, while the separately
recorded **new** particle stress is 0.2376 MPa. This is output/history ordering,
not an extra computational lag. Theta used by mechanics is exported separately
from the newly committed Theta. No history was refreshed merely for output.

Step 2 attempted t=5241669.360228987 s with dt=2617018.641750242 s.
The last recorded Newton iteration was 16:

- bulk relative residual 0.497643; absolute 7.09143e6;
- surface relative residual 0.193975; absolute 515924 Pa;
- bulk scale 1.4242940671e7 and surface scale 2.6597494402e6 Pa,
  unchanged during that solve;
- all 38 returned linear directions across the run passed their fresh checks
  (681 total Krylov iterations); the remaining issue is not a reported linear
  convergence failure;
- early accepted Newton steps reduce the residual, followed by very slow
  progress. This is not convergence and no failed-step fault output is accepted.

At 10 km, the first committed Theta changes from 8000 s to 2240887.76 s. At
fixed V and normal stress this increases frictional resistance by about
4.226 MPa. A **constant-friction/traction diagnostic**, ignoring subsequent
bulk and cohesive feedback, predicts V~2.148e-13 m/s; 153/400 unprescribed
nodes have this diagnostic rate below the unchanged 1e-12 floor. This explains
why lower-bound behavior becomes important, but is **not proof of the cause
of slow coupled convergence**. The failed iterate's active set and accepted
step lengths were not exported by the existing log settings.

The 900.057-s wall cap terminated the attempt with status 124 and peak
4.267 GiB. This was not exhaustion of the configured 30 nonlinear iterations.
`constitutive_normal_2_rank0.csv` belongs to the **unaccepted** linearization;
it must not be interpreted as a realized BP3 state. The process was killed
by the cap, so this attempt is not a test of exception-driven rollback.
No third step, repeat, support adjustment, Vmin change or tolerance change
was launched. The next useful task is a bounded step-2 active/free-set and
step-length audit, retaining these accepted initial/history data. It is not
a reason to restore Airy prestress.

## Timing and server envelope

Existing opt-in timers were enabled with `ASPECT_FAULT_PERFORMANCE=1` and
`ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC=1`; no new performance algorithm was
introduced. The completed initialization gives the following measured scopes:

| Scope | Calls | Seconds |
|---|---:|---:|
| I_h preparation | 2 | 163.77 |
| I_h FE work (nested in preparation) | 792 | 22.4 |
| I_h lookup access (nested) | 792 | 17.9 |
| Particle domains/CPDI | 8 | 20.3 |
| Fault cache build | 2 | 4.49 |
| Fault cache validation | 13 | 0.0281 |
| Bulk Stokes matrix / RHS assembly | 3 / 4 | 4.61 / 4.83 |
| Preconditioner rebuild | 3 | 3.23 |
| Condensed linear solves | 3 | 27.1 |
| B action (nested in solve) | 58 | 8.03 |
| G action (nested in solve) | 58 | 2.75 |
| Surface R/K assembly | 8 | 5.81 |
| History commit (t=0 no-op) | 1 | 0.0000163 |
| Particle advection / sort | 2 / 2 | 0.0897 / 0.0311 |
| Particle-to-FE interpolation | 1 | 0.377 |
| Postprocessing/output | 1 | 0.236 |

Timers overlap: do not add nested B/G/lookup scopes to enclosing solves/I_h.
ASPECT reports 242 s; the external runner reports 243.33 s including startup
and shutdown. I_h dominates completed initialization (~68%). The retained
Cartesian rejection/cache baseline is active: 396 cold batches, 802 rejected
requests, 25,895,578 stored point requests; the next preparation reuses all
396 batches. No discretization or integration tolerance was weakened.

In the capped dynamics attempt, the four completed I_h preparations cost
83.33, 67.37, 69.27, 67.23 s (287.20 s total). The remaining ~613 s includes
setup, CPDI, assembly and the increasingly numerous nonlinear/linear solves;
it cannot be assigned exactly from a killed process's missing final timer
summary. No complete real-step timing breakdown or speedup is claimed. No
additional run was launched solely to repair the performance report.

After the nonlinear continuation issue is reviewed, a reasonable **unmeasured
server starting envelope** for the prepared 48.828-m pilot is one 64-GiB node,
4 physical-core MPI ranks, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1.
Expect roughly four times the smoke's bulk cells/particles and twice its
fault elements; memory/time scaling is not measured. Do not launch that
pilot or extrapolate a first-event runtime while the coarse continuation is
unresolved. There was no targeted performance correction in this patch.

## Artifacts and known boundaries

- `bp3.cc`: current stress-change plugin, fixed-background weak initialization,
  exact side-constraint check, retained/split Theta audit, accepted weak export.
- `bp3_model.h`, `airy_dt0/`, `prestress_audit/`: preserved analytic Airy aids
  and rejected-production evidence; not loaded by the current smoke.
- `perturbation_initialization/perturbation_report.json`: detailed numerics;
  `perturbations_0.csv`, `stations_0.csv`, `initial_columns.csv`: separated data.
- `bp3_notes.md`: pressure, boundary, output-history and initialization semantics.
- `current_design.md` and `specification.tex`: explicit weak background rule.
- `tests/phase_field_fault_surface_system.cc`: background-enabled global and
  inclined local derivative coverage.

The 100-km finite box, incompressible Maxwell bulk versus official nu=.25
elasticity, finite width/additive cohesion, open tips, coarse interpolation,
and the minimum slip rate remain modified-BP3 limitations. No background
traction is inserted into the bulk tensor for visualization. Restart of the
benchmark-specific accumulated slip and background selector is not implemented;
an explicit early fresh-start-only guard prevents a silent background-free
restart. This guard was added after the smoke and is tested separately; the
expensive fresh-start trajectory was not repeated for this guard.

The final benchmark source SHA256 is
`0ac728e99c2f75bda4e81139a4a5a5976e8f7daaf13ed4c5f27dfc80ba617d02`.
`perturbation-source.tar.gz` preserves the source/parameter/report snapshot
and a tracked-worktree patch against HEAD `fb4411915`; the patch includes
pre-existing modifications and is not a claim that all of them were made in
this task. The current executable and each run's actual plugin hash are in
the corresponding resources JSON. No commit or broad repository reset was made.
