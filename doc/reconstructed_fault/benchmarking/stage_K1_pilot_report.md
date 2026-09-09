# K1 approved initialization correction and bounded pilot

Working-tree continuation of `f4032b1824ef4892020af8c58ef981aca03dc0ec`.
The approved production correction is implemented and its boundary/lifecycle/
rollback checks pass. The one-rank pilot is **incomplete and fails its measured
assumption gates**. No spatial/temporal matrix or K2 run has begun. No commit
has been made; unrelated existing changes and historical documents are retained.

This report supersedes the execution status, not the historical evidence, in
`stage_K_progress.md` and `stage_K1_prerequisite_report.md`.

## Production correction and invariant

`PhaseFieldFault::evaluate_reconstructed_fault_localization()` now uses the
converged current phase field as the previous mechanical profile at timestep
zero. Both the surface point response and bulk QP response already call this
shared implementation. This pairs phi_0 with the initialized previous-I_h
snapshot, instead of combining that snapshot with the zero old FE placeholder.

The correction changes **evaluation**, not history:

- The initial coupled kinematic solve is still solved and committed.
- Theta_0 and H_0 are not evolved through the numerical 2 s Maxwell interval.
- Initial cohesive and particle Maxwell histories retain their established
  initialization values; evaluated responses need not equal retained histories.
- I_h,0 remains the previous-normalization snapshot for the first real step.
- No write is made to old_solution. At positive timesteps, including restart,
  the real previous FE phase field remains the input to localization.

This removes the erroneous integrated initial history slip documented in the
prerequisite report. It does not alter an equation, tolerance, iteration budget,
or parameter meaning to admit a root. The user explicitly approved this initial
profile rule, now also recorded in both authoritative specifications.

Files for this correction:

| File | Change |
| --- | --- |
| `source/material_model/phase_field_fault.cc` | One shared timestep-zero previous-profile selection; positive-step path unchanged |
| `doc/reconstructed_fault/current_design.md` | Explicit initial evaluation versus retained-history contract |
| `doc/reconstructed_fault/specification.tex` | Same normative contract |
| `tests/phase_field_fault_prescribed_velocity.cc` | Both production constitutive entry points ignore the old placeholder only at zero; later-step responses must remain sensitive to actual previous phi |
| `benchmarks/reconstructed_fault/uniform_shear/reference.py` | Approved rule is default; former zero-profile case remains an explicitly failing historical reproducer |

The earlier physical-base-iterate correction in `source/simulator/solver.cc`,
its one-/two-rank prescribed-velocity fixtures, nonzero-boundary rollback
fixture, and three phase-iteration-budget-only fixture edits remain separate
working-tree changes. Their reproducer and implementation evidence are in the
prerequisite report. No public interface was added by either production fix.

## Fixture, parameter meaning and reproducibility

The parameter/absolute-error tables in the prerequisite report are unchanged.
In particular, creep cutoffs apply to eta, not to the reduced kappa. For dt
2, 1 and 0.5 s, kappa is respectively 1,980,132.6693, 995,016.6251 and
498,752.0807 Pa s. The latter two are valid despite lying below the minimum
creep viscosity 1e6 Pa s. Only the 2 s level was piloted.

The pilot uses 16 x 64 square cells, 9,216 particles, ell=0.15625 m, AT1 with
normal production activation 0.1, and an open reconstructed fault across the
0.25 x 1 m domain. The bulk is x-periodic; top/bottom prescribe ux=+/-U/2,
uy=0. U(t)=1e-4*[1+0.2*min(t/4,1)] m/s. Adiabatic pressure is 1000 Pa;
dynamic-pressure normalization is only a gauge. Initial tau_xy=1500 Pa and
Theta=200 s are supplied through explicit particle-property mappings.

After normal initial convergence/reconstruction, the benchmark-only plugin
saves the Q1 phase field. At positive timesteps it constrains only independent
phi DoFs to those values, retaining existing dependent periodic/hanging-node
relations. It adds no mechanical constraints. Exact phi preservation is
checked at postprocessing; H is checked by particle ID. `Evolve phase field`
remains false solely for its existing frozen-H meaning. It is not redefined
to freeze phi, and no production solve is bypassed by changing that parameter.

New benchmark files, all under `benchmarks/reconstructed_fault/uniform_shear/`:

| Files | Responsibility |
| --- | --- |
| `pilot.prm`, `fault.txt` | One physical/spatial/timestep fixture and prescribed initial fault |
| `CMakeLists.txt`, `uniform_shear.cc` | Existing ASPECT plugin build; phase-only constraints and accepted-state diagnostic exports |
| `reference.py`, `test_reference.py` | Independent stationary-profile quadrature, scalar interior root, distinct evaluated/retained initial state, self-advanced real-step histories |
| `analyze.py`, `test_analysis.py` | Independent actual-Q1 profile integrals/projections, geometry/volume/normalization checks, exact Q2 boundary traces, scalar and per-node ageing comparisons |
| `run_pilot.py` | One-rank execution with 600 s cap and Linux child peak-RSS accounting |
| `README.md` | Build/run commands, CSV interpretation and failed/incomplete status |

Builds used `cmake --build build-pf-cpdi --target aspect -j4` and the plugin
CMake commands in that README, also with -j4. Both succeeded. Executable:
`/home/ein/repository/aspect/build-pf-cpdi/aspect` (Debug, Voro++ enabled).
SHA256: `1144c79341f60facac2365791ebdf21f10a2c331cb742aa43beb1f61dfa9311a`.

## Pilot execution and convergence

Artifact directory: `/tmp/aspect-k1-pilot/`, including `pilot.log`,
`output-pilot/` accepted CSVs and `analysis.json`. Run command:

```sh
cd /tmp/aspect-k1-pilot
python3 /home/ein/repository/aspect/benchmarks/reconstructed_fault/uniform_shear/run_pilot.py \
  /home/ein/repository/aspect/build-pf-cpdi/aspect \
  /home/ein/repository/aspect/benchmarks/reconstructed_fault/uniform_shear/pilot.prm
```

Measured elapsed time **600.026 s**, child peak RSS **481,736 KiB** (470.45 MiB),
exit status **124** (wall-clock cap). Initialization and t=2 s were accepted;
t=4 s had reached particle advection but not accepted mechanics when stopped.
There is no accepted t=4 or t=6 state and no completed trajectory claim.
The cap was not increased. This run overlapped MPI2 restart verification, so
the elapsed time is a contended observation, not an isolated runtime estimate.

The initial phase solve converged in 13 nonlinear iterations to relative
residual 5.052e-11 against 1e-8; every linear solve took one CG iteration.
The initial coupled solve reached bulk/fault relative residuals
4.367e-15 / 1.808e-9 after nine Newton updates. At t=2 s (actual dt=2 s,
U=1.1e-4 m/s), the frozen phase residual was exactly zero; coupled residuals
were 8.190e-13 / 2.021e-11 after six updates. Its successive accepted updates
rejected 3, 3, 2, 0, 0, 0 line-search candidates. No requested accuracy was
relaxed. Dimensional initial residual scales are not exported by this plugin;
they remain a verification limitation, not inferred from relative logs.

## Assumptions measured before interpreting error

The independent normal integrals split at every crossed Q1 bulk-grid line.
The projected comparison uses independent integrals at the actual three
Gauss points per surface segment and independently assembles the consistent
Q1 mass projection. It is distinct from simply comparing segment midpoints.

| Measurement | Observed | Gate / interpretation |
| --- | ---: | --- |
| Fault length | 0.250001856881 m | Actual, not nominal 0.25 m |
| Maximum abs(y)/ell | 0.00398103 | Fails 1e-6 |
| Maximum segment angle | 0.00552080 rad | Fails 1e-6 rad |
| Endpoints | (0,-0.000622036), (0.25,0.000262280) m | Not the intended centered line |
| Association half-width, both sides | 0.308821593907 m | Measured through production association queries |
| Q1 phi minimum / maximum | 6.13156e-10 / 0.402162 | Physical lower bound respected; prescribed initializer core was 0.6 |
| Maximum phi range along x at fixed y | 0.0596761 | Initial FE profile is not transversely uniform |
| Independent initial H maximum relative error | 2.72013e-7 | Initializer agrees with stationary first-integral reference |
| Initial H maximum along-x range | 1.26456e-8 Pa | Roundoff-scale compared with peak H about 2.95e4 Pa |
| Independent actual FE I_h, length-weighted mean | 39.0625527736 m | Segment-center range 38.4095–39.8366 m; ideal profile gives 108.652564417 m |
| Independent quadrature tightening change | 0 at printed double precision | Meets self-change target; not a proof of absence of bias |
| Projected I_h maximum relative difference | 2.36385e-6 | Fails 1e-6 target; source not yet diagnosed |
| Midpoint I_h relative difference | 0.00557827 | Includes surface projection/interpolation error, not pure kernel error |
| Maximum omitted association-strip fraction | 6.37367e-5 | Fails 1e-6; no renormalization or strip expansion applied |
| Initial C independent particle-projection maximum error | 4.54747e-13 Pa | Roundoff agreement |
| Endpoint diagonal projection mass | 0.00635764 / 0.00636457 m² | Both endpoints have positive particle support |
| Total particle volume, t=0 / t=2 | 0.25000000000000006 / 0.25000000000008793 m² | Domain area preserved to roundoff |
| Individual particle volume / initial uniform volume, t=2 | 0.975777–1.024223 | Nonuniform domains after motion; total volume alone is insufficient |
| Integrated bulk chi V / integrated surface V, t=0 / t=2 | 0.999803580 / 0.999805192 | Fails 1e-4 normalization target |
| V along-fault range / 1e-4 m/s, t=0 / t=2 | 1.83047 / 1.87820 | Grossly fails 1e-4 uniformity target |
| Transverse velocity RMS, t=0 / t=2 | 2.79329e-6 / 2.86341e-6 m/s | Fails 1e-8 m/s |
| Divergence RMS, t=0 / t=2 | 6.89345e-7 / 7.13524e-7 s^-1 | Fails 1e-8 s^-1; weak incompressibility does not imply pointwise zero |
| Prescribed velocity maximum error, t=0 / t=2 | 2.71051e-20 / 4.06576e-20 m/s | Pass; exact Q2 trace recovery at tangential Gauss points |

Initial particle volumes differ from the uniform value only at about 2e-14
relatively. The initial H profile is x-uniform and independently verified,
but the converged FE phase field is already nonuniform before initial
mechanics. Thus the new constitutive initialization selection cannot explain
that initial phase discrepancy. The evidence localizes the next investigation
to the phase-field discretization/assembly/constraints and associated profile
resolution; it does **not** identify a specific CPDI, periodic-domain or
preconditioner defect. No production change or fixture retuning was attempted
to suppress it. Periodic particle-domain correctness is not inferred from
volume totals or lack of observed wrapping. Independent outside-domain tail
control for this nonuniform FE profile is also not established.

## Conditional scalar comparison, not an accuracy pass

For the measured profile, initialize the scalar history exactly once with
tau_xy=1500 Pa, mean retained C0=692.137793349441 Pa and Theta0=200 s. The
independent measured-profile integral is fixed at 39.06255277358289 m. Solve
the strictly monotone scalar residual with an interior bracket V>1e-12 m/s,
traction residual <=1e-7 Pa, using the accepted dt/loading. Subsequent histories
come only from the reference recurrence, never from later ASPECT histories.
At zero, evaluated responses are returned separately while retained histories
remain unchanged. Both accepted states have admissible interior reference roots.

| Quantity | t=0 observed / reference | t=2 s observed / reference |
| --- | --- | --- |
| Mean V (m/s) | 1.47954967479e-4 / 1.47191457491e-4 | 1.56546598119e-4 / 1.55816175987e-4 |
| Retained mean C (Pa) | 692.137793349 / 692.137793349 | 686.336866832 / 686.331075648 |
| Retained mean Theta (s) | 200 / 200 | 148.938959358 / 148.168504442 |
| Evaluated mean tau_xy (Pa) | 1375.404086118 / 1376.852663268 | 1378.196043274 / 1379.575903104 |
| Accumulated real-step slip (m) | 0 / 0 | 3.13093196238e-4 / 3.11632351975e-4 |

Observed evaluated stress is reconstructed from actual FE gradients, old FE
stress, production chi/history data and the actual segment shear tensor, not
an assumed horizontal tensor. Independently recovered QP associations are
checked against production activity and interpolated V before this calculation.
The retained particle stress at zero remains 1500 Pa, not the evaluated value.

Mean errors are around 0.5% for V and Theta, but averaging a nonlinear,
nonuniform fault into a scalar does not produce a valid homogeneous reference.
No 2% trajectory-pass claim is made. A complete independent velocity-profile
error and the t=4/t=6 trajectory are not available.

Separately, advance each vertex's initial Theta using its observed accepted V
and the independent exact ageing formula, without resetting subsequent Theta.
At t=2 the maximum observed update difference is zero at double precision;
the scalar reference increment is -51.8315 s, far above the 2e-8 s isolated
allowance. The timestep-zero retained Theta and H checks also pass. This
validates the observed ageing update even though homogeneous-shear assumptions
fail; it does not validate the scalar reduction for this nonuniform solution.

## Focused verification and limits

After the initialization correction:

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_(prescribed_velocity(_later)?|stage_i_rollback|stage_j)$' -j1
```

**4/4 passed, 549.69 s total**:

| Test | Ranks | Result / time |
| --- | ---: | --- |
| prescribed_velocity | 1 | Pass, 94.48 s |
| prescribed_velocity_later | 2 | Pass, 72.62 s |
| stage_i_rollback | 1 | Pass, 52.07 s |
| stage_j | 2 | Pass, 330.51 s |

The Stage-J test retains its evolving-phase feedback and failure-preservation
assertions; no tolerances or expected outputs were changed for this correction.
The earlier actual checked-in surface/condensed fixture batch passed 7/7
after its budget-only refresh, **before** this initial-profile correction;
that is not represented as a post-correction rerun here.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s benchmarks/reconstructed_fault/uniform_shear -p 'test_*.py' -v
```

**4/4 passed**: initial response/history separation, legacy zero-profile root
rejection, independent real-step history advancement, and exact Q2 boundary
trace recovery including an injected boundary error. `git diff --check` passed.
The independent analyzer exits **2**, deliberately, for incomplete time
coverage and failed assumption gates; this is not a failed Python execution.

The first MPI2 restart pair overlapped the pilot. Creation hit its unchanged
600 s timeout; resume then compared against an incomplete/stale final reference
and failed. These outputs are preserved in `restart-create-timeout/`,
`restart-resume-after-timeout/` and `restart-concurrent-ctest.log` under the
pilot artifact directory. Rerunning the pair alone, with unchanged timeout,
assertions and input, **passed 2/2**:

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_stage_j_restart_(create|resume)$' -j1
```

Creation passed in **336.65 s**, resume in **117.58 s**, total **454.25 s**.
Both use two MPI ranks. Resume verifies restored histories, V, geometry and
bulk state before mechanics, then compares continued feedback against the
fresh uninterrupted run. The failed concurrent comparison is therefore not
evidence of a restart regression; its prerequisite reference was incomplete.

No complete ASPECT suite, additional timestep level, spatial refinement,
parameter retuning or later benchmark family has been run. Stop at K1 review:
the production correction is verified by its focused tests, but the incomplete
pilot is not a validated homogeneous benchmark.
