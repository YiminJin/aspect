# K5: implemented fixture, nonpassing initialization gate

2026-09-12. K4 remains closed as a bounded partially resolved study. This is
the direct implementation task in `stage_5_redesign.md`, not a return to the
old K5 review sequence. No real BP3 timestep has been run.

The plugin and generic capabilities build and the corrected-mesh initialization
converges. It nevertheless fails the official initial-state target: the top
node slips at **22.55 Vinit**. This is not a successful BP3 initialization or
a completed performance pilot. The parameters remain drafts for dynamics.

## Implemented boundary and initialization choices

Physical authority is supplied `SEAS_BP3.pdf` (2021-10-01), especially Table 1
and equations 18, 25, 26; `bssa-2022066.1.pdf` supplies numerical/domain context.
`bp3_notes.md` records the modified-BP3 differences. The old implementation is
not the architecture or parameter authority.

- Proper rotated chart: x=78867.5134594813-x_BP3, y=100000-z_BP3. This displays
  a left-dipping 60-degree thrust while preserving the manager's positive V
  and CCW normal convention. Reversing polyline vertices alone would not
  reverse the slip tensor. The fault extends through the complete box.
- Official side translations, zero top total traction, equilibrated initial
  bottom total traction; physical pressure with no normalization or pore-
  pressure shift. Bottom traction is an explicit finite-domain realization,
  not an official benchmark prescription.
- Official G=32038120320 Pa, damping=4624440 Pa s/m, Vinit=Vp=1e-9 m/s,
  Dc=.008 m, b=.015, f0=.6, Vref=1e-6 m/s; spatial a=.010 to 15 km,
  linear to .025 at 18 km, .025 to 40 km. Deep V=Vp replaces friction equations.
- Official tau0=26546122.365133367 Pa, sigma0=50 MPa and Theta0 retained.
  Additional evaluated cohesive traction augments the Airy shear target, not
  Theta0. The numerical Maxwell initialization interval is 1 s. It does not
  physically advance retained Theta/H/particle stress.
- Frozen Q1 phase is assigned directly from the stationary infinite-line
  distance profile, including top/bottom. ell=400 m, AT1, core=.6, curvature=1,
  cohesion=1 MPa and Gc=1e5 J/m2 define the additional finite-width model.
  H uses the current stationary law and configured .1 activation. No Neumann
  phase solve, stress smoothing, Airy radius regularization, or pressure shift
  was used to obtain initialization convergence.

## Attempts and measured initialization

Artifacts reside in `benchmarks/reconstructed_fault/bp3/`.

1. `initialization01`: missing required particles postprocessor; failed before
   initialization (3.67 s). Corrected the plugin selection only.
2. `initialization02/03`: insufficient support at fault vertex 28 (17.41/17.11 s).
   `initialization03/initial_mesh_0.csv` proved the crossed cell remained 6250 m.
   The refinement function omitted Cartesian coordinates, thus used ASPECT's
   default depth coordinate. Added explicit Cartesian coordinates and an
   outer graded envelope to avoid missing the narrow strip by center sampling.
   No support, particle-density or numerical-tolerance change.
3. `initialization04`: 36106 cells, 1282726 total DoFs, 324954 particles,
   1155 fault elements. All 1615 fault-crossed cells have h=97.65625 m.
   273.1777 s wall, peak 4084000 KiB (3.895 GiB). End step zero only.

| Check | Measured result |
|---|---:|
| Final bulk / surface relative residual | 1.0691e-15 / 1.7591e-12 |
| Final surface RMS | 3.0057e-5 Pa |
| Returned linear directions | all 6 fresh checks passed, 17–19 iterations each |
| Final fresh linear / requested residual | 3.0082e-13 / 8.0683e-13 |
| Deep V absolute error | exactly 0 |
| Retained nodal Theta relative error | exactly 0 against supplied nodal target |
| Fault normal-coordinate departure | 2.91e-11 m |
| Free V / official Vinit | 0.97418–22.55105 |
| Actual constitutive sigma_n sample range | 48.17535–52.62727 MPa |
| Maximum represented q minus Airy fault target, free region | +1.12318 MPa |
| Maximum visualization bulk speed | .0568484 m/s at (71875,0) m |

The surface-intersection Q1-test-weighted constitutive means are p=32.54752,
sigma_n=49.05576 and tau:N=-16.50824 MPa. These are actual surface-equation
moments, **not bulk-column averages** and not pointwise traces. The represented
top shear is 27.94880 MPa versus its Airy target 27.04742 MPa. Theta=8000 s;
V=2.25510e-8 m/s. Between .5–2.5 km, V/Vinit=1.274–3.860; between 2.5–15 km,
1.023–1.278. The hotspot is not just one failed endpoint sample.

`F` in this first CSV is the unrestricted M^-1 R field: the nonzero replaced
deep friction equations can contaminate that representation next to 40 km.
It is not the free residual norm. The solver's restricted mass-consistent norm
above is the actual convergence check. Visualization tau_xx/tau_yy/tau_xy are
the transferred **old FE history**, not accepted current Maxwell stress.

## What the mismatch does and does not establish

The independent C++ Airy checks pass, including spatially varying cohesive
augmentation: maximum fault-traction error 5.22e-8 Pa, top traction zero, and
centered-difference divergence defect 2.08e-6 Pa/m. No sign or analytic
equilibrium mismatch was detected away from the directional corner traces.

The actual discrete initialization is different: bulk mechanics consumes the
transferred FE history, while the surface samples retained parent stress and
domain-integrated surface inputs. Numerical bulk equilibration and finite-width
traction at the intersection do not preserve the analytic pointwise targets.
The accepted solution demonstrates the mismatch; it does not prove that any
particular correction is sufficient.

An additional sensitivity is explicit in the chosen radial Airy extension.
The projected C target ranges .41890–.50131 MPa. Its piecewise-linear q slope
ranges -485.98 to +486.12 Pa/m. The radial derivative coefficient r*q' reaches
-50.65 to +56.11 MPa (even 5–110 km: -17.57 to +19.40 MPa), before its angular
factor. Thus a modest surface-traction variation need not produce a modest
bulk extension. This is **not** evidence to omit the derivative, smooth C, or
change the history transfer. No such workaround was applied.

**Review decision:** choose how to construct a discretely equilibrated bulk
prestress that also realizes the official Vinit/Theta0 under the current
finite-width surface rule. A benchmark-local constrained prestress projection
is a candidate, not yet a derived/verified correction. Alternatively accepting
this startup transient would deliberately change the initial BP3 problem;
that choice has not been made. Increasing the artificial initialization
interval, changing Theta0, pressure or tolerances is not a demonstrated fix.

## Timing: initialization only, not a dynamics performance result

| Scope | Calls | Seconds |
|---|---:|---:|
| I_h/property preparation (nested, count once) | 2 | 147.63 |
| Particle domains/CPDI | 8 | 18.9 |
| Bulk matrix / RHS assembly | 6 / 7 | 9.12 / 8.46 |
| Stokes preconditioner | 6 | 6.53 |
| B actions / coefficient build | 120 / 6 | 16.2 / 1.11 |
| G total | 120 | 5.95 |
| Surface R/K total | 13 | 9.39 |
| Domain/association cache build | 2 | 2.75 |
| Setup initial conditions | 7 | 14.9 |
| Output | 1 | .334 |

The ASPECT total is about 269 s; subprocess total 273.18 s. The listed principal
scopes account for roughly 241 s; unisolated linear/preconditioner applications
and other setup account for the remainder. Do not sum nested FE/lookup timers
into I_h again. Cold I_h=82.52 s; lookup-reusing preparation=65.11 s. Both use
25895578 stored sample requests, 396 batches; MappingCartesian rejection is
eligible and rejects 802 requests. The surface geometry has 1766601 integration
points, uses the straight path, and reports zero general-path segment candidates.

I_h dominates this **initialization**, including a duplicate preparation needed
by the benchmark's current prestress construction. No performance correction
was made on the basis of a nonpassing physical initialization. A separate opt-in
condensed-linear timer has been added and build/unit-tested, but was not present
in this run. No real-step advection, history, or pilot-resolution timing is known.
The fine pilot targets 48.828 m; its resource configuration remains an estimate,
not server-ready evidence. Roughly 4x the fine-strip cells suggests planning
16–24 GiB total memory initially, subject to a short measured pilot after the
initialization issue is resolved. MPI scaling and a recommended rank count are
unmeasured; no strong-scaling campaign is implied.

## Files, tests and commands

Core changes: manager header/source add exact prescribed geometry and selected
essential V with private lift, homogeneous increments, active-set exclusion and
rollback. `source/simulator/solver.cc` uses the existing restricted surface solve
from the first direction and excludes prescribed rows from the fixed scale.
`source/simulator/phase_field.cc` lifts physical phase constraints before the
entry residual, including when all phase DoFs are prescribed. The specification
and current design document these capabilities. Defaults retain previous behavior.

New benchmark files: `bp3.cc`, `bp3_model.h`, `test_model.cc`, `CMakeLists.txt`,
`bp3_smoke.prm`, `bp3_pilot.prm`, `fault.txt`, `stations.txt`, `run.py`,
`analyze_initialization.py`, `bp3_notes.md`, and preserved attempt wrappers/logs.
`unit_tests/reconstructed_fault.cc` adds prescribed-V lifecycle/rollback coverage.

```
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake -S benchmarks/reconstructed_fault/bp3 -B benchmarks/reconstructed_fault/bp3/build -DAspect_DIR=/home/ein/repository/aspect/build-pf-cpdi
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
c++ -std=c++17 -O2 benchmarks/reconstructed_fault/bp3/test_model.cc -o /tmp/aspect-bp3-model-test
/tmp/aspect-bp3-model-test
build-pf-cpdi/aspect-release --test '[fault_prescribed_v],Stage-I*,[phase_field_domain],[fault_domain_quadrature],*cohesive*,MaxwellStress*'
mpirun -np 2 build-pf-cpdi/aspect-release --test '[fault_prescribed_v],Stage-I*'
python3 benchmarks/reconstructed_fault/bp3/run.py benchmarks/reconstructed_fault/bp3/initialization04.prm --cap 900
python3 benchmarks/reconstructed_fault/bp3/analyze_initialization.py benchmarks/reconstructed_fault/bp3/initialization04
```

All builds succeed. The one-rank selection passes **856 assertions / 27 cases**;
two ranks each pass **59 assertions / 11 cases**. The analytic test passes.
The runner refuses to overwrite an existing attempt, so the historical command
above must not be blindly rerun; use a new wrapper after the next decision.
No full ASPECT suite, new two-rank BP3 solve, restart, or real-step test was run.

**Incomplete/unverified work is explicit:** official initial slip and prestress
realization fail; independent support/normalization and discrete free-surface
traction certification are pending; real-step station/history output and restart
are unverified; the added mass-column output and condensed timer build but have
not been exercised in a new BP3 run. The plugin does not yet enforce all physical
initialization guards automatically—do not directly run its draft three-step
inputs. No dynamics or server pilot is claimed complete.

Source HEAD is fb4411915 with uncommitted changes. Attempt resource JSON records
the exact executable/plugin/input SHA256 values; tested source snapshots are
preserved under the benchmark `evidence/` directory. Unrelated working-tree
changes and earlier K evidence remain untouched. No commit was requested here.
