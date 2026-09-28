# K5: frozen mechanical width comparison

## Decision

**Narrowing the mature AT1 profile from ell=400 m to 200 m restores a
substantial amount of mechanical resistance.** On the final local mesh the
increase is **5.5305x for the actual 200-m Q1 input** and **3.0455x for the
600-m input**. This is a frozen mechanical result, not a demonstration that
an evolving RSF trajectory is stable. No further simulation is needed for
this question.

Instructions: `benchmarks/reconstructed_fault/bp3/frozen mechanical width comparison.md`
(the requested `first_long_run/` copy did not exist).
Artifacts: `benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-width-comparison/`.

## Controlled problem and execution

- Same 300x100-km box, fixed straight fault, 100-m fault grid, mature C=0
  law, loading/bulk boundary conditions and uniform G/viscosity.
- Same core phi=.6, AT1, degradation curvature=1, material parameters and
  actual nodal tapered input: amplitude 1e-12 m/s, support 15--18 km,
  center 16.5 km, wavelengths 200 and 600 m. Exported node coordinates and
  input values agree **exactly** between widths.
- Each case initialized its own physical Q1 phase field using the production
  stationary distance profile, computed its own full remote-backend I_h,
  and projected it normally. No old normalization was restored. Both top
  and bottom completion tables were regenerated for each width using the
  continued stationary Q1 profile. The endpoint cell size remains 97.65625 m;
  completion tables are identical between the two meshes of a given width,
  not between widths. Their independent order checks differ by at most
  7.19e-11 m. Immutable prestress input was not recalibrated.
- Holding material parameters fixed implies the implemented degradation
  coefficient m changes from 6.00714756 to 12.01429512 through its existing
  1/ell dependence. The curvature parameter remains fixed. This scalar
  multiplier cancels from normalized chi; no additional parameter was tuned.
- First mesh: existing qualified 48.828125-m patch, 48,699 cells. Its narrow
  profile had a sampled normalization discrepancy of 3.08e-4, motivating the
  **one authorized paired local refinement**. Final patch: 24.4140625 m,
  97,344 cells. The existing mesh utility refines only the buffered probe
  patch (10--23 km, approximately +/-2.5 km normal) and required grading,
  not the entire fault. Exported physical cell geometry is identical between
  widths on each level.
- Four ranks, Release, existing AMG diagnostic setup. The existing callback
  is reached after one fresh-checked, **uncommitted** initialization linear
  direction at Newton iteration zero. This is harness setup, not a loaded
  trajectory or an equilibrated base-state requirement. The reported
  perturbation solves are exclusively A dx=B dV, with homogeneous bulk
  perturbation constraints. No nonlinear trial, timestep or history is
  accepted. The diagnostic verifies unchanged physical/working vectors and V
  and intentionally stops before publication. No state/particle-history
  update is used to obtain these coefficients.
- Native kappa=1.2815248119788472e17 Pa s, from the unchanged artificial
  initialization interval 4e6 s. All tabulated coefficients use the exact
  uniform multiplier 38.33411519295163 to kappa=4.912611976502280e18 Pa s.
  Both A and B scale uniformly: velocity response is unchanged; physical
  pressure and traction responses scale with kappa. No timestep controller
  was changed.

## Work-normalized mechanical response

Let v be the actual Q1 input and use the production owned bulk-QP measure
w=JxW*chi. Define M_v=sum(w*v^2). With S:S=1/2,

    D = sum(w*kappa*chi*v^2)/M_v,
    R_n = sum(w*v*kappa*partial_n u_s)/M_v,
    R_s = sum(w*v*kappa*partial_s u_n)/M_v,
    R = R_n+R_s,       K_mech = D-R.

Positive K_mech is restoring. These are **mechanical shear** coefficients;
instantaneous friction, radiation damping and frictional normal-stress
feedback are not included. The signed parts of relaxation must be added,
not compared as absolute values. Units below: **1e15 Pa s/m**.

| Wavelength | ell | Direct D | R_n | R_s | Relaxation R | Net K_mech |
|---:|---:|---:|---:|---:|---:|---:|
| 200 m | 400 m | 8.594920 | -0.164583 | 8.025253 | 7.860670 | 0.734249 |
| 200 m | 200 m | 17.111613 | -0.738764 | 13.789604 | 13.050840 | 4.060773 |
| 600 m | 400 m | 8.594879 | -0.412414 | 5.766975 | 5.354562 | 3.240317 |
| 600 m | 200 m | 17.111551 | +0.330398 | 6.912861 | 7.243260 | 9.868291 |

At 200 m, only 8.54% of the direct coefficient survives bulk relaxation for
ell400, versus 23.73% for ell200. At 600 m these fractions are 37.70% and
57.67%. The normal-velocity variation along the band remains the dominant
relaxation contribution. Narrowing does not suppress that motion artificially:
it increases the direct coefficient and changes the fraction relaxed by the
actual incompressible bulk solve.

### Mesh and waveform discrimination

| Wavelength | ell | Net at h=48.828125 m | Net at h=24.4140625 m | Change | Actual-Q1 continuum diagnostic | Final difference |
|---:|---:|---:|---:|---:|---:|---:|
| 200 m | 400 m | 0.785358 | 0.734249 | -6.51% | 0.728574 | +0.779% |
| 200 m | 200 m | 3.989098 | 4.060773 | +1.80% | 4.105097 | -1.080% |
| 600 m | 400 m | 3.212364 | 3.240317 | +0.87% | 3.250043 | -0.299% |
| 600 m | 200 m | 9.656699 | 9.868291 | +2.19% | 9.943731 | -0.759% |

The independent waveform diagnostic Fourier-transforms the **actual Q1 hat
functions** analytically, including their taper and higher harmonics, and
uses their exact line mass. For the continuum stationary transverse profile,
the incompressible shear multiplier is
4*k_s^2*k_n^2/(k_s^2+k_n^2)^2. It reproduces the supplied pure-sine numbers:
0.728348 -> 4.118067 and 3.222528 -> 9.975915. Actual-Q1 values in the table
include the measured input spectrum and its own normalization. Doubling
spectral resolution changes them by at most 1.42e-7 relatively; retained
spectral mass differs from exact Q1 mass by at most 4.94e-6.

This is a large-domain continuum waveform diagnostic, **not an exact solution
for the inclined Cartesian FE localization**. Production chi varies along
the fault: primary coefficients above integrate all actual QPs. Independent
profile checks use 61 columns, not one supposedly representative normal ray.
The displayed normal profiles at four nearby coordinates are illustrations,
not the input to an exact FE prediction.

Remaining differences include physical-profile interpolation, Stokes/input
quadrature resolution, and the finite-box response versus the continuum
diagnostic. They decrease on local refinement and are small compared with
the factors 5.53 and 3.05. Finite-boundary error was not separately measured;
the same boundaries and far mesh were retained. These two meshes do not
establish complete spatial convergence, but resolve the requested strong
width effect. No further mesh level or evolving replay was run.

## Normalization and correctness

Column values below are max |integral(h(phi_h))/Ihat_h - 1| over 61 locations
from 15 to 18 km. Native modal mass compares actual sum(JxW*chi*v^2) with
the exact line integral of v^2, separating it from column integration.

| ell | Column error, first mesh | Column error, final mesh | Final max modal mass error |
|---:|---:|---:|---:|
| 400 m | 3.950e-6 | 2.546e-7 | 7.564e-6 |
| 200 m | 3.077e-4 | 2.362e-6 | 1.150e-5 |

Both final checks satisfy 1e-4 without changing support or renormalizing.
An independent enumeration of **all** nearby bulk QPs, including those absent
from the association exports, bounds the final modal mass missing from native
support by 2.915e-6 and the missing direct work by 1.584e-10. These measured
quantities are distinct from the column and native quadrature errors.

Across the eight private A solves:

- Fresh relative linear residual <=8.670e-11 (target1e-10), 16--17 iterations.
- Source/traction work identity relative error <=8.031e-15.
- Signed gradient split closes to <=1.559e-14.
- Independent assembled/action consistency <=2.117e-14; retained centered
  constitutive derivative checks also pass their unchanged bounds.
- Actual Q1 phase versus regenerated profile <=7.483e-14 at checked probe
  QPs; independently reconstructed chi differs by <=5.543e-16 absolute.
- Same fault coordinates, cells, nodal input, source ownership and native
  integration measure verified between widths. No accepted-step file exists.

## Reproduction and provenance

Production executable and BP3 plugin were unchanged. Only diagnostic exports
and selection of the two requested modes were added to the existing test
plugin. No production equations, solver tolerances or friction/state settings
were modified. No broad regression suite or time-evolving replay was run.

Source HEAD: `359ea223cd0fa088ba7c8b03338cc7fe19d4acdc` plus the existing dirty
working tree. Each case's `launch.json` records its resolved parameter file,
mesh/fault/prestress/completion hashes, environment and executable/library
hashes. The diagnostic library hash is
`c51b108bb5e29070f7d3e6cabe7036baef636f4950f5d223e5723bb25a9a5b4b`.
Preparation-only attempts and a sandbox MPI-interface failure were retained;
the latter occurred before ASPECT started. No numerical failure was retried.

Commands, from repository root:

```sh
cmake --build benchmarks/reconstructed_fault/performance/build-gmg --target fault_mechanical_modes -j4
python3 benchmarks/reconstructed_fault/bp3/run_mechanical_width.py 400
python3 benchmarks/reconstructed_fault/bp3/run_mechanical_width.py 200
python3 benchmarks/reconstructed_fault/bp3/run_mechanical_width.py 400 --execute
python3 benchmarks/reconstructed_fault/bp3/run_mechanical_width.py 200 --execute
benchmarks/reconstructed_fault/performance/build-gmg/fault_refine_mesh benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-bulk-refinement/fine/target_cells.txt benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-width-comparison/target_cells_finer.txt
python3 benchmarks/reconstructed_fault/bp3/run_mechanical_width.py 400 --mesh benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-width-comparison/target_cells_finer.txt --suffix=-finer
python3 benchmarks/reconstructed_fault/bp3/run_mechanical_width.py 200 --mesh benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-width-comparison/target_cells_finer.txt --suffix=-finer
python3 benchmarks/reconstructed_fault/bp3/run_mechanical_width.py 400 --execute --suffix=-finer
python3 benchmarks/reconstructed_fault/bp3/run_mechanical_width.py 200 --execute --suffix=-finer
python3 benchmarks/reconstructed_fault/bp3/analyze_mechanical_width.py
python3 benchmarks/reconstructed_fault/bp3/analyze_mechanical_width.py --suffix=-finer
```

The driver refuses to overwrite completed runs. Its ordinary 600-s cap was
not approached. Elapsed times (ell400/ell200): 54.17/49.62 s on the first mesh,
104.31/93.20 s on the final mesh: **301.30 s total**. Maximum reported child
RSS was 2.29 GiB (not an aggregate four-rank node-memory measurement).

Useful artifacts:

- `comparison.json`, `comparison-finer.json`, `coefficients*.csv`;
- `profiles_and_inputs*.png`, `localization_profiles*.csv`,
  `normalization_columns*.csv`;
- Per case: `mechanical_mode_nodes.csv` (actual Q1 inputs),
  `mechanical_mode_qp_rank*.csv` (actual chi/weights/response),
  `mechanical_shear_parts.csv`, Q2 velocity exports, physical Q1 cell exports,
  `surface.csv` (own I_h), and `stationary_profile.csv`.

**Next decision:** treat finite-width accommodation as demonstrated for these
frozen modes. Do not infer nonlinear stability, tune RSF parameters or start
a new trajectory from this experiment alone.
