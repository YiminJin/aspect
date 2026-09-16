# Phase DoF indexing fix and BP3 interpolation check

## Result

The mixed-FE indexing bug is fixed and passes continuous/DG one-rank and DG
two-rank production phase tests. The resumed frozen-Airy BP3 initialization
with `distance weighted average` genuinely converges, but **does not reproduce
the reported removal of the initialization velocity error in this fixture**.
The intended initial-state gate remains unmet; no real BP3 steps were run.
Do not infer a prestress redesign from this result. Compare the exact successful
user fixture with this saved run before another numerical change.

## Indexing defect and correction

Both `PhaseFieldHandler::make_sparsity_pattern()`'s CPDI patch connectivity and
its cached vertex-to-phase map used `vertex_dof_index(v,component_index)`.
Its second argument is a vertex-local DoF index, not a component index. A
preceding DG component invalidates the assumed equality.

Both paths now retrieve cell DoF indices and select
`fe.component_to_system_index(component_index,v)`. The phase element remains
Q1, whose component-local DoFs are its vertices. No physics, CPDI integration,
constraints, ownership or solver criterion changes. The pre-existing essential
phase-state lifting change in this file is preserved, not part of this fix.

`tests/phase_field_precision.cc` independently checks the map using
`system_to_component_index()` and reference support-point coordinates on every
non-artificial cell. The same test then exercises production assembly and
nonlinear solves at zero, tiny and material phase residuals, checks fresh
residuals and verifies restoration of live state. This includes the sparsity
path, not just a standalone FE-index formula.

| Release regression | MPI ranks | Result | Elapsed |
|---|---:|---|---:|
| `cg_phase` | 1 | PASS, 64 cells, 2,249 DoFs | 1.617 s |
| `dg_phase` | 1 | PASS, 64 cells, 3,365 DoFs | 1.617 s |
| `dg_phase_two` | 2 | PASS, same DG discretization | 2.619 s |

DG is enabled for the regression's compositional fields; the BP3 smoke keeps
its original continuous composition discretization. CG and DG one-rank phase
diagnostic values match exactly. The two-rank material-input final residual is
3.56355e-15 versus 3.52127e-15 on one rank, both below the unchanged 4.92033e-12
target. Constant/tiny states meet their precision bounds, material residuals
are not hidden, and all map/live-state assertions pass. No broad suite or 3-D
test was run.

## Controlled BP3 check

Use `airy_dt0/distance_initialization.prm`, including the completed frozen-Airy
4e6 s case. Resolved parameters differ **only** in output directory and
`Particles/Interpolation scheme`: `cell average` -> `distance weighted average`.
The latter uses the existing default `Weight type = linear`. No other settings
are changed, in particular no DG BP3 discretization, Maxwell interval, boundary
condition, support, phase profile, initial state or tolerance change.

The same isolated Airy plugin binary is loaded (SHA256
`00ee39233c264e73803bf356abafdf65b703a9e879ed9b163d15fe1a62ec35a0`). It reads
the frozen initial traction curve; the unfinished offset plugin is not used.
Exported bulk mesh/phase and surface coordinates, Theta0, stored C0 and Ih
compare exactly. The target Airy CSV also matches exactly. The FE stress-history
approximation is deliberately allowed to change with the interpolator; it is
not confused with a change to the supplied particle Airy stress.

| Initialization metric | Cell average | Distance weighted average |
|---|---:|---:|
| max speed [m/s] | 1.37722e-8 | 1.36918e-8 |
| RMS speed [m/s] | 4.00261e-9 | 4.35636e-9 |
| max / RMS speed divided by imposed Vp/2 | 27.544 / 8.005 | 27.384 / 8.713 |
| max dt0 abs(u-u_rigid) [m] | .0568327 | .0563711 |
| RMS dt0 abs(u-u_rigid) [m] | .0154499 | .0166185 |
| free V/Vinit range | .96766--12.66497 | .96812--12.00178 |
| actual sigma_n range [MPa] | 45.6341--52.6082 | 45.7668--52.6464 |
| weak mean sigma_n [MPa] | 50.010535 | 50.009471 |
| free q minus frozen target range [MPa] | -.014562--1.026059 | -.025034--.991727 |
| test-weighted nodal RMS q error [MPa] | .130701 | .119735 |
| bulk physical pressure range [MPa] | 26.83567--35.17513 | 26.62580--35.23420 |

Normal stresses are actual constitutive surface samples, not column averages.
RMS velocity is volume weighted from saved Q2 polynomials; maxima are sampled,
not certified continuous extrema. Deep prescribed V and retained Theta0 have
zero measured error. Both side velocities match the prescribed rigid values
to 7.13e-18 m/s (Float32 output precision).

The new final bulk relative residual is 1.22408e-15 (absolute assembled norm
9.26862e-4, fixed scale 7.57189e11). Surface relative residual is 5.89513e-9
(RMS .100190 Pa, scale 1.69954e7 Pa). Both pass 1e-8. All five returned linear
directions pass their fresh checks; the last estimated/fresh/target values are
8.7767631e-8 / 8.7767714e-8 / 1.0030473e-7. These checks establish convergence
of the discrete solve, not agreement with official initial V/traction.

One new full initialization took 271.49 s and 4,143,824 KiB peak RSS
(3.952 GiB), within its 900 s cap. There was no retry, real timestep, offset
redesign or solver change. The measured runtime is comparable to the baseline
273.75 s; an interim impression that it was slower was not borne out.

## Reproduction and saved evidence

From the repository root:

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/uniform_shear/evolving/phase-floor/tests/build --target phase_precision_tests.release -j4
python3 benchmarks/reconstructed_fault/bp3/airy_dt0/run_interpolation.py cg_phase
python3 benchmarks/reconstructed_fault/bp3/airy_dt0/run_interpolation.py dg_phase
python3 benchmarks/reconstructed_fault/bp3/airy_dt0/run_interpolation.py dg_phase_two
python3 benchmarks/reconstructed_fault/bp3/airy_dt0/run_interpolation.py distance_initialization
python3 benchmarks/reconstructed_fault/bp3/airy_dt0/analyze_interpolation.py
```

Both builds pass; runner invocations are complete and refuse overwrite.
Analysis passes parameter, geometry, frozen-surface/phase/target and genuine
nonlinear/fresh-linear checks. `interpolation_comparison.json` has unrounded
values. Each case retains its log, resource/source hashes and outputs;
`distance_initialization/solution/solution-00000.pvtu` is ParaView-readable.
`git diff --check` and Python syntax checks pass.

Files changed for this follow-up: `source/simulator/phase_field.cc`,
`tests/phase_field_precision.cc`; new benchmark-local CG/DG/MPI/distance input
wrappers, `run_interpolation.py`, `analyze_interpolation.py` and saved diagnostic
evidence; this report, the local README and Stage-K progress entry. Existing
unrelated working-tree modifications and original baseline outputs are
preserved. No commit was requested or made.
