# Bounded bulk-cohesive experiment

This is a distinct default-off constitutive model selected by
`Fault constitutive mode = bulk cohesive`. The authoritative equations and
discrete residual/ledger are in
`doc/reconstructed_fault/particle_free/BULK_COHESIVE_FORMULATION.md`.
The original surface law, production defaults and BP3 package are retained.
The measured outcome must be read in `BULK_COHESIVE_REPORT.md`; passing a solver
or material-point test does not establish connected tip propagation.

## Build and run locally

Run from the repository root. The existing local core build uses deal.II
9.6.2 at `/opt/dealii/9.6-local`, MPI, Trilinos and UMFPACK. Build the core
with the same configuration as `build-particle-free/evolving-phase/native`.
Do not load the P3/PF6a observer DSOs together with the new fixture: they use
the same narrow existing test-access class.

```sh
cmake --build build-particle-free/evolving-phase/native -j2
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/evolving-phase/plugin -DAspect_DIR="$PWD/build-particle-free/evolving-phase/native"
cmake --build build-particle-free/evolving-phase/plugin --target bulk_cohesive_checks -j2
cmake -S benchmarks/reconstructed_fault/particle_free/bulk_cohesive -B build-particle-free/bulk-cohesive/points -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/bulk-cohesive/points -j2
build-particle-free/bulk-cohesive/points/bulk_cohesive_points '[bulk_cohesive]'
```

The standalone runner exports its **actual C++ responses** when
`ASPECT_BULK_POINT_EXPORT` and `ASPECT_BULK_PROFILE_EXPORT` name CSV files.
Run `assess_points.py` and `assess_profiles.py` on these for independent checks.
NumPy is needed; `plot_native.py` also uses Matplotlib. No SciPy is required.

Native primary, serial and bounded (choose a fresh logger label/output):

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py bulk-replay 900 env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 ASPECT_BULK_CHECK_TANGENT=1 ASPECT_BULK_RECOVERY=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/native.prm
```

`ASPECT_BULK_RECOVERY` invokes fully converged nonpublishing native trials and
the existing outer rejection callback at a loaded step, then deterministic
retry. It does **not** claim qualification of automatic timestep cutback.
`ASPECT_BULK_CHECK_TANGENT` requests a test-only independent coupled action
check at 0.11 s; it does not alter the constitutive update.

The archived input overrides record the four selected controls:

| Case | Cells | dt [s] | Mechanics/history quadrature | Physical accepted end [s] |
|---|---:|---:|---:|---:|
| Primary | 16×32 | .0025 | Gauss6 | .16 |
| Temporal | 16×32 | .00125 | Gauss6 | .16 |
| Spatial | 32×64 | .0025 | Gauss6 | .16 |
| Quadrature | 16×32 | .0025 | Gauss8 | .16 |

Source projection is Gauss3 in every case. Sample the same continuum seed
anew at the actual vertices of each mesh; do not prolong the coarse obstacle.
Local a/k/L/q/rate are quadrature histories; changing quadrature is a fresh
trajectory, never an unsupported checkpoint conversion.

For restart, terminate a fresh prefix at the **exact printed step-48 clock**
of the primary, copy its whole `restart/` tree to a fresh output branch,
then set `Resume computation = true` and restore End time .16. Matching the
binary clock avoids creating a different clipped final interval. Preserve
the original checkpoint. The existing checkpoint stores the next pending
clock; compare the new bulk owner's accepted `time`, not that pending clock.

## Read and replay the evidence

`../evidence/bulk_cohesive/` holds reports in JSON, three principal PNG/PDF
figures, source/input identities, bounded command logs and selected actual
exports/checkpoints in `native-evidence.tar.xz`. `archive-manifest.json`
states exactly what was retained; full raw runs remain in
`build-particle-free/bulk-cohesive/runs/` in this worktree.

`assess_native.py RUN --output FILE` checks actual physical fields, positive
integration weights, irreversibility, accepted publication counters, local
relations, observer variance identities and separate work terms.
`compare_native.py PRIMARY REFERENCE --kind temporal|spatial|quadrature --output FILE`
compares common physical times. Spatial/changed-rule stress comparisons use
actual coarse-cell weighted tensor moments; they do not interpolate the local
constitutive histories to invented points. Full plane-strain tensor norms
retain zz and both xy contributions.

The observer is a straight-column, raw-weight diagnostic with old/new interval
masks. It is not automatic ridge reconstruction or physical surface birth.
It receives only constitutive parameters, copied samples and weights; it has
no mutable access to the simulator or its mechanical solution. Explicit
before/after comparisons additionally check every persistent bulk history and
the physical obstacle. New diagnostic records inherit moments, not zero birth
histories. No old surface energy, friction or P3 stress correction is added.
