# Fracture scales and one bounded propagation-focused specimen

[Concise report](../../../../../doc/reconstructed_fault/particle_free/BULK_FRACTURE_REPORT.md)
and [selection before the physical trajectory](../../../../../doc/reconstructed_fault/particle_free/BULK_FRACTURE_SELECTION.md).

- `physical-mechanism.png`, `activation-x-time.png`, `first-activation-locations.png`:
  common physical axes, initial support/endpoints, connected softening versus
  detached activation. Input sets are different physical specimens, not an
  accuracy comparison.
- `early-physical-profiles.png`, `response-and-work.png`, `strip-scales.png`:
  original-state stress/driving/resistance, work/history accounting and the
  independent strip calibration versus actual BE response.
- `propagation-focused.gif`: all 53 accepted frames .12–.185 s, stride .00125 s.
  The imposed loading/seed and fixed geometry are distinguished from calculated
  contours. Crack-core propagation speed and branch selection are unqualified.
- `fracture-scales.csv`: SI units, lengths/slips, coefficients, seed storage.
- `*-centerline.csv`: exact Q1 centerline phase at x-bin centers, band-mean
  |w| and x-bin-averaged cross-fault integral of |w|. This is slip activity,
  not signed surface velocity. `w` is a bulk engineering shear rate.
- `*-early-profiles.csv`: radius-.03125 m compact physical kernel; native
  positive-quadrature cell means for stress/Y, native integrated nodal loads
  and their matching masses for compatible weak drive/resistance.
- `*-history.csv`: initial fracture energy and subsequent storage changes;
  endpoint/trapezoidal work, both viscosities, friction, individual BE numerical
  remainders and residual work. Region phase integrals use Gauss8 physical
  points and Q1 basis. Cap means x>=tip and radius<2.25 ell; outside support
  is physical distance>=initial support radius.
- `*-activation-nodes.csv`: first increments >1e-12 and adjacent accepted-time
  brackets. The report also records the >1e-5 threshold. Far-ahead reporting
  windows are x>tip+.4; they do not enter the equations.
- `*-contours.csv`: Q1 contour thresholds .001,.01,.05,.1,.3,.5, exact line
  first exits and separate raster component diagnostics. Raster subdivisions
  are 4; `topology-check.csv` verifies selected states at 8.
- `*-map-cells.csv`: actual native cell means backing the static map at .175 s.
- `*-strip-envelope.csv`: stationary-profile calibration, not an imposed
  irreversible trajectory. `*-strip-history.csv` uses the actual .185 s load;
  the extended `*-strip-calibration.csv` has its own times/continuation stop.
- `candidate-increment-history.csv`: rapid interval (.17375,.17500], peak
  increment .085608 at (1.25,.5). No speed inference is made.
- `hold-branch-check.json`: independent native-point/weak-load no-growth hold
  candidate. This demonstrates nonunique KKT continuation without publishing
  a second simulator state. `qualification-scope.json` retains failed/unqualified
  scientific findings even though numerical guards pass.

Full native output/checkpoints, the rejected m=3 startup and the source overlay
are under `build-particle-free/bulk-fracture/archives/`, indexed by `archives.json`.
The preceding accepted-127 checkpoint (.15875 s; next pending clock .16) is
retained for later temporal/branch work. Archives keep repository-relative paths;
extract into an empty directory, preserving the retained originals. All archive
payload hashes are verified before delivery. No PDF/SVG/MP4 copies are created.

Native time advancement, trials, acceptance and publication remain in C++.
Python only supplies the independent strip/reference checks and reads completed
exports for analysis/packaging. The recorded native command and environment are
in `logged/`; native execution exits 0 after 148 publications. Source and old
BP3/native binary identities are in `entry.json` and `native-checks.json`.

Source is in `benchmarks/reconstructed_fault/particle_free/bulk_cohesive/`.
The additional fixture uses the existing configured ASPECT build:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free \
  -B build-particle-free/bulk-fracture/plugin \
  -DAspect_DIR=$PWD/build-particle-free/evolving-phase/native -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/bulk-fracture/plugin --target bulk_propagation_checks -j2
cmake -S benchmarks/reconstructed_fault/particle_free/bulk_cohesive \
  -B build-particle-free/bulk-fracture/strip -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/bulk-fracture/strip --target bulk_strip_oracle -j2
```

The exact run was `mpirun -np 4 --bind-to core` with the retained native
`build-particle-free/bulk-parallel/native/aspect-release` and `fracture.prm`.
OMP/DEAL_II/OPENBLAS threads=1, OMP_PROC_BIND=false,
ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1,
ASPECT_BULK_PRESSURE_DIAGNOSTIC=1, ASPECT_BULK_CHECK_TANGENT=1.
Do not overwrite retained output when reproducing it: use a fresh output
directory. The m=3 startup and sandbox MPI failure are separately recorded.
