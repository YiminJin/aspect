# Non-collinear exact-jump transfer evidence

Read [report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_KINK_REPORT.md)
and [formulation](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_KINK_FORMULATION.md).
This is an isolated native FE/owner control, not production propagation.

## Figures

- [Specimen and tip close-up](specimen.png): physical fitted mesh, Gauss history
  points, old segment, prescribed append, closed corner and common-radius
  principal-stress diagnostic. Point counts are not integration weights.
- [Physical stress transfer](stress_transfer.png): before/after share a scale;
  the difference is exactly zero. Physical tau_xy, not auxiliary history T.
- [Histories and pressure](histories_pressure.png): old jump/damage/age retained,
  new tied records, independent normal-pressure traces on both meshes.
- [Admission and energy](admission_energy.png): tied feasibility versus strength,
  and unchanged bulk/bond/fracture inventory.
- [Three-state animation](transfer.gif): loaded incoming state, identical
  geometry-only transferred state, one computed coupled reference timestep.
  Each frame is a saved computed state, repeated for viewing; no interpolated
  dynamics or quantitative propagation speed is implied.
- [Improved original straight-growth comparison](comparison-straight-growth.png):
  early inset, absolute/relative growth differences, common 2.42 s endpoint.
  The 65.4% / .0055521 m discrepancy at 1.8 s remains explicit. Broad process-zone
  advance and the fully decohered front remain separate.

**Plain-language caption.** A kink was prescribed ahead of a loaded old segment.
The new segment was held closed and tied while its force capacity was checked.
Adding its records did not create slip or fracture energy. A separate axial load
would require damage on the oblique segment; that insertion was rejected. The
plots do not show spontaneous direction selection or sliding through the corner.

## Compact data

| File | Contents and units |
|---|---|
| `comparison.csv` | mesh, energy J/m, required multiplier/normal pressure Pa, angles degrees counterclockwise from +x, residual and tangent error |
| `native-*-surface.csv` | actual Gauss x/y m, arclength weight m, branch, a, signed jump m, dimensionless Theta, two normal pressures Pa and tied multiplier Pa |
| `native-*-fields.csv` | physical Gauss6 locations/area weights, full plane-strain stress and p' Pa; common-kernel weights |
| `animation_cells.csv` | plotted physical cell means in the three discrete states; time s, x/y m, tau_xy and p' Pa |
| `local-resolution.json` | full averaged tensor, principal axis, both conjugate maximum-shear planes, trace error and declared comparison gates |
| `native-*-oblique-mode.json` | nonzero virtual append displacement: pressure response, zero normal gap, storage/work and affine identity |
| `native-*-checks.json` | transfer, admissibility, coupled tangent, physical work/loss and BE remainder checks |
| `native-*-accepted.json`, `native-*-physical-step.json` | every history, jump/memory coefficient and geometry generation used in reference restart/continuation |
| `native-owner.json`, `owner-*/*` | actual ASPECT manager/checkpoint runs, atomic rejection and bitwise restart archives |
| `summary.json` | measured gates, timings; no general convergence/propagation claim |
| `preliminary-overload-invalid.json` | retained test-design failure and its correction |
| `parameters.csv` | unchanged physical calibration and selected diagnostic geometry/load values |
| `provenance.json`, `preservation.json`, `changed-files.json` | source/build/input hashes, unchanged production/BP3 hashes and task files |

Surface integration uses physical arclength, including sqrt(1+.5²) on the
append. The radius-.1 m kernel centered at (.5,0) is (1-r²/R²)², compactly
supported and normalized with physical Gauss6 weights. Stress averages use full
3D plane strain (including zz and both xy contributions). Angles describe stress
axes/plane tangents; they are not observed crack-growth directions. The averaging
kernel is diagnostic only; the actual weak strength certificate is unchanged.

The synthetic old damage/reference/age profiles and all boundary/pressure
conventions are in the formulation. New records have a=0 and the virgin clock;
a>0 describes cohesive damage and a=1 full decohesion. No activation time is
inferred from the snapshot. Physical time stays .2 s at insertion; only the
subsequent reference step reaches .202 s. The native owner host scheduler uses
0/.001/.002 solely to exercise checkpoint callbacks, not physical loading.

## Reproduction (sequential, local, bounded)

From the worktree root, with the existing ASPECT/deal.II build available:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free \
  -B build-particle-free/cohesive-interface-kink/build \
  -DAspect_DIR="$PWD/build-particle-free/native"
cmake --build build-particle-free/cohesive-interface-kink/build \
  --target cohesive_interface_kink cohesive_interface_kink_owner -j2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_kink.py kernels
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_kink.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_kink.py owner
MPLCONFIGDIR=/tmp/kink-mpl OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_kink.py
```

The native kernels export the actual mixed operator, full response coefficients
and independent Gauss6 fields under `build-particle-free/cohesive-interface-kink/`.
Those larger solver response files and checkpoints remain separate from the
compact evidence archive. The independent harness checks the existing C++ point
law before using its vectorized port. The owner adapter imports that verified
mechanical state; it does not replace the production outer solver.

The runner preserves previous run directories on rerun and imposes a 180 s timeout
per native invocation. The new archive is a source/evidence **overlay**, not a
standalone ASPECT distribution. Original straight-growth archives are unchanged;
the working comparison PNG is revised, with its original preserved in the task's
entry snapshot and historical archive.
