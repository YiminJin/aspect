# Fixed-mesh bulk MPI and one horizontal-side-release control

Completed 21 October 2026. Source entry f1f9f29cf40c32d1ddcea000b724375b64853532
plus preserved local work. See
[report](../../../../../doc/reconstructed_fault/particle_free/BULK_PARALLEL_REPORT.md)
and [predeclared scope](../../../../../doc/reconstructed_fault/particle_free/BULK_PARALLEL_SELECTION.md).

The direct comparison is `boundary-comparison.mp4` (29 accepted frames,
.15–.185 s), with GIF preview and loading-stop PNG. The entire new experiment
is `released-full-trajectory.mp4` (149 frames). All maps retain the whole domain,
physical axes and common scales. Stress is authoritative physical tau_xy;
rate is bulk slip strain rate, not surface velocity. The native Gauss8-weighted
cell means are a display/averaging choice, not new constitutive histories.
`remote-stress-balance.*` and `remote-profiles.*` independently recover the saved
affine stress, local Y and weak nodal force balance. No history interpolation.

The new loading redirects low damage toward the centerline, but early remote
nucleation joins the seed. Domain-spanning .001/.01 contours are not tip travel.
The .3/.5 contours remain fixed. Spatial accuracy, extended-interval temporal
uncertainty and propagation speed remain unqualified. Original failed evidence
and strict_reference_match_passed=false are retained.

## Native sources and commands

Numerical runs use SHA256
38336e67b35bf4bbee62a3dd8fa17e23111034dbc6c9a9169745d0091ebf58e3,
retained as `build-particle-free/bulk-parallel/native/aspect-numerically-qualified`.
The original launch name was `aspect-release`; its exact command/environment
are in `logged/` and `native-launch.json`. `native-run-source.tar.xz` contains
the actual numerical-run source overlay, with its manifest. The final source
changes only parameter help text after those runs. A final rebuild reproduces
a growing accepted step bitwise (`help-only-regression.json`) and restores the
full four-rank endpoint without physical advancement. Physical defaults/law,
quadrature and tolerances do not change.

The native build reuses the existing configured ASPECT/deal.II build with
separate replacement objects; `build-native.py` and native build records preserve
the local commands. Full rebuilds can use normal ASPECT CMake from the current
worktree. The plugin is configured/built directly:

    cmake -S benchmarks/reconstructed_fault/particle_free \
      -B build-particle-free/bulk-parallel/plugin \
      -DAspect_DIR=$PWD/build-particle-free/evolving-phase/native \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build-particle-free/bulk-parallel/plugin --target bulk_parallel_checks -j2

GCC 12.4, OpenMPI 5.0.6, deal.II 9.6.2, Trilinos 14.2 and the existing
Release configuration are retained. All timing runs set OMP_NUM_THREADS,
DEAL_II_NUM_THREADS and OPENBLAS_NUM_THREADS to 1; OMP_PROC_BIND=false.
The following are recorded commands, not instructions to overwrite retained
outputs. Choose fresh output directories for any future rerun.

    env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_PROC_BIND=false \
      mpirun -np 4 --bind-to core \
      build-particle-free/bulk-parallel/native/aspect-numerically-qualified \
      build-particle-free/bulk-parallel/r4.prm

Run equivalent r1/r2 inputs for the 1/2-rank qualification. The bounded 600 s
wrapper calls are in `logged/bulk-parallel-r1-final.json`, `r2-final.json`,
and `r4.json`. They do not overlap. Final wall times are 118.34/67.24/48.57 s,
with measured 1.76x/2.44x speedups. The direct linear solve remains on rank zero;
this is not scalable distributed factorization.

Native restart inputs are copied from accepted step 47, never edited. Actual
restart/rejection commands add ASPECT_BULK_RECOVERY=1 and use resume-rN.prm.
Two solved trials are rejected, then the normal accepted path publishes once.
All 74 common owner/field files and 17 ledger rows match their uninterrupted
same-rank controls bitwise. `restart-verification.json` records each file.

The sole new physical trajectory is:

    env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 \
      ASPECT_BULK_PRESSURE_DIAGNOSTIC=1 ASPECT_BULK_CHECK_TANGENT=1 \
      OMP_PROC_BIND=false mpirun -np 4 --bind-to core \
      build-particle-free/bulk-parallel/native/aspect-numerically-qualified \
      benchmarks/reconstructed_fault/particle_free/bulk_cohesive/mixed_boundary.prm

External timeout 7200 s; exit 0, 2244.41 s, 148 positive accepted steps to .185 s.
No server job, adaptive mesh, transport, RSF or geometry change is run.

## Evidence and durable data

- `parallel-verification.json`, `costs.csv`: canonical physical-owner/stress
  comparisons, true native residuals, timers and resident peaks.
- `restart-verification.json`: bitwise same-rank restart/rollback/publication.
- `boundary-assessment.json`, `boundary-extra.json`: initial identity, response,
  physical stress/forces/storage, work, energy, region and hold checks.
- `mixed-history.json`, `mixed-contours.json`, `mixed-energy.csv`: physical
  times, onset, cap/remote damage, contour topology and separate energy terms.
- `movie-data.npz`, `mixed-fields.npz`, `remote-fields.npz`: exact float64
  plotting/averaging data; original global-DoF numbering is distinguished by
  grid associations. The original continuum support and represented-zero
  initial field are separate classifications.
- `quadrature/`: lossless float64 original-Gauss8 snapshots at accepted
  0,96,120,121,124,128,139,140,148. Native coordinates/JxW/CellIds are stored once.
  Every compact array roundtrip is verified against the native CSV; the manifest
  pins each original text file and compact copy. Full plane-strain stress is
  retained; zz is reconstructed from -(xx+yy), with both xy norm contributions.
- `qualification.tar.xz`: all canonical nine-mode DG/native-FE/local-history
  records and compact diagnostics for the 1/2/4 controls and restarts, plus
  their accepted-47 native checkpoint. Original large dense text remains in
  staging; its hashes and measured comparisons are retained.
- `mixed-diagnostics.tar.xz`: all phase, weak-force, reaction, cell-mean,
  pressure, resource and accepted-ledger exports; excludes redundant dense
  CSV and empty large-case record placeholders.
- Five complete native checkpoints are retained under
  `build-particle-free/bulk-parallel/saved-checkpoints/`, accepted 95,111,127,
  143,148. Each archive has per-file hashes; the native restart pointer is
  included. The clock is the next pending step. No conversion is performed.

The new raw runs are staged through
`build-particle-free/bulk-parallel/runs -> /tmp/bulk-parallel-runs` to avoid
filling the root disk. They are not deleted. Durable compact data/checkpoints
live in this worktree. Process RSS excludes tmpfs output storage; raw file bytes
and exact identities are in `raw-manifest.json`. Sum of per-rank resident peaks
is not a simultaneous node-memory measurement.

Python scripts only read/export/compare or preserve completed checkpoints;
native C++ owns equations, trial updates, acceptance and time advancement.
`native-run-source-manifest.json`, `final-source-manifest.json`,
`source-documentation-diff.patch`, entry/launch identities and checksums separate
the qualified source from the final help-only correction. All 34 protected BP3
identities and the stateless constitutive source are unchanged.
