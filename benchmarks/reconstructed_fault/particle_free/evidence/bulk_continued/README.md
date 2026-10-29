# One fixed-mesh continued-loading experiment

The selected physical run is complete. No server job, second timestep, changed
mesh, altered law, limiter or production-default change is included.
See [the measured report](../../../../../doc/reconstructed_fault/particle_free/BULK_CONTINUED_REPORT.md).

The native output is `build-particle-free/bulk-continued/runs/continued`.
The retained fine baseline is `build-particle-free/bulk-spatial/runs/refined-qualified`.
Old checkpoints were already in the original hold; the new run initializes
freshly and its prefix through .15 s matches the baseline exactly.

## Reproduce the selected input

From the repository root, with the existing GCC12.4/OpenMPI5/deal.II9.6.2 build:

```sh
env PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:/usr/local/bin:/usr/bin:/bin cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/bulk-continued/plugin -DAspect_DIR="$PWD/build-particle-free/evolving-phase/native"
cmake --build build-particle-free/bulk-continued/plugin --target bulk_continued_checks -j2
build-particle-free/bulk-source-history/native/aspect-release --validate benchmarks/reconstructed_fault/particle_free/bulk_cohesive/continued.prm
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py bulk-continued-native 14400 env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 ASPECT_BULK_CHECK_TANGENT=1 OMP_PROC_BIND=false build-particle-free/bulk-source-history/native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/continued.prm
```

These are the recorded commands, not an instruction to overwrite results.
The wrapper refuses duplicate log labels. For a repeat, use a new output path
and label. The wrapper sets OMP/DEAL_II/OPENBLAS threads to one and records the
actual environment. Validation and native run exit 0; measured wrapper wall
time is 5121.472 s. `parameter-comparison.json` verifies that parsed values
differ from the baseline only in end time, output directory and diagnostic DSO.
The test-only DSO changes the prescribed stop from .15 to .175 s.

The native executable SHA256 remains
`31ad1e59820079e712bdae23732697394cbfe36e8377001fa50cdac2a05b8d5a`.
`selected-source.tar.xz` and `source-manifest.json` preserve selected actual
core/benchmark sources, inputs and analysis helpers against the dirty f1f9f29cf
worktree; this is an evidence capsule, not a replacement BP3 server package.

## Read or regenerate the evidence

```sh
env OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bulk-continued-mpl python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/assess_continued.py
env OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bulk-continued-mpl python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_continued.py
```

- `assessment.json`: initialization, independent work/storage/history checks,
  onset, hold control and diagnostic topology sampling checks.
- `history.json`, `contours.json`, `fields.npz`: all accepted physical times,
  Q1 phase, seed-connected contours, intact-material growth and rates.
- `energy.csv`: endpoint/trapezoidal work, physical storage/dissipation, all
  native BE numerical terms, phase/mechanical/pressure/velocity residual work.
- `continued-loading.mp4`: 149 frames, every accepted state including t=0,
  1500×1100, 7 fps, 21.285714 s. All frames decode successfully. Fixed axes and
  colour scales; physical time is printed. No physical growth speed is implied.
- `continued-loading-preview.gif`: 38 frames, uniform four-step sampling plus
  the endpoint. `continued-loading-stop.png` is MP4 frame 140, physical .175 s.
- `continued-final`, `continued-energy`, `continued-growth`: standalone PNG/PDF.
  Symmetric-log remainder axes are linear inside ±1e-6 J/m. Rate display-bin
  areas are not Gauss weights. Numerical integrals use actual positive JxW.
- `prefix-015.json`, `preservation.json`: identical loading-prefix exports and
  unchanged old evidence/DSOs/core plus all 34 protected BP3 artifacts.
- `raw-manifest.json`: 1196 raw input/output/archive identities without copying
  the multi-GB native outputs. `SHA256SUMS` covers this compact evidence folder.

Analysis uses physical Q1 evaluation, original native q/rate histories and the
full plane-strain norm xx²+yy²+(xx+yy)²+2xy². Neither stress histories nor q
are interpolated into a different constitutive state. Connectivity at levels
.001,.01,.05,.1,.3,.5 is distinguished from an eight-neighbor native activation
graph using the established 1e-12 nodal-change threshold. The former is checked
on finer diagnostic rasters at appearances/joins; this is not native refinement.

## Saved intervals and native restart anchors

`build-particle-free/bulk-continued/saved-checkpoints/accepted-NNNN.tar.gz`
contains the unmodified native `restart/ID/` files and matching last-good
pointer. All 15 archives pass file-by-file checksums against their source
snapshots: steps 95,99,103,107,111,115,119,123,127,131,135,139,143,147,148.
Accepted times and hashes are in `checkpoint-verification.json`.

Retain (.15,.155] s for remote activation and the largest per-step phase
increment, and (.17125,.17375] s for the low-level contour join. No local
increment exceeds .001; the maximum is .000699889 in (.15375,.155]. The five
conservative acceleration flags are retained in `assessment.json`; they do
not by themselves establish abrupt physical growth.

Useful anchors are accepted step 119 at .14875 s (native pending clock .15 s)
and step 135 at .16875 s (pending .17 s). Step 95 at .11875 s precedes original
onset; step 139 at .17375 s precedes the hold switch. Full DG/native-FE/local/
phase/solution exports exist at every accepted state from .15 s; q and phi
exist throughout and direct local rates from .12 s. Signed and absolute
shear, recoverable a, q, all nine DG coefficients and native FE memory remain.

To branch later, unpack a verified archive into a **new** output directory,
preserving its `restart/` tree and pointer; select native `Resume computation =
true` with the same physical owners, mesh and quadrature. Accepted time and
pending native clock differ. A future timestep comparison must respect that
clock rather than relabeling or converting histories. No restart or finer-step
trajectory is launched in this task.

The failed spatial target, earlier energy-screen failures, unresolved branch
findings and `strict_reference_match_passed=false` remain. Current C_E is
measured, but C_T and growth-speed accuracy for this interval are unqualified.
