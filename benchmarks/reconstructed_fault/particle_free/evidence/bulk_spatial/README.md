# One aligned-wide spatial comparison

[Completed report, table and figures](../../../../../doc/reconstructed_fault/particle_free/BULK_SPATIAL_REPORT.md).
Final cap-growth L2 differences are 35.86% on both flanks, with onset seven
timesteps earlier on the finer mesh. Quantitative spatial agreement fails;
broad physical driving-lobe directions persist. `assessment.json` retains all
common-time metrics; `interpretation.json` records the declared-target decision.
`*-fields.npz` retains every physical phase snapshot and original-QP angular
averages. PNG/PDF maps use equal scales across meshes and across domain/zoom.
Native Y values occupy constant midpoint bins for display; these bin areas
are not quadrature weights and do not enter any quantitative measure.

[Selection](../../../../../doc/reconstructed_fault/particle_free/BULK_SPATIAL_SELECTION.md)
fixes one fresh 96x128 trajectory and reuse of the existing 48x64 trajectory,
both at Delta t=.00125 s through .16 s. The physical domain, continuum seed,
length scale, loading, Gauss8/source3, law and native tolerances are unchanged.
`parameter-comparison.json` checks all parsed values: only mesh repetitions,
diagnostic DSO and output directory differ. No history is transferred between
meshes or local quadrature layouts.

Original native data remain at:

- `build-particle-free/bulk-energy/runs/half` (retained coarse trajectory).
- `build-particle-free/bulk-spatial/runs/refined-qualified` (fresh fine trajectory).
- `build-particle-free/bulk-spatial/runs/refined` (initialization-only failed
  observer, no positive physical step; retained with its log).

The first observer sequentially summed 786432 positive JxW values in double
precision and failed its 1e-12 area check. `area-diagnosis.json` shows the
specific error and independent accurate sums. Only that diagnostic accumulator
uses extended precision in the separate spatial DSO; all old DSOs remain intact.
No weight, normalization, law, coverage check or tolerance changes.

Fresh initialization and physical Q1 fields use each mesh's native nodes.
Common growth comparisons subtract each trajectory's own physical initial
field, then evaluate both increments on the same fine physical Gauss8 points.
`initial-reference.json` independently measures each representation versus the
continuum seed with diagnostic Gauss24 and Gauss32. These diagnostic integrations
do not change native quadrature. `initial-common.json` records initial physical
field differences; shared vertices agree exactly. Growth-relative errors are
undefined when the finer growth vanishes, with one-mesh-only growth labeled
explicitly. No mirror registration or first-node matching is used.

Driving Y is calculated from original local q and phi at each mesh's own native
mechanical points before applying the same physical kernel of radius .0625 m.
`kernel-check.json` records normalization accuracy and point counts; positive
physical weights determine the average, not point counts. Local constitutive
histories are never interpolated into another state. Phase interpolation for
plots and common-point comparison is evaluation of the physical Q1 field only.

Reproduce the added native diagnostic build in the matching retained worktree:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free \
  -B build-particle-free/bulk-spatial/plugin \
  -DAspect_DIR="$PWD/build-particle-free/evolving-phase/native"
cmake --build build-particle-free/bulk-spatial/plugin --target bulk_spatial_checks -j 2
```

The native core is unchanged, SHA256
31ad1e59820079e712bdae23732697394cbfe36e8377001fa50cdac2a05b8d5a.
In a **fresh** output directory, the direct launch is:

```sh
env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 \
    ASPECT_BULK_CHECK_TANGENT=1 OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
    build-particle-free/bulk-source-history/native/aspect-release \
    benchmarks/reconstructed_fault/particle_free/bulk_cohesive/spatial_refined.prm
```

Do not overwrite the retained runs. The actual evolving run is bounded by
14400 wall seconds via the existing `refactoring_boundary/run_logged.py`
wrapper. Parsing is also checked with `--validate`. Native exact BE accounting,
source, mechanical/KKT and once-only publication checks remain active.
Postprocessing retains endpoint/trapezoidal work and separate numerical terms;
the historical 1% endpoint screen is not silently replaced.

Data-only reassessment and plots, from the worktree root:

```sh
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/assess_spatial.py \
  /tmp/bulk-spatial-replay
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_spatial.py \
  /tmp/bulk-spatial-replay
```

Python needs NumPy, SciPy and Matplotlib. Processed arrays retain every native
phase snapshot and selected original-QP angular averages. Raw exports remain
in their original directories; this comparison does not duplicate all previous
raw archives. Native restart/rollback infrastructure is unchanged and its prior
qualification is retained; this is not a new restart/MPI campaign. Two meshes
alone establish neither a spatial convergence order nor general propagation
qualification. BP3 and all prior spatial/orientation/energy findings are retained.
