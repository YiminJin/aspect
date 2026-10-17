# PF6a bounded spatial diagnostic evidence

[Report](../../../../../doc/reconstructed_fault/particle_free/PF6A_SPATIAL_REPORT.md),
[pre-run formulation](../../../../../doc/reconstructed_fault/particle_free/PF6A_SPATIAL_FORMULATION.md),
[status](../../../../../doc/reconstructed_fault/particle_free/STATUS.md).
No production/constitutive/default/limiter changes. Unlimited DGQ2 retained.

- `frozen.json`: identical-field versus actually resolved mechanics; 32/64/128,
  compatible accepted FE versus DG, separate actual CPDI fracture, independent
  1D mechanical/projection/moment checks, common gradient and quadrature errors.
- `event.json`: four actual native common-entry steps, eight-step half-dt
  control, full tensor recurrence at nine nodes, phase KKT/irreversibility and
  single publication, independent high-rule energy/gradient integration and
  spatial versus temporal differences. Initial enriched-space KKT defects are
  retained; this import is not a restart or general transfer qualification.
- `frozen-screen.json`: measured decision before the one finer mesh and short
  event; `preparation.json`: nine-point compatible-history verification;
  `parent-continuation.json`: coarse common-entry event versus its parent final
  physical fields. `comparison.png/.pdf` are standalone diagnostic figures.
- `provenance.json`: exact core and observer groups, inputs in run records,
  native budgets/costs, archive hash. All 15 selected native runs succeeded.
  The pre-existing `pf6a-spatial-64-4-extended` belongs to the earlier PF6a task
  and is not relabeled as a run of this campaign.
- `preservation.json`: all 10,423 incoming files remain, with changes confined
  to the benchmark CMake/README and status/audit additions. Production and
  historical evidence bytes are preserved. New files are listed by the archive
  manifest; this task does not commit or push.
- `reference.tar.xz`, `archive-manifest.json`: raw native common-point component,
  moment/jump, gradient/energy/surface exports, logs, exact inputs/observer
  source, source manifests/patches, common data, all source velocities needed
  to reproduce the preparation, and independent reference source closure.
  Identical contents are archived as internal hardlinks. Pinned binaries/DSOs
  remain in the original build run directories and are identified by hashes.
  Frozen first-group source is `diagnostic/observer-first.cc`; later frozen
  group source is `diagnostic/observer-frozen.cc`; current source is the event
  observer. Native core is identical throughout.

Replay without native execution, from any directory, with Python, NumPy, SciPy
and Matplotlib installed:

```sh
python3 benchmarks/reconstructed_fault/particle_free/evidence/pf6a_spatial/replay.py \
  /tmp/pf6a-spatial-UNIQUE-replay
```

The output directory must not exist. Replay verifies every archived file hash,
recomputes common fields (byte-identical text), independent frozen/event
assessments and figures. Numerical replay uses 1e-10 absolute/relative tolerance;
that does not change the original scientific acceptance screens. Original
native runs were serial, one thread, capped at 180 s each, at most two at once.
Their measured times include observer output and initialization. The compiler
include/type failures and preliminary assessment TypeError remain in
`diagnostic/` alongside their successful corrections. No failed evidence or
original PF0–PF6a/WENO artifact is overwritten.

Only this fixed-geometry diagnostic is completed. Particle-authoritative
history/transfer, general CPDI equivalence, AMR/MPI, moving geometry, fault birth,
transport and propagation remain unqualified. No further run is selected.
