# P4a native finite-tip diagnostic

[Report and scientific dependency](../../../../../doc/reconstructed_fault/particle_free/P4A_TIP_REPORT.md).
P3 was closed/committed at `f1f9f29cf`. The two P4a snapshots are frozen,
non-equilibrated native states. No phase/time advancement or physical-response
qualification is claimed. Zero local mechanical driving ahead of the finite
surface is the measured current-model limitation; no new activation law is
introduced.

`tip_quadrature.{png,pdf}` shows the actual native endpoint inside a bulk cell,
mesh, phase contours, normalization rays and every mechanical QP in the close-up.
`figure-measures.json` distinguishes point counts from JxW. Normal integration
is analytic; the rays carry separate surface Gauss weights.
`tip_fields.{png,pdf}` shows actual reconstructed stress and local mechanical
derivatives, with the complete normalization adjoint verified separately.
`aligned.json` and `cut.json` contain independent measured comparisons and
the obstacle KKT check. Physical phase/obstacle and every accepted owner stay
unchanged; the diagnostic never commits a state.

`reference.tar.xz` stores 46 actual native export/input/log files for the final
aligned and cut cases. `archive-manifest.json` hashes each unmodified export.
The archive is 4,402,588 bytes. No ASPECT binary is embedded. `provenance.json`
records the tested binary/plugin/source identities, pre-run screen document
and scope. `logged/` preserves first and final bounded commands and the fresh
default-P3 regression. `p3-default-control.json` records exact comparison of
16 physical/history/constraint exports with the qualified P3 original.
`preservation.json` verifies all 34 protected BP3 identities, manual AGENTS
work and unchanged production source/header/CMake files.

Portable mathematical replay (Python + NumPy, no ASPECT build):

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/p4a_tip/verify.py
```

Native reproduction uses the existing isolated evolving-phase core and
`p3_native_checks` plugin. The test DSO must not be loaded alongside another
fixture defining the same test-friend `MomentCycle`. Build only that target:

```sh
cmake --build build-particle-free/evolving-phase/plugin --target p3_native_checks -j2
```

The maintained native inputs are `p4a_aligned.prm` and `p4a_cut.prm`. Select
new output directories before repeating a run, preserving existing files.
The corresponding explicit diagnostic environment selects the endpoint and
must agree with the prescribed geometry file:

```sh
ASPECT_P4A_TIP=0.5 ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release benchmarks/reconstructed_fault/particle_free/p4a_aligned.prm
ASPECT_P4A_TIP=0.515625 ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release benchmarks/reconstructed_fault/particle_free/p4a_cut.prm
```

The actual qualification commands in `logged/` use a 180 s timeout and final
output directories `build-particle-free/p4a-tip/final/{aligned,cut}`. First
observation exports remain under `build-particle-free/p4a-tip/{aligned,cut}`.
The original P3 plugin is pinned separately at
`build-particle-free/p4a-tip/p3-pinned/`; P3's raw outputs and archive are intact.
No BP3 executable/package was rebuilt or launched.

Independent checks / figure regeneration from actual raw native data:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_p4a_tip.py build-particle-free/p4a-tip/final/cut --output /tmp/p4a-cut.json
MPLCONFIGDIR=/tmp/p4a-mpl OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_p4a_tip.py build-particle-free/p4a-tip/final/cut --output /tmp/p4a-figures
```

The endpoint cut coincides with a native tangential-panel boundary, despite
lying inside the bulk cell. This qualifies that bounded case; arbitrary
off-panel endpoints, finite-tip mechanical solves, physical phase growth,
I=0 activation, nonzero-tip joining and general propagation remain unqualified.
