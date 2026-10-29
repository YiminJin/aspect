# One bounded mesh-orientation fixture

Run from this worktree root. The selected core and original fixture binary are
not rebuilt or overwritten. Only the separate diagnostic plugin is built:

```sh
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/make_orientation_mesh.py
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/bulk-orientation/plugin -DAspect_DIR="$PWD/build-particle-free/evolving-phase/native" -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/bulk-orientation/plugin --target bulk_orientation_checks -j2
```

The mesh generator verifies all bilinear determinant corners before writing
the AVS/UCD input. The mesh is fixed during physical time. Its core has exactly
the old cell sizes, rotated 45 degrees; the annulus and outer rectangle are
conforming. The zero function topography selects native MappingQ1 without
changing the domain. No MappingCartesian approximation is admitted.

These were the direct native commands, each bounded by the existing
`refactoring_boundary/run_logged.py` timeout wrapper (120 / 600 seconds):

```sh
env ASPECT_BULK_ORIENTATION_IDENTITY=1 ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_PROC_BIND=false OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-particle-free/evolving-phase/native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/orientation_identity.prm
env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 ASPECT_BULK_CHECK_TANGENT=1 OMP_PROC_BIND=false OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-particle-free/evolving-phase/native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/orientation.prm
```

Keep the original outputs. A repeat must use an empty output directory via a
separate input override. The identity control exits successfully. The rotated
run **exits nonzero at .1075 s** on the unchanged independent source check,
after 42 accepted publications. It is not a post-onset orientation result.
The test-only identity environment flag controls the benchmark geometry only.

Read-only analysis and plotting of the retained data:

```sh
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/assess_orientation.py benchmarks/reconstructed_fault/particle_free/evidence/bulk_orientation
env MPLCONFIGDIR=/tmp/bulk-orientation-mpl python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_orientation.py benchmarks/reconstructed_fault/particle_free/evidence/bulk_orientation
```

The independent gradient projector uses tensor-product quadratic Lagrange
polynomials in a different ordering from FE_Q, computes the bilinear Jacobian
directly and solves full nine-mode mass systems. Its affine control and
worst-transition error agreement are mandatory checks. Physical diagnostics
decode accepted local histories in numeric CellId order and use each original
QP/JxW. Extra diagnostic quadrature reintegrates Q1 phase/fracture only, never
a local history or constitutive law. Common radial kernels are distinct from
native nodal KKT conditions; all native accepted phase vectors remain unchanged.

The evidence archive contains the exact selected raw states, geometry,
phase-force exports and these scripts at repository-relative paths. Extract to
an empty directory and run the assessor with a fresh output directory for a
read-only replay. No ASPECT executable is required for analysis replay.
The native source/kernel audit, valid/failed logs, preservation identities and
limits are in `doc/reconstructed_fault/particle_free/BULK_ORIENTATION_REPORT.md`.
The area observer in this DSO is explicitly not a normal-line surface observer.
