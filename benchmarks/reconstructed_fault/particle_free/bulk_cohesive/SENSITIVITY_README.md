# Fixed-law flank and right-boundary controls

These are the three user-selected fresh serial trajectories documented in
`doc/reconstructed_fault/particle_free/BULK_SENSITIVITY_SELECTION.md`.
Use the existing bulk-model executable/plugin; no C++ or production input
changes are required. Preserve prior outputs and use fresh logger labels and
output directories when replaying an already completed case.

Run from the repository root, one native case at a time:

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py bulk-sensitivity-fine-gauss8 1800 env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 ASPECT_BULK_CHECK_TANGENT=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/fine_gauss8.prm
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py bulk-sensitivity-square 900 env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 ASPECT_BULK_CHECK_TANGENT=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/square.prm
```

Inspect the numerical controls before the one boundary-distance case:

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py bulk-sensitivity-right-distance 2400 env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 ASPECT_BULK_CHECK_TANGENT=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release benchmarks/reconstructed_fault/particle_free/bulk_cohesive/right_distance.prm
```

All cases start from the same continuum seed and zero histories. The extended
domain is [0,1.5]×[0,1] m with 48×64 cells, dx=1/32 m, dy=1/64 m and Gauss8.
The boundary velocity depends on y and time only, and is unchanged on all four
sides. The enlarged top/bottom portions receive that same affine prescription.
No accepted quadrature histories are transferred between cases.

`assess_sensitivity.py RUN --output JSON` computes initialization error,
regional increments, contour positions, physical energy and native KKT gates.
It independently assembles the physical weak fracture contribution using
exact Gauss2 Q1 integration and the mechanical contribution using actual
exported Gauss points/JxW; it checks the total against native phase-force CSVs.
For the unit or widened domain it verifies actual area coverage, not a hardcoded
unit-area assumption. It does not modify the older unit-square assessment.

Use `--compare REFERENCE --kind quadrature|spatial|distance` to compare common
physical times and the common physical domain. Phase Linf is evaluated on the
union of both rectilinear Q1 meshes; this captures extrema of their difference.
Initial differences and evolution-increment differences are separate. Nodal
weak densities g/mass are compared as diagnostic FE functions, not constitutive
history transfers. For a distance comparison, the first case's right face
x=1 m is an interior cut in the second case; each case's actual boundary
statistics and plots separately use its own x=Lx.

`plot_sensitivity.py EVIDENCE_DIRECTORY` reads the five actual trajectories
(two retained baselines plus three new runs) and exports three PNG/PDF figures
and a line-cut CSV. All maps share physical axes and color scales. The driving
maps show actual Gauss Y; line cuts show N-weighted Y=integral(NY)/integral(N).
Fracture/total rows show native weak nodal densities [Pa]; the underlying raw
g_f, g_m and g are retained in phase-force CSVs [J/m]. Neither representation is
silently substituted for a pointwise phase equation.

`../evidence/bulk_sensitivity/` retains the compact report data, commands,
identities, selected actual exports/checkpoints and checksums. Full raw outputs
remain in `build-particle-free/bulk-sensitivity/runs/`; retained baselines remain
under `build-particle-free/bulk-cohesive/runs/`. Original physical energy-screen
failures and the previous timestep control are preserved, not rerun or tuned.
