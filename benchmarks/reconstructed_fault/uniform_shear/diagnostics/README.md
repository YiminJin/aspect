# K1 initialization diagnostic packet

The approved ownership correction and old/current deduplication comparison are
recorded in `doc/reconstructed_fault/benchmarking/stage_K1_ownership_correction_review.md`.
Open `results-corrected/pre_mechanics_bulk.vtu` together with
`results-corrected/pre_mechanics_fault.vtu` for the corrected initialization;
`results-corrected/phase_and_fault.png` and `transverse_phi.csv` summarize it.
The original `results/` packet is retained as the **pre-correction baseline**;
its three integrity tests intentionally retain the original 68-failure checks.
`compare_ownership.py` checks the corrected data and preservation invariants.
Neither packet contains an accepted mechanics solution.

Follow-up: `doc/reconstructed_fault/benchmarking/stage_K1_cpdi_cause_and_correction_plan.md`
establishes the ownership-gap cause using captured production polygons,
proposes the minimum correction, and separately audits the stationary phase
equation including activation. That document records the pre-approval investigation.
The added `check_cpdi_cause.py` and `check_stationary_equation.py` scripts
reproduce that evidence without solving or modifying the phase field.

This is an **initialization-only diagnostic**, not a rerun or acceptance of
K1 mechanics. Production algorithms, parameters and tolerances were unchanged.
The initial phase field reproduces the reported pilot values exactly. An
upstream CPDI consistency failure is present before the phase solve; see
`doc/reconstructed_fault/benchmarking/stage_K1_diagnostics_report.md`.

Persistent packet directory:

`/home/ein/repository/aspect/benchmarks/reconstructed_fault/uniform_shear/diagnostics/results/`

## Open in ParaView

Open these files together and click **Apply**:

- `pre_mechanics_bulk.vtu`: actual 1,024-cell native Q1 mesh, unsmoothed nodal
  `phi`, native vertex IDs, and exact Q1 gradients at cell centers. Choose
  **Surface With Edges**, color by `phi`. Do not apply smoothing filters.
- `initial_particles.vtu`: actual 9,216 initial positions, integer particle
  IDs, `H`, `domain_volume`, and CPDI consistency diagnostics. Use **Points**
  or a small Sphere glyph. Zoom to y in [-0.016,0.016] to see defective stencils.
- `pre_mechanics_fault.vtu`: production fault exporter, actual 9 vertices and
  8 segments. Includes initialized cohesive traction, initial Theta and
  `phase field fault previous I h` (the initialized I_h snapshot). These are
  initialized histories, **not an accepted mechanical solution**. V_min is
  deliberately not exported as slip rate.
- `fault_normals.vtu`: same actual polyline with cell-data segment normals.
  Use Cell Centers then Glyph for normal arrows if desired.
- `nominal_y_zero.vtu`: separate nominal straight line. Color it distinctly
  and use Wireframe/line width to distinguish it from reconstructed geometry.

`independent_discrete_diagnostics.vtu` is a separate **calculated diagnostic**:
initial residual after periodic elimination and a direct-solve first Newton
direction from exported production stencils. It is not a captured production
iterate. The periodic slave's diagnostic value duplicates its master's for
display; it is not an unconstrained assembled residual entry.

`phase_and_fault.png` and `particles_and_discrete_problem.png` are preview
plots. Native FE nodes, fault vertices and actual particle positions remain
visible. CSV profiles are `centerline.csv`, `along_fault.csv` (fault vertices),
`along_fault_mesh_samples.csv`, `transverse_phi.csv` and `initial_H.csv`.
The stationary AT1 reference is explicitly labeled **intended**, not imposed
as a phase-field constraint. The x-row-mean subtraction in the residual preview
is only a visualization diagnostic; it never changes the simulated field.

`raw/` preserves the original production DataOut bulk files, fault exporter
file, mesh connectivity, phase nodes, full CPDI stencils, particle data,
constraints, effective parameters and log. `particle_stencil_checks.csv`
contains the per-particle numerical checks; `measurements.json` summarizes them.

Accepted t=0 and t=2 velocity/pressure/V/Theta files from the earlier pilot
were stored only under `/tmp/aspect-k1-pilot`, which is absent in this session.
Those accepted states cannot be recovered from summaries. **No accepted
mechanics visualization is supplied or fabricated**, and mechanics was not
rerun to replace it.

## Reproduce the diagnostic

From `/home/ein/repository/aspect`:

```sh
cmake -S benchmarks/reconstructed_fault/uniform_shear/diagnostics \
  -B benchmarks/reconstructed_fault/uniform_shear/diagnostics/build \
  -DAspect_DIR=/home/ein/repository/aspect/build-pf-cpdi
cmake --build benchmarks/reconstructed_fault/uniform_shear/diagnostics/build -j4
cd benchmarks/reconstructed_fault/uniform_shear/diagnostics/build
set -o pipefail
timeout 300 /home/ein/repository/aspect/build-pf-cpdi/aspect ../initialization.prm \
  2>&1 | tee initialization.log
```

The successful diagnostic ends with **exit 1 and the explicit marker
`K1_INITIALIZATION_DIAGNOSTIC_COMPLETE`**, after files have been closed.
The exception deliberately exercises existing rollback before the first
mechanical residual. Exit 124 is a timeout, not a successful export. Preserve
existing output before another run. No positive timestep is executed.

The parameter overlay changes only library/output/postprocessor selection.
It includes the original pilot parameters verbatim. The original pilot
postprocessor is linked to satisfy parameter registration but is not run;
its positive-step phase-freezing slot is inactive throughout this diagnostic.
All initialization data are read through existing public APIs/signals. No
new production interface, friend access or production source edit is needed.

Recreate plots/VTU diagnostics from saved raw output, from repository root:

```sh
MPLCONFIGDIR=/tmp/k1-matplotlib PYTHONDONTWRITEBYTECODE=1 python3 \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/visualize.py \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/build/output-initialization \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/results
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s benchmarks/reconstructed_fault/uniform_shear/diagnostics -p test_packet.py -v
```

Python requires NumPy, SciPy, Matplotlib and VTK's common/XML modules. The
script deliberately imports only non-rendering VTK modules: this installation's
top-level `vtk` import otherwise requires an unavailable OpenVR library.
The packet tests validate file readability/finiteness, unchanged initialization
inputs and faithful recording of the discrepancy; they are not production
acceptance tests legitimizing defective CPDI stencils.
