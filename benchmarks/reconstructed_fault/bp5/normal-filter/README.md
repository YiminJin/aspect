# Experimental BP5 friction-normal filter

Status: **local operator/coupling verification; actual-checkpoint restart
comparison prepared, not executed.** See `report.md`. Filtering is not a
production default or a qualified physical BP5 regularization.

**Direct server execution (no Python launcher or schedule preparation):** use
[`server/README.md`](server/README.md) and the complete `server/R.prm`,
`server/F1.prm`, `server/F2.prm`. R now exports its actual accepted intervals
directly; F1/F2 consume that file. The Python-runner instructions below describe
the earlier optional staging layout and are not required for server execution.

## Build

Use the ASPECT/deal.II/Boost/compiler environment compatible with your checkpoint.
The recovered server checkpoint cannot be deserialized by the current local
environment; do not edit its archive or reset history to work around this.

From the source root (substitute your build directory):

```sh
cmake --build build-pf-cpdi --target aspect -j4
cmake -S benchmarks/reconstructed_fault/bp5 -B benchmarks/reconstructed_fault/bp5/build -DAspect_DIR="$PWD/build-pf-cpdi"
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_steady_initialization bp5_normal_stress_diagnostic -j4
```

The core changes are in `include/aspect/material_model/phase_field_fault.h`,
`source/material_model/phase_field_fault.cc`,
`include/aspect/reconstructed_fault/surface_system.h`,
`source/reconstructed_fault/surface_system.cc`, and the new file-local
`source/reconstructed_fault/normal_filter_internal.h`.
Rebuild the core AND both plugins; the response/diagnostic structures changed.
The benchmark changes are `normal_stress_diagnostic.cc`, `normal_filter_clock.cc`,
`CMakeLists.txt`, and this directory. No BP3 constitutive formula was changed.

## Prepare independent copies

`R.prm`, `F1.prm`, and `F2.prm` are complete inputs, without PRM includes.
They preserve the resolved experiment-A physical settings, ordinary particles,
distance-weighted transfer, true normal stress, eight-panel I_h projection,
boundary loading, tolerances and controllers. The runner clears inherited
`ASPECT_*` investigation selectors, enables qualified sparse B/G and the
pivoted inverse, and enables opt-in timing. Filtered G automatically uses its
exact composed nonlocal action instead of the old local sparse G.

```sh
python3 benchmarks/reconstructed_fault/bp5/normal-filter/run.py prepare \
  --checkpoint /absolute/original/output/restart/01 \
  --fixture /absolute/original/job/fixture \
  --libraries "$PWD/benchmarks/reconstructed_fault/bp5/build" \
  --destination /absolute/new/filter-study
```

Supply the ORIGINAL fault, target-cell and completion files in `fixture`.
Checkpoint identity is checked at step 5612, time `5310111071.5634108` s.
Hashes of every source checkpoint file, staged input and library are recorded;
no hard links or checkpoint clock edits are used. Do not reuse an output directory.

## Execute and assess

```sh
python3 benchmarks/reconstructed_fault/bp5/normal-filter/run.py run /absolute/new/filter-study R /absolute/build/aspect-release mpirun -np 32
python3 benchmarks/reconstructed_fault/bp5/normal-filter/analyze.py /absolute/new/filter-study
# Inspect offline.json and offline.png before F1/F2: compression, overshoot,
# broad profiles, and the direct friction-load change must be acceptable.
python3 benchmarks/reconstructed_fault/bp5/normal-filter/run.py run /absolute/new/filter-study F1 /absolute/build/aspect-release mpirun -np 32
python3 benchmarks/reconstructed_fault/bp5/normal-filter/run.py run /absolute/new/filter-study F2 /absolute/build/aspect-release mpirun -np 32
python3 benchmarks/reconstructed_fault/bp5/normal-filter/analyze.py /absolute/new/filter-study --evolution
```

Replace the MPI launcher consistently with `ibrun` if required. Each process is
capped at 7200 s and ten accepted states. No automatic retry or continuation.
R records constitutive `dt` values; F1/F2 add those as a MIN cap and reject any
schedule shortened by the existing safety controllers. Comparisons use summed
actual intervals, not differences of rounded absolute timestamps. Convergence
success and fresh-linear flags are required, not exit zero alone.

If a branch fails to accept this common clock, the single permitted retry is:

```sh
python3 benchmarks/reconstructed_fault/bp5/normal-filter/run.py half-retry /absolute/new/filter-study /absolute/new/filter-study-half
```

Then execute R/offline/F1/F2/analysis on that new directory. It runs TEN
half-sized intervals (a shorter common interval), not twenty replacement steps;
the original attempt plus retry therefore cannot exceed twenty accepted states
per branch. All branches restart from the original histories. A second retry
is rejected. Do not use this to bypass another type of physical/solver failure.

## Interpretation and output

`raw` bypasses filtering; `projected` solves M z=b; `helmholtz` solves
(M+L² K)z=b. L=0 is projection, not raw. Select modes under
`Postprocess / BP5 normal diagnostic / Friction normal input` and
`Normal filter length` (metres). F1/F2 provisionally use 100/200 m. Their
acceptability on the actual checkpoint remains unassessed; 200 m is not
automatically preferable. No clipping, window recentering or state smoothing.

If 200 m plainly oversmooths broad features, `analyze.py --extra50` adds only
the authorized 50-m offline fallback to the same data. Before staging any
filtered runs, `run.py prepare --secondary-length 50` selects 100/50 m in a
new package. Do not repeat R merely for packaging: the unchanged R output,
execution record, actual clock and offline assessment may be copied together
after checking their checkpoint/input/library/binary hashes. No other lengths
or loading scans are authorized.

The initial offline export is explicitly the **first resumed Newton base**:
checkpoint bulk unknowns/rates and committed histories evaluated with the
pending constitutive timestep. It is not a reconstructed accepted-step-5612
current stress (that precommit quadrature state is not in the checkpoint).
All four offline representations use this identical captured mechanical field,
its actual mu, and production M/K/M_mu. No history advances during capture.

`normal_initial_operator.csv` contains full-fault matrices and raw loads;
`initial_normal_*` contains native window samples. Accepted-state files keep
raw p, stress tensors and history separate from `friction_normal` and the
filtered coefficients. `normal_profile`'s `weak_normal_sum` is raw; its
`existing_production_weak_normal` is the actual friction input's consistent
projection. `closure_error` checks the RAW pressure/deviatoric/background
projection identity. `normal_filter_*` provides the corresponding explicit
loads and coefficients, including residual/shear/friction/damping.

The analysis writes only `offline.png` and `evolution.png`, with JSON summaries.
Full-fault coefficient RMS uses consistent M; raw window RMS uses native work
weights. Retained-particle statistics use unique real particles with equal
particle weights and are labeled separately, not equated to work-QP statistics.
The chord metric includes smooth curvature. Friction attribution is skipped
if map/weight equality fails; it is never silently interpolated between maps.
The standard `accepted_steps.csv` retains iterations, active counts, line-search
alpha and fresh checks. Timer output separates `Fault: normal filter` and
`Fault: G normal filter`; compare with each branch's elapsed time.

Python analysis requires NumPy, SciPy and Matplotlib (not pandas).
`python3 .../normal-filter/test_analysis.py` exercises its offline pipeline on
synthetic data only. No synthetic result is evidence about BP5 behavior.
