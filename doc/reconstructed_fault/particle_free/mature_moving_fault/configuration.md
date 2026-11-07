# Runnable examples and code navigation

[Overview](../MATURE_MOVING_FAULT.md) · Prerequisite: [attempt lifecycle](lifecycle.md)
· Optional next: [retained alternatives](alternatives.md)

## Complete bounded examples

Run from the repository root in the existing configured build environment.
The maintained [run_material_native.py](../../../../benchmarks/reconstructed_fault/particle_free/transported_objective/run_material_native.py)
**generates a complete input**, fault file and isolated output directory. It does
not read an old developer run or a checkpoint unless `--restart` is explicitly
supplied. Its dependencies are Python 3, the configured ASPECT executable, native
checker library and the local MPI runtime. This repository's bounded runner uses
`/opt/openmpi/5.0.6`; adjust that environment-specific runner path when porting.
These are tiny nondimensional-scale fixtures in SI units, not BP3 inputs.

Build core and matching checkers sequentially (at most four workers):

```sh
# Preserve retained executables before rebuilding (see the correction report).
cmake --build build-particle-free/native --target aspect.exe.release --parallel 4
# Install the tested candidate at stage-c/bulk-spin/aspect-release, then restore
# the retained canonical binary if it is a protected comparison executable.
cmake -S benchmarks/reconstructed_fault/particle_free/transported_objective/native \
  -B build-particle-free/transported-objective/stage-c/bulk-spin/checks \
  -DAspect_DIR="$PWD/build-particle-free/native"
cmake --build build-particle-free/transported-objective/stage-c/bulk-spin/checks \
  --target transported_bulk_spin --parallel 4
```

For a newly configured environment, the checker project is
[transported_objective/native/CMakeLists.txt](../../../../benchmarks/reconstructed_fault/particle_free/transported_objective/native/CMakeLists.txt);
configure it with `Aspect_DIR` pointing to the ASPECT build, using the same
compiler/deal.II installation. No old output template is needed.

The constant-viscosity direct example (two ranks, two physical steps):

```sh
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/run_bulk_spin.py guide-direct \
  --representation 'direct physical stress' --profile-method 'bounded MacCormack' \
  --inflow 'accepted trace' --support 'full normal cross-section' \
  --phase '.35*max(0,1-(x/1.4)^2)^4*max(0,1-((y-.3*x)/.6)^2)^4' \
  --constitutive-spin 'bulk slip subtracted' --speed .4 --shear .5 --dt .05 --end .1 --level 4 --eta 1 \
  --ranks 2 --seconds 120 --energy --stage-energy --true-checks --audit
```

The layered-viscosity variant (four physical steps, otherwise the same model):

```sh
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/run_bulk_spin.py guide-layered \
  --representation 'direct physical stress' --profile-method 'bounded MacCormack' \
  --inflow 'accepted trace' --support 'full normal cross-section' \
  --phase '.35*max(0,1-(x/1.4)^2)^4*max(0,1-((y-.3*x)/.6)^2)^4' \
  --constitutive-spin 'bulk slip subtracted' --speed .4 --shear .5 --dt .05 --end .2 --level 4 \
  --eta '1+.5*tanh(4*x)' --ranks 2 --seconds 120 --energy --stage-energy
```

The smooth layer multiplies reference viscosity 10 Pa s; $G=10$ Pa. The domain
is $[-2,2]^2$ with a 16×16 mesh, initial fault from $(-.8,-.24)$ to $(.8,.24)$,
initial material-label spacing .1 m, growth disabled and a .1 s initialization
interval. Initial stress is zero. Prescribed left/right/bottom velocity combines translation and shear; remaining mechanical conditions are the runner's existing natural
boundary conditions. Incoming-age regularized friction remains enabled.

These examples select the corrected constitutive spin and shear; the historical
`dc`/`dl` consolidation regressions retained total spin and their original loading. Outputs are never overwritten;
choose a fresh label on repeat. The complete generated `input.prm` includes:

```text
# This excerpt is explanatory; the runner above creates the complete file.
set Bulk history representation = DG trial stress
set Stress rotation = Jaumann DG material Lie
set Constitutive spin = bulk slip subtracted
set Material stress representation = direct physical stress
```

`DG trial stress` is the retained legacy spelling selecting independent DG bulk
storage; it does not force auxiliary T storage. The separate material selector
chooses direct physical stress. Without it, `separated` remains default.
`Q1 remap`, zero tensor inflow and stationary-profile association also remain
default; this example explicitly selects the qualified optional combination.
Do not copy only this excerpt into a production restart.

## Native DGQ2 phase option

The historical examples above retain continuous Q1. The independent stress
space stays DGQ2 for both phase choices. To select the new persistent phase
field, include these native settings in the complete input:

```text
subsection Discretization
  set Use discontinuous phase discretization = true
  set Phase polynomial degree = 2
  subsection Stabilization parameters
    set Use limiter for discontinuous phase solution = true
  end
end
```

The native implementation reuses `Discontinuous penalty` for SIPG. Whole-cell
phase bounds, material SSPRK3 transport, explicit traces and constrained growth
are described in [DG design](../DGQ2_PHASE_FIELD_DESIGN.md). The native moving
fixture still disables growth. Q1 checkpoints cannot be loaded as DGQ2.

The selected four-case layered comparison uses the separately preserved binary
and matching checker/provider in `build-particle-free/dgq2-phase/candidate/`:

```sh
python3 benchmarks/reconstructed_fault/particle_free/dgq2_phase_field/run_coupled.py \
  plus plus-half minus plus-fine
```

This is a multi-hour, four-rank **selected comparison**, not a quick smoke test.
The launcher refuses to overwrite existing runs. To create a short fresh case
with the same physical conventions, use a new label:

```sh
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/run_layered_campaign.py dg-guide \
  --candidate build-particle-free/dgq2-phase/candidate \
  --phase-discretization DGQ2 --spin 'bulk slip subtracted' \
  --background-shear 3.7 --translation 2 --halfwidth 6 \
  --level 4 --dt .0125 --end .025 --ranks 4 --seconds 180
```

`--phase-discretization Q1` deliberately selects the historical control. The
runner records exact executable/library/input hashes. Native operator tests and
fixed-fault cohesive growth use the drivers in
[`dgq2_phase_field/`](../../../../benchmarks/reconstructed_fault/particle_free/dgq2_phase_field/);
[report](../DGQ2_PHASE_FIELD_REPORT.md) records their measured scope. Neither
these fixtures nor the layered campaign specify the dimensional BP3-like crustal
paper example.

## Checks and outputs

Find each run under
`build-particle-free/transported-objective/stage-c/material-runs/<label>/`:

| File | Meaning |
|---|---|
| `input.prm`, `fault.txt` | Complete resolved inputs |
| `command.json` | Arguments, command, binary/plugin/input hashes, elapsed time and exit code |
| `run.log` | Residuals, linear/outer iterations, final gates and optional energy observations |
| `output/material-summary.csv` | Accepted clocks, history/geometry publication counts and invariant diagnostics |
| `output/material-*.csv` and other checker exports | Bulk tensor/profile and surface state used by the comparison reducers |
| `output/true-stress-source.csv` (direct analytical option) | Independent source projection, decomposition and directional derivative checks |
| `output/mature-projection-inputs.csv` | Explicit profile/gradient input check against an analytic constant tensor |
| `output/restart/` | This fresh fixture's checkpoint, with existing metadata |

Exact exported filenames are recorded by the run; the regression
[comparison](../../../../benchmarks/reconstructed_fault/particle_free/transported_objective/check_stress_history_consolidation.py)
checks physical bulk/surface values, tensor norms, work traction and publication
counts. [New-task report](../STRESS_CONSOLIDATION_DOCUMENTATION_REPORT.md) records
actual runs/results, including independent spatially varying source checks. A
successful process alone does not establish projected-traction or physical
energy accuracy.

## Equation-to-code and modification map

| Quantity/operation | Main entry point | Change here when… |
|---|---|---|
| $\beta,\kappa,F$ | [stress_law.h](../../../../include/aspect/material_model/phase_field_fault/stress_law.h), [stress_reaction.cc](../../../../source/material_model/phase_field_fault/stress_reaction.cc) | Changing the selected material reaction; audit residual/tangent/publication together |
| $2\mathcal L[a]$ | [localize_surface_values](../../../../source/material_model/phase_field_fault/dg_objective.cc) | Changing interpolation/association requires corresponding work/normalization checks |
| $A_h$ | [TensorAdvection](../../../../source/material_model/history/tensor_advection.cc), [distributed rows](../../../../source/material_model/history/tensor_advection_distributed.cc) | Changing numerical transport or inflow treatment |
| $P_h$ | [TensorField::project](../../../../source/material_model/history/tensor_projection.cc) | Changing common mass algebra, field count, ghost publication |
| $Q\tau Q^T$ | [objective_stress.h](../../../../include/aspect/reconstructed_fault/objective_stress.h), [stress_evolution.cc](../../../../source/material_model/phase_field_fault/stress_evolution.cc) | Changing objective integration/order; preserve full history coverage |
| $X^+,\phi^+,I^+$ | [material_geometry.cc](../../../../source/material_model/phase_field_fault/material_geometry.cc), [profile_transport.cc](../../../../source/material_model/phase_field_fault/profile_transport.cc), [manager.cc](../../../../source/reconstructed_fault/manager.cc) | Changing material geometry/profile; retain joint rollback and obstacle |
| Friction/age | [fault_friction.cc](../../../../source/material_model/rheology/fault_friction.cc), [surface_system.cc](../../../../source/reconstructed_fault/surface_system.cc) | Changing friction response or its derivative/age convention |
| Physical stress access | [physical_stress.cc](../../../../source/material_model/phase_field_fault/physical_stress.cc) | Adding an observer; avoid reconstructing T/R/B in a mechanics consumer |
| Coupled correction/inner solve | [reconstructed_fault_stokes.cc](../../../../source/simulator/solver/reconstructed_fault_stokes.cc) | Changing outer convergence or mechanical solve; keep terminal rates and gates |
| Accepted transaction | [core.cc](../../../../source/simulator/core.cc), [material_history.cc](../../../../source/material_model/phase_field_fault/material_history.cc) | Changing lifecycle/ownership; test rejection and restart |

A direct-mode first reading can skip stationary native-source roundoff machinery,
separated B/R reconstruction, prescribed PF5 transport controls, cohesive
formed-side/lifting histories and production BP3 setup. They remain supported
only within their separately documented restrictions; see [alternatives](alternatives.md).

The new `run_bulk_spin.py` wrapper selects the dedicated tested candidate core
and matching checker and defaults its explicit constitutive selector to bulk
spin. `--constitutive-spin 'total material'` produces a paired historical-law
control. The generic runner retains its previous defaults and accepts explicit
`--core`/`--checker` overrides. Never infer a retained binary's law from HEAD.
Native spin/rotation references are written to `bulk-spin-native.csv`; the
throughgoing prescribed fixture additionally writes `bulk-spin-prescribed.csv`.
The bounded matrix and independent reducer are `bulk_spin_matrix.py` and
`assess_bulk_spin.py` beside the runner. Their inputs are synthetic mature-fault
cases, not production BP3 or cohesive-interface qualification.
