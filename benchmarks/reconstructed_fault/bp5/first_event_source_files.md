# Sources for the eight-panel loading-driven first-event configuration

This inventory describes the tested working tree based on
`359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`. Rebuild both ASPECT and the selected
plugin against the same headers/compiler/MPI/deal.II installation. Do not copy
the local Linux shared library as a portable server binary.

## New since `server-30km-loading`

ASPECT source/header differences:

1. `include/aspect/material_model/phase_field_fault.h`: stores the fixed
   surface-quadrature subdivision count.
2. `source/material_model/phase_field_fault.cc`: declares/parses that parameter,
   constructs composite three-point surface quadrature and validates completion
   counts against it. The Q1 space/mass and normal tolerances are unchanged.
3. `source/reconstructed_fault/surface_system.cc`: the earlier normal-feedback
   control supports prescribed frictional pressure with the bulk work measure
   and propagates that choice into G. The first-event configuration keeps
   **true normal-stress feedback**, so this control is not selected.

Plugin source/header differences:

1. `benchmarks/reconstructed_fault/bp3/bp3.cc`
2. `benchmarks/reconstructed_fault/bp3/work_replay.h`
3. `benchmarks/reconstructed_fault/bp5/steady_initialization.h`

These three incremental plugin differences are guarded normal-control diagnostic
hooks. They are inactive in the server target, which defines only
`ASPECT_BP5_STEADY_INITIALIZATION`, not `ASPECT_BP5_NORMAL_CONTROL`.
**No new plugin runtime edit is needed to select eight panels.** Selection is
through the parameter file plus matching regenerated completion data.

## Complete ASPECT working-source overlay relative to the base revision

The packaged `source.patch` retains all nine changed runtime source/header files,
not just the two new normalization files. This avoids losing earlier qualified
initialization and bound-interpolation corrections:

```
include/aspect/material_model/phase_field_fault.h
source/material_model/phase_field_fault.cc
include/aspect/material_model/rheology/fault_friction.h
source/material_model/rheology/fault_friction.cc
include/aspect/reconstructed_fault/utilities.h
source/reconstructed_fault/utilities.cc
source/reconstructed_fault/manager.cc
source/reconstructed_fault/surface_system.cc
source/simulator/assemblers/reconstructed_fault_stokes.cc
```

The friction files contain the earlier configured-law inverse-state helper;
the chosen loading initializer does not call that old inverse-state procedure.
The utility/manager/assembler changes retain bound-safe velocity interpolation.
Additional test changes are included in the source patch as provenance, not as
server runtime plugins. Apply the patch only to its recorded base; on an
already modified checkout inspect/apply the incremental changes instead of
blindly applying it twice.

## Plugin sources shipped and compiled

```
plugin/bp3/bp3.cc
plugin/bp3/bp3_model.h
plugin/bp3/first_event.h
plugin/bp3/output_schedule.h
plugin/bp3/mature_fault.h
plugin/bp3/work_replay.h
plugin/bp3/matched_resolution.h
plugin/bp5/steady_initialization.h
plugin/bp5/startup_time_step.cc
plugin/bp5/CMakeLists.txt
```

The initializer sets the authoritative projected-mixture nodal state with
`R_VW=0.8` and constructs the native weak background. The timestep plugin
retains the 0.02 weighted-log-state predictor. Event, output and restart code
is reused unchanged. No investigation plugin is loaded on the server.

## Inputs that must accompany the sources

- `I h surface quadrature subdivisions = 8` in both fresh and resume inputs.
- `Weakening region length = 30000`, with the existing 3000-m transition.
- The **27,720-profile** completion file, not the old 3,465-profile file.
- The unchanged prescribed fault and saved mesh inputs.
- Loading initial-state ratio 0.8, true normal-stress feedback, fully frictional
  mature fault and the existing physical/controller/solver settings.

The new short-check report and package qualification record determine launch
readiness; this source inventory alone is not an event or temporal-convergence
qualification.
