# GMG diagnostics and BP3 timestep cleanup — September 26, 2026

Requested scope: remove temporary production GMG debugging code after the
Intel 26.0 rebuild resolved the server failure, and remove BP5 timestep
settings from the maintained BP3 runtime. No fault equations, constitutive
updates, solver tolerances or standard timestep-cap values were changed.

## Changes

- Removed the diagnostic environment switch, per-rank log output, coefficient
  and diagonal observations, diagnostic MPI collectives, scratch-vector
  probes and eigenvalue exception logging from the two production GMG files.
  Those files now match their pre-instrumentation HEAD contents exactly; the
  working diff before removal contained only the inspected instrumentation.
  The GMG implementation, including its original positivity assertion,
  diagonal computation and eigenvalue estimation, is preserved.
- Moved `fault_gmg_diagnostics.h` out of `source/` into `server_gmg/`, with
  the two standalone probe includes adjusted. These opt-in reproduction tests
  remain buildable without installing debugging code into production.
  The runner now relies on Stage-I/probe verification and no longer enables
  or expects the removed production observer. Historical outputs are intact.
- Removed `state_startup.cc` from the BP3 library and moved its source to
  [retired_state_startup.cc](retired_state_startup.cc), outside any build target.
  The BP3 plugin no longer declares/registers `BP5 state startup` or its
  settings. Its two controller-audit CSVs are consequently no longer emitted;
  the independent accepted-step output still records the selected timestep.
- Removed the BP5 controller and its subsection from the maintained raw,
  filter20 and filter40 PRMs. First-event inherits the cleaned filter20 input.
  Updated the existing generator text to emit the same controller list;
  the Python generator was not executed and no new Python script was used.
- Kept the explicit rejection of BP5 checkpoints and experimental environment
  switches: these protect BP3 inputs and do not limit timesteps. Separate BP5
  benchmark sources/settings are outside this cleanup and remain intact.

## Server migration

Rebuild ASPECT with the working Intel 26.0 stack and rebuild the BP3 plugin
against that matching ASPECT build. Load the new BP3 library. Use:

```text
subsection Time stepping
  set List of model names = convection time step, reconstructed fault time step
end
```

Delete the entire old `subsection BP5 state startup`, including `Maximum
logarithmic state change` and `Record timestep selection`; merely removing
the plugin name leaves undeclared parameters in an old input. The standard
convection/fault-law restrictions and ASPECT first/global/growth caps remain.
There is no replacement BP5-style state-change restriction.

The maintained repository PRMs still have first/global caps `100 / 4e6` s.
Keep the user's newer `4e6 / 4e7` s caps when updating the server input if those
are desired. Saved `output-test-amg/original.prm` and `parameters.prm` are
execution evidence, not maintained inputs, and have not been edited.

## Verification

Local toolchain: GCC 12.4, OpenMPI 5.0.6, deal.II 9.6.2, Release. This is not
an independent Intel 26.0 server qualification.

| Check | Result / evidence |
|---|---|
| Rebuild modified ASPECT | Passed; [core-build.log](core-build.log) |
| Fresh standalone BP3 plugin build | Passed; [plugin-build.log](plugin-build.log) |
| Standalone GMG probe/helper and Stage-I plugin builds | Passed; [test-build.log](test-build.log) |
| Raw/filter20/filter40/first-event PRM parsing with rebuilt BP3 library | All passed; `validate-*.log` (except the intentionally obsolete input) |
| Add only the retired BP5 selector to an otherwise valid input | Rejected, as expected; [reject-bp5-selector.log](reject-bp5-selector.log) |
| Add only the retired BP5 subsection to an otherwise valid input | Rejected, as expected; [reject-bp5-subsection.log](reject-bp5-subsection.log) |
| Updated GMG runner, one-rank Cartesian Stage-I initialization | Passed; [stage-i/run.log](stage-i/run.log), final printed bulk/fault residuals `9.165224e-08 / 0`, verifier passed, normal end-time termination |
| Source and rebuilt BP3 symbol audit | No production GMG observer or BP5 startup registration remains |
| Shell syntax and `git diff --check` | Passed |
| Recorded supplied-output hashes | Unchanged; [output-preservation.log](output-preservation.log) |

`--validate` checks parameter declarations and syntax; it does not exercise a
full BP3 trajectory or restart. Negative parsing tests use ASPECT's quiet
failure path, so their logs show MPI abort rather than detailed parameter
messages. Both tests differ from the passing filter20 validation input only
by the indicated retired setting. The initially attempted saved-run negative
check is also retained in `validate-retired.log`.

No full-resolution BP3 run, long trajectory, checkpoint continuation, new
multi-rank run or server run was launched. The Stage-I initialization is a
regression check of newly cleaned GMG code, not a repeat of the earlier fault
localization experiments. The logs contain sandbox network-interface warnings;
positive checks and Stage I nevertheless complete successfully.

## Preservation

[before.tar.gz](before.tar.gz) preserves the edited source/configuration files
as they stood before cleanup, including the complete BP3 plugin directory.
[removed-core-diagnostics.patch](removed-core-diagnostics.patch) preserves the
removed production observer implementation. Existing uncommitted changes in
other source files were not reverted. Reports, archives, run outputs and
checkpoints were not cleaned or overwritten.

[binary-hashes.txt](binary-hashes.txt) identifies the local rebuilt executable
and libraries; [stage-i/provenance.txt](stage-i/provenance.txt) records the new
Stage-I run. Full transient build/run directories are under
`/tmp/aspect-bp3-gmg-cleanup-build`, `/tmp/aspect-gmg-cleanup-tests`, and
`/tmp/aspect-gmg-cleanup-stage-i`; compact evidence is retained here.
