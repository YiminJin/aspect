# Instructions to Codex: clean the maintained BP3 plugin and reduce output

Please implement a focused cleanup of the maintained BP3 plugin, its production parameter files, and its plotting readers. Inspect the current checkout and its repository instructions first. Preserve the working physical model, solver selection, boundary conditions, particle interpolation, normal-stress filter, timestep controller, and accepted-state update order.

These instructions are based on inspection of the supplied `plugin(1).tar`, containing `output.cc`, `monitor.cc`, `execution_environment.h`, and the other BP3 plugin files. The archive contains neither production PRMs nor plotting scripts; locate their actual maintained versions in the checkout. Treat the findings below as leads to verify against that checkout, rather than permission to delete similarly named core functionality.

## 1. Audit the environment switches before deleting them

`execution_environment.h` contains a rejection list, not a list of enabled features. `unexpected_execution_switch()` rejects a launch if any listed variable exists, including when its value is `0`. Removing an entry can therefore enable an inherited experiment to affect an ordinary BP3 run.

Search the matching ASPECT core, plugin sources, launch scripts, and maintained tests for actual consumers of each listed variable. Distinguish executable reads from documentation references. Produce a short table with the variable, consumer, purpose, and disposition:

- Remove entries whose consumers have actually been retired or cannot be loaded by the production executable.
- Retain a small, documented guard for live experimental selectors that could alter production execution. State why each remaining entry is rejected and where it is consumed.
- Do not silently unset environment variables in C++, and do not introduce a blanket rejection of `ASPECT_*`. Preserve useful profiling controls such as `ASPECT_FAULT_LINEAR_PERFORMANCE`.
- Verify `ASPECT_FAULT_INTERFACE_MODES` specifically. In the previously inspected core it selected an optional few-mode interface-preconditioner correction; it was not a friction or normal-stress-filter parameter. Confirm its current consumer before removing its guard. Do not reactivate or redesign that preconditioner during this cleanup.

There are also two actual diagnostic selectors inside `output.cc`: `ASPECT_BP3_LENGTH_COUPLED_DIAGNOSTIC` and `ASPECT_BP3_LENGTH_FULL_AUDIT_FROM`. Remove these hidden production controls by using the existing explicit diagnostic parameters where sufficient, or by moving the experiment to its dedicated diagnostic plugin. Update the corresponding launchers. Avoid replacing two hidden controls with a large new parameter hierarchy.

Document the few remaining supported runtime controls in the README. Presence-based selectors must be **unset**, not set to zero.

## 2. Separate diagnostics from required model initialization and checks

The current `BP3Output::execute()` calls `export_work_replay()` every accepted step. Passing `write_files=false` suppresses file streams but still performs substantial bulk-cell and particle evaluation, weak-load reconstruction, and MPI work. Its returned vector is discarded by the caller.

Split responsibilities before gating this function:

- Preserve the first-real-step Maxwell-publication check and its small `first_update_maxwell.csv` report.
- Preserve accepted-state/Theta checks, compression and boundary checks, fixed-geometry and completed-`I_h` checks, and the current native weak-traction consistency check unless an equivalent replacement is verified.
- Put redundant historical common-FE/parent-P0 comparisons and detailed replay output behind explicit diagnostics. Avoid computing a comparison solely to discard it when diagnostics are disabled. In particular, distinguish the particle sampling needed for the first Maxwell check from sampling needed only for subsequent comparison files.
- Do not simply put the entire function inside `if (diagnostics)`: it currently also contains correctness assertions.

`capture_work_invariants()` gathers all particle IDs and initial H values onto every rank, retaining a global map that is also serialized. Report the map's measured memory and serialized size on an available representative case. This is a potential scaling and checkpoint-size cost. Preserve the audit in this cleanup unless an existing scalable mechanism can provide the same check, including after particle migration and restart. Do not quietly drop the invariant or construct a new distributed-history subsystem as part of this task.

The restored monitor is not merely an output plugin: it loads the stationary boundary profile and configures the normal-stress filter. Keep its initialization and signal ordering. Gate only diagnostic sampling/copies and writers, not the entire monitor.

Keep raw quadrature, incoming-particle, full-state, work-replay, and initial-mesh CSV dumps off in ordinary production. Retain mesh verification even if the initial mesh dump becomes optional. Reuse the existing `Write detailed diagnostics` and `Audit full state every step` controls where practical, with clear documentation of what each enables.

## 3. Make scheduled profiles the canonical full-fault output

Currently `cumulative_slip.csv` appends every fault vertex at every accepted step, independently of output scheduling. Scheduled `profiles/fault_<step>.csv` already contains cumulative slip, velocity, state, shear traction, and normal traction. This duplicates the slip history at a much denser cadence.

Implement the following production policy:

| Output | Production policy |
| --- | --- |
| `accepted_steps.csv` | Every accepted step; retain timestep, peak velocity, solver and consistency summaries. |
| `stations.csv` | Every accepted step at the existing stations. |
| `restored_growth.csv` and first-event summary | Retain the existing inexpensive summaries. |
| `profiles/fault_<step>.csv` and `profiles.csv` | Canonical scheduled full-fault history, retaining every fault vertex. |
| `cumulative_slip.csv` | Stop writing the duplicate every-step table by default. Provide an offline export from saved profiles if a legacy reader needs this schema. |
| Native visualization and particle visualization | Keep the existing coordinated heavy-output schedule. |
| Raw/full-state diagnostic dumps | Explicit diagnostic runs only. |

Update maintained plotting scripts to read `profiles.csv` and its referenced profile files. Preserve support for old cumulative-slip files where inexpensive. The legacy exporter must identify that it contains **saved profiles only**; it cannot recreate unsaved timesteps.

Use these explicit settings for the normal production PRM:

```text
subsection Postprocess
  subsection BP3
    set Profile slip interval = 0.1
    set Profile time interval = 31557600
    set Heavy output slip interval = 0.1
    set Heavy output time interval = 31557600
    set Audit full state every step = false
  end
  subsection BP3 restored monitor
    set Write detailed diagnostics = false
  end
end
```

The time intervals above are seconds. Preserve explicit settings in dedicated detailed-output experiments. The supplied plugin's current profile defaults are 0.01 m and 3155760 s; align production defaults/documentation deliberately, not by silently overriding explicit user parameters. Both slip and time conditions can trigger output. Keep forced initial, graceful-final, and existing event-milestone profiles. Keep the existing rule that a heavy-output step also writes a profile.

The slip criterion should retain its current meaning: the maximum absolute change of cumulative signed slip from the last saved profile. It is not a fixed number of timesteps and is not accumulated absolute slip distance. Do not change this definition during cleanup.

Keep slip integration on **every accepted physical timestep** using the committed slip rate, with no artificial step-zero increment. Continue publishing/checkpointing the full slip vector every step. Output thinning must never thin constitutive updates, state checks, event detection, or slip integration.

Use each saved profile's velocity for its instantaneous event classification. Do not treat a slip difference divided by the interval between sparse profiles as the event's peak velocity. Retain the every-step peak-velocity summary and label temporal resolution accurately in plots.

Reduce temporal duplication before reducing spatial resolution or numeric precision. Keep full precision for saved profile quantities. Do not introduce HDF5 or a custom binary format merely for this cleanup. If every-step full-fault data is essential for a particular diagnostic, make it an explicit short-run option, disabled in production, rather than adding another overlapping scheduling system.

## 4. Preserve restart behavior and output integrity

The output checkpoint is version 5. Apparently unused fields are still part of its serialized layout. Prefer preserving that layout during this pass, with named compatibility helpers/comments. If a format change is genuinely necessary, introduce an explicit version and a tested version-5 reader; never delete fields while leaving the version unchanged.

Preserve cumulative slip, previous Theta, event history, audit baselines, and output-schedule reference states. Do not reconstruct physical history from CSV files. Do not replace the initial H baseline with restart-time H. Keep rejection of incompatible BP5 checkpoints even if the BP5 name looks like leftover code.

Check these concrete output issues:

- `restored_growth.csv` currently writes its header only at step zero. A new restart output directory needs a header based on file existence/emptiness, not simulation step.
- Saved schedules include their intervals. Decide and document how an explicitly changed output cadence is applied on restart; if supported, preserve the last-written slip/time reference while applying the new intervals. Do not change physical restart identity because only output frequency changes.
- Advance a schedule only after successful payload and index writes. Coordinate write failures across MPI ranks so one rank does not throw while others enter later collectives.
- Avoid duplicate index rows on resume. Detect output newer than the loaded checkpoint; do not silently append conflicting histories or automatically delete unrelated later data. Use the existing supported restart/branch convention and explain it.
- Checkpoint metadata copying currently includes `profiles.csv` but does not itself copy the referenced profile payloads. Ensure the documented restart/plot workflow resolves those files correctly. Do not claim that an index alone preserves the old profile history, and do not solve this by copying the entire growing output tree at every checkpoint.

Preserve the completion postprocessor that commits the heavy-output schedule after native visualization/particle/fault output. Do not remove it as an apparently empty or duplicate plugin.

## 5. Remove only verified stale pieces

- Audit the unsupported `Mature prestress file` parameter, which is declared but required to be empty in this restored model. Remove it from maintained PRMs/docs or retain a clearly deprecated empty-only compatibility entry; never silently accept a nonempty value.
- Check whether the zero-traction registration is still used on any boundary before removing it. Bottom Dirichlet conditions alone do not prove the registration is unused everywhere.
- Verify constant/function references before removal. Some historical-looking constants still determine prestress. Do not change prestress while cleaning comments or constants.
- In the monitor, replace the hardcoded denominator in Omega with the same authoritative `D_c` accessor used by the model. Preserve the distinction between committed Theta and the incoming Theta used by mechanics.
- Retire irrelevant diagnostic sampling windows or derive them from current geometry and weakening length. Avoid adding many production parameters just to preserve old experiments.
- Shorten the README to model assumptions, supported settings, output schemas/cadence, build/run, and restart instructions. Remove broken references to retired evidence directories; use version history for obsolete implementation narratives.

## 6. Validate the changes and report their effect

Use the existing small restored-BP3 fixture and focused output/restart checks. A new full first-event server run is not needed for this cleanup.

1. Compare before/after runs with identical physical settings. Check accepted times, V, Theta, cumulative slip, tractions, event decisions, and required audit outcomes within established tolerances. Output cadence must not affect these results.
2. Check diagnostics off/on on the same fixture. Confirm disabled diagnostics avoid their files and unnecessary work, while required assertions and first-step checks still execute.
3. Exercise slip/time-triggered profiles, forced initial/final/event output, and a split restart between scheduled writes. Verify headers, valid profile references, no duplicate saved states, and unchanged slip integration. Test an existing version-5 checkpoint if compatibility is retained.
4. Build the plugin in the maintained Release configuration and parse the actual updated production PRM. Use existing checks instead of creating an extensive new test framework.
5. Report file counts and bytes for the same accepted-state sequence before and after, separating summaries, profiles, native visualization, diagnostics, and checkpoints. Also report the number of fault vertices, accepted steps, and saved profiles. The removed table scales with vertices times accepted steps; scheduled profiles scale with vertices times saved profiles. Do not promise a fixed reduction factor before measuring it.

Deliver the patch, updated PRM/README/readers, a small environment-variable disposition table, validation results, and measured output sizes. List any audit or compatibility machinery deliberately retained and why. Keep this work separate from solver redesign or further changes to BP3 physics.
