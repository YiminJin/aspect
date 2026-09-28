# Modified BP3: prepare long-run output and simplify particle properties

Implement the following in the current repository. The reviewed inputs were `bp3(9).cc`, `original(2).prm`, and the K5 cleanup/GMG reports. Inspect the actual helper headers and launcher before editing; they were not included in the uploaded source. Prepare and briefly verify the implementation, then provide commands for me to launch the long simulation myself.

## Scope and baseline

Preserve the qualified wide, fully frictional modified-BP3 configuration as a reproducible reference. Make a separately named long-run configuration and output directory. Keep the 200 x 100 km domain, physical fault, loading, RSF parameters, mature C=0 treatment, work measure, paired endpoint corrections, accepted-history publication, and solver acceptance criteria. Retain the qualified velocity-block GMG choice and AMG reference. This GMG prototype retains assembled fine operators: do not replace it with a different full Stokes solver merely by changing the `Stokes solver type` parameter.

The supplied parameter file is still a seven-step saved-clock experiment. Remove its replay cap/completion selection, saved-clock environment dependency, seven-step termination, and short wall-time limit from the long-run configuration. Retain the real adaptive timestep controllers. Make the physical end time and graceful wall-time stop explicit launcher options. Distinguish a first-event pilot from a recurrence-cycle run: the existing first-event completion flag does not establish a complete recurrence cycle. Do not automatically launch a long run or parameter campaign.

## 1. Remove the special refinement around 40 km

Find the actual saved-mesh generator/fixture rules. `Strategy = BP3 saved mesh` means editing the displayed minimum-refinement function alone may do nothing. Remove the special 40-km neighborhood refinement in the bulk and any corresponding extra fault-node refinement. Retain the ordinary fault-band resolution and appropriate grading through the localization support.

Report actual bulk cell sizes normal to/along the fault and reconstructed-fault spacing, especially at 15–18 km and the former refinement boundaries. Keep the phase-field length scale at 400 m. Use the ordinary current resolution for an exploratory pilot; do not label it spatially converged. If h is approximately 100 m, ell/h is approximately 4; a later approximately 50-m comparison would assess resolution while keeping ell fixed. This ratio alone does not qualify earthquake nucleation or peak rates.

Regenerate mesh-dependent inputs with the existing generation procedure. Check whether prestress coefficients, boundary-completion tables, coordinate hashes, and the hard-coded 1236-node guard must change. Preserve the physical initialization and replace obsolete fixture-specific checks with checks against the new inputs. Do not bypass consistency checks or reuse incompatible tables. Record new input hashes once at launch.

## 2. Use `initial composition` for strengthening

Remove the dedicated `BP3Strengthening` particle-property class and its registration from the maintained path. Extend the selected fields of the existing `initial composition` particle property to include `strengthening`, and map that compositional field to the property name/component actually exported by the local implementation. Keep the existing theta/state and Maxwell-stress mappings correct; do not guess component indices from upstream documentation because this branch uses different parameter names.

Preserve the current physical semantics. The removed plugin evaluates `BP3::depth_fraction(current_position[1])` after particle advection every timestep. Initialization alone would instead carry that value with the particle and gradually move the spatial friction transition.

First reuse an existing generic, field-selective refresh/prescription facility if available. If absent, implement a small opt-in selected-field refresh within the existing `initial composition` property: default empty, enabled only for `strengthening` here, evaluating the active initial-composition model at current particle positions at the same lifecycle stage as the old plugin. Keep the BP3 formula in its initial-composition model. Never reset theta/state, stress, or inert H as part of this refresh. Preserve default behavior for unrelated models and any existing prescribed-field behavior. Do not enable a prescription that resets all compositional fields.

Verify initialization and a deliberately displaced particle in the transition zone against the former spatial rule; small plate displacement in a short replay alone is not an adequate test. Removing a particle property changes the checkpoint layout: qualify fresh checkpoints from the new configuration, and do not claim old checkpoints are automatically compatible.

## 3. Coordinate native visualization using accumulated slip

Enable ASPECT's native visualization and particle writers plus the reconstructed-fault writer. Replace the custom `bulk_*` writer in this configuration. Use native parallel time-series metadata, physical field names, and suitable output interpolation for the FE order. Bulk mesh/fields, particles, and reconstructed-fault geometry/properties must share one accepted-state output decision and timestamp.

Keep cumulative slip updated at every accepted real timestep using the existing committed-rate convention:

\[
\delta_j^n=\delta_j^{n-1}+\Delta t_n V_{j,\mathrm{committed}}^n.
\]

Use the authoritative existing slip storage where possible. For heavy output, compare each node with its own slip at the last heavy output:

\[
D_n=\max_j\left|\delta_j^n-\delta_j^{\mathrm{last\ output}}\right|.
\]

Write when D reaches 0.1 m. Compute the maximum consistently across MPI ownership. This is neither the single-step slip nor the difference between two spatial maxima. Evaluate only after acceptance; rejected solves must not advance histories or output references. If a step crosses several thresholds, write its actual accepted state once, with its actual time and increment. Do not fabricate intermediate states or force solver timesteps just to meet output thresholds.

Always write the initial and final gracefully accepted states. Add a configurable maximum physical interval, with a proposed default of one year, so quiet intervals remain visible; allow disabling this time trigger. Update the shared reference only after the scheduled outputs succeed. Audit execution order: the decision must be available before each writer runs. Gate file writing, not the whole BP3 postprocessor, particle update, statistics, or history verification.

Preserve the distinction between retained FE stress history and accepted current constitutive stress. Native visualization does not itself resolve that distinction. Label available fields clearly. If exporting current constitutive stress, use the correct accepted cache or retained working inputs; never evaluate a second Maxwell update using newly committed history merely to produce a plot.

## 4. Keep inexpensive cumulative-slip data

Retain compact station time series and one summary row per accepted timestep, including time, dt, maximum V/location, solver iterations/residuals, and relevant history checks. These records are needed to resolve event timing even when heavy output is sparse.

Save reconstructed-fault profiles for cumulative-slip plots at every heavy output and on a separate lightweight schedule. Proposed defaults are a maximum slip change of 0.01 m or a maximum physical interval of 0.1 yr, plus first onset, event end, and final state. Make these configurable and implement them without thousands of unnecessary bulk/particle files.

Profiles must include stable fault/node identifiers, down-dip distance, coordinates, accepted time/step, cumulative signed slip, V, and Theta. Use explicit SI units and an index of profile times/files; useful tractions can be included with their measure documented. Preserve the original cumulative-slip zero across restart. Provide a small offline plotting script for delta(s,t), event-relative slip, and optionally Vp*t minus delta. Build plots from recorded profiles, not reintegration of sparse V samples. State that any requested interpolated profile is interpolated, not a solved state.

The 0.1-m visualization threshold is not a timestep-accuracy target. Dc is 0.008 m here. Preserve the RSF controller independently and report the accepted timestep/slip scales without loosening its settings to reduce output.

## 5. Reduce checkpoints and diagnostic dumps

Use periodic restart checkpoints independently of graphical output. Proposed settings, subject to local parameter support:

```text
subsection Checkpointing
  set Steps between checkpoint = 0
  set Time between checkpoint = 1800
  set Number of checkpoints to keep = 3
end
```

Here 1800 is wall-clock seconds. Retain checkpoint-on-graceful-termination. Remove per-step checkpointing and automatic checkpoint-directory copies for every new event peak. Keep compact onset/peak/end metadata. Remove the assumption that the latest checkpoint necessarily represents the immediately preceding accepted step; label any retained checkpoint with its actual time.

Disable routine full-particle `mature_history_*`, raw-QP/DoF CSV dumps, and redundant investigation files. Inspect `export_work_replay` and its callees as well as `write_audit_state`; some exports occur outside the current audit switch. Separate essential calculations and history checks from file export. Keep full diagnostics explicitly available for short investigations. Replace repetitive per-step success messages with compact statistics where appropriate.

Checkpoint cumulative slip, all output references/clocks/counters, and event-tracking state. Ensure restart cannot duplicate profile rows or leave time-series indices pointing to outputs beyond the resumed checkpoint. Use an explicit restart output policy and preserve the prior run's evidence.

The last supplied GMG report still documents an unresolved frozen-Ih restart limitation. Check whether it has since been fixed. If present, report this as a resumability blocker; include the minimal fixed-mesh restoration correction and its focused verification before advertising a recoverable long run. Do not silently change normalization on resume or expand this into general curved/evolving-boundary formulation work.

## 6. Bounded verification and handoff

Verify the new output decision with a small focused test covering changing maximum-slip location, threshold crossing, a rejected step, and save/load. Check field-selective strengthening refresh separately. On the actual new configuration, run only a short fresh/restart continuation sufficient to verify native output timestamps, particle/fault fields, preserved histories, and restart output numbering. Force small output thresholds in this test rather than waiting for an earthquake.

For behavior-preservation comparisons, hold the mesh fixed while checking output/property refactoring; assess mesh removal separately because it can legitimately change numerical results. Reuse existing reference evidence and established field tolerances. Report the known initial GMG stress-equivalence qualification separately. Do not rerun the full historical campaign or relax a failed criterion.

Provide a concise change report, actual mesh statistics, effective long-run parameters, output/checkpoint schedule, observed bytes per output and projected storage range, and exact fresh/restart commands using the qualified Release binary/plugin and GMG backend. State what was tested and what remains unqualified. Stop after this preparation so I can launch the expensive run.

Reference: ASPECT documents wall-clock checkpoint timing in its [checkpoint parameters](https://aspect-documentation.readthedocs.io/en/latest/parameters/Checkpointing.html). Upstream [initial-composition particle code](https://github.com/geodynamics/aspect/blob/main/source/particle/property/initial_composition.cc) illustrates why initialization, prescribed updates, and local field names must be checked rather than assumed equivalent.
