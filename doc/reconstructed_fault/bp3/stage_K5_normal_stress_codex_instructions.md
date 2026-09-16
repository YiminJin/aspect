# Stage K5: locate the normal-stress anomaly

Continue the current reconstructed-fault BP3 work. Read `stage_K5_first_cycle_theta_stop_analysis.md`, the current K5 configuration, and the relevant implementation. Reuse completed checks and existing outputs.

## Goal and scope

Determine where the negative normal-stress samples occur, how they affect the friction equations, and whether prescribed deep slip is transferred consistently into bulk deformation. Recommend one targeted next action.

Step 11 reports sampled normal stress of [-12.2348, 90.8439] MPa, while the twelve projected station values are [50.0805, 52.5636] MPa. The sample locations are unknown. Missing deep-slip contributions, a concentration near the 40 km transition, across-band stress variations, and bottom-boundary effects are hypotheses to distinguish—not established causes.

Implement only the narrow Theta-checker correction if still needed, plus diagnostic instrumentation. Keep the physical model, pressure treatment, prescribed-slip cutoff, phase profile, mesh, integration scheme, and production history law unchanged. Do not extend RSF to the bottom, smooth or clip normal stress, or launch a first-event continuation in this task.

## 1. Check prescribed deep-slip transfer

Trace prescribed V from surface constraints through surface-to-bulk interpolation, crack deformation, particle history, and bulk assembly. Identify the relevant functions and whether prescribed nodes are accidentally excluded by a free-node or active-equation mask.

Verify that the deep V=Vp region contributes consistently to the model's deformation through the bottom. Respect the existing perturbation formulation: distinguish total deformation from background-subtracted deformation. A zero perturbation at V=Vp is not by itself evidence of missing slip.

Check continuity of the actual transferred deformation across the 40 km junction, including the element that spans it. Inspect V, accumulated slip, h(phi)/I_h, and the deformation quantity actually consumed by the bulk. Do not infer correct transfer solely from exported nodal V.

## 2. Locate and interpret the stress extrema

Prefer diagnostics from the saved step-11 state. Use the same sampling and stress evaluation as production; identify the time/history level being evaluated.

Capture the minimum and maximum with MPI-safe sample identities and physical coordinates, along-fault coordinate, signed distance across the fault band, cell identity, integration/coupling weight, and contribution to free, prescribed, or mixed support. At each extremum, report pressure, deviatoric normal traction, and background contribution separately, with the compression-positive sign convention explicit.

Export the full projected normal-traction profile along the fault, together with V and accumulated slip. Compare these with raw samples near the extrema and near the 40 km junction. Use a small local sample export or plot; avoid dumping every bulk sample.

Trace exactly which normal-stress quantity enters the friction residual and its Jacobian. Identify any nonlinear treatment before projection. Quantify whether negative samples contribute materially to free-node friction equations; positive station values or a small sample weight alone are not sufficient to dismiss them.

## 3. Theta correction and execution budget

If already corrected, reuse the validated checker. Otherwise replace only its cancellation-prone reference evaluation with an independently specified, cancellation-safe evaluation; retain the 1e-12 tolerance. Verify the reported near-bound cases and one deliberately wrong-history case. Add concise failure details identifying the node and numerical discrepancy. Do not weaken the assertion or change production Theta evolution.

Use existing outputs and checkpoint evaluation first. If a solve is necessary, perform at most one replay from the completed step-11 checkpoint identified as `restart/03`, through step 12 only. Preserve the checkpoint and original outputs, use a separate diagnostic output directory, and retain the original stepping controls. Capture diagnostics before a possible postprocessing failure. Record the source/binary used.

If execution is unavailable or exceeds the existing job budget, deliver the instrumentation and exact replay command, explicitly marking the missing evidence. Do not substitute a fresh multidecade run, resolution sweep, or changed-loading experiment.

## Deliverable and stopping condition

Write a short report with sample locations, the prescribed-slip transfer finding, the relationship between raw/projected/friction-consumed stress, and one recommended next action. Include only the source references and CSV/plot evidence needed to support the decision.

State whether extending the RSF-solved section is justified by the evidence. If an implementation defect is found, describe the smallest repair for review; do not apply a physics or transfer change during this diagnostic task.

Stop once the cause is sufficiently localized to choose that next action, or the single-replay budget is exhausted. Report any remaining uncertainty directly; do not broaden testing or polish the report further.
