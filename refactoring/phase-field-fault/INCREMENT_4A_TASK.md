# Increment 4A · Transport accepted bulk stress through the ordinary Stokes solve

Work in the ASPECT 3.1 `pf-rsf` worktree. Implement and verify the first part of Increment 4. In a normal mature-fault calculation with a stationary reconstructed curve and a prescribed phase profile, advect the accepted physical DGQ2 stress with the accepted bulk velocity. The next Stokes solve and its section balance must use the transported stress. The accepted endpoint must contain the resulting physical stress. This operator will also serve the moving-fault calculation.

## Start from the author's local state

Read `AGENTS.md` and the active plan, status and verification guide under `refactoring/phase-field-fault/`. Inspect HEAD, the working diff and the local 3B review repairs. Preserve incoming work. The author reports the bulk-to-fault rate tangent passed on one and two MPI ranks. Record the existing test and evidence if available. Reuse that check if this change touches its inputs.

The published baseline inspected for this assignment was `e4338dd`. Use the author's local descendant or working changes. Check the disposition of the four 3B findings (phase range, configured phase model, registered energy test and installed label reference). Record any genuinely open item under its existing assignment. Shared phase configuration is due before a second consumer of the configured phase functional. Bulk stress transport itself consumes the stress field and velocity. It has no need for the crack model or its postprocessor.

## Implement one production path

1. **Trace the parent before editing.** In the published tree, `FaultStokesCoupling::prepare_parent` reads the old DG stress on each cell, rotates and relaxes it, and supplies it to section sampling and Stokes assembly. Locate the current equivalent. In the donor, trace the *direct physical stress* path through `transport_material_history`, `TensorAdvection::assemble/advance` and `rotate_material_history`. Cross-check the selected exterior-stress and subcycling settings against the frozen mature input. Keep the sibling donor read-only.
2. **Decide how much ASPECT advection assembly to reuse.** Inspect the release cell, interior-face and boundary-face assemblers, the stage-vector interface, local mass inverse and parallel scatter. For each piece, record the release symbol, required input/output and your choice. Use the release implementation wherever it gives the selected spatial action. If time factors, matrix assembly, boundary data or scratch ownership prevent direct use, identify the exact mismatch. Consider a small shared extraction and test its existing caller if changed. Keep any dedicated code to the unmatched operation. The earlier donor experiment `3a21d59cf` and revert `319b10b94` are evidence if available. Avoid copying a complete scratch/copy-data hierarchy.
3. **Transport the existing independent stress components.** Use accepted velocity throughout the attempted interval, the donor's componentwise material-advection form, physical quadrature, upwind traces, local mass inverse and explicit SSPRK3 with the donor's CFL subcycling. Account for the velocity-divergence term where the weak form requires it. Evaluate prescribed exterior **parent stress** on locally incoming faces at the SSPRK stage times. Respect neighbor ownership and exchange current stage values. Check supported meshes and face types against the production caller. Existing stress compositions already supply the accepted coefficients. Assign transport to one path so each coefficient advances once.
4. **Connect transport to the existing mechanics.** On a physical attempt the data flow is accepted stress → transported trial stress → existing bulk-slip-subtracted rotation, relaxation and projection → Stokes load and section balance → accepted endpoint. Use the established accepted velocity and slip inputs for spin. Preserve the initialization convention, including zero elapsed physical transport at initialization. Repeated nonlinear evaluations must see the same parent. A rejected attempt must discard its candidate and recompute using the new duration. Keep the accepted stress authoritative in its existing fields. Add the smallest normal configuration needed to select the supported transport and inflow, with existing stationary inputs retaining their recorded behavior.
5. Keep the FE transport operation independent of fault geometry, RSF and the phase-field observer. Follow ASPECT file and naming conventions. Use the existing ordinary simulator lifecycle and the fault model's own rheology, with upstream elasticity disabled. The fault curve, phase profile and material labels stay at their existing locations in the 4A fixtures. Geometry motion, phase transport or growth, a general fault manager and disk checkpoint/restart belong to later decisions.

## Evidence required before handoff

Use the FE verification workflow for the new spatial operator. Write the componentwise weak action and stage-time convention in the working record. Establish tolerances from FE representation, quadrature, output precision or a refinement comparison. Use small fixtures, fresh output directories and at most four local workers total.

| Question | Discriminating check |
| --- | --- |
| Is the transport direction, magnitude and component mapping correct? | On a Cartesian mesh prescribe a constant translation and an affine physical stress with distinct component slopes. Supply exact time-dependent upstream values. Compare each transported component to its translated profile at known physical times. Reverse the velocity to switch the inflow side. Include zero-velocity identity for the transport operation. |
| Does the upwind face term use the neighbor's current state? | Put a stress jump across an interior face carrying nonzero normal velocity. Check the signed transfer against independent face quadrature, with the relevant face crossing the partition on two ranks. Compare one- and two-rank owned-cell results by CellId/component. Include at least one case with multiple SSPRK substeps. A continuous affine field alone cannot expose a missing jump term. |
| Does transported stress drive mechanics? | Run an ordinary parameter file with nonuniform parent stress, nonzero transport and stationary fault geometry. Check a component or weak action of the **actual assembled Stokes stress load**, or an independently predicted change in a solved section observable, and check accepted DG stress. Choose data for which feeding the old untransported stress would fail. Show that rotation of anisotropic transported stress has the expected sign and magnitude in a nonzero-spin case. Use independent fixture data for the expected results. |
| Are retry and existing callers correct? | Compare a rejected longer attempt followed by a shorter accepted interval with a clean run at the same accepted times, from fresh initial data. Check DG stress, label age and slip, and publication count. Rerun the corrected stationary AT1 regression, the relevant stationary/tangent checks, and a feature-disabled release control. If a release assembler changes, run an ordinary scalar-DG regression that reaches the changed code. |

Register a compact test exercising the production path. Record actual test commands, rank/thread counts, measured errors, tolerance rationale and any gaps in `STATUS.md`. Link the assignment from `PLAN.md`. Give a source walkthrough from accepted coefficients to the Stokes load and endpoint publication, including the stage/ghost exchange. Keep file references copyable as required by `AGENTS.md`.

Stop for author review after 4A. The later Increment 4 acceptance case moves the curve and profile in layered viscosity and compares two joint publications with the frozen donor at 12.5 and 25 ms. Passing the stationary-geometry checks here establishes one component of that comparison.

## Source entry points relative to the target worktree

```text
source/reconstructed_fault/stokes_coupling.cc
source/material_model/phase_field_fault.cc
source/simulator/assemblers/advection.cc
include/aspect/simulator/assemblers/advection.h
refactoring/phase-field-fault/PLAN.md
refactoring/phase-field-fault/VERIFICATION.md
../aspect-pf-rsf-particle-free/source/material_model/phase_field_fault/stress_evolution.cc
../aspect-pf-rsf-particle-free/source/material_model/history/tensor_advection_distributed.cc
```
