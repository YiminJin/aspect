# Increment 3A · A stationary mature fault reconstructed from a spatial phase field

**Assignment for Codex.** Implement and verify one 2-D mature fault on a fixed Cartesian mesh. Initialize a spatial DGQ2 phase ridge, reconstruct its trace once from a seed, use its normalized profile in both directions of bulk–fault coupling, and run a short 60° case in which the normal ASPECT solve determines slip rate. Stop at the review of this result.

## Starting point

Work on `pf-rsf` from `c272572691fbd8fab449b7b6fa0db96e49058f30` or a descendant containing its fixes. Read the root `AGENTS.md` and the project `PLAN.md`, `STATUS.md` and `VERIFICATION.md`. The seven stationary review findings are closed. Retain their regression cases and existing reference values, tolerances and clocks. Preserve incoming work. Use the sibling donor read-only. The source revision for the inclined control below is `abbc9b0c0574c9f4de7a64d9662d539b93599f2b`. Keep the recorded Increment 2 donor revisions for their own comparisons.

Put this assignment at `refactoring/phase-field-fault/INCREMENT_3A_TASK.md` and point the active-assignment lines in the plan and working record to it. Preserve the rest of the roadmap. Work through A, B and C in order as separately reviewable changes. Continue through routine implementation decisions. A genuinely missing scientific choice blocks only its dependent work. Record the exact discrepancy and evidence before proposing a change to the model.

## A. One phase field and one trace

1. Use the existing static DGQ2 composition `phase_field` as the authoritative bulk field. Initialize it once. The test initializer may use signed distance and the donor’s mature profile. All subsequent reconstruction, normalization and mechanics must read the FE solution. Evaluate the field on the owning cell, respecting DG traces. Keep phase fixed through accepted and retried timesteps.
2. Extend or factor the current stationary geometry into the reconstructed-fault manager. Give it ownership of the ordered trace, permanent material-label IDs, station and segment coordinates, oriented tangents/normals, and point-to-trace associations. Both the normalization code and Stokes coupling must consume this geometry. Preserve one owner for accepted age, rate and cumulative slip. Keep the RSF law and section solve in their present roles. Give the manager enough responsibility that deleting the old stationary geometry implementation leaves one working geometry path.
3. Use the donor’s exercised seed-to-ridge initialization and boundary-contact convention. Test an intentionally displaced seed against a known straight ridge. Record pre-fit and post-fit normal errors at interior labels and endpoint contact independently. Keep label order and orientation fixed. For the old constant-phase controls, retain the supplied trace because a constant field has no identifiable ridge. Route these through the same geometry and transfer code.
4. Read these target entry points first. At the baseline, `StationaryFault::localization()` returns inverse box width, `prepare_parent` rejects nonconstant phase and unresolved knots, and `solve_rates` computes one stiffness outside the point loop. Remove each assumption only when its replacement has an exercised caller.

```text
source/reconstructed_fault/stationary_fault.cc
source/reconstructed_fault/stokes_coupling.cc
source/material_model/phase_field_fault.cc
../aspect-pf-rsf-particle-free/source/reconstructed_fault/manager.cc
```

## B. Make the spatial field change the mechanics

1. Follow the mature donor’s selected degradation law, integration support, boundary treatment and Q1 normalization projection. At a bulk point use its FE phase to compute `h = 1/g(phi) - 1` and its associated station to evaluate `I_h`. The localization is `chi = h/I_h` on supported points. Identify the donor mode and write down how its measure handles a truncated endpoint. Distinguish zero weight from an invalid normalization on active support. Preserve the constant-phase limit. A fitted width, a Gaussian substituted for the weight, or a peak value read once cannot establish this result.
2. Carry pointwise `chi` through the section residual and its rate derivative, the slip-stress load returned to Stokes, the accepted-parent slip subtraction in the spin, and the accepted physical stress. Preserve the accepted time levels and single publication after a successful step. Account separately for the weight in the bulk integration measure and the local stiffness in the traction response. Test the resulting derivative using the nonuniform profile.
3. Check a nonconstant vector of Q1 label rates and a nonconstant virtual test field. Compare work accumulated from the actual section loads against work computed with the returned interpolated rate in the bulk. For a scalar traction sample t_q, both contractions use JxW_q · chi_q · t_q and the same Q1 shape functions. Calculate the two sides through their respective production operators. Check the section tangent with an independent directional difference at fixed parent and bulk iterate. Separately compare the physical stress projected into DGQ2 on cells crossed by section knots against independently overintegrated cell moments. Revisit both the quadrature and the old nodal writeback. An algebraic identity at common quadrature should agree to scaled roundoff. Report that defect separately from FE/quadrature errors under refinement.
4. Use the fault-owned weak load and ordinary ASPECT Stokes/accept/retry path. Keep all dependencies on ASPECT’s original elasticity rheology out of this model, including its outputs, assemblers and stress-history machinery. Keep independent physical DG stress in the existing named fields. Avoid new Simulator hooks unless an actual unmet lifecycle need is demonstrated against the existing hooks.

```text
../aspect-pf-rsf-particle-free/source/material_model/phase_field_fault/normalization.cc
../aspect-pf-rsf-particle-free/source/material_model/phase_field_fault/constitutive.cc
../aspect-pf-rsf-particle-free/source/reconstructed_fault/surface_system_bulk_work.cc
```

## C. Run these controls and stop

Use a metre-wide square centered at the origin for the new geometry controls. Use `32 x 32` and `64 x 64` Cartesian cells for the refinement comparison. The coarser grid is sufficient for the free-RSF mini-run if it resolves the selected profile. Use fresh output directories and at most four total local workers.

| Control | Input | Acceptance evidence |
| --- | --- | --- |
| Existing stationary limits | Original homogeneous and nonuniform stationary fixtures, with their existing rejection/retry and review regressions | Existing comparisons pass unchanged through the shared implementation. Preserve the pressure-datum, closed-fault and endpoint checks. |
| Vertical profile | Use the donor mature distance profile with peak 0.6. Test length scales 0.15625 and 0.078125 m at the same prescribed rate. Translate the ridge from x = 0 to x = 0.07 m for a separate run. For a fit check, shift the straight seed 0.015625 m in x relative to the ridge, leaving its endpoints on the horizontal box faces. | The fitted ridge follows the translated FE field and improves the seed residual. Plot a central transect of phase, weight and returned slip strain. The narrower profile concentrates strain while preserving integrated slip within measured quadrature/FE error. Compare against separate high-order integration of the FE profile. |
| Inclined transfer | Use the donor’s 60° trace through the center, ordered bottom to top, with endpoints at x = ±1/(2√3) on y = ±0.5 m. Hold V = 0.005 m/s in a diagnostic fixture. Keep internal section knots cutting cell interiors. Exercise an offset seed as a separate geometry check. | Compare station and orientation against straight-line geometry, including length 2/√3 m. Check an interior station and a boundary-truncated station, normal and shear traction, work and projected stress. Match donor inputs and report profile/pressure representation differences for each donor comparison. |
| Inclined free RSF | Use the same 60° geometry and frozen spatial phase, the fitted trace, and the target’s supported traction-anchored box configuration. Start from the existing target stationary fixture for constitutive data and loading, adapting coordinates to the centered box. Release every label rate after initialization. Run initialization and at least two accepted physical timesteps. | Report accepted rates, age and slip by permanent label ID, physical DG stress at matched bulk points, normal compression used by RSF, and section residual. Compare the traction to an independent contraction of the sampled physical stress and pressure using the documented prestress convention. Show a nontrivial rate response to the applied loading. Repeat on two ranks with actual work on both and compare to one rank by label ID and cell/sample identity. |

Use these donor inputs to reproduce the prescribed-rate control and identify its diagnostic callback. That fixture fixes all boundary velocities, uses a different pressure convention, and disables seed fitting. Use it for quantities with matched inputs. For free RSF compare to an independently matched donor trajectory wherever possible. If no matched donor exists, state which quantities lack that comparison, and provide residual, independent traction and time-update checks. A prescribed-rate run cannot qualify the free-rate result.

```text
benchmarks/fixed_fault/stationary.prm
../aspect-pf-rsf-particle-free/benchmarks/reconstructed_fault/bp5/inclined_native_history_reference.prm
../aspect-pf-rsf-particle-free/benchmarks/reconstructed_fault/bp5/clean_stress_cycle.cc
```

**Review boundary.** Set tolerances from the preserved references, solver tolerances or independent quadrature/refinement evidence. Record commands, rank/thread counts, absolute errors, scales and tolerances in the existing working record. Include one source walkthrough from the initialized FE coefficients to an accepted DG stress component, and identify the two production callers of the manager. Label a partial result accurately if any gate remains open. Keep generated run trees outside version control.

Scope ends here. Phase evolution, tip growth, advection, moving geometry, multiple faults, lower-crust layering, gravity, production BP3 runs and disk checkpoints belong outside 3A. Diagnostic angle, prescribed speed and manufactured loading belong in fixtures. Preserve the donor and keep publication, cluster submission and later increments for separate assignments.
