# Increment 6A · Composition method selection and the shared Stokes Picard loop

Work on `pf-rsf`. The reviewed starting point is `0d605a75af43bf6e21ab45d0cb52bc04117d5885` on ASPECT 3.1.0. Check the actual HEAD and working tree first. Read `AGENTS.md` and the current plan and working record. Implement the two changes below as separately reviewable changes. Use normal ASPECT entry points and the existing test drivers. Preserve the numerical methods and supported inputs. Keep substantive core edits small enough that each can be explained to the maintainer of that function.

## 1. Give explicit DG SSPRK3 a compositional field method

Make `explicit DG SSPRK3` a value in `Compositional field methods`. This name identifies the supported pure material-advection algorithm. A mixed input with fields `companion, tracer` should be able to say:

```text
subsection Compositional fields
  set Compositional field methods = field, explicit DG SSPRK3
end
```

- Use one authoritative per-field method value from parameter parsing through `Introspection` and `AdvectionField::advection_method()`. Dispatch the explicit case inside the ordinary composition switch. Have that case call the existing `Transport::advance_composition`, emit the existing per-field completion signal once, and preserve the common update of the linearization point after the loop. The midpoint phase case continues to defer its update to the coupled stage.
- Keep existing inputs working. Translate the branch’s `SSPRK3 fields` name list at parse time, after the normal default/list expansion. A named ordinary `field` becomes explicit. Accept consistent redundancy if both spellings select the same field. Continue to reject duplicate names within the old list, unknown names, or a name that conflicts with an explicitly incompatible method such as `static` or `midpoint fault phase`. Use the resolved method for validation even when the old list is empty. Separate the stage-time inflow evaluator from field selection where needed. A second lasting boolean selector would recreate the ambiguity.
- Audit every reader of `explicit_compositions.selected`, `compositional_field_methods`, and `AdvectionFieldMethod::fem_field` whose behavior depends on this change. In particular check FE grouping, ordinary assembler registration, volume and DG-face sparsity, mature-material admission, midpoint-provider admission, and basis checks on restart. The whole FE group can dispense with the ordinary composition matrix when none of its fields needs implicit assembly. An all-explicit group is one example. For mixed fields sharing a base element, ASPECT may legitimately store the implicit companion’s sparsity in the **first** field’s block, even when that field is explicit. Preserve this layout. Avoid unrelated edits to switches that already do the right thing.
- Keep existing qualification limits and error checks. These include 2-D and 3-D supported scalar DGQ2 boxes, accepted Q2 velocity, stage-time inflow, fixed mesh, and the restrictions on reactions, splitting, constraints and incompatible solver combinations. The phase role still selects bounded Bernstein transport. Signed generic or chemical tracers keep their existing basis. The three physical stress components retain the batched material-history path used before Stokes. Keep the existing ASPECT elasticity path independent.
- Update parameter help and the user-facing compositional-fields documentation. Show one mixed example and describe the compatibility spelling. Retain fixtures that actually exercise the old spelling, including defaults and restarts. Avoid rewriting historical benchmark inputs or frozen reference data to manufacture agreement.

## 2. Make the shared Picard loop readable at its call sites

Keep the two existing scheme entry points. Extract their one shared Stokes iteration into a clearly named private Simulator operation, for example `solve_stokes_by_picard_iteration()`. The no-advection scheme invokes it directly. The single-advection scheme solves temperature and composition once, then invokes it. Follow the existing source-file/template instantiation style. Update the signal documentation to identify both callers.

The pre-assembly signal prepares the coupled section response at the current iterate and can request rebuilding the matrix and preconditioner. The convergence signal checks the returned solution against that response before allowing the iteration to stop. Keep their order, iteration counters, pre-refinement limits, per-iteration postprocessing, failure/retry behavior, and final nonlinear-solver notification. Whole-step history is published after the repeat decision. Retain all of this for both Picard callers and preserve the behavior with empty signal slots. The growing-fault case uses the separate validated coupled Newton/phase route. Leave that solver and the genuinely iterated-advection schemes in their present owners. No new solver-scheme option is needed.

## 3. Verify the two changes

Use the existing controls, adding a small test only where it can distinguish a regression. Build with at most four local workers total. Use the existing Release and Debug configurations as appropriate and keep one numerical thread per MPI rank for numerical comparisons.

| Question | Evidence to return |
|---|---|
| Do old and new syntax select the same algorithm? | Run the mixed 2-D tracer with both spellings from the same initial data. Compare keyed accepted coefficients and clocks for both fields, including the implicit companion. Show the explicit and implicit algorithms still give distinguishable results. Exercise the existing single-entry broadcast behavior and one conflicting legacy input. |
| Does setup see the same method as the solver? | Check an all-explicit group and a shared-base group with the explicit field first and last. Show method identity, matrix or sparsity-block evidence for the relevant composition group, and a working implicit solve. Check that no ordinary implicit advection assembler/solve is used for the explicit field. Preserve the ordinary per-field signal and validated stage-time boundary behavior. |
| Are existing scalar fields and checkpoints still usable? | Run the existing bounded phase and signed tracer controls, including the 3-D slab on two ranks, mixed-basis and legacy-basis restart checks. Resume at least one compatible checkpoint across the old and new parameter spellings and compare with its uninterrupted endpoint. Exercise a previously supported rejection such as a selected reacting or constrained field under the new syntax. |
| Do the mature callers retain their different timestep orders? | Use a fixed-phase fault control with the direct no-advection scheme, the existing physical-stress transport control, and the layered moving-fault case with single advection. Show that phase advances once per attempt where selected, that Picard can continue after bulk convergence when the returned fault response disagrees, and that acceptance/retry publishes physical history once. Reuse existing observables and negative controls. |
| Are the growing-fault and stock routes intact? | Run the small existing `growing_fault` control. Use a stock feature-off Picard run and matrix-layout control. Compare iteration output/solver notifications with the pre-change executable where relevant. The working record documents a toolchain-level near-zero pressure difference for `postprocess_iterations_iterated_Stokes`; use a matched baseline if it recurs. Keep the frozen 5B strict audit and its documented strip-15 exception intact. |

Record the commands, rank/thread layout, expected failures, numerical comparisons and any verified change in matrix storage. Use fresh output directories. A runtime or memory claim needs like-for-like measurements. One source-level explanation of which block is avoided is sufficient when reporting a structural change only. The algebraic SSPRK3 operator itself is already qualified and needs fresh derivation only if its implementation changes.

## 4. Handoff and boundary with 6B

Update the current plan and working record with the selection rules, the two Picard call paths, the separate growing-fault path, affected upstream hunks, verification and remaining limitations. Give copyable paths to the two smallest code changes that explain the design. Commit the scoped changes locally when complete and stop for the author’s 6A review.

Reserve the overall source-linked implementation guide, installed donor-independent reproduction, final parameter/limitation survey, matched resource measurements and per-example qualification summary for **6B**. The accepted strip-15 exception belongs to the frozen reproduction. Physical front convergence for Example 2 and the regional gravity example retain their own scientific qualification.

## Files to start from

```text
include/aspect/parameters.h
include/aspect/simulator.h
include/aspect/simulator_signals.h
source/simulator/parameters.cc
source/simulator/solver_schemes.cc
include/aspect/simulator/solver/explicit_composition.h
source/simulator/solver/explicit_composition.cc
source/simulator/advection_field.cc
source/simulator/introspection.cc
source/simulator/assembly.cc
source/simulator/core.cc
source/simulator/checkpoint_restart.cc
source/material_model/phase_field_fault.cc
source/reconstructed_fault/stokes_coupling.cc
source/simulator/solver/midpoint_fault.cc
benchmarks/explicit_dg_transport/mixed.prm
benchmarks/layered_moving_fault/layered.prm
benchmarks/growing_fault/onecut.prm
doc/sphinx/user/methods/compositional-fields.md
```
