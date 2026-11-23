# Increment 4B · Shared explicit DG transport for stress, phase and passive tracers

Implement this assignment on the ASPECT 3.1.0 `pf-rsf` worktree at `3c603715a13525dd8c1138a1cf125f25f5a2a1f8` or a descendant containing the six 4A review repairs. Inspect incoming changes and preserve them. Read the root `AGENTS.md`, `PLAN.md`, `STATUS.md` and `VERIFICATION.md`. Use the installed `verify-fe-operator` workflow for the operators changed here.

**Deliverable.** One SSPRK3 advance used in production by the existing three-component stress transport, by a moving, growth-disabled Bernstein phase field, and by an ordinary passive DG composition. A fault-free parameter file must be able to select explicit transport for one eligible composition while another uses the release implicit scheme. Run the bounded phase calculation through ASPECT's normal timestep and solution vectors. This is the transport part of Increment 4. The following assignment binds the transported phase to moving fault geometry and qualifies the layered mature-fault case.

## 1. Design from the three callers

Trace the repaired stress implementation, the release composition dispatch, and the donor's phase transport before deciding where a shared operation belongs. Use the sibling donor or its verified numerical source in the closeout archive. Compare these properties explicitly:

| Caller | Behavior to preserve |
| --- | --- |
| Physical stress | Transport the accepted signed xx, yy and xy DGQ2 coefficients as one batch using accepted velocity. Preserve QGauss(8) cell and face rules, local mass action, existing CFL/substep rule, exterior parent stress at stage time, and collective nonfinite diagnostics. Rotation, Maxwell relaxation, projection, section solve and accepted stress/label publication stay with their current owners. |
| Phase | Use the donor's native semantic phase field and quadratic Bernstein coefficient meaning for the new bounded route. Preserve its physical mass (Lobatto 4), spatial transport (Lobatto 5), mean-based CFL/substep rule, prescribed stage-time inflow, subdivision enclosure and mean-preserving scaling on entry and after each stage. A bounded polynomial can have a coefficient outside [0,1]. Read inputs and expose results through an actual phase FE field, including correct initialization at generalized support points. Existing static phase fixtures keep their accepted inputs and behavior. |
| Passive composition | In an ordinary ASPECT run, advance a selected scalar DG field by material advection with its selected inflow data. Allow an unbounded signed tracer; use the original per-field implicit default for its companion. Field names and component positions are user inputs. The transport solver must be usable without fault labels, a phase model, or fault-specific plugins. |

Share the SSPRK3 substep/stage schedule, stage-time boundary evaluation and candidate semantics. Share FE and communication code wherever their inputs and numerical action agree. Basis, quadrature, admissibility, substep estimate and treatment of the result belong at the appropriate caller or existing component. Keep the batched stress exchange. The scalar phase and signed stress may require different substep counts during the same physical step. Avoid a second copy of the stage loop, a hard-coded list of fault fields inside the shared solver, or an abstract framework for hypothetical integrators. A thin stress adapter can keep the existing public call readable.

## 2. Integrate with the smallest ASPECT change

- Use an opt-in, **per-composition** selection, orthogonal to how particles, volume of fluid, static and prescribed fields obtain their values. Keep the existing default and parameter files valid. Begin with pure material advection on the supported fixed, conforming, nonperiodic 2-D Cartesian DGQ2 setup. State the supported combinations in the parameter help. Keep any broadly named option honest about those limits.
- Inspect `Simulator::assemble_and_solve_composition`, the applicable solver schemes, boundary composition handling, and the existing field/FE registration. Prefer an additive dispatch at the narrowest existing call site and short declaration/parse calls following local convention. Keep the explicit implementation out of `core.cc` and `parameters.cc`. First look for a design requiring no edit to `core.cc`. The core rule in `AGENTS.md` permits a necessary, justified hunk.
- The 4A fault path uses *no Advection, iterated Stokes* and requests an attempt-local stress before Stokes. Ordinary tracers use composition dispatch. Preserve a single advance per physical attempt for each selected field. The same FE operator can serve both call sites. Read the accepted parent and the velocity prescribed by the supported scheme; state the velocity time level used. Do not advance an already updated field during a nonlinear repeat, round-trip stage values through the published solution, or change another field's boundary clock. Keep feature-specific storage, constraints and diagnostics near their owner.
- Preserve ASPECT reaction semantics. A material model can return a whole-step composition increment. The first explicit route is for pure advection. Diagnose unsupported active reactions, splitting, limiters or solver/mesh combinations where needed, at the selected field, without disabling valid unrelated fields. Never silently omit a reaction or apply an increment at all three stages. Temperature, original ASPECT elasticity, BDF1, moving meshes, growth/evolution and coupled moving geometry are outside this task.
- Keep the numerical baseline. Use the material form with its velocity-divergence contribution and upwind fluxes, including the correct measure at inflow. At stage times `t_k`, `t_k + h`, and `t_k + h/2`, use the appropriate state, apply any phase bound treatment, and ensure the next stage reads current neighbor coefficients. Preserve the 4A behavior for rank-local invalid data and failed attempts. Use the branch's existing lifecycle for publication and rollback.

## 3. Prove the shared implementation works

1. **Release default and public selection.** Register a small, ordinary fault-free parameter-file regression containing two DG compositions in a nonzero flow. Advance one via SSPRK3 and the other via the normal implicit method. Verify the implicit field against its default-only control at common clocks. Choose a curved profile, jump or pulse for which the explicit and implicit results measurably differ, and check the explicit field against an independent translation/refinement or discrete reference. An affine check alone could pass under either method. Exercise the field indices with a swap or a decoy. Check a nonzero reaction on the selected explicit field gives its documented diagnosis.
2. **Moving bounded phase through normal ASPECT execution.** Translate an inclined ridge with growth disabled and nonzero displacement across a cell face and an MPI partition. Use positive and reversed translation on small meshes. Compare to the translated initial profile or a matched donor control. Report displacement, peak, normal width, extrema or certified enclosure, and the fault-normal localization integral at a translated station if defined for the chosen ridge. A conserved global integral or surface energy alone cannot establish that the ridge actually moved. Check the discrete material balance, including stage-weighted boundary flux and phase-weighted discrete velocity divergence. In the limiter tests include both a physically bounded polynomial with an out-of-range coefficient and a real in-cell overshoot at admissible cell mean. Check the limiter on stages, conservation of the mapped cell mean, and collective treatment of an inadmissible mean. Keep this phase computation usable with zero reconstructed faults.
3. **Accepted state and parallelism.** Compare one and two ranks by global field/cell identities with a nontrivial partition flux. On a forced retry, compare phase and affected histories with a clean run at matching accepted times. Verify zero physical duration causes zero transport and each accepted interval advances once. Use the existing retry harness wherever possible.
4. **Preserve 4A.** Rerun the signed partition jump, receiving-cell Stokes load with its stale-parent negative control, the varying-spin ordering test, rank-local NaN/overflow and checker corruption cases, retry and stationary/tangent regression for affected paths. Rerun the relevant 3B phase-only and feature-off controls. Use scoped Debug/Release checks with at most four local workers. Record the actual commands, rank/thread layouts, tolerances and results. Use fresh directories. Measure a matched small release case for gross changes in work, halo exchanges and field copies.

## Handoff for maintainer review

Make shared transport, its ordinary caller, and bounded phase integration separately reviewable where practical. Record the accepted-to-candidate-to-publication call path for each real caller, including who applies the stage limiter. Identify the existing upstream cell/face machinery reused and the specific reason for any separate spatial operation. For each substantive upstream core hunk, give the file/symbol, the requirement, the reason an existing extension point did not suffice, and the test for the old path. Review both the increment diff from `3c603715a` and relevant protected files against `v3.1.0`, so inherited core changes are distinguished from new ones. Keep the record in the existing `STATUS.md`.

Finish these three working consumers and stop for review. The full layered moving-fault control, with two joint publications at 12.5 and 25 ms, remains the later acceptance criterion for Increment 4. Keep its frozen references accessible.

## Source landmarks

Paths are relative to the target worktree.

```text
source/simulator/solver/dg_stress_transport.cc
include/aspect/simulator/solver/dg_stress_transport.h
source/simulator/solver_schemes.cc
source/simulator/assemblers/advection.cc
source/simulator/introspection.cc
source/reconstructed_fault/stokes_coupling.cc
refactoring/phase-field-fault/VERIFICATION.md
../aspect-pf-rsf-particle-free/source/phase_field/dg_transport.cc
../aspect-pf-rsf-particle-free/include/aspect/phase_field/dg_bounds.h
../aspect-pf-rsf-particle-free/include/aspect/fe_dgq_bernstein.h
../aspect-pf-rsf-particle-free/doc/reconstructed_fault/particle_free/NATIVE_DG_BERNSTEIN.md
```
