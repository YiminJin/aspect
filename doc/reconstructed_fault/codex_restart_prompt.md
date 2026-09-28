# Codex Restart Prompt: Reconstructed-Fault Surface RSF

Do **not** modify code in the first pass.

Read `reconstructed_fault_surface_rsf_design.md` as the authoritative mathematical/architectural specification. Then inspect the current repository implementation of:

- reconstructed-fault geometry/property storage;
- phase-field handler and point evaluation;
- particle manager / particle-domain handler;
- material-model / RSF rheology code;
- Stokes nonlinear assembly and tangent-modulus infrastructure;
- current particle-to-fault projection code;
- current reconstructed-fault tests and VTU postprocessor.

The existing code is potentially reusable, but it is **not an architectural constraint**.

Your first task is to produce an **architectural migration plan only**.

The plan must include:

1. A brief restatement of the new mathematical workflow.
2. A table of existing classes/files categorized as:
   - keep unchanged;
   - reuse with modification;
   - replace;
   - remove/deprecate.
3. A recommendation between:
   - explicit coupled fault-slip unknowns in the global block system; or
   - an exact condensed/Schur-complement implementation using the replicated fault.
4. Exact proposed interfaces/data flow for:
   - current `I_h`;
   - surface `V`, `Theta`, `T_coh`;
   - effective trial stress;
   - fault residual;
   - `A`, `B`, `G`, `K_V`.
5. MPI ownership/reduction semantics.
6. Cache construction and invalidation.
7. A staged implementation sequence with small independently testable commits.
8. A mandatory finite-difference Jacobian verification plan.
9. Any place where the current code conflicts with the new mathematics.
10. Any mathematical ambiguity that must be resolved before coding.

Important constraints:

- Current scope is 2-D bulk + 1-D ordered reconstructed-fault polyline.
- `K=1` inside the normal influence strip.
- One particle contributes to at most one fault.
- Cohesive traction is a surface history variable, not a particle field.
- `I_h` is measured from the actual FE phase field, not from an extended core phase field.
- Keep the IMEX state treatment; do not restore the full unstable `dTheta/dV` feedback.
- Do not interpolate `dmu/dV` to bulk quadrature points to mimic the old local tangent.
- Do not force the new mathematics through old APIs if those APIs are no longer appropriate.
- Do not implement branching, coalescence, 3-D faults, or overlapping influence bands in this restart.

After producing the plan, stop and wait for review before writing code.
