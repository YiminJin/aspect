# Repair the stationary fault coupling

This assignment starts from the two donor-matched fixtures on `pf-rsf` at `e91e801bab`. The physical inputs, reference snapshots, and comparison tolerances are the baseline. Begin each simulation from its initial conditions in a fresh output directory. Keep the midpoint closeout and the three paper examples in the plan. Inclined geometry, growing phase and the regional application start in later increments.

## Implementation prompt

```text
Read AGENTS.md and the current assignment, verification table and status in
refactoring/phase-field-fault. Implement the stationary fault coupling repair
described below in the current worktree. Work through the build, the small
numerical tests and the handoff. Keep the two independent donor references
frozen. Use fresh initial conditions for all fault runs. Stop after this
maintenance increment and present the code for architecture review.
```

1. **Retain the accepted mechanics.** Follow the existing nonuniform three-label case through the Q1 section residual, rate transfer, bulk Stokes solve and DG stress writeback. Keep its pressure and traction convention, bulk-slip-subtracted spin, Maxwell interval at initialization, zero physical aging/slip at initialization, and once-per-accepted-step aging law. Use the same implementation for the homogeneous case. Isolate changes in behavior from code movement in separately verifiable edits.
2. **Give the fault model an independent Stokes coupling.** Both cases run with `Formulation / Enable elasticity = false`. The fault owns its constitutive law and physical stress history. Its active path must have zero dependencies on ASPECT’s original elasticity rheology, including its header, helpers, old/rotated field conventions, elasticity reaction or transport, both elasticity-specific output types and elasticity-specific assembler or visualization branches. Generic Stokes/FE/field mechanisms remain available. Update the actual parameter files and validation. Assign physical stress field names, declare field types explicitly, and look up components by their declared roles. The discarded prototype places no compatibility constraint on these names.
3. **Preserve virtual work.** For the current iterate write the stress as the plane-strain viscous contribution plus `tau_offset`, the rotated/relaxed parent plus the current localized slip correction. The ordinary viscosity output supplies the Maxwell coefficient. Register one additive fault stress term on the Stokes system via the existing assembler registration mechanism. For velocity test function `v_i` it adds `-sum_q JxW_q * (tau_offset_q : symgrad(v_i)_q)` to the velocity RHS, including on RHS-only assembly. Obtain the gradient at the current quadrature point from FEValues. Another assembler's temporary gradient buffer has no per-quadrature lifetime across assembler calls. Supply the tensor directly from the state owner or through one narrowly typed generic additional-material output when needed. Keep the existing plane-strain trace contribution exactly once and incompressible continuity. Ordinary viscosity, pressure, gravity, quadrature, constraints and linear solver remain ASPECT responsibilities. The generic vector body-force output has a different test function. Useful release analogues are `tests/anisotropic_viscosity.cc` for registration and `tests/additional_outputs_03.cc` for the output request.
4. **Give production components durable owners.** Use `MaterialModel::PhaseFieldFault` for the material and `MaterialModel::Rheology::RateAndStateFriction` for the scalar law. Pair the rheology header and source under their respective rheology directories. Name traction and its fixed-age speed derivative in the result. Keep label identity and projection with the fault geometry. Let the sectional mechanics use the geometry and law. Move collective bulk sampling, staged rates and global DG writeback to the coupling owner. Locate concise preparation and acceptance hooks at the required times. Remove fixture-specific concrete material casts in shared solver/assembly code. Keep fixed or stationary names where they accurately describe a regression or supported geometry.
5. **Make outputs and acceptance physically meaningful.** Trace density to its EOS arguments and prevent a stress in Pa from being used as a material fraction. Read accepted deviatoric stress from the fault's DG fields for the supported physical output. Exercise that output on an accepted endpoint with nonzero history. With upstream elasticity disabled, release stress visualizers can reconstruct a viscous stress. Name the supported physical output in the examples and documentation. Avoid cloning the upstream stress visualization suite. Resolve consistency between section rates and the converged Stokes solution inside the nonlinear convergence or recoverable-failure path. Stage the result. Commit after the timestep manager decides to advance. A genuine failure or forced repeat discards the trial state.
6. **Retire prototype persistence.** Remove the fork's fault-only serializer, readers, v1/v2 schema keys, RTTI material trailer and tests devoted to migrating or corrupting those records. At the pinned head the trailer is the fork's only change in the common checkpoint file. Check the working diff for later unrelated edits before removing those hunks. Convert the lifecycle harness to a fresh-run retry comparison. Omit checkpoint writes and reads from the fault fixtures. Give a clear model-local input diagnostic if a fault run requests disk continuation or a fault checkpoint while full fault history cannot be restored. Keep the release checkpoint service available to its other users. This task does not require deleting run directories, donor archives or existing files on disk.
7. **Verify and report.** Run the scoped tests in VERIFICATION.md, including the normal ASPECT setup with the fault off. Use reliable Python comparison failures under both ordinary and optimized Python. If a test exposes a difference, locate the first disagreeing assembly, iterate or accepted endpoint and record the cause. Update the current status with the diff, exact commands, results, outstanding gaps and one ordinary step in source order. Show paths in the format requested in AGENTS.md.

## Review prompt

Use this text in the CLI's **Custom review instructions**, or as a regular review-only message in the same project chat. It specifies the base explicitly and covers edits before and after a commit.

```text
Review the full phase-field fault change from ASPECT 3.1.0 (0d97d2d331)
through the current working tree. Use git diff 0d97d2d331 -- and inspect
relevant untracked source, tests and instructions. Also inspect the repair
since e91e801bab. Review only. Give prioritized findings with a file, symbol
or line, a triggering input, the consequence and a discriminating test.

Which production class or core edit exists only because of this test specimen?
For each suspect, identify its actual callers and decide whether it is fixture
code, a legitimate restriction of the current solver, or necessary production
physics. In particular, justify the plane-strain trace correction.

Trace one nonuniform solve with upstream elasticity disabled. Verify the sign,
test gradient, RHS-only assembly, density inputs, accepted stress output,
coupled convergence, retry and publication. Look for indirect use of upstream
elasticity conventions. Check that fresh starts need no fault checkpoint and
that ordinary ASPECT users retain the release path. Check paired rheology
files and the ownership of global FE/MPI work. Distinguish observed defects
from future limitations and state exactly what you ran.
```

In the Codex app or IDE, type `/review` in the existing worktree thread. If only the base-branch and uncommitted presets appear, use the applicable preset for the built-in pass, then paste the block above as an ordinary message for the full review. To open the CLI's custom prompt, enter `/review`, choose **Custom review instructions**, paste, and submit. [Codex code review](https://learn.chatgpt.com/docs/code-review)
