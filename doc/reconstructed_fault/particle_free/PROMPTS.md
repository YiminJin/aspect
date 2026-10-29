# Codex task prompts

## First task: PF0 and PF1

Paste the following from a fresh session in the new worktree:

```text
We are starting verification of the particle-free phase-field RSF formulation
in worktree aspect-pf-rsf-particle-free, branch pf-rsf-particle-free, based on
the propagation branch.

Read the applicable repository instructions and:
  doc/reconstructed_fault/particle_free/README.md
  doc/reconstructed_fault/particle_free/PLAN.md
  doc/reconstructed_fault/particle_free/VERIFICATION.md
  doc/reconstructed_fault/particle_free/STATUS.md
  doc/reconstructed_fault/particle_free/particle_free_history.tex

If the supplied ROOT_AGENTS_APPEND.md section has not yet been integrated,
merge it into the effective repository-root instruction file while preserving
existing instructions. Report which instruction files apply.

Execute PF0 and PF1 completely. Audit actual accepted-history laws and particle
dependencies, run one bounded existing baseline if the local build is available,
and implement/run independent reference checks V1--V4. Use existing test
conventions and an explicit local runtime budget. Create SOURCE_AUDIT.md and
PF1_REPORT.md, and maintain STATUS.md.

Do not change production constitutive behavior in this task. In particular,
do not silently add redistribution or spin, remove residual history, change
phase caps/filters/material parameters, or port the WENO limiter yet.
Distinguish exact transformations, discrete replacements, and physical changes.
A nonzero commutator or failed compact closure is useful evidence.

Continue routine edits/checks without asking at every step. If source behavior
is ambiguous, finish independent useful checks and present the alternatives
with source locations and mathematical consequences. Report measured defects
and a concrete PF2 proposal with real source paths, selected histories, and
minimal verification commands.
End this task after PF0/PF1; production implementation is the next task.
```

## Resume the same task

```text
Resume the particle-free task from STATUS.md and actual repository state.
Read the latest user instruction, applicable AGENTS.md files, PLAN.md, relevant
verification cases, SOURCE_AUDIT.md, and the current phase report.
Reconcile branch, revision, dirty changes, and completed checks before continuing.
Do not repeat discovery or tests unless inputs changed or a concrete unresolved
risk requires it. Complete the requested scope and update the next action.
```

## Start a later implementation phase

Replace bracketed choices with the selected scope:

```text
Proceed with [PF2, or the named range of phases] using the recorded source audit
and verification evidence. The selected physical model is [name the audited
model and any explicitly accepted constitutive change].

Read STATUS.md, PLAN.md, relevant cases, and previous reports. Implement the
smallest selectable change, preserve the baseline/default path, and verify
behavior against declared references and tolerances. Continue through the named
phases when their technical gates pass. If a scientific choice is unresolved,
complete independent useful work and present that concrete decision instead of
guessing or tuning around it. Record commands, measured results, limitations,
and the next action.
```

This template does not preselect an unresolved physical change. PF0/PF1 should
make the proposed PF2 choice concrete.
