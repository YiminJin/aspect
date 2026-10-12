# Start a dedicated propagation session

Use a new Codex session for this research task. Prefer a separate Git worktree and a task branch such as pf-rsf-propagation, based on the exact current refactored commit you want to preserve. Keep the existing session/checkout for BP3 runs and maintenance. A new session does not require starting the implementation from scratch.

## Install without overwriting repository guidance

1. Extract this package into a staging directory outside the active ASPECT checkout. Do not unzip it over the repository root: an existing AGENTS.md must be preserved.
2. Identify the current tested/refactored base commit. pf-rsf-refactor is the known historical branch, but a newer branch or uncommitted fixes may now contain the intended baseline. A worktree created from a commit does not automatically include uncommitted changes. Preserve and identify those before choosing the base; do not stash, reset, or commit another session's work automatically.
3. Create or select the dedicated worktree and task branch. Keep its build and output directories separate. Codex can perform this setup from the existing repository after the baseline is clear; it should not switch the old session's checkout.
4. Read the existing root and applicable nested instructions. Append the supplied AGENTS.md task block to the dedicated worktree's root AGENTS.md, without replacing existing content. If an AGENTS.override.md is active, do not create or remove an override to bypass existing rules; read both and make the task instructions explicit in the prompt.
5. Put PLAN.md, STATUS.md, and this START_HERE.md in benchmarks/reconstructed_fault/propagation/. This is a proposed task-document directory; P0 may adjust it to repository conventions and update the references together. Runtime results should use a separate output directory.
6. Start Codex from the dedicated worktree root. Use the launch prompt below. If setup has not yet been performed, give Codex the actual absolute path to this staging package so it can read and install it safely before P0.

The standard instruction filename is AGENTS.md (plural). A file only inside a benchmark folder is not sufficient to convey this task to a session launched at the repository root. Do not change global ~/.codex settings for this one task.

## Initial prompt to paste

We are starting a dedicated ASPECT task to verify non-frozen phase field and, later, fault propagation. The current refactored code has not yet been established to work with evolving phase field. The existing BP3/refactor work and its running jobs must remain separate.

Read the existing repository instructions and the supplied propagation handoff: AGENTS.md, PLAN.md, and STATUS.md. If they are still in the staging package I provide, install the task documentation without overwriting existing instructions. Establish the actual intended refactored baseline; use a separate worktree/branch and build/output directories where needed, preserving all existing work. pf-rsf-refactor is a historical hint, not a substitute for checking the current repository.

Execute P0 only. Inspect the production phase-field and coupling code paths, history/irreversibility handling, the frozen-field and fixed-geometry assumptions, and the existing small fixtures. Produce P0_report.md with real symbols and file paths, a capability/gap table, and a concrete minimal P1 phase-field-only test proposal with commands, diagnostics, and estimated cost. Update STATUS.md.

Do not modify production physics, unfreeze the phase field, implement geometry extension, or start a production cluster run in this stage. A known short existing smoke test is acceptable if a matching executable is already available. Complete the audit and stop with the P1 proposal for review. Do not implement the whole roadmap.

## When P0 is reviewed

Use the same new session for P1, P2, and subsequent approved stages. Do not create another session at every small stage. Update the status and evidence files so compaction or a later replacement session does not lose scientific decisions. If a stage proposal needs revision, change the plan before changing the model.

For a later stage, an instruction such as “Proceed with P1 as revised in P0_report.md; complete its checks and report before P2” defines the intended scope. Multiple stages can be authorized together later if the work becomes routine.

## File roles

| File | Role |
| --- | --- |
| AGENTS.md | Concise task rules to merge into existing root instructions |
| PLAN.md | Scientific sequence, experiments, evidence, and later roadmap |
| STATUS.md | Small factual handoff updated as work proceeds |
| START_HERE.md | Setup guidance and the initial prompt |

Official guidance for instruction discovery: https://developers.openai.com/codex/guides/agents-md

This package contains planning infrastructure, not an executable prm, code changes, or a completed repository audit.
