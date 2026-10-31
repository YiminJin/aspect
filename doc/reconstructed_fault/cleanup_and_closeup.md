Perform a thorough disk cleanup of this worktree, concentrating on `build-particle-free`, and prepare a handoff for a new research-review thread. Execute the cleanup; do not stop after producing a proposal.

The worktree occupies over 80 GB. Previous cleanups recovered only 5 - 10 GB each. This time, inspect all major space consumers and remove obsolete generated artifacts systematically. There is no arbitrary recovery quota: finish when the remaining substantial disk usage has a concrete justification.

1. Establish the actual disk usage and active dependencies.

Read applicable repository instructions and the relevant status/evidence indexes. Record the absolute worktree path, Git HEAD, branch, and existing tracked/untracked changes.

Measure allocated disk usage for the worktree, `build-particle-free`, its major subdirectories, and the largest files. Record filesystem free space. Descend into large directories rather than relying on the supplied top-level listing.

Identify running simulations/builds, executable and plugin dependencies, active output directories, and checkpoint references. Do not terminate jobs or alter files they are using. Do not follow symlinks into other worktrees, shared installations, or external run directories.

2. Identify the protected material precisely.

Protect all unique source code, including uncommitted and untracked source, parameter files, scripts, documentation, and patches. Being ignored by Git does not make a file disposable.

Protect the original, still-unreviewed BP3 first-event results and their complete restart dependencies. Verified duplicate copies may be removed, but do not prune the original BP3 trajectory during this cleanup.

Protect the current RSF-episode continuation checkpoint and complete dependencies. The latest supplied report identifies step 748, t=374.58003334330067 s, on four ranks; verify whether a newer accepted checkpoint now supersedes it. Retain the current restart generation and one useful preceding complete fallback, including all rank files, mesh/history records, and metadata required by the restart mechanism.

Preserve one authoritative, complete server continuation package, its source/provenance manifest, launch instructions, and required validation evidence. Distinguish this from the separate BP3 first-event package.

Retain the working executable/plugin builds needed for current BP3 and cohesive-RSF work, including their runtime dependencies. Preserve any additional build only if its specific purpose is still active.

3. Reduce completed experiments to their necessary evidence.

For completed or superseded experiments, retain a compact evidence set sufficient to support the conclusions we still rely on: exact inputs, recoverable source version, commands, reports, quantitative summaries, and the raw data needed to reproduce retained figures or convergence comparisons.

Preserve evidence of unresolved failures and limitations as well as successful tests. Keep essential regression/restart fixtures. Do not erase a negative result merely because a newer experiment exists.

However, do not preserve an entire historical run directory merely because a report mentions it. Identify the actual supporting files. Large visualization sequences, repeated exports, temporary diagnostics, redundant checkpoints, and duplicate runs can be deleted when they no longer support an active analysis or unique scientific claim.

Do not assume an expensive simulation is disposable merely because it could theoretically be rerun. Preserve unique, costly results still relevant to the paper.

4. Remove obsolete generated data, largest consumers first.

Inspect every experimental family, including `pf*`, `p2-*` through `p5*`, `bulk-*`, `coulomb-*`, `mature-*`, `cohesive-interface-*`, `bp3-*`, `runs`, and `archive`. These names identify inspection targets, not blanket deletion targets.

Remove verified disposable artifacts such as:

- Superseded build trees, object files, binaries, and caches whose exact source/configuration remain recoverable and which are not current dependencies.
- Duplicate baseline/plugin builds and repeated test-run directories.
- Intermediate visualization files and redundant field exports from completed diagnostics.
- Obsolete restart generations, preserving complete required restart sets.
- Temporary unpacked package copies and superseded archives.
- Redundant source snapshots, but only after verifying that their exact contents—including any dirty changes—remain recoverable.

Inspect `archive` rather than exempting it. Moving files into another directory on the same filesystem does not reclaim space. Do not create a full backup of everything being deleted, and do not count moving files to Trash as recovered space.

Do not delete anything merely because it is old, large, ignored by Git, or has a particular filename extension.

5. Execute with an explicit deletion manifest.

Before each deletion batch, record the exact paths, allocated sizes, reason for deletion, and any retained replacement/evidence location. Validate that deletion targets are inside the intended cleanup scope and exclude protected material.

I authorize deletion of verified disposable generated artifacts within `build-particle-free` under these rules. Proceed without requesting confirmation for each batch. If an item is genuinely ambiguous, preserve that specific item, explain the uncertainty, and continue cleaning other items.

Do not use blanket `git clean`, reset source changes, remove the worktree, or delete shared dependencies. For large cleanup candidates outside `build-particle-free`, report their paths and sizes separately.

Do not reorganize retained run paths unnecessarily; existing scripts and restart inputs may depend on them.

6. Verify the result without regenerating the deleted data.

After cleanup, compare disk usage and filesystem free space against the initial measurements. If they disagree substantially, investigate hard links, snapshots, or deleted-but-open files as appropriate; do not kill processes to force recovery.

Check that source changes remain intact, retained reports point to available evidence or explicitly documented replacements, and the protected packages/checkpoint sets are complete.

Use existing lightweight validation commands where useful. Any startup or zero-advance restore check must use an isolated temporary output location and must not overwrite protected results. Do not rerun the historical test matrix, rebuild obsolete variants, or launch new scientific simulations.

Continue inspecting until every retained top-level item over 500 MiB has a specific explanation of what it contains and why it remains necessary. If little space can be recovered, demonstrate exactly which protected data account for the remaining footprint.

7. Deliver a cleanup report and research handoff.

Place these documents in the repository’s existing documentation structure, outside disposable build output:

`CLEANUP_REPORT.md`:
- Before/after worktree and build-directory sizes.
- Actual filesystem space recovered.
- Largest deleted groups and their justification.
- Largest retained groups and their justification.
- Locations of current executables, checkpoints, server packages, and scientific evidence.
- Anything that now requires rebuilding or re-extraction.
- A concise retention policy for future test outputs.

`RESEARCH_HANDOFF.md`:
- Exact worktree, source version, and uncommitted-work status.
- Current build/run commands and environment requirements.
- BP3 status, explicitly separating verified observations from unknown completion status.
- Current propagation/RSF status and unresolved numerical/scientific limitations.
- The distinction between production BP3 and the newer cohesive-interface formulation.
- The revised paper plan: BP3; controlled viscoelastic-crust experiment; fault propagation. No realistic subduction application.
- Immediate next work: inspect existing BP3 results and prepare variable, timestep, and energy maps connecting the mathematics to implementation.
- Links to authoritative design documents and retained evidence.

This task is cleanup and handoff only. Do not change constitutive laws, solver behavior, physical parameters, or scientific conclusions, and do not launch a server run.
