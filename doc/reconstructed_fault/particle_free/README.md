# Particle-free phase-field RSF: working package

Target worktree: `aspect-pf-rsf-particle-free`.
Target branch: `pf-rsf-particle-free`.
Prepared: 15 October 2026.

Local PF0/PF1 execution is complete: see [STATUS.md](STATUS.md),
[SOURCE_AUDIT.md](SOURCE_AUDIT.md) and [PF1_REPORT.md](PF1_REPORT.md).
In this installation the derivation is [../particle_free_history.tex](../particle_free_history.tex)
and the duplicate standalone plan is [../particle_free_plan.md](../particle_free_plan.md).
The root instructions are already integrated; no `ROOT_AGENTS_APPEND.md` was
provided. The installation instructions below describe the original package.

The goal is to determine which particle-free history representation preserves
the intended model, then verify a bounded implementation. The propagation branch
is the starting point. Its source and build have not been inspected in the
environment that prepared this package.

## Install and start

1. Create the worktree from the intended propagation revision. For example,
   from the propagation checkout, after checking `git status --short --branch`
   and `git rev-parse HEAD`, use:

   ```sh
   git worktree add -b pf-rsf-particle-free ../aspect-pf-rsf-particle-free HEAD
   ```

   This assumes the branch and destination do not already exist. Worktree
   creation uses the committed revision; uncommitted propagation edits need a
   deliberate transfer if they are part of the intended starting state.
   Do not reset or clean the original checkout.

2. Extract the package into a temporary directory. Merge its documentation
   subtree into `doc/reconstructed_fault/particle_free/` in the new worktree.
   Preserve existing files rather than blindly overwriting them.
3. Merge `ROOT_AGENTS_APPEND.md` into the repository-root `AGENTS.md`.
   If absent, create `AGENTS.md` with that content. Do not install the fragment
   as `AGENT.md` or rely on instructions stored only below `doc/` to govern
   source-code work. An existing root `AGENTS.override.md` takes precedence
   at that level: preserve its instructions and integrate the task entry point
   into the effective file if needed; do not bypass existing guidance.
4. Place the revised derivation here as `particle_free_history.tex` and,
   optionally, `particle_free_history.pdf`. This package does not duplicate it;
   PF0 records the actual file hash. If available, place the historical limiter
   and Zhu et al. paper in a reference location or record their existing paths.
   Their absence does not block PF0/PF1, only a later source-specific limiter audit.
5. Launch a fresh Codex session from the repository root after installing the
   instructions. Use the first prompt in `PROMPTS.md`.

`AGENTS.md` is the recognized instruction filename. A root instruction explicitly
directs Codex here; planning filenames are not assumed to have special automatic
behavior. Keep the root addition concise and load the detailed plan on demand.

## Files and ownership

| File | Purpose |
|---|---|
| `particle_free_history.tex` | User-provided mathematical reference and limitations |
| `PLAN.md` | Phases, dependencies, deliverables, and decision points |
| `VERIFICATION.md` | Independent references, observables, acceptance criteria |
| `STATUS.md` | Current state, evidence links, decisions, and resume action |
| `PROMPTS.md` | Initial task and continuation prompts |
| `SOURCE_AUDIT.md` | Created in PF0: actual history laws and source locations |
| `PF1_REPORT.md`, etc. | Created when the corresponding work is performed |

Use existing repository test conventions. Select one small test directory and
one ignored output directory during PF0; record their paths in `STATUS.md`.
Do not create a new testing framework or empty report templates.

## First-task outcome

PF0/PF1 audits source and verifies algebra using independent references. It does
not change production constitutive behavior. It determines whether:

- The propagation implementation matches the cohesive construction.
- Existing histories/geometry require nonzero commutator terms.
- Differences are discretization changes rather than algebraic equivalence,
  especially for CPDI phase integration.
- A specific minimal particle-free implementation is justified next.

Do not label the entire method verified when only these checks pass.
The first candidate is a fixed-geometry, constant-parameter cohesive path,
conditional on the source audit. Direct mature stress transport is a useful
reference/control, not an instruction to change propagation into a mature model.

## Instruction-file references

Official guidance, checked 15 October 2026:

- [AGENTS.md discovery and scope](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [Execution plans](https://developers.openai.com/cookbook/articles/codex_exec_plans)

The second reference is an archived workflow example. This package uses the
general idea of maintained plans, not its model recommendations or exact template.
