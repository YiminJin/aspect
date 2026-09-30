# Codex instructions for reconstructed-fault development

The authoritative design specification for the reconstructed-fault
framework is:
    doc/reconstructed_fault/current_design.md
    doc/reconstructed_fault/specification.tex

For every task related to reconstructed-fault development:

1. Read the relevant sections of the specification before editing code.
2. Treat the specification as authoritative for scientific algorithms,
   architecture, ownership, and MPI design.
3. Treat the current source tree as authoritative for existing APIs,
   actual class/function names, and existing reusable infrastructure.
4. If the specification conflicts with the current implementation,
   stop and report the conflict instead of silently choosing a new design.
5. Implement only the stage explicitly requested by the user.
6. Do not introduce duplicate physical/numerical parameters when an
   existing ASPECT/phase-field interface already provides the quantity.
7. Reuse existing deal.II and ASPECT infrastructure where the
   specification requires it.
8. Do not use the core-phase-field algorithm for the reconstructed fault.
9. Do not use the old PhaseFieldRSF architecture as the design basis.
   Isolated numerical routines may be reused only where permitted by
   the specification.
10. Do not redesign scientific algorithms without explicit approval.
11. Inspect callers, tests, and relevant headers before changing an
    existing interface.
12. Run relevant non-destructive tests after implementation and report
    what was and was not tested.
13. Follow: `doc/reconstructed_fault/refactoring.md`
    for code-quality, assertion/validation, readability, and
    responsibility-boundary guidelines. Prefer straightforward scientific
    C++ whose structure follows the mathematical algorithm. Do not
    introduce additional abstraction merely to reduce line count or
    eliminate small amounts of duplication.

## Reconstructed-fault session recovery

- Read doc/reconstructed_fault/CURRENT_STATUS.md for implementation
  status, evidence, and unresolved questions. Treat suggested next
  tasks there as context. The user's current instruction determines
  the active task; do not automatically resume a scientific experiment.

## Refactoring workflow

For tasks explicitly designated as refactoring:

- Module boundaries are defined in doc/reconstructed_fault/refactoring.md.
  M1 particle domains/CPDI and M2 phase-field implementation are frozen;
  M3 reconstructed-fault infrastructure, M4 fault material physics, and
  M5 coupled mechanics are the active refactoring scope. Only the stage
  explicitly selected by the user is authorized; boundary-interface changes
  and prerequisite cleanup require separately selected tasks.

- Follow doc/reconstructed_fault/refactoring/refactoring_plan.md.
  The plan describes the roadmap; it does not authorize executing
  every stage. Perform only the stage/pass selected by the user.

- Confirm the working directory, branch, and local changes before
  editing. Keep source edits, builds, and outputs within this
  refactoring worktree or its designated directories. Do not modify
  the separate scientific-test worktree.

- Preserve numerical algorithms, parameter defaults, MPI ownership,
  state-publication semantics, and checkpoint compatibility.
  Report suspected numerical bugs separately from structural changes.

- Keep code movement, structural changes, formatting, and numerical
  fixes distinguishable in commits or diffs.

- When the user reports manual edits, reread the affected files and
  inspect staged and unstaged changes before continuing. Preserve
  unrelated edits.

- Use the agreed stage-specific verification. Record pre-existing
  failures separately; do not loosen tolerances or change expected
  numerical results to make a refactor pass.

- Finish the selected pass with a concise review report: revisions,
  changed responsibilities/interfaces, verification results, remaining
  limitations, and one proposed next task. Await the user's selection
  before beginning another pass.
