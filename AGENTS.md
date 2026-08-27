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
