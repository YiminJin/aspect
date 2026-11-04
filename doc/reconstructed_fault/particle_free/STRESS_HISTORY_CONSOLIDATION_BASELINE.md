# Stress-history consolidation starting baseline — 4 November 2026

The user selected numerical-behavior-preserving consolidation, not continuation
of finite-end approximation experiments. Before restructuring, the actual
worktree is `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, HEAD
and local origin tracking reference `69e8a10a93a08718d31af961197899e320657c04`.
No fetch or push was performed. The subsequent baseline commit records the
completed local direct physical-stress implementation and final-publication
slip-guess fix distinctly from this consolidation.

The native source matches the tested direct-stress provenance. Its full unit
suite passed183170 assertions/152 cases. Retained comparisons cover constant
± and layered + direct/separated cases, refinement, rejection/cutback and
2→4-rank continued restart. The unchanged constant/default controls pass;
the older/newer layered arithmetic-bound failure remains recorded and the
fresh corrected separated control is the comparison reference. See
[DIRECT_TRUE_STRESS_REPORT.md](DIRECT_TRUE_STRESS_REPORT.md). The subsequent
[frozen comparison](FROZEN_FINITE_END_REPORT.md) changed analysis/docs only.

`benchmarks/reconstructed_fault/particle_free/evidence/stress_history_consolidation/`
contains `entry.json`, `entry.patch`, and `entry-source.tar.gz`. Entry identifies
actual source/build inputs, dirty/manual work, five current binaries and33
already protected controls. Five additional `consolidation-entry-*` copies in
the existing stage-c controls directory preserve the current binary/checker
pairs before any rebuild. Runtime OpenMPI libraries remain at their existing
locations. Binary identity is established by hashes, not embedded revision text.

This baseline retains defaults, direct/separated representation and checkpoint
meanings, incoming-gradient spin lag, Gauss rules, convergence/publication
criteria, mature/cohesive separation, and all existing failures. Refactoring
equivalence is not new scientific qualification. No output/checkpoint or manual
file is deleted; inherited39/45 native-pilot-frame and temporary-log losses
remain. The active consolidation will report separately against this baseline.
