# Q1 workaround verification

One new Stage-I initialization was run with `gmg_q1.prm`, the unchanged existing
Release core and the matched Stage-I verifier plugin. Exit status is zero.
The core was not rebuilt in this follow-up. GCC 12.4/OpenMPI/deal.II 9.6.2 is the
local stack; this is not the Intel-server result.

The run log confirms `MappingQ1`, the Stage-I verification marker, and final
printed normalized bulk/fault residuals `9.165224e-08 / 0`. The diagnostic log
contains 28 completed eigenvalue estimates and four completed consumers.
This qualifies the bounded local fresh-start workaround; it does not qualify
long trajectories, restart equivalence or the complete server Stage-I case.

Input, flattened original PRM, build log, run log, diagnostics, executable/plugin
hashes (in provenance) and exit status are adjacent. All six current failing
server-output files passed the `server-inputs.sha256` preservation check. The
existing core hash also passed `core-before.sha256`. A compact snapshot of the
decisive server probe/PRM/log is under `server-context-release-observed/`, so
later updates to the upload directory do not erase this comparison.

Shell syntax and whitespace checks passed. No production C++ or BP3/BP5 input
was edited for the workaround, and no Python scripts were used.
