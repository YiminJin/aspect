# Native FE BP3 server package provenance — 2026-10-14

Base: fault-propagation at c33edaaeb145e5d3dce155bb815552ef80efc5f9,
plus the current uncommitted implementation/documentation changes. These changes
are already applied in the source snapshot; do not reapply source-changes.patch.

The archive contains 11989 regular ASPECT/source-data files and
74 preserved source symlinks, including bundled World Builder
and Catch, plus the separate BP3 job directory. It includes every tracked file
at its current worktree contents and the required untracked BP3 runtime,
fixtures/reports and any source/include/build-helper additions. It excludes
.git, local build caches/binaries, untracked historical run output, credentials
and installed compiler/MPI/deal.II/Trilinos/p4est/Voro++ libraries.

The archive-root source-inputs.sha256 manifest has SHA256:
5170863e8655a4a83ee94aa341d7c08a9f68c9e514c4fe41afe683179ab6d9e8
It names ASPECT source files relative to the archive root. SHA256SUMS covers
all regular delivered files; source-symlinks.json records link targets.

All BP3 plugin C++ and header files are exact copies of the maintained ordinary
plugin. plugin-build-flags.patch records the only CMake difference: compiler
options preserving finite-value checks, precise Intel arithmetic and disabled
FP contraction. BP3_LOCAL_OSCILLATION_TEST remains OFF; the bottom-test and
accelerated-loading definitions are absent. No physical law/default is changed.

The standalone input is the section-8 fresh full-BP3 candidate, with only
relocatable executable/plugin/fault/output paths and native memory observations
added. Fine cells 3.90625 m, requested surface 10 m, q6, actual BP3 spatial
friction/state/loading, filter 20/full bottom, stationary accepted FE history,
no insertion/deletion, no checkpoint/resume and abort on nonlinear failure.
Timestep caps 4e6/4e7 s and unweighted log-state bound .2 are retained from the
latest intended local BP3 inputs. Physical/end/wall and event stops are in
parameters.prm; no server job is launched by this package.

Earlier unchanged-source physical evidence: local sections 7B,7C/7D and 8 reports
in aspect-source/benchmarks/reconstructed_fault/bp3_native/. Their numerical
limits remain in force. Current package checks are in VERIFICATION.md. Server
compilers/module names, MPI layout and fine/event results have not been verified.
Local tested core executable SHA256:
1ba2f7f43b37b75b1462d9cd5e48998fc8cc58b891e943829be8240f70776d12
It is not included or usable as a substitute for the server build.
