# R7b — bounded verification gap closure

Qualified core revision: **`6a3781277f2f594d52cd248d84f5b6fdc98fdeb1`**.
No production source, header, parameter-default, ownership or numerical change.
The only committed change since qualified post-R6 `0f66d9869` is the user's
53-line CMake package check. R7b adds local test fixtures, evidence and review
updates; it does not implement R8 or qualify production BP3 physics.

The immutable reference is `build/reference-aspect`, SHA256
`f954b70bb1aae8168e2473dc989fbf8eaee1c5ca770121905c04bae90e2866a3`,
copied before rebuilding the former post-R6 unity executable. Its old embedded
banner is not its source identity: provenance is the post-R6 qualification and
artifact hash. The new frozen artifact is `build/aspect-r7b-qualified`, SHA256
`ed74d4d4ad1cfa5d1bd4fbade496d9748ab298c24a259a4612a94d4df375b60d`.
[Manifest](evidence/qualified-manifest.json) records source identity, configs,
plugins, fixture hashes and preservation of all 65 unrelated incoming files.
Four existing guidance/status documents receive intentional R7b updates.

## Fresh checks

Stack: GCC 12.4, OpenMPI 5.0.6, deal.II 9.6.2, Trilinos 14.2/Epetra, Voro enabled,
Release with unity/PCH ON. Builds use the existing pinned floating-point flags.

| Check | Result / scope |
|---|---|
| Default Epetra configure and full core rebuild | PASS; final executable carries `6a3781277` |
| Maintained BP3 plugin, local oscillation diagnostics OFF | Build target and library-load/parameter validation PASS; this is not a new BP3 runtime qualification |
| Explicit Tpetra on actual deal.II 9.6 | Correctly rejected; `configure-reject-tpetra.log` |
| Package-check branches | 9/9 configure-only mocked cases pass; no actual deal.II 9.8/Tpetra compile claim |
| Ordinary AMG, AMG-BFBT, melt, GMG | Four matched serial cases pass, including actual backend log markers; statistics, available field output, solver decisions and work counts match |
| Ordinary phase/fault-disabled particles | Two ranks, through step 5; native mesh/particle outputs present; empty output gate asserted. Same-build and baseline-to-candidate checkpoint restarts reproduce steps 4/5 particle IDs, positions, properties, both RNG streams and owned bulk DoF values exactly |
| Prepared particle projection cache | Two ranks with nonempty owners: exact cold/warm interpolation and support; one cold rebuild, no warm rebuild. All-rank domain regeneration increments generation at exactly unchanged volumes, rebuilds once, then reuses |
| Empty-owner cache | Separate manufactured fixed-geometry probe: admitted particle counts 0 and 1. Both ranks pass the same generation/rebuild/value checks; no new production API or MPI operation |
| Cohesive/legacy restart | Original accepted-step checkpoint and restored history/V/geometry/bulk checks pass, including candidate loading the reference checkpoint. Step-two decisions and accepted checkpoint payloads match; known step-two line-search exhaustion persists in every trajectory |
| Singular surface factorization | Complete current Stage-F diagnostic/invalidation pass marker before intentional exit 1, on both executables; original stale-diagnostic test is untouched |
| Missing two-rank units | 7 cases per rank: 59,278 and 2,556 assertions pass (`phase_field_fault_ih_accuracy`, `phase_field_fault_ih_cache`, `fault_slip_restart`) |
| Focused Debug assertions | 69 assertions / 4 cases pass for boundary contact and condensation, compiled with `DEBUG` and Debug deal.II; not a full Debug simulator |

**206/206 exact comparison/outcome checks pass** in
[evidence/comparisons.json](evidence/comparisons.json). No compared numerical
values, solver decisions or work counts differ. Elapsed time, output-directory
names, VTK creation comments and gnuplot Date/Time headers are excluded. Gnuplot
rows and VTK data are exact; ordinary full-precision audit data are also exact.
Copied resume directories are not treated as proof of execution: fresh resumed
log markers and newer per-rank audit files for steps 4/5 are required.

## Fixture failures retained separately

The initial one-cell ridge-fit input (`empty-cache`) fails before cache probing:
a structural vertex has no phase-field support. Prescribed fixed geometry
(`empty-fixed-cache`) reaches the old I_h postprocessor, whose unconditional
rank-local sentinel assertion assumes admitted particles on every rank. Neither
is an MPI cache failure or a successful empty-rank test. The separate cache-only
plugin initializes finite varying nodal values; its first input omitted the
required `particles` postprocessor (`empty-owner`) and was rejected. The final
`empty-owner-qualified` input includes it and passes on both executables.
These trial inputs/logs remain available; no production threshold was changed.

Initial fixture build/configure errors (SolCx path, missing domain header,
particle-manager index, and a target requested before CMake regeneration) remain
in their original logs. Successful final logs are `build-plugin-qualified.log`,
`configure-empty-owner.log` and `build-empty-owner-final.log`. The initial
comparison script assumed `.txt` for native ASCII particle output; ASPECT emits
`.gnuplot`. `comparison-development.json` retains that harness failure; the final
comparator checks the actual files and excludes only their Date/Time headers.

The cache extension uses an existing const getter plus a **test-only** const cast
to invoke native domain regeneration collectively. `R7B_EMPTY_OWNER` enables
manufactured nodal initialization only in the dedicated empty-owner plugin;
the balanced case retains the original constitutive postprocessor. No ownership
or public-interface change is implied.

## Reproduction

Run from the repository root. Inputs retain absolute artifact/output paths for
this worktree; change their common prefix when relocating. Existing logs and
runtime output directories are protected against overwrite by the runners.
Preserve the baseline binary before rebuilding. Exact commands, exit status,
rank count, environment, timeout and peak RSS for each run are in its JSON log.

```sh
export OMPI_CXX=/opt/gcc/12.4.0/bin/g++
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
export ASPECT_SOURCE_DIR="$PWD"
cmake -S . -B build-refactor-post-r6-gcc12-unity -DASPECT_USE_TPETRA=OFF
cmake --build build-refactor-post-r6-gcc12-unity -j3
cmake -S benchmarks/reconstructed_fault/refactoring_r7b/plugin \
  -B benchmarks/reconstructed_fault/refactoring_r7b/build/plugin \
  -DAspect_DIR="$PWD/build-refactor-post-r6-gcc12-unity"
cmake --build benchmarks/reconstructed_fault/refactoring_r7b/build/plugin -j2
cmake --build benchmarks/reconstructed_fault/post_r6_cleanup/build-gcc12/maintained -j2
cmake -S benchmarks/reconstructed_fault/refactoring_r7b/debug \
  -B benchmarks/reconstructed_fault/refactoring_r7b/build/debug \
  -DDEAL_II_DIR=/opt/dealii/9.6-local -DCMAKE_BUILD_TYPE=Debug
cmake --build benchmarks/reconstructed_fault/refactoring_r7b/build/debug -j2
python3 benchmarks/reconstructed_fault/refactoring_r7b/check_cmake.py
python3 benchmarks/reconstructed_fault/refactoring_r7b/run_checks.py reference
python3 benchmarks/reconstructed_fault/refactoring_r7b/run_checks.py candidate
python3 benchmarks/reconstructed_fault/refactoring_r7b/compare.py
python3 benchmarks/reconstructed_fault/refactoring_r7b/record_manifest.py
```

The actual unsupported-stack rejection used a separate configure directory with
`-DASPECT_USE_TPETRA=ON`; do not flip the qualified build cache. The Debug/unit/
plugin-validate commands are recorded verbatim in their evidence JSON files.
The wrappers impose a 240-second runtime limit per case and 20 GiB aggregate RSS
limit; no case hit either limit. `run_checks.py` checks exits; `compare.py` also
requires intended failure/pass markers and matched data. A nonzero exit alone
never qualifies a known-failure fixture.

## Reused evidence and remaining limits

Post-R6 qualified ON/OFF builds, 964 assertions/39 cases per build, eight installed
header consumers, 2D/3D instantiation symbols and 961 matched mature/retry/restart/
birth comparisons per build remain evidence for **their recorded revision**.
The source/include tree is exactly unchanged through `6a3781277`; this is the
basis for reuse, not an assertion that those campaigns were newly rerun. The
repaired four-rank frozen AMG/GMG campaign also remains historical evidence.

Not qualified here: Intel/26.0, real deal.II 9.8/Tpetra, Voro-OFF, a full Debug
simulator, 3D runtime, complete test suite, production BP3 resource use/first event,
mid-publication failure injection, comprehensive AMR/migration/reordering, or
rank-asymmetric cache invalidation. Consistent all-rank cache entry remains a
caller contract; the local validity predicate is unchanged and does not enforce
agreement. No new production MPI correctness defect was demonstrated. The known
cohesive step-two nonconvergence is unchanged, not resolved.

R7b is complete for review. Recommended next bounded task: resolve the imported
Tpetra-option compatibility contract (a fail-fast guard/proposal, not a backend
port) before upstream preparation. No R8 work or production correction is made.
