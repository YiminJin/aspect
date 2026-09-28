# Server coarse-probe follow-up, 2026-09-26

**Newer corrected Release results:** The latest upload now runs all three
context cases. Both Cartesian cases fail with NaNs; the Q1 parent case passes.
See [the isolated failure and Q1 workaround](mapping_cartesian_workaround.md).
The next task is a Stage-I run with the prepared `gmg_q1` input, not another
context rerun. The baseline-only discussion below describes the earlier upload;
the Release output path has since been replaced by the corrected failure output.

## Latest copied context outputs: baseline only

The directories were found under `server_gmg/output-coarse-context-debug/` and
`server_gmg/output-coarse-context-release/` (not `gmg_server/`). Both runs pass,
but their actual probe files contain **only**:

```text
case=q1_active refinements=0
...
coarse_probe_pass=1
```

Each file is 32 lines with one case; none of `cartesian_active`, `q1_parent`,
or `cartesian_parent` appears. Evidence:
[Debug probe](output-coarse-context-debug/coarse_diagonal_probe.txt),
[Release probe](output-coarse-context-release/coarse_diagonal_probe.txt).
Thus these outputs repeat the successful baseline; they do not yet test the
mapping/hierarchy hypotheses. No new conclusion about the Stage-I NaN follows.

The fixture previously selected context cases using
`ASPECT_FAULT_GMG_COARSE_CONTEXT=1`, set by `run.sh`. Merely loading the previous
context PRM did not select those cases. The `q1_active` label shows that the
updated plugin ran its baseline branch; the copied files do not record the
launch environment, so the exact reason the switch was inactive is unknown.
This was an avoidable weakness in the fixture and generic success check.

The selection is now explicit in the PRM, saved in `parameters.prm`, and works
for direct launches. **Rebuild the coarse plugin from the updated source and
use the updated context PRM.** For a manually maintained server PRM, add:

```text
subsection Postprocess
  subsection Fault GMG coarse probe
    set Test mode = context
  end
end
```

No environment switch selects the test cases anymore. `baseline` remains the
default for the original `coarse.prm`; `ASPECT_FAULT_GMG_DIAGNOSTICS` still
independently controls raw-diagonal logging. An old plugin cannot parse the new
parameter, so it fails instead of silently selecting the baseline.

The runner now calls [check_context_output.sh](check_context_output.sh), requiring
exactly these three case headers and three successful results:

```text
case=cartesian_active refinements=0
case=q1_parent refinements=6
case=cartesian_parent refinements=6
```

Copy the updated checker along with the runner. For direct launches, also run:

```bash
bash check_context_output.sh /path/to/new-output/coarse_diagonal_probe.txt
```

Run the corrected **Release context** case into a new directory using the
instructions below. The three context cases remain the next bounded task;
do not infer their completion from the directory name or repeat the baseline.
No core rebuild is needed for this selection correction. If all three pass,
the subsequent instrumented Stage-I measurement described below remains useful.

Local validation of this correction: the plugin rebuilt successfully, and a
direct ASPECT launch with `ASPECT_FAULT_GMG_COARSE_CONTEXT` explicitly unset
executed and passed all three cases. The checker accepted the full context
result and rejected the supplied baseline-only result and a truncated two-case
result. Numerical output matches the previous local context reference exactly.
Evidence: [verification-context-selection-local](verification-context-selection-local/).
The supplied files were preserved and hash-checked. No core source or numerical
algorithm changed in this correction; no Python scripts were used.

## What the new evidence establishes

Both supplied server runs pass, including Release with eight AVX512 lanes:
[Debug](output-coarse-debug/coarse_diagonal_probe.txt),
[Release](output-coarse-release/coarse_diagonal_probe.txt).
Both production inverse entries, production matrix-vector diagonal entries,
test-compiled raw diagonals and independent FEValues references agree. The
production inverse values are identical between the two runs:
`4.6875000000630927e-12` and `4.6875000000630919e-12`.

This establishes that the production operator can compute this one-cell Q1-mapped
diagonal correctly on the server in Release. It does **not** establish that the
original Stage-I problem is fixed or that every optimized operator path works.
The [previous Stage-I evidence](output-gmg-release/fault_gmg_rank0.log) still
shows the first observed NaN at level-zero inverse-diagonal DoF 16. The newly
quoted CG exception does not show whether diagnostics were enabled or establish
its call stack; it should not replace that more specific saved evidence.

The supplied build information resolves several previous questions:

- [environment.txt](environment.txt): `mpicxx` invokes **icpx/IntelLLVM
  2024.0.0**, with Intel MPI 2021.11. This is not evidence of classic `icpc`
  being the C++ compiler.
- [CMakeCache.txt](CMakeCache.txt) and [config_aspect.sh](config_aspect.sh):
  ASPECT is DebugRelease, unity build ON, direct `icpx` C++ compiler, and already
  appends `-fno-finite-math-only -fp-model=precise -ffp-contract=off`.
- [detailed.log](detailed.log): deal.II 9.6.0 was also configured with IntelLLVM
  2024.0.0, DebugRelease and 512-bit vectorization. Common exported flags include
  `-march=native`, `-qopenmp-simd`, `-fno-finite-math-only`; Release adds `-O2
  -funroll-loops -fstrict-aliasing`; Debug adds `-Og -ggdb
  -ffp-exception-behavior=strict`. Different dependency libraries are linked for
  Debug and Release.

Do not repeat a recommendation to add the three ASPECT FP flags: they are
already configured. The generic cache entry `CMAKE_CXX_FLAGS_RELEASE=-O3` alone
does not establish the final optimization level; ASPECT builds its two targets
using deal.II's mode-specific flags. The verbose unity compile command remains
the authoritative record of actual flag ordering. Configuration agreement is
not runtime library/hash proof, but there is no demonstrated compiler/MPI mismatch
in these files and no established Intel compiler bug.

## A gap in the first reduced model

Source inspection found two concrete differences worth isolating:

| Setup | Passing original coarse probe | Failing Stage-I setup |
|---|---|---|
| Mapping class | Explicit `MappingQ1<2>` | `MappingCartesian<2>` selected for flat box geometry |
| Level-zero cell | Active; one level | Parent of refined cells; seven levels |

See [construct_mapping](../../../source/simulator/core.cc) and
[coarse_diagonal_probe.cc](coarse_diagonal_probe.cc). On a unit square the
mathematical mapping is the same, but the setup implementations differ. Neither
difference has yet been shown to cause the server failure. The standalone probe
also bypasses material projection and the full coupled initialization/history
of allocations. If mapping/hierarchy tests pass, those contextual differences
remain relevant, as do binary identity and earlier memory corruption.

The probe now offers three **new** cases through `coarse_context`:

1. `cartesian_active`: change only the mapping from the completed baseline.
2. `q1_parent`: retain Q1 mapping but add six uniform refinement levels.
3. `cartesian_parent`: combine both Stage-I setup choices.

Every case still checks only the 18-DoF, unit-square level-zero operator against
the same analytic and FEValues references. It solves no Stokes, advection, RSF
or phase-field equations. The parent cases create 4096 fine cells for hierarchy
setup, with no physical evolution. A comparison mismatch is recorded and the
remaining cases are evaluated; a thrown exception/signal can stop earlier.
The completed `q1_active` baseline is not repeated by this case selector.

## Run this next on the server

First rebuild **only the plugin** against the same existing ASPECT build used
for the failing Stage-I run, preserving all original executables and outputs.
No compiler-option changes or dependency rebuilds are needed for this step.
Transfer the updated `coarse_diagonal_probe.cc`, `coarse_context.prm`, `run.sh`, `check_context_output.sh`
and package CMake file along with their existing includes.

```bash
export GMG_SOURCE=/absolute/path/to/aspect-source
export GMG_ASPECT_BUILD=/absolute/path/to/existing/DebugRelease-build
export GMG_CONTEXT_BUILD="$SCRATCH/gmg-context-${SLURM_JOB_ID}"
mkdir "$GMG_CONTEXT_BUILD"
set -o pipefail
cmake -S "$GMG_SOURCE/benchmarks/reconstructed_fault/server_gmg" \
  -B "$GMG_CONTEXT_BUILD/plugin" -DAspect_DIR="$GMG_ASPECT_BUILD" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  2>&1 | tee "$GMG_CONTEXT_BUILD/configure.log"
cmake --build "$GMG_CONTEXT_BUILD/plugin" -j2 --verbose \
  2>&1 | tee "$GMG_CONTEXT_BUILD/plugin-build.log"

# Run inside a compute allocation, exactly one rank.
bash "$GMG_SOURCE/benchmarks/reconstructed_fault/server_gmg/run.sh" coarse_context \
  "$GMG_CONTEXT_BUILD/release" "$GMG_ASPECT_BUILD/aspect-release" \
  "$GMG_CONTEXT_BUILD/plugin/libfault_gmg_coarse_probe.release.so" ibrun -n 1
```

If a case fails, preserve the result and stop here; the failed case identifies
which setup difference to investigate. A matched Debug context run may then be
useful. If all pass, the reduced model still lacks the triggering context; take
the next measurement in Stage I rather than trying more global compiler flags.

## If context tests pass: raw diagonal in the original failing run

An additional observational change is prepared in
[matrix_free_operators.cc](../../../source/simulator/solver/matrix_free_operators.cc).
With the existing `ASPECT_FAULT_GMG_DIAGNOSTICS=1` switch, it records four stages
of a small (at most 128 global DoFs) level-zero A-block diagonal to **stderr**:

- `initialized`: after allocating/zero-initializing the vector, before the helper;
- `raw`: immediately after `MatrixFreeTools::compute_diagonal`;
- `constrained`: after setting constrained entries to one;
- `inverse`: after the original reciprocal loop.

It also records mapping type, compressibility/dilation flags and viscosity table
width. It does not zero, repair, recompute, or replace any diagonal. It adds no
MPI collective and does not change the existing assertions or solver controls.
It is off by default. Use one rank for readable stderr. The existing per-rank
`fault_gmg_rank0.log` still records the broader setup and stops on invalid data.

Apply this observational change to the matching server source and rebuild with
the **existing cache**, saving the verbose build output. Include the existing
`fault_gmg_diagnostics.h` and local-smoothing instrumentation. Then run one
bounded initialization through the runner's `gmg` case using its matching
`libfault_gmg_stage_i.release.so`. The runner sets the diagnostic switch,
captures stderr in `run.log`, refuses an existing output directory, and has a
300-second timeout. The prior README gives the full command syntax. Retain the
old binary or record its hash before rebuilding.

Interpret the first bad stage, not just the later CG exception:

| Observation | Localization |
|---|---|
| Initial owned entries not zero | Allocation/initialization or prior corruption |
| Initial entries zero; raw free entries NaN | Basis evaluation, kernel, integration, helper accumulation or inputs in this context |
| Raw free entries good; constrained entries bad | Constraint handling/indexing or corruption |
| Constrained free entries good; inverse bad | Reciprocal loop/code generation or corruption at that boundary |
| All diagonal stages good; CG still fails | Inspect existing scratch-action and eigenvalue markers; the previous diagonal localization does not describe this execution |
| Instrumentation makes failure disappear | Record sensitivity; do not call it a numerical fix |

No extra broad configuration dump is needed now. Please return the new runner
result directory and plugin build log. If Stage I is needed, also return its
full `run.log` and verbose Release unity compile command. Confirm whether the
old coarse and Stage-I runs used the same executable without an intervening
rebuild, and whether the just-quoted CG failure had diagnostics enabled. The
copied output directories do not contain executable hashes or full stderr.

## Local validation and limits

All three context cases pass locally with GCC 12.4, OpenMPI, deal.II 9.6.2 and
four SIMD lanes, both before and after compiling the new observer. The complete
96-line numerical probe outputs compare byte-for-byte equal. The new stderr
shows zero initialized entries, finite raw diagonals near `2.133333333304619e11`,
and the expected reciprocals in all three cases. The core and plugin builds
succeeded; shell syntax and whitespace checks passed. Evidence is preserved in
[verification-context-local](verification-context-local/). This does not test
IntelLLVM/AVX512 or reproduce the server failure.

No completed Stage-I/BP3/BP5 runs were replayed locally in this follow-up, no
Python scripts were used, and the supplied server artifacts were hash-checked
unchanged. A source fix for the NaN is still unproven; these changes only narrow
the next observation.
