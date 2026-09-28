# Server GMG NaN diagnosis (2026-09-25)

**Resolved toolchain issue (user report, September 26):** GMG runs properly
after rebuilding both deal.II and ASPECT with `intel/26.0` instead of
`intel/24.0`. This supersedes the debugging next steps below; preserve the
historical evidence without repeating the completed tests.

**Cleanup:** production GMG logging, raw-diagonal observers and scratch-vector
probes have been removed. The standalone probes remain opt-in benchmark targets;
their shared helper now lives in this directory, not in `source/`. `run.sh`
checks the Stage-I/probe verification result and no longer sets or requires
`ASPECT_FAULT_GMG_DIAGNOSTICS` or `consumer_done`. The old `gmg_uninstrumented`
case is retained as an alias for compatibility. Descriptions of core diagnostic
logs and their installation below describe the historical investigation, not
the cleaned production code. The removed core patch is preserved under
[`gmg-bp5-cleanup-evidence`](../bp3/gmg-bp5-cleanup-evidence/).

This package diagnoses the reported Intel-build failure in
`SolverCG<distributed::Vector<double>>::solve(ABlockOperator<2,2,double>,
DiagonalMatrix<...>)`. It does not change the fault equations, timestep policy,
solver tolerances or smoother parameters. The optional `gmg_q1` input selects
the geometry-mapping workaround documented below.
No Python is used. Local verification and its limitations are recorded below.

**Latest result:** The corrected server Release context run fails for both
MappingCartesian cases and passes for MappingQ1 on the refined hierarchy.
See [the Q1 workaround](mapping_cartesian_workaround.md) and new `gmg_q1.prm`.
The PRM-only flat-box workaround now passes both local and server Release
Stage-I initialization. The new `output-gmg-q1.tar` contains the server
verification, with final bulk/fault residuals `9.165231e-08 / 0`. No core/deal.II
rebuild is needed. The linked analysis distinguishes this tested bypass from
the still-unresolved dependency defect and records a source-level alternative.
Do not repeat the completed Stage-I test merely to rebuild context.

**Earlier context-output check:** The first supplied `output-coarse-context-*` files
contain only `case=q1_active refinements=0`, so they repeat the baseline.
Context selection now uses the explicit PRM entry `Postprocess / Fault GMG
coarse probe / Test mode = context`, and the runner verifies all three expected
cases. Rebuild the test plugin and update the PRM/runner/checker; see the
[latest analysis](server_context_analysis.md). The context cases remain pending
on the server; no core rebuild is needed for this test-selection correction.

**Latest update (September 26):** Both server coarse probes pass, including
Release/AVX512. IntelLLVM and existing precise-FP flags are now verified from
the supplied configurations. Read the
[context analysis and next bounded test](server_context_analysis.md)
before running anything further. It adds mapping/parent-cell comparisons and
opt-in raw-diagonal logging in the original setup. No numerical fix is claimed.

**Earlier copied Stage-I Debug/Release outputs:** Release already has a NaN at
level-zero inverse-diagonal DoF 16, before eigenvalue estimation. Debug has the
analytic expected value for the same coefficient. Read the
[result analysis and one-cell follow-up](server_debug_release_analysis.md)
first; it supersedes the broad initial diagnosis below. The new `coarse` runner
case requires `libfault_gmg_coarse_probe` and one rank, and solves no equations.

## What the message establishes

The reconstructed-fault outer solver is FGMRES. In the current source,
`with_velocity_preconditioner()` calls `PreconditionChebyshev::estimate_eigenvalues`
on every velocity level. deal.II 9.6 uses a Jacobi-preconditioned CG/Lanczos solve
there, matching the reported template types. A failure at CG iteration 1 with
NaN is therefore consistent with **smoother setup**, not evidence of failure
of the outer coupled Newton/FGMRES solve. The full backtrace and diagnostic
phase markers must confirm the call site. See the
[deal.II 9.6 Chebyshev documentation](https://dealii.org/9.6.0/doxygen/deal.II/classPreconditionChebyshev.html).

Potential causes still to distinguish include invalid level coefficients or
diagonal, an operator/constraint/partition problem, a degenerate CG seed, and
compiler/vectorization or dependency incompatibility. “Intel mpicxx” alone
does not identify the actual underlying compiler or establish a compiler bug.
Record `mpicxx -show` and `mpicxx --version` before changing compiler options.

## Code to transfer and build

Use the same current modified ASPECT source as on the server, including existing
reconstructed-fault changes. The new instrumentation is in:

- `source/simulator/solver/stokes_matrix_free_local_smoothing.cc`
- `source/simulator/solver/fault_gmg_diagnostics.h` (**new file; include it**)
- `source/simulator/solver/matrix_free_operators.cc` (September 26 raw-diagonal observer)
- This entire `benchmarks/reconstructed_fault/server_gmg/` directory.

The test plugin compiles the existing `tests/phase_field_fault_stage_i.cc`, with
its existing test-access headers; PRMs include `tests/phase_field_fault_stage_i.prm`
and `tests/phase_field_fault_ih.txt`. Keep the checkout layout. Do not overlay
an old source archive or mix the BP3 runtime library into this fixture.

Build on the server against its own deal.II/MPI/compiler installation. Reuse the
known working server configure options (Voro++ ON is required by the fixture).
Use a separate build directory and capture the actual verbose compile commands.
For example, from the source root after loading the existing server modules:

```bash
module list 2>&1 | tee server-modules.txt
mpicxx -show
mpicxx --version
export GMG_SOURCE="$PWD"
export GMG_BUILD="$PWD/build-gmg-diagnostic"
export GMG_DEAL=/work2/11463/yiminjin/stampede3/software/dealii/9.6/INSTALL_PREFIX
# Replace INSTALL_PREFIX with the installed deal.II prefix from your existing
# CMakeCache.txt; the path to solver_cg.h is not necessarily that prefix.

cmake -S "$GMG_SOURCE" -B "$GMG_BUILD" \
  -DDEAL_II_DIR="$GMG_DEAL" -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_BUILD_TYPE=Release -DASPECT_WITH_VORO=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# Include your other existing server configure options above.
set -o pipefail
cmake --build "$GMG_BUILD" --target aspect -j4 --verbose 2>&1 | tee gmg-build.log
cmake -S benchmarks/reconstructed_fault/server_gmg \
  -B "$GMG_BUILD/server-gmg" -DAspect_DIR="$GMG_BUILD" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$GMG_BUILD/server-gmg" -j4 --verbose 2>&1 | tee gmg-plugin-build.log
export GMG_BINARY="$GMG_BUILD/aspect-release"
export GMG_PLUGIN="$GMG_BUILD/server-gmg/libfault_gmg_stage_i.release.so"
```

The first build should preserve the server's failing optimization/FP options;
do not simultaneously change compiler, MPI, mesh, precision, and diagnostics.
Record the old and new CMake caches and compile commands. In particular retain
any existing `ASPECT_ADDITIONAL_CXX_FLAGS`; the example does not supply them.
Debug information (`-g`) can be added for the backtrace without selecting Debug
libraries. A Debug ASPECT build requires a compatible deal.II Debug installation.
Do not change SIMD width through an ASPECT-only macro: that can change the ABI
relative to deal.II. Compiler/MPI/ABI consistency is required for every plugin.

## Small matched model and bounded runs

The model is the existing coupled Stage-I fixture: unit square, 64×64 cells,
36,864 initial particles, Q2 velocity, Q1 pressure, frozen AT1 phase, horizontal
fault, nonuniform initial stress/composition, fixed velocity boundaries,
rate-dependent friction with true normal feedback. It runs **initialization
only** (`End time=0`), exercising the reconstructed-fault velocity GMG path.
Its verifier requires genuine coupled convergence and committed bound states.
It is not a reduced physical BP3 trajectory or an RSF time-accuracy test.

Three inputs differ only by backend or diagnostic switch:

| Input | Purpose |
|---|---|
| `amg` | Matched control, same fine coupled equations and tolerances |
| `gmg` | Velocity GMG; runner sets `ASPECT_FAULT_GMG_DIAGNOSTICS=1` for level diagnostics and scratch operator probes |
| `gmg_uninstrumented` | GMG with the new diagnostic disabled; detect sensitivity to the extra probe/initialization |

Run **sequentially inside a compute allocation**, initially on one rank, then
two ranks. For Stampede3, use the allocation's supported `ibrun`; elsewhere pass
`mpirun -np N`. Every invocation has a 300 s TERM timeout plus a 30 s kill grace.
The runner refuses an existing output directory, captures provenance and retains
failures; it does not retry or loosen a criterion. Site launcher cleanup after
timeout should be checked before starting the next run.

```bash
export GMG_RESULTS="$SCRATCH/gmg-debug-${SLURM_JOB_ID}"
mkdir -p "$GMG_RESULTS"
export GMG_RUN="$GMG_SOURCE/benchmarks/reconstructed_fault/server_gmg/run.sh"

# Optional tiny C++ diagnostic check: no ASPECT simulation, includes empty ranks
# and injected bad data on only one rank, with collective failure verification.
ibrun -n 2 "$GMG_BUILD/server-gmg/check_gmg_diagnostics"

bash "$GMG_RUN" amg "$GMG_RESULTS/amg-1" "$GMG_BINARY" "$GMG_PLUGIN" ibrun -n 1
bash "$GMG_RUN" gmg "$GMG_RESULTS/gmg-1" "$GMG_BINARY" "$GMG_PLUGIN" ibrun -n 1
bash "$GMG_RUN" gmg_uninstrumented "$GMG_RESULTS/gmg-off-1" "$GMG_BINARY" "$GMG_PLUGIN" ibrun -n 1
bash "$GMG_RUN" gmg "$GMG_RESULTS/gmg-2" "$GMG_BINARY" "$GMG_PLUGIN" ibrun -n 2
```

Stop to inspect the first failure; do not run the rest blindly. If GMG fails
only at two ranks, add the matched AMG two-rank control. If all small cases pass,
one same-model run at the original failing rank count tests partition/empty
coarse-level sensitivity, subject to allocation limits; it does not establish
that the full BP3 mesh is fixed. Do not launch an earthquake run for debugging.

The runner intentionally clears inherited `ASPECT_*` experimental switches,
after recording them, and sets thread counts to one. It leaves site MPI paths
and binding settings intact. It records the command, wrapper/compiler identity,
CPU information, loaded-library paths, binary/source hashes, dirty diff and logs.
Save `module list`, both `CMakeCache.txt` files, `compile_commands.json`, and
`gmg-build.log` alongside these outputs; the runner cannot infer their paths.

## Read the diagnostics

Enable the same instrumentation in any already-approved failing input by setting
this environment variable **after** sourcing any environment cleanup script:

```bash
export ASPECT_FAULT_GMG_DIAGNOSTICS=1
```

Only the exact value `1` enables it; unset or `0` disables it. All ranks must
agree (checked collectively before diagnostic work). It only instruments the assembled reconstructed-fault
velocity-GMG path. It does not instrument generic full matrix-free Stokes GMG.
Backend selection remains exclusively the ordinary PRM setting. The temporary
velocity-preconditioner handler does not parse the generic Matrix Free PRM
subsection, hence this observation-only environment switch.
Each rank appends `output/fault_gmg_rankR.log`, flushing phase markers before
potential failure. Setup repeats for each coupled direction, labeled by step
and nonlinear iteration. The diagnostic does not change the estimator seed,
eigenvalue algorithm, iteration counts or degree used by the actual smoother.
The scratch probe uses a deterministic mean-free vector; it is a diagnostic
first-step calculation, not a complete replay of deal.II's internal CG.

```bash
grep -E 'setup_begin|level=|first_bad|eigenvalues_|consumer_' \
  "$GMG_RESULTS/gmg-2"/output/fault_gmg_rank*.log
tail -100 "$GMG_RESULTS/gmg-2/run.log"
cat "$GMG_RESULTS/gmg-2/exit-status.txt"
```

| Last marker / observation | Interpretation and next evidence |
|---|---|
| `setup_begin`, no `setup_dofs_done` | Failure constructing hierarchy/constraints; obtain backtrace before examining CG |
| `setup_dofs_done`, no `material_done` | Coefficient evaluation/projection/transfer failed |
| `viscosity first_bad_cell` | Active level coefficient nonpositive or nonfinite; cell ID, batch/lane/q and rank are recorded |
| `diagonal_begin`, no inverse summary | Failure while computing/inverting diagonal; debug assertions/backtrace are useful |
| `inverse_diagonal first_bad_global` | Diagonal problem exists before eigenvalue CG; inspect that global DoF and coefficient/constraint path |
| Invalid `probe_Dinv_r` / `probe_A_Dinv_r` | First Jacobi or matrix action already fails; compare rank/compiler/optimization with identical model |
| `r_dot_z>0`, but `z_dot_Az<=0` or nonfinite | Unexpected nonpositive/nonfinite energy in this probe; evidence for operator/constraint/coefficient investigation, not a reason to switch outer solver |
| Probe finite, `eigenvalues_failed level=L` | Failure localized to actual CG/Lanczos estimate on L; retain full exception/backtrace and try the FP-option comparison below |
| All `eigenvalues_done`, then `consumer_begin` without `consumer_done` | Failure occurred after smoother setup; inspect original exception and coupled solve log |
| Instrumented passes, uninstrumented fails | Extra calls/layout may expose initialization or optimization sensitivity; do not adopt diagnostics as a fix |

Empty coarse-level ranks are allowed. Viscosity scans inspect **active** SIMD
lanes only and report padded-lane counts separately. Diagonal scans inspect
owned entries only. Bad-data flags are reduced before diagnostic throws;
exception logging around the estimator adds no collective. Finiteness uses
IEEE bit classification so compiler finite-math assumptions do not remove
the diagnostic checks. Min/max are local, not global extrema. This does not
prove that inactive-lane arithmetic inside a kernel is harmless.

## Backtrace and controlled compiler comparison

If the small one-rank input fails, use it under GDB on a compute node, after the
runner has produced `input.prm`. Copy that file and change its output directory
to a new path before rerunning. For a one-rank launcher:

```bash
ibrun -n 1 gdb -batch \
  -ex 'set pagination off' -ex 'set breakpoint pending on' \
  -ex 'catch throw' -ex run -ex 'thread apply all bt full' \
  --args "$GMG_BINARY" /absolute/path/to/new-gdb-input.prm \
  > "$GMG_RESULTS/gdb.log" 2>&1
```

`catch throw` stops at the first C++ exception, which may be an unrelated handled
exception. Inspect that location; if needed continue interactively to the CG
throw. With debug symbols the source-level call chain should show whether it
is the Chebyshev estimate. Avoid suspending one rank of a large collective run
while expecting the remaining ranks to finish. For an MPI-only failure start
with the per-rank phase logs and use the site's MPI debugger if needed.

After preserving the original failure, make **one separate** ASPECT build with
the same Intel/MPI/deal.II dependencies and `-g -O0 -fp-model=precise` appended
through `ASPECT_ADDITIONAL_CXX_FLAGS` (retain other necessary existing flags).
Verify the verbose command places these after the inherited optimization flags.
Rebuild the test plugin against that build and run the same smallest failing
input/rank count in a new directory. Intel documents `-fp-model=precise` as
restricting transformations that can alter floating-point results:
[Intel FP model](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-0/fp-model-fp.html).
Use the syntax supported by the actual wrapper's underlying compiler; do not
apply Intel options if it invokes GCC. Do not use Fortran-only `-fpe0` advice
for the C++ build.

This comparison changes compilation, not the model. A pass narrows the cause to
optimization/FP/initialization sensitivity; it does not prove an Intel bug or
qualify a numerical fix. Precompiled deal.II/Trilinos/Kokkos code still retains
its original build flags, so an ASPECT-only `-O0` failure cannot rule out a
dependency issue. Only if evidence points there should a separate consistent
dependency build or SIMD comparison be planned. Do not loosen tolerances,
hard-code eigenvalue bounds, suppress NaNs, or replace Lanczos/pivot rules to
make the reproduction pass.

## Evidence to return

Return the smallest failing case directory (including every rank log), its
matched successful control if any, full backtrace, modules, compile commands,
both build caches, wrapper/compiler versions, hashes, and any original BP3
failure log/PRM. Include whether it fails at one rank, only multiple ranks,
only the full BP3 mesh, or only without instrumentation. The failure is still
unresolved until those server results identify a cause.

## Local verification

Final Release build and focused checks passed with GCC 12.4 / OpenMPI 5.0.6.
Compact logs, rank diagnostics, hashes and CMake caches are preserved under
[verification-local/](verification-local/). The core retained its existing
`-fno-finite-math-only -ffp-contract=off` additional flags; this is not the
server's Intel compilation environment.

| Check | Result |
|---|---|
| C++ failure checks, one and two ranks | Positive vector accepted; negative, Inf and NaN on only rank zero rejected on every rank, including empty peer |
| GMG diagnostics on, one rank | Coupled verifier passed; final normalized bulk residual 9.165224e−8, surface 0 |
| GMG diagnostics on, two ranks | Coupled verifier passed; final normalized bulk residual 9.165219e−8, surface 0 |
| GMG diagnostics off, one rank | Coupled verifier passed; final normalized bulk residual 9.165224e−8, surface 0; no diagnostic file |
| AMG control, one rank | Coupled verifier passed; final normalized bulk residual 5.089815e−8, surface 0 |
| Log comparison, one-rank GMG on/off | Printed fresh-linear solve records identical; no claim of full-field bitwise equivalence |
| Shell syntax / diff whitespace | Passed |

Each instrumented rank logged four GMG setups, 28 completed eigenvalue estimates
(levels 0–6 each time), four completed consumers, and no bad-data flags. All
fresh-linear checks and the existing 1e−6 coupled nonlinear criterion passed.
Each model ran in about eight seconds locally; this is not server timing/scaling
evidence. MPI launch required leaving the local socket-restricted sandbox.

Executed core SHA256:
`eef974905d3cca3063c5ae3937ca6cdd27bda3d8d3a5a6ba7dfa2fb199ac6743`.
Executed test-plugin SHA256:
`013d8aa090f94993fed3043f7ad1ade0fb32e5101e1e088de091210fd53845bd`.
The full temporary runs remain in `/tmp/aspect-gmg-observer-{one,two,off,amg}`;
the compact copied evidence is the durable navigation target.

The Intel server failure has **not** been reproduced or fixed. No full BP3 run,
physical time evolution, timestep change, broad suite or server submission was
performed. The original phase-field/RSF changes and saved outputs remain intact.
