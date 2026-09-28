# Analysis of the copied server Debug/Release results

**Superseding follow-up:** The requested server coarse Debug/Release probes
have both passed, and compiler/configuration information has been supplied.
See [the September 26 context analysis](server_context_analysis.md); do not
repeat the baseline instructions or missing-configuration requests below.

## Verified localization

The first observed bad value is in the **level-zero inverse velocity diagonal**,
before any Chebyshev eigenvalue estimate or coupled preconditioner application.
The earlier CG/Lanczos explanation identifies a downstream consumer of the bad
diagonal; it does not identify the origin of the NaN.

Evidence: [Release diagnostic](output-gmg-release/fault_gmg_rank0.log),
[Debug diagnostic](output-gmg-debug/fault_gmg_rank0.log),
[Debug solver log](output-gmg-debug/log.txt),
[Release solver log](output-gmg-release/log.txt).

| First setup, level 0 | Debug | Release |
|---|---|---|
| MPI ranks / SIMD lanes | 1 / 8 (AVX512) | 1 / 8 (AVX512) |
| Velocity DoFs / cell batches | 18 / 1 | 18 / 1 |
| Active / padded lanes | 1 / 7 | 1 / 7 |
| Active viscosity | 24999999999.663513 | 24999999999.663513 |
| Inverse diagonal | minimum 4.6875000000630919e-12; maximum 1 | first nonfinite entry: global DoF 16, `-nan`; finite entries all 1 |
| Next stages | All 7 levels and eigenvalue estimates complete; 4 consumers complete overall | No `eigenvalues_begin` or `consumer_begin` |

The resolved `parameters.prm` files differ only in the Debug/Release plugin
filename and output directory. Both banners show ASPECT 3.1.0-pre, deal.II 9.6.0,
Trilinos 15.0.0, p4est 2.8.5, one MPI rank, and 4096 cells on seven levels.
Debug reaches `Reconstructed-fault Stage-I solve: verified`. Their banners do
not identify a source SHA. Matching PRMs do not prove matching source builds.

For one unit-square Q2 velocity cell with homogeneous Dirichlet boundaries,
16 boundary DoFs are constrained and two central velocity DoFs remain free.
For the incompressible bilinear form `2 eta epsilon(u):epsilon(v)`, each free
diagonal is `(128/15) eta`, hence its inverse is `15/(128 eta)`.
This predicts `4.687500000063092e-12`, matching Debug. It follows from the
central scalar basis `16 x(1-x)y(1-y)`, whose squared x and y derivatives each
integrate to `128/45`. These magnitudes are far from floating-point overflow
or underflow. The existing diagnostic logs only the **first** invalid entry;
they do not establish whether DoF 17 is also NaN.

## Code region implicated, and what remains unproven

The bounded suspect region is
[ABlockOperator::compute_diagonal](../../../source/simulator/solver/matrix_free_operators.cc),
around lines 865–898 in this checkout:

1. `MatrixFreeTools::compute_diagonal()` evaluates the Q2 basis vectors through
   `ABlockOperator::cell_operation()`.
2. That calls gradient evaluation, `inner_cell_operation()` (multiply symmetric
   gradients by twice viscosity), then gradient integration.
3. Constrained diagonal entries are set to one.
4. The raw diagonal is inverted, `local_element = 1./local_element`.

The raw diagonal is not present in the copied logs. Thus the first NaN could
arise in evaluation, integration, accumulation, or inversion; current evidence
cannot distinguish them. Production `vmult()` uses `gather_evaluate()` and
`integrate_scatter()` around the same inner kernel, so comparing it against the
diagonal path is useful. The positivity assertion before inversion is a Debug
`Assert`; Release omits it. That explains why invalid data can reach CG, but
does not explain why the data become invalid only in Release.

Source inspection did **not** establish an uninitialized coefficient flag:
level `is_compressible` and `pressure_scaling` are assigned during material
evaluation, and both branches assign level `enable_prescribed_dilation=false`
and `enable_newton_derivatives=false`. The logged active viscosities agree.
Neither observation rules out earlier memory corruption or other bad mapping
or scratch data. Seven padded SIMD lanes are present, but their presence alone
does not prove a lane-masking defect.

Release optimization, undefined behavior, an incompatible build/library, or
an optimized deal.II path remain hypotheses. Debug and Release may also use
different dependency binaries. **A compiler bug has not been demonstrated.**
No multi-rank exchange, long timestep, or long earthquake trajectory is needed
to reach this particular failure. Changing CG tolerances, iteration limits,
RSF parameters, or timestep policy is not supported by this evidence.

## Next bounded test: one cell, no fault solve

The new [coarse_diagonal_probe.cc](coarse_diagonal_probe.cc) plugin uses the
copied coarse viscosity and builds a separate level-zero Q2/Q1 operator. It
calls the **production** ABlock diagonal and matrix-vector implementations
from the loaded ASPECT executable. It compares them with:

- a raw diagonal from the deal.II helper and a separately compiled test kernel;
- an independent FEValues quadrature diagonal;
- the analytic central-basis result above.

It writes all production inverse entries (including explicit bit-based finite
flags), the two free-DoF comparisons, and flushed stage markers to
`output/coarse_diagonal_probe.txt`. A mismatch exits unsuccessfully after writing
the comparisons. A thrown exception or signal may end it earlier; preserve the
last marker and full stderr. The test-only tolerance is relative `1e-11`.
The [coarse.prm](coarse.prm) harness has one cell, no advection or Stokes solve,
no particles, no reconstructed fault, and no physical time evolution. It requires
exactly one MPI rank. No production equations, controls, or kernels were changed
for this follow-up.

Build only this small package against the **existing matching** server ASPECT
build(s), preserving their optimization options. From the source root, use new
plugin build and result directories; substitute the actual existing build paths:

```bash
export GMG_SOURCE="$PWD"
export GMG_RELEASE_BUILD=/absolute/path/to/existing/release-build
export GMG_DEBUG_BUILD=/absolute/path/to/existing/debug-build
# They may be the same path for a DebugRelease build.
export GMG_PROBE_BUILD="$SCRATCH/gmg-coarse-plugin-${SLURM_JOB_ID}"
mkdir "$GMG_PROBE_BUILD"
set -o pipefail
cmake -S "$GMG_SOURCE/benchmarks/reconstructed_fault/server_gmg" \
  -B "$GMG_PROBE_BUILD/release" -DAspect_DIR="$GMG_RELEASE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  2>&1 | tee "$GMG_PROBE_BUILD/configure-release.log"
cmake --build "$GMG_PROBE_BUILD/release" -j2 --verbose \
  2>&1 | tee "$GMG_PROBE_BUILD/build-release.log"
cmake -S "$GMG_SOURCE/benchmarks/reconstructed_fault/server_gmg" \
  -B "$GMG_PROBE_BUILD/debug" -DAspect_DIR="$GMG_DEBUG_BUILD" \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  2>&1 | tee "$GMG_PROBE_BUILD/configure-debug.log"
cmake --build "$GMG_PROBE_BUILD/debug" -j2 --verbose \
  2>&1 | tee "$GMG_PROBE_BUILD/build-debug.log"

# Inside a compute allocation; each runner refuses an existing result directory.
bash "$GMG_SOURCE/benchmarks/reconstructed_fault/server_gmg/run.sh" coarse \
  "$GMG_PROBE_BUILD/result-debug" "$GMG_DEBUG_BUILD/aspect-debug" \
  "$GMG_PROBE_BUILD/debug/libfault_gmg_coarse_probe.debug.so" ibrun -n 1
bash "$GMG_SOURCE/benchmarks/reconstructed_fault/server_gmg/run.sh" coarse \
  "$GMG_PROBE_BUILD/result-release" "$GMG_RELEASE_BUILD/aspect-release" \
  "$GMG_PROBE_BUILD/release/libfault_gmg_coarse_probe.release.so" ibrun -n 1
```

The plugin macro follows the selected ASPECT build modes; `CMAKE_BUILD_TYPE`
alone cannot turn a Release-only ASPECT build into Debug. Do not load a Debug
plugin into Release or change vector width independently of deal.II. The
production operator remains compiled with the ASPECT flags; the test kernel
uses the plugin's compile flags, which must be retained with the results.

| Result, with finite analytic/FEValues reference | Next interpretation |
|---|---|
| Production inverse bad, production action good, test raw good | Focus on production diagonal-specific code, helper instantiation and reciprocal loop; capture raw values there next |
| Production inverse and test raw bad, production action good | Focus on basis evaluation/integration or helper accumulation used by diagonal construction |
| Production action also bad | Shared kernel, mapping/constraints, initialization or dependency path remains implicated |
| One-cell Release passes | Reduced setup did not reproduce; inspect the original fixture's level-zero raw diagonal/mapping/lifetimes in its failing context |
| Reference also bad | Check build/dependency consistency and earlier corruption before interpreting comparisons |

These are localization rules, not proofs of a compiler defect. A passing isolated
test does not establish that the coupled fixture is fixed. Stop after this pair
and inspect the output; do not repeat the completed BP3/BP5 experiments.

## Additional data needed

Please retain the full **runner result directories**, including `run.log`
(stdout and stderr), provenance, exit status and probe output, plus the new
plugin build logs and compile commands. For the **original failing build**, also
copy:

1. ASPECT and deal.II `CMakeCache.txt` (both modes if separate), and the actual
   Release compile command for `matrix_free_operators.cc` or its unity source.
2. `mpicxx -show`, `mpicxx --version`, module list, CPU and `ldd` information.
3. Full exception/backtrace or job stderr from the instrumented failure.
4. Executable/plugin hashes and the source diff used to build them, if available.

The supplied ASPECT `log.txt` files do not contain the full failure stderr.
No Intel compiler or matching server dependency stack is available locally.

## Local validation of this follow-up

The new plugin compiled with GCC 12.4/OpenMPI and passed the one-rank Release
probe against the existing `build-tmp/aspect-release`, with four SIMD lanes and
local deal.II 9.6.2. Both free entries passed all comparisons; production raw
action diagonals were approximately `2.133333333304619e11`, compared with
FEValues `2.1333333333046201e11`. See
[verification-coarse-local/coarse_diagonal_probe.txt](verification-coarse-local/coarse_diagonal_probe.txt)
and its adjacent build, provenance and run logs. This validates the fixture,
not the server's Intel/AVX512/deal.II 9.6.0 path. The core executable was not
rebuilt in this analysis follow-up.

During fixture development, a missing gravity-model selection and an unallocated
test raw-diagonal vector were corrected. The latter caused a local fixture
crash; GDB localized it to the test vector access. Production already allocates
its diagonal before calling the helper, so this was **not** a reproduction or
explanation of the server failure. Initial development outputs remain under
`/tmp/aspect-gmg-coarse-one*` and `/tmp/aspect-gmg-coarse-gdb*`; the final passing
run is `/tmp/aspect-gmg-coarse-validated`.

Shell syntax and whitespace checks passed. SHA256 checks verified all 11
copied server-output files were unchanged; the manifest is preserved with the
local verification evidence. No Python scripts, BP3/BP5 trajectories, coupled
fixture reruns, or numerical behavior changes were part of this follow-up.
