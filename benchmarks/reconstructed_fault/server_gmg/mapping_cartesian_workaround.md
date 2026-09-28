# Cartesian matrix-free failure: isolated path and Q1 workaround

## Toolchain resolution reported September 26

The user reports that recompiling both deal.II and ASPECT with `intel/26.0`
instead of `intel/24.0` resolves the GMG failure. This supersedes the pending
upstream investigation and the need to use Q1 as the remedy for that rebuilt
stack. The earlier tests below remain historical evidence. The precise
compiler/dependency defect and the new build provenance were not independently
established in the subsequent BP3 AMG timestep audit; no repeat GMG tests are
requested.

## Server Stage-I workaround verified (September 26)

The newly supplied [output-gmg-q1.tar](output-gmg-q1.tar) contains a successful
**server Release Stage-I initialization**, not just the standalone diagonal
probe. The archive was inspected with `tar -tf` and `tar -xOf`, without
extracting over existing outputs or rerunning the model.

- `output-gmg-q1/log.txt`: OPTIMIZED mode, deal.II 9.6.0, AVX512/eight doubles,
  one MPI process, 4,096 cells on seven levels; four condensed linear solves
  and four velocity-GMG setups; `Reconstructed-fault Stage-I solve: verified`;
  normal end-time termination. Final printed normalized bulk/fault residuals
  are `9.165231e-08 / 0`, with total wall time about 19 seconds.
- `output-gmg-q1/parameters.prm`: `block GMG`, `local smoothing`, initial
  topography `function` with expression and maximum height both zero, and
  `End time = 0`. `original.prm` identifies the Release Stage-I plugin.
- Archive SHA-256:
  `b15a0d043a2835c2c8580726f569c80c67db9fdc1aaa74235591e69345aac248`.

The archive has the five standard output files, but no separate stderr,
per-rank GMG diagnostic log, executable hash or runner provenance. Thus mapping
selection follows from the saved parameters and the inspected mapping factory;
it was not independently printed in this server log. The successful Stage-I
verification is directly recorded. This qualifies the workaround for the
one-rank initialization fixture, not a long BP3/BP5 trajectory or MPI scaling.

**Practical fix:** retain the zero-topography-function block below for the
affected flat-box server runs. No further data or repeat of these completed
experiments is needed to use that workaround. It bypasses the failing mapping
path without changing the box coordinates or finite-element degree: MappingQ1
describes geometry; velocity remains Q2 and pressure remains Q1.

**Minimal source-level alternative for an affected server build:** in
`construct_mapping()` in `source/simulator/core.cc`, replace the Cartesian
return in the zero-topography branch:

```diff
       if (Plugins::plugin_type_matches<const InitialTopographyModel::ZeroTopography<dim>>(initial_topography_model))
-        return std::make_unique<MappingCartesian<dim>>();
+        return std::make_unique<MappingQ1<dim>>();
```

This makes the geometry-mapping choice explicit in the source, so the PRM can
keep its usual zero-topography model. It is a bypass of the dependency path,
not a repair of MappingCartesian. It affects every model reaching that factory
branch, including AMG runs, and may change setup cost, geometric-search fast
paths and rounding. The curved-geometry branch remains MappingQCache. For now
the PRM workaround is narrower and already server-tested. The source change
above is **proposed, not applied or compiled**; no production code was changed
in this evidence update. A maintained GMG-only alternative would need to
substitute Q1 consistently for both active and level MatrixFree objects and
preserve its lifetime; changing only the coarse-diagonal computation would
leave the failing matrix-vector path intact.

**Underlying defect remains unresolved.** The Cartesian-only raw diagonal and
matrix-vector failures identify a mapping-dependent matrix-free path; they do
not prove a particular uninitialized variable, aliasing violation, AVX512 bug,
or Intel compiler miscompilation. The finite FEValues reference does not test
all of MatrixFree's geometry setup. The next bounded task, if pursuing an
upstream fix, is to extend the existing one-cell probe to find the first bad
geometry quantity: compare the forward Jacobian used during MatrixFree setup
with its cached inverse Jacobian/JxW and then the first evaluated gradient.
That new observation can distinguish bad geometry setup from an evaluation
kernel failure before choosing a dependency patch or compiler workaround.
Additional repeated Stage-I runs, CG/timestep changes, or broader compiler-flag
experiments are not justified by the present evidence.

## New server evidence (September 26)

The corrected Release context probe now executes all three cases. Its failure
assertion is reporting a real numerical mismatch, not a parameter parsing issue.
Evidence: [coarse_diagonal_probe.txt](output-coarse-context-release/coarse_diagonal_probe.txt).

| Case | Production inverse diagonal | Test raw diagonal / production action | FEValues reference |
|---|---|---|---|
| `cartesian_active`, no refinements | Both free entries NaN | Both NaN | Finite, about 2.1333333333046201e11 |
| `q1_parent`, six refinements | Both correct, about 4.687500000063092e-12 | Both correct | Finite, same reference |
| `cartesian_parent`, six refinements | Both free entries NaN | Both NaN | Finite, same reference |

The first and third cases have `coarse_probe_pass=0`; the middle has `=1`.
This establishes a reproducible failure associated with **MappingCartesian
in the server's matrix-free path**. It occurs without a fault model, coefficient
projection, CG, multigrid iteration, or multiple MPI ranks, and also occurs when
the coarse cell is active. A refined hierarchy is therefore not required.
Because the raw diagonal and matrix-vector action both fail, the reciprocal
loop and Chebyshev seed are not sufficient explanations. FEValues with the same
Cartesian mapping still computes the reference correctly.

The precise faulty instruction inside the server's deal.II/IntelLLVM stack is
not yet established. The differing mapping implementations enter different
matrix-free setup paths; a compiler bug or a specific uninitialized field has
not been demonstrated. This is stronger localization than the original CG
exception, but not an upstream source fix. The latest copied Debug directory
still contains the old `q1_active` baseline, so it does not establish Debug
results for these three cases. The Release directory also contains an older
`statistics` file; the explicit per-case results and failure exception determine
the outcome of this run.

## Practical workaround for the flat-box Stage-I model

Use ASPECT's existing **MappingQ1** path. A PRM-only way to select it in this
checkout is to use a zero-valued initial-topography function:

```text
subsection Geometry model
  subsection Initial topography model
    set Model name = function
    subsection Function
      set Coordinate system = cartesian
      set Function expression = 0
      set Maximum topography value = 0
    end
  end
end
```

Apply this to the **Stage-I simulation input**. The prepared
[gmg_q1.prm](gmg_q1.prm) includes the existing GMG Stage-I input and adds exactly
this block. The existing mapping factory in
[source/simulator/core.cc](../../../source/simulator/core.cc) selects MappingQ1
for this non-curved geometry when the topography plugin is `function`. For a
flat box, Q1 represents the same affine coordinate mapping as MappingCartesian.
[Box::add_topography_to_point](../../../source/geometry_model/box.cc) adds a
vertical displacement proportional to the function value; with value zero,
vertex coordinates stay unchanged. Maximum topography must also be zero to
keep the reported depth/height consistent.

This is a workaround using an existing input interface, not a new physics or
solver parameter. It selects Q1 for the simulator, including its GMG hierarchy.
The geometry, fault parameters, physical equations, tolerances, timestep settings
and backend remain those of the original flat-box fixture. Extra setup cost and
floating-point rounding differences are possible between mapping implementations.
The local bounded verification below does not qualify changed mapping choices
for an existing long-run restart or a non-flat/non-box geometry.

**The standalone context probe intentionally constructs its own mappings.**
Adding this block to `coarse_context.prm` will not change its explicit Cartesian
test cases. Preserve the failed probe as evidence; do not remove its assertion
or make it report success by suppressing the Cartesian cases.

## Completed server test: invocation retained for reference

The server result above completes this task. These commands document the test;
they are not a request to repeat it.

No ASPECT or deal.II rebuild is needed for this PRM workaround. Use the same
Release executable and the matching **Stage-I verifier plugin**, not
`libcoarse.release.so`. Copy the updated runner and `gmg_q1.prm`, retaining the
existing package includes. Inside a one-rank compute allocation:

```bash
# Set these to your existing source, executable, and Stage-I plugin paths.
export GMG_SOURCE=/absolute/path/to/aspect-source
export GMG_BINARY=/absolute/path/to/aspect-release
export GMG_STAGE_I_PLUGIN=/absolute/path/to/libverify_stage_i.release.so

bash "$GMG_SOURCE/benchmarks/reconstructed_fault/server_gmg/run.sh" gmg_q1 \
  "$SCRATCH/gmg-q1-stage-i-${SLURM_JOB_ID}" \
  "$GMG_BINARY" "$GMG_STAGE_I_PLUGIN" ibrun -n 1
```

If built using the supplied package CMake file, the equivalent plugin is named
`libfault_gmg_stage_i.release.so`. The runner requires a new output directory,
sets `ASPECT_FAULT_GMG_DIAGNOSTICS=1`, captures stdout/stderr and provenance,
and has the existing 300-second timeout. It requires both the Stage-I verification
marker and `consumer_done`. Expected new evidence is finite diagonals, completed
eigenvalue estimates, and `Reconstructed-fault Stage-I solve: verified`.
If the raw-diagonal observer is in the executable, stderr should identify
`MappingQ1` in its mapping type. Return the full runner result directory. There
is no reason to repeat the already-failed Cartesian context cases first.

The complete one-rank Stage-I workaround is now verified on the server by the
archive described above. The separate runner diagnostics were not included
in that archive.

## Local verification

The new `gmg_q1` Stage-I initialization passed on one rank with the existing GCC
Release core (no core source changes or rebuild). Its raw-diagonal observer
confirms `MappingQ1`; all 28 level eigenvalue estimates and four consumers
complete. Final printed normalized residuals are bulk `9.165224e-08`, fault `0`,
and the coupled Stage-I verifier passes. This is a new workaround qualification,
not a replay to rebuild context. It is not a server/Intel qualification or a
claim of bitwise equivalence for complete field outputs.

Evidence and provenance are under [verification-q1-local](verification-q1-local/).
Shell syntax/whitespace checks passed. The copied failing server files and core
executable were hash-checked unchanged. No Python scripts were used, no existing
BP3/BP5 input was modified, and no long trajectory or restart was launched.
