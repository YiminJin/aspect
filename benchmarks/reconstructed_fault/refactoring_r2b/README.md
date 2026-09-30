# R2b cell-profile geometry extraction verification

This pass extracts only private cell-profile geometry preparation within
`PhaseFieldFault`. Reference binaries, plugins, inputs and outputs remain in
`build-refactor-r2a/` and `../refactoring_r2a/`; qualified R1 artifacts remain
unchanged too. The candidate has a fresh `build-refactor-r2b/` build and this
directory's generated inputs, outputs and evidence. See the
[rolling review](../../../doc/reconstructed_fault/refactor_review.md) for results.

Use the [R1 toolchain configuration](../refactoring_r1/README.md), substituting
`build-refactor-r2b` for the core build and `refactoring_r2b/plugin-build` for
the maintained BP3 plugin build. Retain GCC 12.4.0, OpenMPI 5.0.6, deal.II 9.6.2,
Release/unity/PCH, two build jobs, and the existing floating-point flags.
After configuring both builds, run from the worktree root:

```sh
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
cmake --build build-refactor-r2b -j2
cmake --build build-refactor-r2b/tests -j2 --target \
  phase_field_fault_ih phase_field_fault_ih_mpi phase_field_fault_ih_no_composition
cmake --build benchmarks/reconstructed_fault/refactoring_r2b/plugin-build -j2
python3 benchmarks/reconstructed_fault/refactoring_r2b/stage_inputs.py
bash benchmarks/reconstructed_fault/refactoring_r2b/run_checks.sh
python3 benchmarks/reconstructed_fault/refactoring_r2b/compare_states.py
python3 benchmarks/reconstructed_fault/refactoring_r2b/compare_states.py --reference refactoring_r1
python3 benchmarks/reconstructed_fault/refactoring_r2b/verify_extraction.py
```

Inputs and logs refuse overwrites. Each run is bounded and records its command,
environment and exit status; the shell stops on a failed invocation. MPI needs
local socket access outside the sandbox. R1/R2a outputs are never regenerated
or overwritten. The input stager changes only build/plugin/output paths, with
the same initialization budget already qualified in R1; physical parameters,
test assertions and tolerances remain unchanged.

The existing accuracy/cache cases run on one and two ranks. Lifecycle cases
exercise both backends on one/two ranks and the no-composition path on one rank.
Both six-step BP3 trajectories compare against R2a and qualified R1 at matching
rank counts, using their existing exact comparison rules for profiles, bulk
fields, particles, native fault history/I_h and solver decisions. Work counters
are compared exactly; only geometry wall time is excluded from those records.

Additional paired R2a/R2b lifecycle runs set the existing
`ASPECT_DISABLE_IH_VALUE_CACHE=1` switch. They exercise repeated integration
using warm geometric traversals. Comparisons require identical counters and
statistics, successful existing assertions, and nonzero traversal reuse.
They do not change cache criteria or constitute a new numerical test.

`verify_extraction.py` uses recorded entry snapshots, checks the moved geometry
statements and retained integration tail, and verifies preservation of all
other source/header/test files. Source diffs and reference hashes are retained
in ignored `evidence/`. This pass does not rerun the full test suite, production
BP3, restart/rollback, or a 3D simulation.
