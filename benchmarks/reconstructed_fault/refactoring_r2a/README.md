# R2a qualification

This harness reuses the qualified R1 cases and exact comparison rules. It does
not change their tolerances or repair the recorded R1 fixture failures. Read the
[rolling review](../../../doc/reconstructed_fault/refactor_review.md) for the
baseline state, source delta, results and limitations.

Reference: `build-refactor-baseline/` and `../refactoring_r1/` (relative to this
directory). Candidate: `build-refactor-r2a/` from the worktree root, plus this
directory's `plugin-build/` and `output-*/`. Generated artifacts are ignored and
retained locally. Never run a reference build/test target against edited source.

Use the R1 configuration and toolchain commands from its
[README](../refactoring_r1/README.md), substituting `build-refactor-r2a` for the
core build and `refactoring_r2a/plugin-build` for the maintained BP3 plugin
build. Keep GCC 12.4.0, OpenMPI 5.0.6, deal.II 9.6.2, Voro++, Release/unity/PCH,
and `-fno-finite-math-only -ffp-contract=off`. Start with two build jobs.
Then, from the worktree root:

```sh
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
cmake --build build-refactor-r2a -j2
cmake --build build-refactor-r2a/tests -j2 --target \
  phase_field_fault_ih phase_field_fault_ih_mpi \
  phase_field_fault_ih_no_composition phase_field_fault_stage_i_rollback
cmake --build benchmarks/reconstructed_fault/refactoring_r2a/plugin-build -j2
python3 benchmarks/reconstructed_fault/refactoring_r2a/stage_inputs.py
bash benchmarks/reconstructed_fault/refactoring_r2a/run_checks.sh
python3 benchmarks/reconstructed_fault/refactoring_r2a/compare_states.py
python3 benchmarks/reconstructed_fault/refactoring_r2a/verify_move.py
```

The input stager refuses to replace existing inputs. It changes only build,
plugin and output paths in R1's qualified PRMs, retaining the immutable R1
fixture data. Original rollback PRMs are also exercised directly so their stale
shell output filter cannot hide the actual acceptance/restoration markers.
The existing branch tool copies preserved R1 checkpoint slot 02 (accepted step
4 at 400 s) into a new output directory; the candidate resumes steps 5 and 6.

Each invocation is bounded and records its command, numerical environment,
elapsed time and exit code. MPI needs local socket access outside the sandbox.
The runner continues after a failure; its final shell status is not a suite
verdict. Inspect individual logs and comparisons. No parameter probes, full
test campaign or production BP3 run is included.

The field comparer retains R1's absolute and pointwise relative error measures,
including separate zero-reference errors. It compares matching rank counts:
both uninterrupted trajectories (steps 0–6) and baseline-checkpoint continuation
(steps 5–6), including every owned bulk component, all particle properties,
fault profiles, final native I_h/history and solver decisions. It requires
exact equality, and checks lifecycle statistics, cell-cache work (excluding
geometry time), and accepted-update/rollback markers separately.

`evidence/separate-commands.json` records ordinary compilation of the original
material file and new normalization file with neither unity nor PCH.
`separate-link-commands.json` records the focused executable link replacing the
normal unity object with those separate objects and the other, unchanged
material sources. This checks actual 2D/3D template definitions and linking;
it does not claim 3D simulation coverage. `verify_move.py` uses recorded entry
hashes/spans to check exact movement and preservation of unrelated local edits.
