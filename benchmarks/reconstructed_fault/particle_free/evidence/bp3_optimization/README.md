# Bounded stationary mature DG optimization evidence

Version control retains reusable replay/reduction scripts and selected result
summaries. Raw run/launcher logs, per-run inputs, detailed source manifests and
large archives/checkpoints remain local at the paths described below. Historical
snapshot hashes identify the measured source; they are not rewritten at commit.

Reviewed base `b2dfdebe43437d6028b06d4f9f862cc80a00cc71`, branch
`pf-rsf-particle-free`, 17 October 2026. Actual changed source hashes are in
`source-state.json`; the regenerated server archive carries the complete source
snapshot/manifest. No server/full-event launch. Neither reference algorithms nor
particle controls are a physical accuracy reference. **strict_reference_match_passed=false**.

[Measured report](../../../../../doc/reconstructed_fault/particle_free/BP3_DG_OPTIMIZATION_REPORT.md)
and [formulation](../../../../../doc/reconstructed_fault/particle_free/BP3_DG_OPTIMIZATION_FORMULATION.md).

- `assessment.json`: final sequential timing pairs, RSS, nested native timers,
  exclusive coupling timings, counts, accepted sequences/iterations, residuals,
  publication guards, full tensor/history/load comparisons and MPI/restart/cutback.
- `operator-checks.json`: independent native source/allowance/QP probes and B/G/C
  action checks, including native hanging constraints and physical pressure scaling.
- `kernel.json`: actual 5776-node projection kernel and manufactured filtered
  surface algebra. Shared normal action/filter solve is excluded from product
  timing; this is not native full-grid bulk-work quadrature.
- `stationarity.json`: full initial/final ID-coordinate maps match exactly for
  66176 and 184448 points; effective integrator `BP3 stationary samples`.
- `parameter-validation.json`: all four native validations, parameter equivalence,
  file hashes and ordinary-plugin test-macro exclusion.
- `regression.json`: repeated collective wall-stop and scheduled-output restart
  results; both compared continuations are exactly equal to their controls.
- `entry-preservation.json`: all 12758 entry files remain; all 382 incoming
  untracked/manual files and both previous source archives are unchanged. Only
  listed authorized implementation/documentation files differ.
- `matrix-progress.json`: all 21 final sequential matrix actions exit zero.
  `runs/` has input, command/artifact hashes, complete logs, RSS records and compact
  acceptance checks. `wall/` and `logged/` preserve bounded wrapper logs/commands.
  Large raw fields and checkpoints remain under
  `build-particle-free/bp3-optimization/`, outside this compact bundle.

Final performance comparisons use **qualified-reference/optimized-small/larger**,
two ranks and one thread/rank, two positive 200000 s steps. Earlier timings and
the engineering failures are retained but are not final performance results.
The reference flag reproduces the original copied base binary's compared fields
exactly. `operators-small`, `operators-small-v2` preserve bulk-matrix overwrite
failures; v3 preserves the owned/ghosted-vector diagnostic harness failure.
The `bp3-opt-units` preliminary negative-filter run selected two saved-state
tests without required environment; the intended default suite subsequently
passes 126 cases/149074 assertions. No acceptance tolerance was loosened.

Local replay uses the existing Release build at `build-particle-free/native`.
Build local fixture plugins separately from ordinary server plugins:

```sh
cmake -S benchmarks/reconstructed_fault/bp3_native/tests/collective_stop \
      -B build-particle-free/bp3-optimization/plugins \
      -DAspect_DIR="$PWD/build-particle-free/native" \
      -DBP3_TEST_NONROOT_WALL_STOP=OFF \
      -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++
cmake --build build-particle-free/bp3-optimization/plugins -j 2
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py \
  bp3-opt-replay-units 60 env LD_LIBRARY_PATH=/opt/openmpi/5.0.6/lib \
  OMP_PROC_BIND=false build-particle-free/native/aspect-release --test
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py \
  bp3-opt-replay-reference 240 python3 \
  benchmarks/reconstructed_fault/particle_free/run_bp3_optimization.py \
  replay-reference --mode reference
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py \
  bp3-opt-replay-optimized 240 python3 \
  benchmarks/reconstructed_fault/particle_free/run_bp3_optimization.py \
  replay-optimized
```

Use fresh labels/output directories; the runner refuses existing ones and is
only a bounded local harness. `--mesh larger`, `--operators`, `--ranks 1/4`,
`--restart`, `--candidate-rejection`, `--outer-rejection`,
`--nonlinear-rejection` and `--adaptive` select the recorded controls. Actual
arguments and full environment are in the individual records. `completed_matrix.py`
and `validate_inputs.py` preserve the one-off orchestration used here; they refer
to existing local build paths and are not server runtime controllers.

To reduce the preserved raw runs again (requires NumPy):

```sh
python3 benchmarks/reconstructed_fault/particle_free/analyze_bp3_optimization.py \
  build-particle-free/bp3-optimization --output /tmp/bp3-optimization-assessment.json
```

The server package's `qualification/` contains this evidence and the unchanged
previous qualification under `inherited-startup/`. Its ordinary native workflow
remains direct `ibrun ./plugin/aspect-release startup.prm` then
`startup-resume.prm`; see the packaged README/resource sheet. Full-grid memory,
checkpoint/dense-output cost, conditioning and first-event accuracy remain
server questions.
