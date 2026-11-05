# ASPECT architecture refactor evidence

`entry.json`, `entry.patch` and `entry-source.tar.gz` preserve the actual dirty
incoming source, binaries and prior protected artifacts. `frozen-sources.json`
identifies the final source to build/verify. HEAD alone is not either identity.
The maintained design/report are under `doc/reconstructed_fault/particle_free`.

The seven case runner replays the incoming `mature-step-*` arguments into fresh
`architecture-*` output directories. It runs sequentially and refuses overwrite.
The comparison reducer retains the declared bulk/surface bounds and additionally
checks physical tensor/work traction, clocks, publications and solver decisions.

```sh
cmake --build build-particle-free/native --parallel 4
cmake --build build-particle-free/transported-objective/stage-c/checks --parallel 4
cmake --build build-particle-free/transported-objective/stage-c/bp3-plugin --parallel 4
ctest --test-dir build-particle-free/native -R '^unit_tests$' --output-on-failure
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/run_aspect_architecture.py dc sc dl sl default reject restart4
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/check_aspect_architecture.py
```

Use fresh explicit labels for any rerun. Runtime libraries and actual commands
are recorded with each run. Baseline/current stationary and ordinary non-fault
checks cover normal dispatch. Build/launch failures are retained separately.
No server trajectory, physical campaign, cleanup or checkpoint conversion is
part of this work. Previous cleanup availability losses remain unchanged.

Completed evidence: `equivalence.json` (seven moving cases),
`lifecycle-crosschecks.json`, `selected-configurations.json`,
`dispatch-equivalence.json` (stationary/non-fault), `serial-comparison.json`,
`unit-tests.log`, `independent-compile.json` and `final-provenance.json`.
All 51 protected files match. Native input/command copies are in `runs/`;
full exports remain in the recorded local build/output directories. The first
stationary MPI startup failure and first link failure are retained. No historical
cleanup loss was repaired or reclassified.
