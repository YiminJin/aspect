# Commands and environment

These commands describe the completed sequential local experiment in the
matching worktree. They are not a server launch or a standalone ASPECT package.
Existing run directories are protected against accidental replacement.

```sh
cmake -S benchmarks/reconstructed_fault/particle_free \
  -B build-particle-free/cohesive-interface-turn/build \
  -DAspect_DIR="$PWD/build-particle-free/native" -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/cohesive-interface-turn/build \
  --target cohesive_interface_kink cohesive_interface_turn_solver -j2

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_turn.py main-turn
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_turn.py main-scan
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_turn.py followup
python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_turn.py
python3 benchmarks/reconstructed_fault/particle_free/qualify_cohesive_interface_turn.py
python3 benchmarks/reconstructed_fault/particle_free/qualify_cohesive_interface_turn.py --regression
MPLCONFIGDIR=/tmp/cohesive-turn-mpl \
  python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_turn.py
python3 benchmarks/reconstructed_fault/particle_free/package_cohesive_interface_turn.py
```

The seven coarse previews ran through their original loading/hold; the report
uses their identical incoming trajectory only through .45 s. Follow-up runs
use .45 s as the endpoint before candidate-end proximity. All spatial/time
controls start freshly. The historical undersized/full-Q2 pilots remain in the
diagnostic logs; they are not prerequisites to rerun or successful controls.
The sharp regression uses the retained `cohesive-interface-kink/native-16` data.

Build: OpenMPI mpic++, deal.II 9.6.2, Release/O2, UMFPACK, ASPECT's configured
includes; explicit `-fno-finite-math-only -ffp-contract=off`. Kernels and evolution
are serial; no server job or new MPI qualification. Package metadata records
Python/library versions. Native JSON timers include operator/field export;
evolution summaries include local ownership, certification and checkpoint work.
Timing runs did not overlap.

`raw-kernel-index.json` gives paths, sizes and hashes for retained large native
files. The ZIP contains source, compact selected operators/checkpoints,
PNG/GIF/CSV/JSON data and diagnostic logs, excluding those large raw responses.
JSON checkpoints are experimental same-build/geometry archives; the strict
operator/geometry identity guard is retained. They are not ASPECT production
checkpoints or a portable checkpoint conversion.

`diagnostics/interruption.json` records an explicitly interrupted final small
regression setup. Its empty partial directory/log remain under the work folder;
the command was resumed after verifying the previous process had stopped.
Accepted histories were unaffected. The original bitwise-default regression
expectation also failed; `default-regression.json` retains that fact and the
measured roundoff differences under unchanged physical tolerances.
