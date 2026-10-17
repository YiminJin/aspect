# Collective wall-stop local MPI regression

This is a test-only build of the maintained small 60-degree strengthening mature
fixture (`section7b-fe-perturbed.prm`), with accepted unlimited domain-free DGQ2,
Gauss6, fixed mesh/profile/particles and dt=200000 s. No server job is involved.
The production server adapter never defines `ASPECT_BP3_TEST_NONROOT_WALL_STOP`.
No new production parameter/environment selector or elapsed-time sleep is used.

Build two dedicated directories against a matching Release ASPECT build. Example
from the source root (replace `Aspect_DIR` with your build):

```sh
cmake -S benchmarks/reconstructed_fault/bp3_native/tests/collective_stop \
  -B build-particle-free/collective-stop/injected -DAspect_DIR=/absolute/aspect/build \
  -DBP3_TEST_NONROOT_WALL_STOP=ON
cmake --build build-particle-free/collective-stop/injected -j 4
cmake -S benchmarks/reconstructed_fault/bp3_native/tests/collective_stop \
  -B build-particle-free/collective-stop/control -DAspect_DIR=/absolute/aspect/build \
  -DBP3_TEST_NONROOT_WALL_STOP=OFF
cmake --build build-particle-free/collective-stop/control -j 4
```

The Python regression utility is **only a local test harness**, requires NumPy
for field comparisons and uses two MPI processes with a 240 s timeout per case.
Use a fresh work directory; existing cases are never overwritten. If the native
binary is elsewhere, pass `--binary /absolute/aspect-release` to each run.

```sh
python3 benchmarks/reconstructed_fault/bp3_native/tests/collective_stop/regression.py run --work build-particle-free/collective-stop --case wall
python3 benchmarks/reconstructed_fault/bp3_native/tests/collective_stop/regression.py run --work build-particle-free/collective-stop --case control
python3 benchmarks/reconstructed_fault/bp3_native/tests/collective_stop/regression.py run --work build-particle-free/collective-stop --case wall-resume
python3 benchmarks/reconstructed_fault/bp3_native/tests/collective_stop/regression.py run --work build-particle-free/collective-stop --case scheduled-control
python3 benchmarks/reconstructed_fault/bp3_native/tests/collective_stop/regression.py run --work build-particle-free/collective-stop --case scheduled-split
python3 benchmarks/reconstructed_fault/bp3_native/tests/collective_stop/regression.py run --work build-particle-free/collective-stop --case scheduled-resume
python3 benchmarks/reconstructed_fault/bp3_native/tests/collective_stop/regression.py check --work build-particle-free/collective-stop
```

The injected library makes rank 1 alone request wall termination at accepted
step 2. Rank traces prove ordinary output and all other stop requests were not
due; all ranks must agree on stop/heavy/profile. Terminal fields/native profile,
indices and the step-2 checkpoint must exist. Lifecycle assertions retain one
publication per accepted positive step. Continuation to step 5 must match the
uninterrupted accepted timestep sequence and every DG coefficient, physical FE
field, reconstructed stress, surface load/state/slip and full particle properties
by stable ID. Separate scheduled cases retain the step-3 split regression with
fields at 0/3/5 and no spurious step-4 dump. Per-rank native-child peak RSS and
full native timing tables are retained. Neither path changes physical equations.
