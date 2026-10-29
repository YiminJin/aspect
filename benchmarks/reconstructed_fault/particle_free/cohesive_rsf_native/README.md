# Default-off native ASPECT cohesive–RSF pilot

Native evolution uses ASPECT's distributed mixed solver, general tensor history,
actual acceptance/cutback and native checkpoints. Python is used only for bounded
test orchestration and reading accepted CSVs. Production/BP3 is not selected.

Build this worktree **with its local changes**, not HEAD alone. Tested: GCC12.4,
deal.II9.6.2 (MPI/Trilinos/p4est/zlib/SUNDIALS), Trilinos14.2, OpenMPI5.0.6, ASPECT Release.

```sh
cmake -S . -B build-particle-free/cohesive-rsf-native/native \
  -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/opt/dealii/9.6-local \
  -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  -DASPECT_UNITY_BUILD=ON -DASPECT_PRECOMPILE_HEADERS=ON \
  -DASPECT_WITH_PYTHON=OFF
cmake --build build-particle-free/cohesive-rsf-native/native -j2
cmake -S benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native \
  -B build-particle-free/cohesive-rsf-native/checks-native \
  -DAspect_DIR="$PWD/build-particle-free/cohesive-rsf-native/native"
cmake --build build-particle-free/cohesive-rsf-native/checks-native -j2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
mpiexec --bind-to none -n 2 build-particle-free/cohesive-rsf-native/native/aspect-release \
  benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/pilot.prm
```

Run from the worktree root. Omit `--threads`: it enables threading and does not
take a count. The pilot and native full restart use **two ranks**. Small ownership
controls additionally cover 1/2/4 ranks; this is not a claim of general scaling.

`pilot.prm` explicitly checks initialization from the retained .440 s loaded
snapshot/kernel under build-particle-free. This needs that archived reference.
It verifies physical tensor, pressure, full storage and K/N before publication;
it does not replay future states or authorize general representation conversion.

Alternatively copy the supplied **native restart anchor's** `restart` directory
into a new `build-particle-free/cohesive-rsf-native/reproduced-pilot` directory:

```sh
mkdir build-particle-free/cohesive-rsf-native/reproduced-pilot
unzip -n benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/native-start-checkpoint.zip \
  -d build-particle-free/cohesive-rsf-native/reproduced-pilot
mpiexec --bind-to none -n 2 build-particle-free/cohesive-rsf-native/native/aspect-release \
  benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/resume.prm
```

Native resume needs no prototype input. Keep `last_good_checkpoint.txt` with its
referenced numbered checkpoint; branch into a new output directory. The native
owner saves the accepted physical clock; ASPECT saves the next pending interval.
The endpoint guard closes only floating-point clock gaps. All physical/numerical
configuration and same-rank-count requirements remain checked.

`qualify.py` creates isolated bounded local runs, records exact commands, inputs,
binary hash, wall time and test-only controls. `analyze.py` reads only accepted
native CSVs (NumPy, Matplotlib, Pillow). `verify.py` independently compares the
saved lifecycle controls and entry protection hashes. `qualify.py validate
--label PARAMETER-CHECK` validates all nine inputs without advancing state.
Full solver/checkpoint artifacts stay in
build-particle-free; compact PNG/GIF/CSV/JSON evidence is indexed under
`benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native`.

See `doc/reconstructed_fault/particle_free/COHESIVE_RSF_NATIVE_FORMULATION.md` and
the report for equations/ownership, measured qualification and limitations.
This is the declared finite-wall specimen. Existing point/ahead normal-reaction
accuracy failures, causal path hypothesis and physical R remain unchanged.

For a fresh checkout without the separately retained prototype/checkpoint
archives, the small general-history control starts from physical initial data:

```sh
mpiexec --bind-to none -n 2 build-particle-free/cohesive-rsf-native/native/aspect-release \
  benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/general.prm
```

Use a new output directory for each run. The loaded .440 s pilot and its native
restart still require the qualified archives described above; source commits
omit those archives and dense/rendered output. Committed compact results record
the original measured build, not a rerun after each commit.


The qualified endpoint is 0.462375 s. The attempted extension toward 0.465 s
hit the retained 7.5 m formed-length budget; it was not bypassed. The target run
ends before that guard. This is a bounded native pilot, not long-run boundary or
quantitative propagation qualification. The source/point-traction failures and
causal path hypothesis remain explicit in the report.
