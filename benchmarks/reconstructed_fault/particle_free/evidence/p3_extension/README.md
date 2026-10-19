# Bounded P3 synthetic straight append

See [the report](../../../../../doc/reconstructed_fault/particle_free/P3_SYNTHETIC_EXTENSION_REPORT.md).
`selection.json` fixes the contract, fixture and screens before numerical runs.
`serial.json`, `mpi2.json` and `verification.json` contain measured results and
the exact qualification boundary. The logged commands retain native build,
serial/two-rank, process-restart, full unit and independent-reference results.

`reference.tar.xz` is a compact portable copy of all 31 native exported files,
including before/after geometry/history archives. `archive-manifest.json` records
every member. Replay needs Python and NumPy, without an ASPECT build:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/p3_extension/verify.py
```

Native reproduction from a build of this source, using the existing bounded
logger (300-s per native command). Substitute another isolated ASPECT binary
if needed; do not rebuild a protected BP3 executable.

```sh
mkdir -p build-particle-free/p3-review/serial build-particle-free/p3-review/mpi2
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py p3-review-serial 300 env ASPECT_P3_OUTPUT=$PWD/build-particle-free/p3-review/serial OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release --test '[p3_extension],[p3_extension_storage]'
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py p3-review-mpi2 300 env ASPECT_P3_OUTPUT=$PWD/build-particle-free/p3-review/mpi2 OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false mpirun -np 2 build-particle-free/evolving-phase/native/aspect-release --test '[p3_extension],[p3_extension_storage]'
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py p3-review-restart 300 env ASPECT_P3_RESTART=$PWD/build-particle-free/p3-review/serial OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=false mpirun -np 2 build-particle-free/evolving-phase/native/aspect-release --test '[p3_extension_restart]'
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_p3_extension.py build-particle-free/p3-review/serial --compare build-particle-free/p3-review/mpi2 --output build-particle-free/p3-review/assessment.json
```

The explicitly selected process-restart test is hidden from the default suite
because it consumes the preceding files. It rebuilds native DG ownership and
continues the serial pre-append archive to the exact uninterrupted record.
`unit_tests/CMakeLists.txt` additionally registers the bounded two-rank test.
The entire native default suite passes 131 cases / 165436 assertions.

The candidate is a **synthetic native-component transaction**, not a simulator
outer-loop finite-tip geometry transaction or production material checkpoint
conversion. No physical step/phase solve occurs. General finite-tip evolution,
virgin-material activation and nonzero-tip birth are unqualified. P2's strict
branch failure remains false; all 34 protected BP3 identities were verified.
The original raw directories and P2/P3 binaries remain under
`build-particle-free/p3-extension/`; the failed first compile log is retained
under `refactoring_boundary/evidence/p3-native-build.*`. No scientific failure
is removed or reclassified.
