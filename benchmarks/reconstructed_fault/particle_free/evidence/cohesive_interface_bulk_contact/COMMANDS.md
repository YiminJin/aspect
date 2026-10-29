# Commands and scope

Run from `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, baseline
f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and sequential timing runs.
Existing output directories are intentionally protected against overwrite.

```sh
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-bulk-contact/build -DAspect_DIR="$PWD/build-particle-free/native" -Ddeal.II_DIR=/opt/dealii/9.6-local/lib/cmake/deal.II -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/cohesive-interface-bulk-contact/build --target cohesive_interface_path -j2
```

The native executable's optional arguments after `order compress` are numerical
lifting half-width, numerical normal-test half-width, `cartesian` or `fitted`,
then a file of physical diagnostic coefficients. Defaults retain the original
Cartesian mesh and mesh-scaled widths. Fitted geometry uses MappingFEField Q4,
all surface knots and both support boundaries. Native Q2/P1disc or Q3/P2disc
mechanics, Gauss6 assembly/Gauss8 verification and mean-zero p′ are retained.
There is no new production parameter.

With `B=benchmarks/reconstructed_fault/particle_free`:

```sh
python3 $B/investigate_cohesive_interface_bulk_contact.py --identity
python3 $B/investigate_cohesive_interface_bulk_contact.py
python3 $B/investigate_cohesive_interface_bulk_contact.py --case fitted-Q3
python3 $B/investigate_cohesive_interface_bulk_contact.py --case offset96
python3 $B/investigate_cohesive_interface_bulk_contact.py --case narrow-Q3
python3 $B/check_cohesive_interface_lifting.py
python3 $B/assess_cohesive_interface_bulk_contact.py
python3 $B/evolve_cohesive_interface_bulk_contact.py
python3 $B/repeat_cohesive_interface_bulk_contact.py
python3 $B/fresh_cohesive_interface_bulk_contact.py
python3 $B/evolve_cohesive_interface_bulk_contact.py --own-time
python3 $B/analyze_cohesive_interface_bulk_evolution.py
MPLCONFIGDIR=/tmp/cohesive-bulk-contact-mpl python3 $B/plot_cohesive_interface_bulk_contact.py
python3 $B/verify_cohesive_interface_bulk_contact.py
```

The native C++ signed solver is reused through the unchanged shared library
from the preceding path fixture. The two full-run scripts change only their
isolated process's factory/archive roots; no persisted default or production
owner is replaced. Native physical states are solved, not replayed.

Frozen coefficients/geometry come from saved `work96/final.json`. Short
controls start at .42025 s from the original `work-temporal-incoming.json`,
with old surface records unchanged and appended tied/virgin records explicit.
Each bulk response is reconstructed independently; its initialization difference
is reported and never published into the original trajectory. The separately
initialized fresh controls avoid this import entirely. The original .44 s
load-to-hold transition and .445 s endpoint are unchanged.

Every short control exercises actual native contact-loss rejection, correct
retry, rebuilt-context JSON restart and duplicate-publication rejection. These
are benchmark checkpoints, not production ASPECT checkpoints. Fresh causal
extensions additionally assert unchanged old response/stress columns, old
histories, physical time and storage before accepting geometry once.

Each native kernel has a 300 s timeout. No server job, long propagation run,
MPI expansion, R calibration, pressure lag, constitutive change or tolerance
relaxation is performed. Original failed results stay in their original
evidence directories. Cleanup storage changes are separately logged with
SHA256 and restoration instructions under `maintenance/disk_cleanup_2026_10_27`.
