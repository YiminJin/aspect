# Bounded mature resolution comparison

Six matched fresh serial stationary trajectories: CC, FC, FF with unlimited
DGQ2 and native particle controls. Gauss6 mechanics, unchanged Gauss3 DG source,
three positive dt=200000 s steps to T=600000 s; fixed 59-node surface. FC imports
only the complete coarse step-zero Q1 phase, including hanging values and its
coarse exterior lattice. It does not import constitutive history or I.

[Report](../../../../../doc/reconstructed_fault/particle_free/PF6_MATURE_RESOLUTION_REPORT.md),
[pre-run formulation](../../../../../doc/reconstructed_fault/particle_free/PF6_MATURE_RESOLUTION_FORMULATION.md),
[status](../../../../../doc/reconstructed_fault/particle_free/STATUS.md).
The numerical JSONs report neutral backend differences, not accuracy errors
against DG or particles. Every prior failure remains, including the failed
native-quadrature/reference criteria and `strict_reference_match_passed=false`.

- `measured-summary.json`: all matching times, full embedded tau/Cauchy tensor
  norms, common physical weights, fixed interior/endpoint regions, actual native
  weak/background-measure effects and a common geometric surface functional,
  V/state/slip, cancellation, jumps, costs and acceptance checks.
- `independent-verification.json`: exact physical-vertex prolongation,
  nonnodal field/localization/I agreement, all-nine-mode saved-coefficient
  reconstruction and physical tau.
- `final-checks.json`: matching initial controls, same artifacts/1031 native
  source hashes, unchanged 34 material defaults, 32 byte-identical retained DG
  field files, default particle reproduction, all entry files retained and old
  archive digests. `unit-results.json`: 55 cases/2625 assertions pass.
- `comparison.svg`/`.png`: final native surface projections/state; grey regions
  are the fixed 125 m endpoint windows. Curves are diagnostic evidence only.
- `reference.tar.xz`, `archive-manifest.json`, `provenance.json`: portable inputs,
  selected common fields, actual weak loads/mass, complete final DG coefficients,
  initialization/reference fields, logs/records, failures, source snapshots,
  entry/task patches and pinned binaries. Raw full outputs and preliminary builds
  remain under `build-particle-free/mature-resolution/`.

Offline replay (destination must be new) launches no ASPECT or MPI:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/replay_pf6_mature_resolution.py benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_resolution build-particle-free/NEW-mature-resolution-replay
```

Replay verifies every archive member, reruns the mathematical/measurement
reductions exactly, checks 32 retained field hashes and default particle fields,
and reproduces six unfiltered common-point traction exports. Common-point trace
ownership is explicit; normal-cut CSVs retain both interface traces. Step-zero
committed T/Z/tau are unloaded; its separately recorded kinematic solve is not a
positive physical history interval. Particle T columns are a tau placeholder
and positive-step Z is NaN; those quantities have no independent particle storage.

Build commands used (GCC12.4, deal.II9.6.2, one thread/rank):

```sh
cmake --build build-particle-free/native -j 2
cmake -S benchmarks/reconstructed_fault/bp3_native/server/plugin -B build-particle-free/mature-resolution/bp3-plugin -DAspect_DIR=$PWD/build-particle-free/native -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_C_COMPILER=/opt/gcc/12.4.0/bin/gcc -DCMAKE_CXX_FLAGS=-DASPECT_BP3_LOCAL_BOTTOM_TEST
cmake --build build-particle-free/mature-resolution/bp3-plugin -j 2
cmake -S benchmarks/reconstructed_fault/bp3_native -B build-particle-free/mature-resolution/observers -DAspect_DIR=$PWD/build-particle-free/native -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_C_COMPILER=/opt/gcc/12.4.0/bin/gcc -DBP3_COUPLED_PLUGIN=$PWD/build-particle-free/mature-resolution/bp3-plugin/libbp3_restore_150x50.release.so
cmake --build build-particle-free/mature-resolution/observers --target bp3_coupled_checks -j 2
```

Entry provenance: the first full-tree capture overlapped the first four edits.
Their original bytes/hashes were recovered from the preceding study's pinned
final-source archive. Entry bytes and the reconstructed `incoming.patch` were
verified against the original manifest; `overlapping-capture.patch` preserves the
initial capture. `task-only.patch` separates this task from earlier manual work.

All primary cases use these exact SHA256 identities:

| Artifact | SHA256 |
|---|---|
| aspect-release | `eb90af8d0a93242aa423e563681f4cc0cd161aa33e7ef6eca7e33d5d149c4719` |
| BP3 plugin | `54bb4908ffa30427e8aec213a03f1699b951a5417d325becf172dc5c34f019c3` |
| Coupled observer | `f746ca6d7ebbd2ad694fcc12b64800e52fd0608b53e8e0dbb83b29dccf582d1f` |

The native matrix used the established bounded logger and
`run_pf6_mature_resolution_cases.py`. The initial strict CSV header rejected
CRLF before setup; its raw input/log remain in `runs/dg-fc`, while successful FC
is `runs/dg-fc-final`, using separately retained LF inputs. Future preparation
emits the required LF convention. No physics or tolerance was changed for that
input correction. Drivers refuse existing output directories. A separately
selected fresh repeat can use, for example:

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py NEW-mature-FC 710 python3 benchmarks/reconstructed_fault/particle_free/run_pf6_mature.py NEW-dg-FC --backend dg --build-directory build-particle-free/mature-resolution --steps 3 --quadrature 6 --refinement 7 --phase-reference build-particle-free/mature-resolution/inputs-lf/coarse-phase.csv --common-points build-particle-free/mature-resolution/inputs-lf/fine-points.csv --seconds 700
```

Eight native attempts cost 743.68/3600 s; primary six cost 696.52 s. Retain the
failed 1% temporal-load screen and all old errors. Fine temporal uncertainty,
continuum spatial order, mature restart/MPI, phase evolution, transport, AMR,
moving geometry and the first-event trajectory are outside qualification.
No limiter/production adoption, commit or subsequent run is selected.

Completed offline replay verified all **1260 members**, reproduced both complete
measurement/reference JSONs exactly, verified 32 retained field hashes and
reproduced six raw traction exports byte for byte. See `portable-replay.json`.
Archive size: 213109952 bytes; SHA256:
`92a1e76500b3da0222802ca0ba9b4b0a4f0e5777104bab48c1a7374ce552f2a6`.
