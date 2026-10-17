# Gauss12 and bounded local-transfer evidence

[Measured report](../../../../../doc/reconstructed_fault/particle_free/PF6_LOCAL_REPORT.md),
[formulation and explicit amendments](../../../../../doc/reconstructed_fault/particle_free/PF6_LOCAL_FORMULATION.md).
The accepted subset is frozen fixed-surface 64→67-cell remapping/evaluation and
1/2/4-rank native restart. Coupled hanging-node phase evolution and automatic
AMR remain guarded. The original Gauss6 and strict-reference failures remain
failed; the deep sparse-particle support failure also remains explicit.

`verification.json` separates these outcomes. `reference.tar.xz` contains all 13
final same-build cases: retained-state Gauss6/12/24/32 evaluation, short Gauss12/24/32
trajectories, uniform half/continued restart, local 1/2/4-rank transfers and
restore-only restarts, plus serial-to-four redistribution. It includes all
compact exports, actual checkpoints, the matching executable/observer, inputs,
argv/source hashes/patches, complete modified/untracked production source,
observer/test access, independent checks and compact intermediate/failure logs.
Full older outputs and pinned failed-run binaries stay in their original local
directories. No failed attempt is relabeled as a final successful case.

`archive-manifest.json` hashes every member. `provenance.json` hashes the archive
and retains the native records and original uniform-transfer archive hash.
Absolute original paths are provenance, not portable launch commands.

Replay with Python/NumPy, without launching ASPECT or MPI:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/replay_pf6_local.py benchmarks/reconstructed_fault/particle_free/evidence/pf6_local build-particle-free/NEW-local-replay
```

The output directory must be new. Replay verifies archive hashes, independently
evaluates DG/Q1 fields and integrals, checks normal-ray/surface normalization,
target means, native hanging traces, every exported Cartesian Voronoi domain,
fixed IDs/coordinates, rank ownership, and all compared persistent/native
histories and weak loads across restarts. Same-count RNG streams replay exactly;
cross-count native RNG replay is unavailable with population changes disabled.
Matching fresh four-rank streams does not establish stochastic restart replay.

The independent Voro library counterexample is `voro_closed_patch.cc`. Actual
command used in this environment:

```sh
/opt/gcc/12.4.0/bin/g++ -O2 -I/home/ein/local/voro++/0.4.6/include/voro++ benchmarks/reconstructed_fault/particle_free/voro_closed_patch.cc /home/ein/local/voro++/0.4.6/lib/libvoro++.a -o build-particle-free/quad-local/voro-closed-patch
build-particle-free/quad-local/voro-closed-patch
```

It independently compares closed-patch insertion and target area with midpoint
bisectors. The recorded library/source/binary hashes accompany its result.
The native adapter verifies every intended candidate is resident and retains
the original positive-volume/physical-coverage guards. A separate native failure
with all four candidates inserted demonstrates genuine one-layer insufficiency
at 256 total cells; no guard is bypassed and no physical domain is fabricated.

Native cases use one thread/rank, core binding, frozen RK2 advection and a
180-second cap; jobs are sequential with at most four MPI ranks. See the parent
README and preparation runner for separately authorized new cases. Load only
the transfer observer DSO; other tests defining `MomentCycle` cannot share its
process. Limiters and production defaults remain unchanged.
