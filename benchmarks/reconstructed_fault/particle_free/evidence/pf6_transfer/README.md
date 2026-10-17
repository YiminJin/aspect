# Fixed-reference transfer evidence

The [measured report](../../../../../doc/reconstructed_fault/particle_free/PF6_TRANSFER_REPORT.md)
and [pre-run rules](../../../../../doc/reconstructed_fault/particle_free/PF6_TRANSFER_FORMULATION.md)
define this bounded experiment. `verification.json` records passing transfer,
ownership/lifecycle and restart checks **separately** from the retained false
native-quadrature screen and prior false strict-reference comparison.

`reference.tar.xz` contains all ten final native cases: full 1/2/4 ranks,
half/checkpoint and same-rank restart, plus serial-to-four restart. It includes
all compact exports, actual checkpoints, one matching executable/observer pair,
flattened inputs/argv, source hashes/patches, complete modified/untracked
production source, observer/independent checker, and compact failed/capped-run
logs. `archive-manifest.json` hashes every member; `provenance.json` hashes the
archive and retains original per-run records. Full intermediate outputs and
matching intermediate binaries remain in their original local build directories.
Prepared-only records do not count as successful runs. No failed evidence is
overwritten by the final cases.

From the repository root, replay without launching ASPECT (Python with NumPy):

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/replay_pf6_transfer.py benchmarks/reconstructed_fault/particle_free/evidence/pf6_transfer build-particle-free/NEW-transfer-replay
```

The output directory must be new. Replay verifies every archived byte, then
independently evaluates complete DG/Q1 polynomials, target-cell L2 projections,
physical stress/energy/weak loads, fracture actions, fixed particle IDs/positions,
rank ownership and restart fields/RNG. It also reproduces the **failed** native
Gauss6-versus-Gauss24/32 screen; this does not cause a false passing qualification.
The old `strict_reference_match_passed=false` belongs to the preserved
[publication evidence](../pf6a_publication/README.md); it is not re-evaluated or
repaired by this remapping study.

The native source/observer build is pinned by SHA256 in every record. Native
commands use one thread/rank, core binding, `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1`
and at most four ranks; each case has a 180 s cap. Absolute old paths in archived
inputs are provenance, not portable launch commands. Use the preparation runner
with a fresh label for any separately authorized new native run. See the parent
README for build and lifecycle instructions; use the transfer DSO alone.

This archive qualifies the specified uniform Cartesian/fixed-surface fixture.
It does not qualify automatic/local AMR, lossy phase/obstacle coarsening, surface
remeshing, moving geometry, transport, propagation or general PF5 completion.
