# Homogeneous DG mesh-data evidence

[Measured report](../../../../../doc/reconstructed_fault/particle_free/PF6_DOMAIN_FREE_REPORT.md),
[pre-run rules](../../../../../doc/reconstructed_fault/particle_free/PF6_DOMAIN_FREE_FORMULATION.md).
The selected default-off mode restores the 64→256→64 diagnostic without a domain
object, domain integration or extra particles. Its independent publication check
and normalization/derivatives remain. The separate legacy deep CPDI case still
fails its original coverage guard. Original Gauss6 and strict-reference failures
are retained in their original evidence packages.

`verification.json` distinguishes physical equivalence, representation error,
transfer, MPI/restart, negative controls and retained limitations. Native input/
source/build hashes, argv, timings and logs accompany every case. The archive
contains all 17 bounded transfer/configuration cases, actual restart files, one
matching executable/observer, modified/new production source, test access, the
supplied physical seed and independent checks. The separate legacy FE control's
21 CSVs and prior matching CSVs are included. Original raw outputs and pinned
binaries remain in their run directories.

The point-state `*-domain-free.csv` files report actual requested-domain flags,
the manager's private domain-handler pointer and geometry-only associations.
These are object-absence observations, not fabricated access counters: source
constructs the handler only when requested, and access without it throws.
Selected particle CSVs contain no domain-volume/box columns. All 144 physical
IDs/coordinates remain fixed. Native changed-count RNG replay is unavailable;
population changes are disabled and stochastic replay is not claimed.

Verify archive hashes and independently replay the checks with Python/NumPy:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/replay_pf6_domain_free.py benchmarks/reconstructed_fault/particle_free/evidence/pf6_domain_free build-particle-free/NEW-domain-free-replay
```

The output directory must not exist. Replay launches no ASPECT/MPI job. It checks
all-mode remaps, the actual seed's representation, common physical fields and
energies/weak loads, I/DI/D²I against Gauss24/32, persistent/native/point fields
across rank/restart comparisons, domain absence and the byte-identical legacy
control. `archive-manifest.json` hashes every member; `provenance.json` identifies
the archive and previous local evidence. Original absolute paths are provenance,
not portable native launch commands.

Actual bounded drivers run from the repository root through the established logger:

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py NEW-serial-driver 900 python3 benchmarks/reconstructed_fault/particle_free/run_pf6_domain_free_cases.py serial --prefix NEW
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py NEW-negative-driver 900 python3 benchmarks/reconstructed_fault/particle_free/run_pf6_domain_free_cases.py negative --prefix NEW
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py NEW-mpi-driver 900 python3 benchmarks/reconstructed_fault/particle_free/run_pf6_domain_free_cases.py mpi --prefix NEW
```

These are separately authorized new native runs, not part of replay. Rebuild the
core and `pf6_transfer_checks` observer first. The driver refuses existing labels.
The source/observer is frozen before preparation; each case pins its own inputs
and binaries. Keep the surface fixed and the representable-phase coarsening rule.
Coupled hanging-node phase evolution, heterogeneous material projection,
automatic AMR, transport, moving geometry and propagation remain unqualified.
