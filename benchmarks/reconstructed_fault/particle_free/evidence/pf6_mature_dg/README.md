# Bounded stationary mature DG evidence

[Measured report](../../../../../doc/reconstructed_fault/particle_free/PF6_MATURE_DG_REPORT.md),
[pre-run scope](../../../../../doc/reconstructed_fault/particle_free/PF6_MATURE_DG_FORMULATION.md),
[spatial figure](spatial-reference.svg).

DG/FE compatibility passes; spatial accuracy and the 1% timestep screen for
perturbation loads do not. `verification.json` preserves these distinct results
and `strict_reference_match_passed=false`. All 18 native attempts, including
both original failed ports, are recorded; actual native cost is 1278.61/3600 s.
No phase evolution, automatic AMR, transport, moving geometry or first-event
trajectory was launched. The production/default and limiter selections remain.

`measured-summary.json` contains per-time full-tensor norms, physical-profile
representation errors, quadrature/timestep/spatial differences, point/cell-trace
ownership, jumps and costs. `publication-summary.json` retains native all-point
errors, derived bounds and obsolete net-stress-only allowance failure counts.
`cancellation-summary.json` records independent T versus physical tau amplitudes.
`independent-math.json` checks the fixed-chi mature transformation and all nine
Q2 modes/mass integration, including x²y².

`reference.tar.xz` contains the exact inputs, argv/source/build/output hashes,
logs and replay data for every case; selected complete authoritative stress,
DG/FE history, unfiltered point-traction exports; production source/header
snapshots and task-only/incoming patches; one core and matching primary/final
observer artifacts. The archive omits redundant gate/half-step outputs and some
fine intermediate exports. All original complete outputs, checkpoints from
prior stages, pinned binaries and failures remain untouched in their original
worktree directories. `archive-manifest.json` hashes every included member;
`provenance.json` records the archive identity and selection.

Offline replay verifies every archive member, independently rechecks the
mathematics and reproduces the recorded norms, cancellation ratios and raw
traction export. Destination must not exist; it launches no ASPECT or MPI job:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/replay_pf6_mature.py benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_dg build-particle-free/NEW-mature-replay
```

Build commands and physical settings are in the report. Actual native execution
used the established logger with `run_pf6_mature.py` and the selected
`run_pf6_mature_cases.py` gates/trajectories/sensitivity/spatial stages. Drivers
refuse existing case directories. A separately selected repeat can use fresh
labels, for example:

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py NEW-mature-dg 710 python3 benchmarks/reconstructed_fault/particle_free/run_pf6_mature.py NEW-dg --backend dg --steps 10 --seconds 700
```

Rebuild against this worktree's core and maintained BP3 library first. Original
absolute paths in the packet are provenance, not portable launch commands. The
maintained plugin uses `ASPECT_BP3_LOCAL_BOTTOM_TEST` (uniform strengthening,
physical Vp=1e-9) and leaves the oscillation-test option off. Particle control
explicitly selects the existing D3 convention; legacy defaults are unchanged.

Use `authoritative-stress-N.csv` tau columns for physical history. Independent
DG/FE T/Z are separate; particle T columns echo tau and positive-step Z is NaN,
while surface-profile Z=0 denotes absent independent storage. The old particle
`physical-fe` history column was a velocity placeholder and is unused; later
observer output labels it NaN. Common physical-identity gates resolve static
Theta's native property-name alias. Raw nodal surface profiles are consistent
weak-load projections. `derived/*-raw-tractions.csv` contains unfiltered point
tractions at common Gauss3 diagnostic points; native Gauss6/8 mechanical
integration is separate. Both cell traces remain in the normal-cut files;
common-point RMS comparisons exclude ambiguous face traces and report jumps
separately. Step-zero committed history and the kinematic solve are distinct.

Mature DG restart/MPI, coupled rejection/retry, material heterogeneity, moving
geometry, automatic AMR, fault birth and propagation are not qualified here.
Keep the independent native publication check. Prior failed quadrature/reference
and legacy-domain evidence remain in their original packages.

The completed offline replay verified all 1567 archive members and reproduced the
measured norms, cancellation ratios and unfiltered traction table exactly; see
`portable-replay.json`. Archive size is 141989708 bytes; SHA256 is
`eec5f1d5465026977aa4a08c7264f2e9e5695addb632b3b5327800b3558b3881`.
`final-checks.json` verifies preservation, recorded input/output hashes and defaults.
