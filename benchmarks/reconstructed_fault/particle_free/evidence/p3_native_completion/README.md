# P3 native frozen append

[Report and captions](../../../../../doc/reconstructed_fault/particle_free/P3_NATIVE_COMPLETION_REPORT.md).
`geometry_histories.{png,pdf}` and `physical_consequences.{png,pdf}` show actual
native plateau/taper exports. The old sinusoidal result is retained separately.
`selection.json` fixes the input and screens before tests; `original.json` and
`sensitive.json` report independent measurements. Pointwise error increases
are retained alongside improved L2 error and physical cell integrals.

`reference.tar.xz` contains native exports **and actual simulator checkpoints**
for six serial processes: fresh, pre-append restart/retry and post-commit
restore, for each of two loaded histories. `archive-manifest.json` hashes every
file. `provenance.json` records the tested core/plugin/source identities;
`logged/` includes commands, timeouts, successes and original failed attempts.
`preservation.json` records all 34 protected BP3 identities and manual work.
P2 strict branch matching stays false. This is a zero-time geometry transaction;
physical finite-tip evolution remains forbidden.

Portable independent replay (Python + NumPy; no ASPECT build needed):

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/p3_native_completion/verify.py
```

Native runs use the isolated evolving-phase build, never a BP3 executable.
Build the maintained `p3_native_checks` target against the same core/headers.
The working raw directories are `build-particle-free/p3-native-completion/runs/`.
Initial failed fixture attempts remain there under `failed-no-particle-observer`,
`failed-bootstrap-native-I` and `failed-zero-observation-rate`; their errors
also remain in `logged/`. The pre-correction empty-constraint restart outputs
are retained under `runs/before-constraint-rebuild/`; their original archive
is `build-particle-free/p3-native-completion/before-constraint-reference.tar.xz`.
No original BP3/P2/P3 data were removed.
The standalone `p3_native.prm` sets the original output directory; select a
**new** output directory when reproducing, preserving checkpoints and evidence.
The required frozen particle environment is explicit:

```sh
ASPECT_P3_CASE=original ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release benchmarks/reconstructed_fault/particle_free/p3_native.prm
python3 benchmarks/reconstructed_fault/particle_free/prepare_p3_restart.py build-particle-free/p3-native-completion/runs/original build-particle-free/p3-review-before --side before
ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release build-particle-free/p3-review-before/input.prm
python3 benchmarks/reconstructed_fault/particle_free/prepare_p3_restart.py build-particle-free/p3-native-completion/runs/original build-particle-free/p3-review-after --side after
ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_PROC_BIND=false build-particle-free/evolving-phase/native/aspect-release build-particle-free/p3-review-after/input.prm
```

For the single plateau control, set `ASPECT_P3_CASE=sensitive` on the fresh
run and override its output directory in an input including `p3_native.prm`.
Restart loads the profile label from native postprocessor status. The helper
copies checkpoints to new branches and modifies only the copied selection
marker (01 before, 02 after); it refuses existing destinations. Every qualified
native command was bounded by 300 seconds; native suite by 180 seconds.

To regenerate figures from raw native files:

```sh
MPLCONFIGDIR=/tmp/p3-mpl python3 benchmarks/reconstructed_fault/particle_free/plot_p3_extension.py build-particle-free/p3-native-completion/runs/sensitive --metrics benchmarks/reconstructed_fault/particle_free/evidence/p3_native_completion/sensitive.json --output build-particle-free/p3-review-figures
```

The native weak functional is observed at zero interval, not relaxed through
another initialization timestep. q=Z/I and the physical phase obstacle are
native owners. The temporary admissible observation rate affects no load for
this constant-friction, zero-damping fixture and never changes birth histories.
General nonconstant-friction/RSF birth is not qualified. Actual native Stokes
caches are invalidated, but a finite-tip mechanical solve remains guarded.
