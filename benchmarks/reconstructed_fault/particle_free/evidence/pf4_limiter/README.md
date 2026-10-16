# Portable bounded PF4 limiter evidence

Read [the report](../../../../../doc/reconstructed_fault/particle_free/PF4_LIMITER_REPORT.md).
No default/constitutive change; serial fixed-grid prescribed transport only.
All eight native runs use the same core; full source/input/compiler/run identities
are in `provenance.json` and archived `record.json` files. Raw binaries, logs,
preliminary assessments and incoming preservation manifest remain in the ignored
`build-particle-free/` directories. No existing evidence was overwritten.

- `local-and-snapshots.json`: final77 checks, physical-coordinate independent
  indicators/reconstruction, Q2 quadrature rank, covariance and boundary controls.
- `threshold-calibration.json`, `threshold-selection.json`: initial73 calibration
  gates and timestamped choice C=.1 before limited validation, snapshots at4/32.
- `assessment.json`:66 transport gates, all grids/cases, L1/L2/extrema, initial
  projections, norm changes, front widths, means, fractions and separate costs.
- `operator-control.json`: fresh original PF4 operator/source/publication control;
  matrices, loads, lifecycle, accepted and metrics byte-identical to old evidence.
- `reconstruction-diagnosis.json`: bounded frozen-snapshot mode/weight diagnosis
  and roundoff comparison of new unlimited fields with the original PF4 fields.
- `standalone.json`, `units.log`, `pf1-control.json`: both production translation
  units without unity/PCH;628 native assertions and94 independent checks.
- `comparison.png/.pdf`: plots of exported native DG fields and independent
  smooth-error measurements. Prescribed transport only, no coupled qualification.
- `preservation.json`: incoming files and authorized changes, original handoff,
  manual instructions/derivation, plans and all prior evidence unchanged.
- `reference.tar.xz`: all compact CSVs and records for eight runs, no particles,
  executable, PCH or large native output. Archive hash in provenance.

Build from the worktree using the existing build/toolchain:

```sh
cmake --build build-particle-free/native -j4
cmake --build build-particle-free/plugin -j2
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf4-limiter-repeat-cal4 --input benchmarks/reconstructed_fault/particle_free/limiter_calibration.prm --plugin build-particle-free/plugin/liblimiter_checks.release.so --postprocessor 'PF4 limiter checks' --seconds 180
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf4-limiter-repeat-32 --input benchmarks/reconstructed_fault/particle_free/transport_32.prm --plugin build-particle-free/plugin/liblimiter_checks.release.so --postprocessor 'PF4 limiter checks' --seconds 180
```

Use unused labels and repeat transport4/8/16 as needed. The postprocessor's
`Calibration only` switch affects only local/snapshot diagnostics. No simulator
runtime limiter selector was added. Supplementary native local KXRCF covariance
assertions were added after validation without changing the kernel or C; all
validation binaries/plugins remain pinned. Do not recalibrate from validation.

Replay the portable numerical assessments without ASPECT (NumPy required):

```sh
mkdir -p /tmp/pf4-limiter-replay
 tar -xJf benchmarks/reconstructed_fault/particle_free/evidence/pf4_limiter/reference.tar.xz -C /tmp/pf4-limiter-replay
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_limiter.py /tmp/pf4-limiter-replay/pf4-limiter-local-final-4 /tmp/pf4-limiter-replay/pf4-limiter-calibration-32 --calibration --output /tmp/pf4-limiter-replay/local.json
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_limiter.py /tmp/pf4-limiter-replay/pf4-limiter-validation-4 /tmp/pf4-limiter-replay/pf4-limiter-validation-8 /tmp/pf4-limiter-replay/pf4-limiter-validation-16 /tmp/pf4-limiter-replay/pf4-limiter-validation-32 --output /tmp/pf4-limiter-replay/transport.json
```

`local.json` and `transport.json` reproduce the corresponding checked JSONs.
Operator control can also be replayed with `assess_transport.operator_reference`.
The source reuse reference is the neighboring `aspect-weno` checkout, read only;
it was never compiled or modified as part of this comparison.
