# Portable PF4 timestep/scaling/finite-width evidence

[Report](../../../../../doc/reconstructed_fault/particle_free/PF4_STABILIZATION_REPORT.md).
All paths are private experiments. Unlimited stays the production/default path
and smooth baseline; no general physical stress bounds or coupled qualification.

- `assessment.json`:149 final transport/ledger gates, errors/extrema, positive and
  negative peak attenuation, exact analytic overlap, front widths, common-time
  full-field differences, raw/RK-weighted call norm changes, means, held-map
  diagnosis, limited fractions and costs.
- `local.json`:45 independent scaling/SSPRK3 algebra gates: largest common factor,
  all tensor modes, means, direct norm contraction, constants, covariance,
  idempotence and explicit inadmissible-mean rejection.
- `weno-control.json`:77 unchanged local/snapshot gates on the rebuilt kernel.
- `compatibility.json`: independent original PF4 operator; byte-identical
  unlimited/prior-WENO controls; instrumentation nonintrusiveness and idle scaling.
- `standalone.json`, `units.log`, `pf1.json`: both affected DG translation units
  without unity/PCH;628 assertions/8 native cases and94 mathematical identities.
- `formulation.md`: selected setup, numerical conventions and derived quadratic
  ledger, established before transport results were inspected.
- `provenance.json`: branch/core/plugins/input/source identities, exact run limits,
  wall times, peak child RSS and archive identity. `preservation.json` verifies
  incoming files and previous evidence; `final-checks.json` records replay.
- `comparison.png/.pdf`: actual native DG fields and measured fixed-mesh temporal
  differences; prescribed noncommitting transport only.
- `reference.tar.xz`: compact CSVs/records for seven final native runs plus the
  previously saved32-cell snapshots used by the unchanged WENO checker. Native
  binaries, particles, PCHs and checkpoints are excluded. Raw/preliminary states,
  full binaries and logs remain under ignored `build-particle-free/` paths.

The original and final comparison families have byte-identical fields; the
repeat adds explicit logging of any mean outside the exact benchmark envelope.
Actual scaling means have zero excess at every measured call. The prescribed
scalar bounds[-1,1] apply only along the benchmark reference tensor, with one
common factor for the entire tensor. No componentwise clipping. GL9 control
points are the envelope target; between-point polynomial extrema are reported
separately. Mean1.1 is a deliberate rejection control, not repaired.

For fresh native reproduction use unused labels and the same configured builds:

```sh
cmake --build build-particle-free/native -j4
cmake --build build-particle-free/plugin -j2
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf4-stabilization-repeat-local --input benchmarks/reconstructed_fault/particle_free/stabilization_local.prm --plugin build-particle-free/plugin/libstabilization_checks.release.so --postprocessor 'PF4 stabilization checks' --seconds 180
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf4-stabilization-repeat-32-4 --input benchmarks/reconstructed_fault/particle_free/stabilization_32_4.prm --plugin build-particle-free/plugin/libstabilization_checks.release.so --postprocessor 'PF4 stabilization checks' --seconds 180
```

Use `stabilization_16_1.prm`, `stabilization_32_1.prm`,
`stabilization_32_2.prm` for the remaining family. At most two native runs at a
time, one rank/thread. `Time divisor`/`Local only` are test-observer controls,
not production parameters. The finite width is read from existing ell=.15625m;
delta=ell/ln9, held constant across meshes. Compare to the entire overlapping
analytic pulse. No initial limiting or interface/history conversion.

Portable replay needs Python/NumPy, not ASPECT. From the repository root:

```sh
mkdir -p /tmp/pf4-stabilization-replay
 tar -xJf benchmarks/reconstructed_fault/particle_free/evidence/pf4_stabilization/reference.tar.xz -C /tmp/pf4-stabilization-replay
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_stabilization.py /tmp/pf4-stabilization-replay/pf4-stabilization-local-verified --local --output /tmp/pf4-stabilization-replay/local.json
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_stabilization.py /tmp/pf4-stabilization-replay/pf4-stabilization-final-16-1 /tmp/pf4-stabilization-replay/pf4-stabilization-final-32-1 /tmp/pf4-stabilization-replay/pf4-stabilization-final-32-2 /tmp/pf4-stabilization-replay/pf4-stabilization-final-32-4 --output /tmp/pf4-stabilization-replay/assessment.json
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_limiter.py /tmp/pf4-stabilization-replay/pf4-stabilization-weno-control /tmp/pf4-stabilization-replay/pf4-limiter-calibration-32 --calibration --output /tmp/pf4-stabilization-replay/weno.json
```

These three JSONs reproduce the shipped assessments byte-for-byte. The larger
field assessment takes longer than scalar-log checks because it independently
reads and integrates all common-time fields. To regenerate the figure (Matplotlib):

```sh
MPLCONFIGDIR=/tmp/pf4-mpl OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_stabilization.py --run /tmp/pf4-stabilization-replay/pf4-stabilization-final-32-4 --assessment /tmp/pf4-stabilization-replay/assessment.json --output /tmp/pf4-stabilization-replay/comparison
```

Per-call Q sums, RK-weighted contributions and remaining transport/RK changes are
separate measurements. None is a full mechanical energy balance. Experimental
WENO timestep sensitivity remains unresolved. No subsequent stage was run.
