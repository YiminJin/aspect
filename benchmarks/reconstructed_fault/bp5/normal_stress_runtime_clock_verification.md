# Runtime A/B clock verification — 2026-09-22

The complete PRMs are `normal_stress_experiment_A.prm` and
`normal_stress_experiment_B.prm`; run instructions are in
`normal_stress_AB_README.md`. No BP5 restart trajectory was launched here.

## Implemented boundary

- Core `post_resume_time_step` runs once after all checkpoint state is restored
  and before the first `start_timestep`. Only a finite, positive, MPI-identical
  reduction of the saved pending dt is allowed. It adjusts pending time around
  the restored accepted origin, leaving old dt and the step number unchanged.
  No slot means no change and no extra collectives.
- Benchmark `BP5 recorded half steps` reads A's four accepted CSV rows directly.
  It validates the original restored pending clock before requesting its half,
  supplies subsequent caps through the existing timestep manager, and checks
  the actual clock before every solve. No checkpoint layout or bytes change.
- The second half includes an explicitly recorded endpoint-rounding correction
  of at most one absolute-time ULP. The exact same four endpoints are required.
- Constitutive laws, quadrature, residual/linear tolerances, history publication,
  and original timestep models are unchanged.

## Checks performed

Release ASPECT and both loaded BP5 libraries built with `-j4`. The two new test
targets also built successfully. Both full A and B PRMs passed ASPECT `--validate`
using those libraries (syntax/parameter checks, not BP5 runtime qualification).

```
benchmarks/reconstructed_fault/bp5/build/test_normal_stress_clock
```

Passed: actual captured late-time dt values, eight half-step clocks, exact common
endpoints, rounding corrections, bad/incomplete/extra CSV records, nonfinite and
malformed values, and rejection of a shortened actual timestep.

```
python3 benchmarks/reconstructed_fault/bp5/test_normal_stress_diagnostic.py
```

Four tests passed, including prior diagnostic tensor/mass, plot, staging and
clock-only archive regression coverage. The archive-edit path is retained for
old experiment reproduction; the new direct inputs do not use it.

```
python3 benchmarks/reconstructed_fault/bp5/test_restart_clock.py \
  build-pf-cpdi/aspect-release benchmarks/reconstructed_fault/bp5/build --ranks 1
python3 benchmarks/reconstructed_fault/bp5/test_restart_clock.py \
  build-pf-cpdi/aspect-release benchmarks/reconstructed_fault/bp5/build --ranks 2
```

All **8 one-rank** and **9 two-rank** expected outcomes passed:

| Case | Expected and observed |
|---|---|
| Fresh small nonconstant-temperature Box | Ordinary checkpoint written |
| Keep pending dt | Resumed clock unchanged |
| Half pending dt | First resumed solve starts at the half endpoint |
| Zero / increased / infinite dt | Rejected before a resumed mechanical solve |
| Rank-dependent positive dt (two ranks) | Rejected collectively |
| Recorded controller | Eight half-step solves, exact prescribed endpoints |
| Additional safety cap | One accepted half step; next reduced step rejected before mechanics |

Before the first resumed solve the test verifies unchanged previous dt and step
number, and exact agreement of the three restored bulk-vector norms with their
values inside the restart callback. The core does not write these vectors or
particle/fault histories. Source checkpoint SHA256 hashes remained unchanged.
This is a generic tiny Box lifecycle test, not a BP5 fault-history trajectory.

Detailed local logs/results were retained in
`/tmp/bp5-restart-clock-1-jbpf_sx9/` and
`/tmp/bp5-restart-clock-2-9dfzyikx/`. The checked-in test reproduces them without
BP5 input data. It makes fresh temporary directories and does not retry failures.

The existing focused coupling regression also passed:

```
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_surface_dynamic_pressure$' -j1
# 1/1 passed, 32.20 s (32.22 s total).
```

## Input equivalence and limitations

The A/B parameter dictionaries differ only in output directory, four versus
eight accepted steps, B's added timestep controller and its A-trajectory path.
A retains the supplied original run's physical and solver settings. Neither
file includes another PRM.

The actual late BP5 A/B experiment and new native stress/centerline exports
remain to be exercised on the server. Both branches must restore independent
unedited copies of original `restart/01` (accepted step 5612), not the final
diagnostic checkpoint or A's newly advanced state. Rebuild all loaded plugins
against the modified simulator header. Retain the server's separately accepted
restart-normalization allowance; this change does not alter that check or the
production integration tolerances.
