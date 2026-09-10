# Conservative cold-lookup rejection: bounded result

Retain the optimization. Cold preparation of the existing nine fine profiles
decreases from 19.126755 to 3.854614 seconds (4.962x). No equation, I_h profile,
termination rule, support width, reference-cell tolerance, integration tolerance,
or MPI ownership rule changes. The verified warm lookup reuse is retained.

## Invariants and scope

Only exact degree-one MappingQ/MappingQ1 with axis-aligned affine hypercubes
uses early rejection. The global enclosure contains every cell's
reference-tolerance-expanded box, plus outward roundoff padding. Unsupported
mapping types (including subclasses), degrees, or cell geometry use the
original search. The enclosure is rebuilt with invalidated mesh/mapping
lookups; existing mesh-deformation clearing remains in force.

Filtering removes only proved-outside requests. Surviving requests retain
order and use the original RemotePointEvaluation handshake and shared-cell
search. The private Batch stores the map back to original adaptive-request
indices. Missing entries stay missing, not found with zero phase field.

## Evidence

`cold-comparison.json` is produced by `compare_cold_rejection.py` and requires
equal counts, full found/missing sequence digest, and all nine integrals printed
with round-trip precision. Both modes report 811,863 found and 522 missing
requests, digest 4277628729994074550. Every integral matches exactly at that
precision. Optimized cold/warm vectors additionally pass direct equality.
Warm preparation takes 0.315248 seconds. Both fine runs pass 262,217 assertions.

| Matched nine-profile harness | Full-request baseline | Early rejection |
|---|---:|---:|
| Cold preparation (s) | 19.126755 | 3.854614 |
| Whole harness (s) | 21.218310 | 8.283742 |
| Peak RSS (KiB) | 309,044 | 307,928 |

The baseline retains its full lookup maps through completion, as does the
optimized path. Both comparison modes carry request-index vectors; their
approximately 3.1 MiB storage on this fixture is new relative to the old
production batch layout. The small RSS difference is not claimed as a memory
optimization. These are single measurements, not scaling/error bars.

Affected checks only:

```
timeout 120 build-pf-cpdi/aspect-release --test '[phase_field_fault_ih_cache],[phase_field_fault_ih_accuracy]'
timeout 120 mpirun -np 2 build-pf-cpdi/aspect-release --test '[phase_field_fault_ih_cache],[phase_field_fault_ih_accuracy]'
python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/performance/run_cold_rejection.py rejection
python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/performance/run_cold_rejection.py baseline
```

One rank: 59,199 assertions / 4 cases pass (`cold-unit-one-rerun.log`).
Two ranks: 59,199 and 2,435 assertions / 4 cases pass (`cold-unit-two.log`).
Controls exercise the captured outside point, inside/on/outside boundary
requests, accepted/rejected reference-tolerance offsets, four-cell shared-vertex
multiplicity, per-owner sample values, changed ordering, empty requesting ranks,
mesh refinement invalidation, and unsupported degree-two mapping fallback.
The existing analytic/distributed I_h accuracy tests are retained. No full
trajectory or all-profile benchmark is used.

Release builds use -j4. The first timing command failed before testing because
/usr/bin/time is absent; the preserved corrected invocation passed. The first
baseline harness measurement did not yet retain maps because of an incremental
build timing mismatch; its 23.79-second run is preserved as
`cold-baseline-pre-retention.*` and excluded from the matched memory comparison.
The matched runs use the same executable SHA recorded in their resource files.
All runs stayed below 120 seconds; aggregate test execution was well below ten
minutes. No timeout was retried.

## Handoff

Production changes: `include/aspect/material_model/phase_field_fault.h` and
`source/material_model/phase_field_fault.cc` only for this task. The private
lookup interface now returns Batch instead of the bare remote evaluator.
Tests and the opt-in harness are in `unit_tests/phase_field_fault_ih.cc`;
`current_design.md` and `specification.tex` record the conservative guard.
`retained-lookup-aspect-release` preserves the pre-guard executable.

Changes are uncommitted. No known unsafe or incomplete production change remains.
The earlier performance report's unfinished final addendum remains unfinished;
this review does not silently replace it. Unsupported-mapping speed, full-fault
memory scaling, and trajectory speedup remain unmeasured, not blockers for this
bounded result. Further cold-I_h work is separate; no normalization redesign or
larger benchmark campaign is started.
