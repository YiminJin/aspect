# Horizontal moment-consistency comparison — partial qualification

Historical first-pass report. The subsequent pressure audit/correction and
completed four-step/MPI qualification are recorded in
[the follow-up report](../moment-qualified-final/report.md). The stopped runs
and measurements below are preserved unchanged.

The first accepted update supports the proposed mechanism: ordinary history
publication turns an almost equilibrium-invisible stress mode into a resolved
weak load. Native retention avoids that jump; the horizontal moment publication
also avoids it while retaining the physical shear loading. **This is not a
completed four-step qualification or a selected production correction.** Both
alternative branches stop at the unchanged pressure-compatibility guard before
their second real solve completes. No solver correction, tolerance change,
two-rank replay, inclined test, or BP5 trajectory was attempted.

## Results

All branches are independent fresh starts with identical physical parameters.
Only `History mode` and output directory differ between the three complete PRMs.
The actual real timestep is 0.1 s, beta = 0.999999999999999 and kappa =
99999.999999999942 Pa s. Initialization evaluates a stress response but retains
zero stress history: its evaluated tensor is **not** published as the first
physical history update. Publication comparisons below therefore exclude step 0.

| Branch | Accepted real steps | First transfer jump J (Pa m) | Status |
|---|---:|---:|---|
| A: production | 4 | 0.0672938746322797 | Baseline reproduced |
| B: native_history_reference | 1 | 1.77149955571401e-14 | First publication passes; step 2 blocked |
| C: horizontal_moment_update | 1 | 1.02720485271835e-11 | First publication passes; step 2 blocked |

Reversing native quadrature summation gives a first-step assembly-roundoff
measurement of 5.69222128533178e-14 Pa m. B's jump is 0.311 times this measurement.
The noncancelling first-step scale F_abs is 75981.0487587967 Pa m: the sum of
absolute individual tensor/test-gradient/JxW contributions before cancellation
and constraint elimination. No extra localization weight is inserted.

C reduces A's jump by **6.551164011e9**. Its prescribed absolute allowance is
max(1e-9, 1e-10 F_abs) = 7.59810487588e-6 Pa m; it also clears the independent
1e4 reduction requirement. The component jumps are:

| C quantity | Value |
|---|---:|
| J_proj (Pa m) | 9.47854179515e-12 |
| J_map after complete production transfer (Pa m) | 4.17627845310e-12 |
| Projection/map signed inner product ((Pa m)^2) | -8.84533454151e-25 |
| Largest single free-load difference (Pa m) | 5.32984036660e-13 |
| Maximum fitted zeroth/first tensor-moment errors (Pa) | 2.26630e-13 / 2.06602e-14 |
| Maximum final working-history zeroth/first errors (Pa) | 3.66022e-12 / 3.12014e-12 |

The signed cross term is retained in `comparison.json`; norms were not added.
The independently implemented periodic Q2 weak-load calculation agrees with
native J to 3.16e-19 Pa m or better in B/C, and 5.56e-17 in A steps 1–3.
It includes both velocity components and all free rows, with periodic node
identification and prescribed top/bottom variations removed. The fixture has
no hanging nodes. The C++ calculation uses actual FEValues and constraint
distribution; the independent calculation constructs polynomial test functions.

### Physical loading and resolved response

At the first real accepted state, all three branches have **identical exported
velocity, all four velocity-gradient components, current stress tensor and
incoming native stress**, at every saved native point. Thus first-step response
agreement verifies identical incoming data; it does not establish later response
agreement after the new publications have been consumed.

| First publication | B | C |
|---|---:|---:|
| Retained domain mean shear (Pa) | 500.000762686476 | 500.000762686461 |
| Top tangential weak reaction (Pa m) | 125.000190671611 | 125.000190671612 |
| Bottom tangential weak reaction (Pa m) | -125.000190671612 | -125.000190671613 |

C/B relative differences are 3.0241e-14, 8.6402e-15 and 8.2991e-15 respectively,
well below 1e-6. The physical mean increment is measured from retained zero
history, not by subtracting the uncommitted timestep-zero response. Flat-wall
pressure has no tangential reaction contribution here.

The first current native tensor's cell-centered, JxW-weighted Frobenius RMS is
3.10892568746 Pa in every branch. C removes 3.10892568746 Pa in that same norm.
Its retained parent **xy-component**, equal-parent within-cell RMS is
3.77298861749e-12 Pa, versus A's 1.54453181434 Pa. These tensor and component
statistics are different measures and must not be plotted as interchangeable.
B's parent RMS is marked -1 (unconsumed shadow data), not interpreted as stress.
Tangential native stress variation at matched normal Gauss ordinates is only
1.09847e-10 Pa in the first real state, supporting the fit's required symmetry.

A's subsequent jumps are 0.116882455500144, 0.168485606323044 and
0.218677546437235 Pa m. Parent shear RMS grows through 3.08905098547,
4.63393493401 and 6.17998896691 Pa. These reproduce the preceding clean-cycle
baseline. Current mean shear reaches 2000.66275555750 Pa at step 4; ordinary
publication changes it to 2000.88265819697 Pa. C suppresses this publication
change in the first tested update without deleting the approximately 500-Pa
physical increment.

## Stopping condition and limitations

Both B and C fail at step 2 with incompatible pressure component
**1.6047464209644725e-14**, above the existing roundoff/nonlinear bound
**8.71377001866816e-15**. This is the exact guarded condition, not evidence of a
large transfer error. We have not established whether changing this guard would
be justified. Its repair was explicitly excluded. Failed step 2 is not an
accepted state, and no result is inferred from it.

All accepted first real states have relative nonlinear residual
2.68246706589034e-10, one Newton update, 26 Krylov iterations, alpha=1; the
configured nonlinear target remains 1e-8. All surface rates are prescribed in
this fixture. A's remaining states converge with unchanged checks. B/C cannot
yet demonstrate absence of next-step re-equilibration, four-step accumulation,
or MPI publication invariance. The conditional two-rank C run was not launched.

One initial B setup attempt failed after initialization because the benchmark
called an unsupported MPI vector-min overload. The benchmark call was corrected;
the failed attempt is preserved in `native-setup-symbol-failure/`. This was not
a physical or solver alteration. Its wall time was 5.75 s.

The original particle hash was sampled before particle creation and recorded
`0 0`: **it is invalid evidence of particle identity**. The final plugin moves
that diagnostic to accepted timestep zero and rejects an empty population. This
diagnostic-only fix is compile-checked, not simulation-checked. No run was repeated
after the numerical stopping condition. Saved initial native fields, sorted by
physical coordinates, agree byte-for-byte; their SHA256 is recorded in
`comparison.json`. Identical PRMs and frozen-particle position assertions provide
additional evidence, but do not replace the missing original particle hash.

A used an earlier diagnostics-only plugin version without final-history binary
export/reverse-summation measurement. Its first three transferred fields are
independently recovered from the next accepted state's working history. No A
step-4 independent binary reconstruction is claimed. Exact per-run sources,
PRM, plugin and hashes are preserved beside each log.

## Implementation and reproducibility

New benchmark: `../moment_cycle.cc`, derived from `../clean_stress_cycle.cc`.
New parameter:

```
subsection Postprocess
  subsection Moment cycle
    set History mode = production
    # Alternatives: native_history_reference | horizontal_moment_update
  end
  subsection Clean stress cycle
    set Compact output = true
  end
end
```

Production defaults are unchanged. The default-empty
`PhaseFieldFault::benchmark_retained_stress(CellId, Point)` callback returns
**unrelaxed** retained stress. Only branch B installs it; plugin-owned cell-local
Gauss history is frozen during each solve and replaced after acceptance. All
consumers redirected are:

1. Bulk frozen Maxwell weak load, throughout the domain.
2. Native bulk-work surface constitutive evaluation, plus its optional diagnostic
   line evaluation.
3. Particle Maxwell candidate calculation, so B's shadow stresses use the same
   incoming tensor, although they never feed B mechanics.

The unsupported particle-domain surface measure explicitly rejects this hook.
No cancellation force is added, and the beta weighting is applied only by the
existing constitutive machinery. The callback is not checkpointed; this is a
fixed-mesh fresh-start diagnostic, not a replacement production architecture.

`MomentCycle` receives narrowly scoped friend access in the material model and
simulator to reuse the original Maxwell formula and particle-transfer method.
The transfer probe saves/restores the live solution, including on exception,
and applies physical constraints only to its private working copy. At real
accepted steps C replaces parent tensors with the full fitted accepted tensor;
it does not add old history twice. Ordinary accepted history updates otherwise
remain intact. This postprocessor publication is not qualified for restart,
advection, general geometry, or production failure-atomic history replacement.

ASPECT files touched for these hooks:

- `include/aspect/material_model/phase_field_fault.h`
- `include/aspect/simulator.h`
- `source/material_model/phase_field_fault.cc`
- `source/reconstructed_fault/surface_system.cc`
- `source/simulator/assemblers/reconstructed_fault_stokes.cc`

These files already contained unrelated working changes, which are preserved.
The benchmark also adds its CMake target, compact-output option, three complete
`moment_*.prm` files, runner, offline analyzer and mathematical tests. No changes
to pressure-compatibility logic or production tolerances were made.

From repository root, build with:

```sh
cmake --build build-pf-cpdi --target aspect -j4
cmake -S benchmarks/reconstructed_fault/bp5 -B benchmarks/reconstructed_fault/bp5/build -DAspect_DIR="$PWD/build-pf-cpdi"
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_moment_cycle -j4
```

The exact runner commands used, one branch at a time, were:

```sh
python3 benchmarks/reconstructed_fault/bp5/run_moment_cycle.py production --binary build-pf-cpdi/aspect-release
python3 benchmarks/reconstructed_fault/bp5/run_moment_cycle.py native_history_reference --binary build-pf-cpdi/aspect-release
python3 benchmarks/reconstructed_fault/bp5/run_moment_cycle.py horizontal_moment_update --binary build-pf-cpdi/aspect-release
```

The runner sets the required diagnostic environment, uses one rank and a hard
120-s cap, and refuses to overwrite existing directories. **These commands are
provenance, not a recommendation to rerun the blocked controls.** A future
reviewed continuation must use a new `--root`.

| Branch | Wall seconds | Reported child peak RSS (MiB) | Exit |
|---|---:|---:|---:|
| A | 8.645 | 286.72 | 0 |
| B | 7.994 | 287.35 | 1 |
| C | 6.740 | 287.17 | 1 |

The binary identifies revision `33228369d` with a dirty working tree; tested
binary SHA256 is
`e9535adf2384429ad4e3adb0cc7589b7a47a43efde003a6d21cfe5b951176cc3`.
Per-run plugin/source hashes are in `execution.json`. Current source differs
from those snapshots only by the final hash-timing correction and harmless
benchmark cleanup; do not replace the preserved snapshots with current files.

Offline verification commands and results:

```sh
python3 benchmarks/reconstructed_fault/bp5/test_moment_cycle.py             # 3 passed
python3 benchmarks/reconstructed_fault/bp5/test_clean_stress_cycle.py       # 1 passed
python3 benchmarks/reconstructed_fault/bp5/test_stress_cycle_analysis.py    # 2 passed
python3 benchmarks/reconstructed_fault/bp5/analyze_moment_cycle.py benchmarks/reconstructed_fault/bp5/moment-consistency
```

The analyzer passes independent-load, matched-coordinate, initial native-field
identity and signed-vector checks and writes `comparison.json`. No figure is
needed to distinguish the measured jumps. `summary.csv` retains each accepted
state; small probe CSVs retain representative tensor/moment values. Compact
binary fields preserve matched native locations without domain-wide CSV output.

**Next decision:** review the separately excluded tiny-RHS pressure-compatibility
stop before authorizing completion of B/C and MPI qualification. The first
publication establishes a promising horizontal-fixture result; it does not yet
establish a multi-step remedy, explain the full BP5 normal-stress bands, or
justify applying this symmetry-specific fit to an inclined fault.
