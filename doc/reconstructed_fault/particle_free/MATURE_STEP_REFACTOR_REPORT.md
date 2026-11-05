> **Historical, superseded architecture:** these results describe the incoming tested implementation. The later user-selected [ASPECT refactor](ASPECT_ARCHITECTURE_REPORT.md) removes its dedicated driver and supporting executor. Retain this report as evidence; use [the current guide](MATURE_MOVING_FAULT.md) for architecture.

# Mature moving-step refactor — 4 November 2026

Selected scope: executable mature driver, explicit accepted/candidate/cache
meanings, explicit projection/gradient/profile dependencies, common physical
stress law and one current reading guide. Baseline is `a22214d57f3ce809b3b20c9883b853c96e04033f`
on `pf-rsf-particle-free` in the original worktree. No tracked incoming changes;
untracked historical/manual files remain. The prior direct/publication baseline
`e4b74c556` is retained. No remote fetch/push or production run is selected.

## Implementation

The [current reading guide](MATURE_MOVING_FAULT.md) is the authoritative
code-level description, linked from current design/specification and maps.
Historical scientific reports are retained as evidence.

- `mature_material_step.cc` exposes the complete mature attempt and final
  publication/rollback callback; its projected outer solve calls the unchanged
  coupled inner solver. Generic advection/mechanics dispatch remains shared in
  `solve_timestep_equations`. Other formulations retain their existing wrapper.
- `mature_material_internal.h` separates the retained accepted parent, staged
  geometry/profile candidate, transport workspace and projection-correction
  cache. Nonowning const views supply accepted velocity, interval and staged
  profile/cells. Mature physical stress and separated trial/total-memory scratch
  have explicit names. Cohesive physical histories are not merged.
- Material transport, geometric/profile staging, rotation, projection and
  transaction bookkeeping now have separate numerical entry points. Projection
  receives a field evaluator. Profile transport and rotation receive the same
  accepted velocity explicitly. Shared raw tensor evaluation is distinct from
  physical stress reconstruction.
- `mature_stress_endpoint` expresses the physical law without T/R/B bookkeeping.
  `physical_history_stress` observes either storage with an explicit state;
  the compatibility entry supplies its currently bound state. Reaction projection
  receives the complete field evaluator and state. Comparing correction caches
  no longer temporarily rebinds the model's active correction pointer.
- `dg_accepted.cc` and the serialized schema/representation meanings are unchanged.
  No new state is serialized, no old history is converted. Separated storage,
  Q1 remap/zero inflow and all other defaults remain. Spin lag, quadrature,
  reaction coefficients, inner/outer tolerances and final slip guess remain.

## Declared bounded verification

Six current core/checker/support binaries were copied before rebuild. Entry
source hashes match the preceding tested consolidation. Evidence lives in
`benchmarks/reconstructed_fault/particle_free/evidence/mature_step_refactor/`.
The controls are the existing `consolidate-after-{dc,sc,dl,sl}` runs and default,
rejection and restart controls. Replays use the identical saved inputs and at
most four MPI ranks, sequentially. No experiment or physical parameter sweep.

Screens fixed before the refactor: every matching accepted exported bulk value
within 5e-13 absolute and surface value within 5e-12; compare reconstructed
physical tensor/work traction separately, plus publication counts and mechanical
iteration sequences. Retain exact failures if a screen does not pass.
The full existing unit suite and rebuilt matching DSOs are required. The native
source check now also supplies an independent profile/gradient view and verifies
an analytically constant projected tensor within 3e-12, without changing the
bound accepted state. This catches hidden reads of model-global fields.

## Measured results

Core and all four matching native checker libraries build. Eight affected
translation units independently compile without unity/PCH. The existing unit
suite passes **183170 assertions in 152 cases** (7.93 s). Repository AStyle 2.04
was applied to the extracted/touched material kernels and driver, without an
unrelated repository formatting pass. No numerical criterion was relaxed. The first build overlapped final edits; only the
frozen-source build and final consistency rebuild supplied the tested binary.

All seven sequential native runs pass. They use the retained 16² inputs:
constant/layered viscosity, direct/separated storage, default Q1/zero-inflow,
completed-step rejection, and continued restart. All seven have identical
linear-solver and projection-outer iteration sequences to their respective controls; accepted clocks
and geometry/history publication counts match. The direct constant run also
exercises the existing affine/source/rotation checks. The new explicit
profile/gradient check measures **1.96081e-15**, below its unchanged declared
3e-12 bound. It leaves the bound accepted state intact.

| Comparison to preserved consolidation run | Max bulk export difference | Max surface export difference | Physical tensor L2 | Work-traction max difference |
|---|---:|---:|---:|---:|
| Constant direct | 8.89e-16 | 5.43e-20 | 8.08e-17 | 8.18e-18 |
| Constant separated | 1.09e-16 | 5.43e-20 | 1.77e-17 | 1.09e-19 |
| Layered direct | 1.78e-15 | 1.63e-19 | 6.88e-16 | 5.75e-17 |
| Layered separated | 6.63e-18 | 0 | 1.08e-18 | 5.43e-20 |
| Default | 3.38e-15 | 1.74e-18 | 2.14e-15 | 2.36e-16 |
| Rejection | 1.34e-15 | 8.14e-20 | 6.76e-16 | 3.99e-17 |
| Restart on four ranks | 6.80e-17 | 5.43e-20 | 1.87e-17 | 1.55e-18 |

Rejected line-search candidate counts also match. An initial exact log-text
comparison flagged two starting step lengths; their maximum numerical difference
is 1.44329e-15. `solver-text-comparison.json` preserves that failed text comparison;
`solver-decisions.json` records the numerical values and equal discrete branch
counts for all seven runs. No solver behavior or acceptance criterion was changed
to obtain this result.

These are roundoff-level, not universally bitwise, comparisons. Bulk/surface
export maxima combine columns with different units and are regression screens;
physical stress and work traction are reconstructed separately. Every declared
5e-13/5e-12 screen passes. Native wall times in table order were
18.60, 11.09, 23.66, 18.15, 15.51, 22.77 and 10.79 s. The cases have different
lengths/diagnostics; these single timings do not establish a speedup.

The four-rank restart continues the preserved **local test** checkpoint at .1 s
through .15/.2 s. It also agrees with this refactor's uninterrupted layered-direct
run within 2.47e-15 bulk / 1.07e-14 surface. The forced completed-step rejection
agrees with the retained matched accepted cutback schedule within 1.58e-15 /
1.25e-14. See `lifecycle-crosschecks.json`. These checks exercise the changed
ownership/publication boundary; they are not production checkpoint conversion.
Escaping-exception and recoverable-no-candidate branches retain their original
restoration/invalidation/observer distinctions by source extraction; this task
did not add separate runtime fault injection for each exception location.

All **45 protected files** match their saved hashes: the prior 39 plus six entry
copies. `dg_accepted.cc` serialization is unchanged. `final-provenance.json`
identifies actual source and binaries; the executable's printed `a22214d57`
revision alone does not identify this dirty-source build. Exact inputs, command
records, compact counters and the analytical dependency check are retained.
Historical/manual untracked files, outputs and checkpoints were preserved.

## Qualification boundary and next action

Implemented and locally verified: the selected mature driver/state/interface
refactor and common physical law, including direct/separated, constant/layered,
default and lifecycle controls. Numerical behavior is preserved within the
measured roundoff bounds. Physical accuracy and production readiness are not
newly qualified. Finite-end approximation, endpoint energy/flux errors, cohesive
energy/tip failures, moving AMR/birth and broader Stage C remain open. Production
BP3 remains distinct from cohesive-interface development. The inherited loss
of 39/45 native-pilot frames and temporary logs remains unrepaired and recorded.

The next justified scientific experiment, only if selected, remains the frozen
endpoint-aware tensor-space comparison with matching work integration. The
refactored driver and physical-stress interface now provide its reviewable
implementation boundary. No scientific campaign was resumed here.
