# Active-path cleanup record

The subsequent user-selected consolidation is implemented; see
[TIP_CONSOLIDATION_REPORT.md](TIP_CONSOLIDATION_REPORT.md). Native phase is
Bernstein-only, Q1/CPDI and the replicated endpoint solvers are retired, and
stress transport uses owned rows on all ranks. The table below preserves the
**pre-deletion caller audit**, not a list of interfaces that remain active.
Independent serial transport and direct bulk-inverse references now belong to
benchmark/unit-test support. Remaining historical material/history algebra is
retained only where shared callers or checkpoint interpretation still require it.

| Candidate | Actual callers and retained evidence | Proposed boundary |
|---|---|---|
| Separate Q1 phase and CPDI assembly in `source/simulator/phase_field.cc` (`initialize`, assembly and setup), legacy block construction in `source/simulator/introspection.cc`, particle-domain graph in `source/simulator/core.cc` | Historical particle/CPDI controls, retained FE-history endpoint path (`reconstructed_fault_endpoint_phase.cc`); native composition route explicitly rejects CPDI fallback | Remove only after an explicit retirement decision for these physical routes; archive their executable/input identities first. Do not remove general FE_Q, MappingQ1 or ordinary ASPECT pressure/velocity infrastructure. |
| Native nodal DGQ2 compatibility in `PhaseField::Q2Bounds`, `BulkTipCoupling::transform_phase` and constructor matrix conversion | `--phase-basis nodal` controls; `analyze_native_bernstein.py`; the explicit `MomentCycle::import_parent` converter interprets the old nodal material archive | Native production can specialize after this task. Keep benchmark conversion with evidence readers; old checkpoints must remain identified/rejected, not silently reinterpreted. General FE_DGQ remains needed by ordinary fields and tensor histories. |
| Replicated endpoint algebra in `source/simulator/solver/bulk_cohesive_phase.cc` (`Simulator::solve_bulk_cohesive_phase`) and historical `reconstructed_fault_endpoint_phase.cc` | First is still selected by nonintegrated bulk cohesive modes and `Bulk tip solver = benchmark direct`; the latter owns a distinct interface formulation. `bulk_tip/checks.cc::compare_linear_oracle` is a separate small independent verification oracle | Separate the benchmark-only tip oracle from active formulation code before considering deletion. The distinct cohesive-interface physics is not obsolete merely because integrated bulk-tip uses another solver. Retain direct-reference tangent evidence. |
| Serial tensor advection in `source/material_model/history/tensor_advection.cc` | `PhaseFieldFault::make_dg_transport`, `project_dg_transport_field`, `advance_dg_transport` in `material_history.cc`; production `stress_evolution.cc` and `dg_accepted.cc` call these wrappers. `assemble/advance` dispatch to distributed methods when selected. Reference transport/native/limiter checkers also call the wrappers | Trace selector coverage before moving serial implementations into benchmark support. Projection is also used by independent initial/reference field construction. Keep distributed mature SSPRK3 and physical inflow semantics; do not delete the dispatch interface while production callers still use it. |
| Experimental limiter support under `benchmarks/.../support/serial_dg_limiter.cc` | `limiter_checks.cc`, `stabilization_checks.cc`, transported native reference checks; it is already benchmark support | Delete only genuinely unreferenced helpers; preservation of historical evidence does not require every experiment to remain an active interface indefinitely. |

Use `rg` on these symbols and selectors again at cleanup entry. Caller status is
source-dependent, not a promise that every legacy path must remain executable.
Retained reports/manifests are evidence; preserving their interpretation is
separate from retaining production dependencies. Inherited losses remain 39/45
pilot frames, early temporary logs and the unavailable old `bulk_continuation.zip`.
