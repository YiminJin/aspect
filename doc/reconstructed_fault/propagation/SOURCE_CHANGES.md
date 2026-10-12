# Cumulative propagation changes

## Latest P2 accuracy and energy assessment

**No production source functions changed.** Direct irreversibility, recoverable
H, unloaded initialization, surface Z, bulk-conjugate mechanics, consistent I
and generic stress projection retain their incoming hashes. The governing energy
question is derived but not implemented. Added `accuracy_normalization.py`,
`accuracy_stress.cc` (`MomentCycle::pattern/execute`, existing native friend),
`run_accuracy_stress.py`, `analyze_accuracy_stress.py`, and
`accuracy_profile_energy.py`; one optional CMake plugin target. These isolate
frozen surface accuracy, native stress update/transfer drift and energy variations.
New report/derivation and status/design/spec/index updates record failed accuracy
and unresolved energy gates. See [report](P2_accuracy_energy_report.md) for exact
functions, commands, source identities and preserved incoming evidence.


## Latest direct P2 irreversibility revision

[P2_irreversibility_report.md](P2_irreversibility_report.md) lists exact functions.
Only phase_field.h/phase_field.cc production changes add the optional accepted
phase lower bound, correct contact KKT/projection and immutable solve snapshot,
with guards against equality mode and failed-step continuation. Default false,
initial relaxation, H and all work/normalization operators are retained.
The test-access forwarding, observer KKT, replay plugin, fresh runner, analysis
and documentation are separate diagnostic/documentation changes. Actual diffs
are in `irreversibility_evidence/`; incoming files are preserved in the verified
3,154-artifact snapshot. All other incoming production changes are unchanged.


## Latest P2 coupling repair

The [task report](P2_coupling_report.md) lists exact production functions and
separates the new history/work representations from test and documentation work.
Material parser/header/history/constitutive files add optional Q1 Z publication
and consistent point q=Z/I; surface_system.cc dispatches a distinct cohesive work
path, and surface_system_bulk_work.cc supplies its shear adjoint and ds resistance
with matching linearization. Generic projection, CPDI, H and mature defaults are
unchanged. `coupling_evidence/production-revision.diff` is relative to the incoming
recoverable stage; `specification.diff` is the actual governing-document diff.
Native history/tangent/MPI/default checks pass; three evolving trajectories and
identical-H replays retain resolved tail healing, so feedback qualification fails.

## Latest recoverable-energy revision

[P2_recoverable_report.md](P2_recoverable_report.md) lists the exact changes relative
to the preserved history-diagnosis state. Production adds an optional pure
`compute_recoverable_phase_driving_force`, selection in `compute_history_candidates`,
an unloaded branch in `evaluate_initial_cohesive_particle_values`, parser settings
and restart provenance markers in `initialize`, with declarations in the material
header. Legacy defaults and histories are unchanged. This is an explicit governing
revision; no generic projection or mechanical assembly is modified. Native local
checks pass, but three coupled trajectories reveal further transfer/work issues.
New benchmark observers/configurations/runner/checker and documentation are separate
from these production changes. No stress-scaling or P3 qualification is claimed.


## Latest history diagnosis

No production source function or executable changed. Added
`history_driving.cc::HistoryDriving::execute` (native fixed-phase traction paths),
`history_driving.prm`, its CMake plugin target, and `analyze_history.py` (analytic
checks, unchanged coupled repeats, comparisons/plots). Added test-only
`PhaseFieldFaultTestAccess::maxwell_coefficients`, a pure native wrapper. Updated
status/design documentation without changing governing equations. All 755 prior
pinned artifacts were preserved first. See [P2_history_report.md](P2_history_report.md)
for exact audited functions, the unresolved physical decision and commands.


Baseline: `de7beaa65846c7372711f7922abf1c5d9a745ee6`, current
`fault-propagation` worktree. This inventory is based on actual tracked diffs
and untracked files, not a list of intended changes. Nothing is committed or
pushed. Incoming work is retained; the source/evidence snapshots distinguish
successive experiments. See the linked reports for numerical limits.

## Latest bounded material study

No production source or core executable changed in the material comparison.
`material_feedback.cc`, `material_feedback.prm`, `straight_core06.txt`,
`run_material.py`, `analyze_material.py` and the benchmark CMake target add
fresh material initialization, read-only native calibration/history probes,
normalization checks, local slip curves and plots. The shared
`coupled_feedback.cc` exposes protected observer reuse and reads prescribed
core instead of two diagnostic 0.8 constants. Production initialization already
used actual core/Gc. Documentation and the new material evidence preserve the
previous high-load failures. See [P2_material_report.md](P2_material_report.md)
for checks and actual scientific limits; the cumulative production changes
below are from earlier selected stages.

## Production code

| File and functions | Problem and resulting behavior | Setting / default | Evidence |
|---|---|---|---|
| `source/simulator/phase_field.cc`: `declare_parameters`, `parse_parameters`, `initialize_phase_guess`, `solve_bound_constrained_phase`, `constrain_phase_newton_system`, `assemble_phase_field_system`, `evolve_phase_field`; `include/aspect/phase_field.h` (`uses_bound_constraints`, solver options/signatures) | Equality solve violated phase bounds. Optional projected Newton reuses native CPDI assembly/CG; explicit trial state drives residual/Jacobian/energy. Fresh rounded guess relaxes at existing H; later solves start from accepted old_solution, with scratch until acceptance. | `Use bound constraints=false`; opt-in box [0,1], no previous-phase lower bound. | P1 integration: equilibrium/bounds/contact release, remote elimination, scratch poisoning, failure restoration and identical-H repeat. Equality failures retained. |
| `source/simulator/solver_schemes.cc`: `solve_single_advection_no_stokes`, `solve_single_advection_single_stokes`, `solve_single_advection_iterated_newton_stokes` | Generic predictor phase was unsuitable for downstream consumers. After acceptance copy only phase into current_linearization_point; preserve native old/older history rotation. Phase-only path prepares dependent data. | None. | P1 publication/poisoning, distinct-history checkpoint and exact resumed holds/ramps. |
| `source/material_model/phase_field_fault/history.cc`: `prepare_reconstructed_fault_phase_state`; declaration in material header | Initialize q/previous I_h after relaxed accepted phi, or recompute transient current I_h subsequently. Never regenerate H from relaxed phi. | None. | Bitwise native-H invariance; normalization consistency; initialization-failure checks. |
| `source/reconstructed_fault/manager.cc`: parameter declaration/parsing, `reconstruct_initial_faults`; manager header | Narrow profile support omitted active phase tails. Explicit verification support derives full normal width from actual Box/fault, builds particle/Stokes associations before cohesive data; existing serialized widths remain fixed. | `Association support=stationary profile` unchanged; optional `full normal cross-section` restricted to fixed single straight through-going 2D Box fault. | Final straight serial/two-rank/restart controls: zero omitted source, identical H/phase; padded control unchanged. |
| `source/simulator/solver/reconstructed_fault_stokes.cc`: `solve_reconstructed_fault_condensed_system`, its fresh continuity residual compatibility check and `pressure_assembly_scale` | Zero base velocity gave zero arithmetic allowance despite nonzero Newton correction. Include actual correction assembly scale in the existing roundoff bound, preserving convergence tolerance/cap and RHS guard. | None. | Stage-I: 20,141 assertions / 15 cases, including zero-base/nonzero-correction and genuine incompatible-flux rejection; coupled mechanical entry succeeds. |
| `source/material_model/phase_field_fault.cc`: `declare_parameters`, `parse_parameters`; `include/aspect/material_model/phase_field_fault.h` | Select the I_h-only representation, fixed during a run. | New `I h projection=consistent` default; optional `row sum lumped`. | Default straight and mature compatibility pass; lumped straight/padded startup and holds/ramps pass. |
| `source/material_model/phase_field_fault/normalization.cc`: `project_normalization_integrals_to_fault` | Positive normal integrals can give negative consistent coefficients. Optional b_i / row_sum(M)_i uses the same assembled arclength system and distributed reduction; validates Q1 shapes/weights and every denominator. | Uses the new setting above; no shared matrix or generic projection changes. | Native frozen 0.1/0.05 m projection agrees with independent calculation within 3.03e-14 scaled error; constant/conservation, MPI, cache restoration pass. Coarse P2 spatial error remains 35–66%; not qualified for that field. |
| `source/material_model/phase_field_fault/history.cc`: `initialize` and post-resume callback | Prevent a current I_h representation from being mixed with stored previous I_h from another representation. Optional marker in existing serialized property registry; default layout unchanged; native cache invalidation retained. Initial and terminal previous I_h use the selected current normalization through unchanged native publication. | No separate setting. Marker presence is provenance, not a physical field. | Same-representation changed-history restart passes serial/two ranks; both mismatch directions fail explicitly; copied pre-change consistent checkpoint loads. No new coupled commit/rollback qualification. |

The mechanical candidate/max-history law, RSF/friction equations, particle
transport law, generic particle projection and generic extrapolation coefficients
have not changed. The existing frozen-RK2 switch is a named test restriction.

## Test and diagnostic code

`tests/phase_field_test_access.h` exposes native phase assembly, bounds and
scratch solve; `tests/phase_field_fault_test_access.h` exposes native material
normalization and Maxwell coefficients. `unit_tests/reconstructed_fault.cc`
adds the pressure arithmetic regression. Untracked `tests/phase_field_controlled_history.*`,
its expected-output directory and checker implement the original controlled-H
probe. `benchmarks/reconstructed_fault/propagation/` contains the later 1D
reference, box prototype, production-integration and coupled observers, PRMs,
runners/checkers, and retained evidence. Test-prescribed H belongs only to P1;
the P2 observer reads mechanically produced H and verifies trial nonpublication.
The normalization follow-up adds frozen nodal capture without modifying H,
`normalization_frozen.py` for analytic-Q1 transverse integration and both surface
representations, `check_normalization.py` for explicit numerical/lifecycle gates,
and `run_normalization_compat.py` for two existing fixtures. The P1 observer
checks distributed constants/conservation and native projection of frozen samples;
the runner exposes `--lumped` and the isolated `--spacing` repeat.

## Documentation and evidence

`AGENTS.md` retains the committed instructions and appends successive user scope
selections. `current_design.md` and `specification.tex` record reviewed production
extensions. The benchmark index links propagation work; the propagation docs
contain PLAN, STATUS and chronological reports, including failed equality,
failed support and failed coupled evidence. `COUPLED.md` now gives the complete
current model and distinguishes the subcritical control from revised loading.
CSV/log/JSON artifacts and manifests under the untracked benchmark directory
are evidence, not production code. Existing manifests remain immutable; the
normalization baseline snapshot preserves all 273 prior pinned artifacts.

The machine-readable inventory alongside this report lists every tracked and
untracked path present at this audit, including evidence. The normalization follow-up is included; the associated manifest pins exact
source/build/evidence identities.
