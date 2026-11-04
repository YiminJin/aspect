# Narrowed Stage C evidence

The current interpretation is in
[the coupled report](../../../../../../doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md)
and [cohesive diagnosis](../../../../../../doc/reconstructed_fault/particle_free/STAGE_C_COHESIVE_DIAGNOSIS.md).
`final-provenance.json` records the verified worktree/branch/HEAD, exact dirty
source, builds, checker libraries and every new run's command/input/hash/exit.
`current-source.tar.xz` is a verified source recovery payload, not a checkout.
Entry patches and source hashes retain the incoming work.

- `material-comparison.json/png`: independent exported-Q2 characteristic,
  geometric localization, curve and surface reaction comparisons; finite errors
  are measured rather than called exact. Terminal ULP interval is retained in
  JSON and excluded from physical-time plotting.
- `material-sensitivity.json`: separate final-state and one-step differences,
  spatial/temporal controls and local cost; interior temporal error is not
  monotone, so there is no completed temporal qualification.
- `lifecycle-comparison.json`: complete export comparisons for rejection and
  two-to-four-rank restart. Matched rejection is byte-identical. Heterogeneous
  final control differs by a 1.4e-17 s timestep rounding and passes the same
  predeclared per-column roundoff comparison; its failed byte screen is explicit.
- `cohesive-energy-decomposition.json/csv/png`: existing 8/16/32-step endpoint
  energy outputs, independent accounting identity, signed/absolute sums, dt
  scaling and cost. All fracture-relative physical screens still fail.
- `tip-diagnosis.json` and signed trace CSVs: unchanged legacy/objective/half-dt
  localized loading; first failure is force-residual merit decrease, before
  admission. Diagnostic repetitions remain failed evidence.
- `preservation.json`: original protected controls/checkpoints, 220 supplied
  server exports, 328 cleanup-critical files, pre-C and current-entry controls.

Reproduce reductions from the worktree root with the scripts in
`benchmarks/reconstructed_fault/particle_free/transported_objective/`:
`assess_material_native.py` (case names recorded in material-comparison.json),
`assess_material_sensitivity.py`, `assess_material_lifecycle.py`,
`diagnose_stage_c_energy.py`, `diagnose_stage_c_tip.py`, and finally
`record_coupled_provenance.py`. Reductions launch no simulation. Native run
inputs/logs/checkpoints remain under
`build-particle-free/transported-objective/stage-c/material-runs/`; failed runs
are not removed or relabelled. New local runners refuse existing labels.

No production checkpoint was resumed or converted. The original BP3 first
event remains incomplete; the earlier loss of39/45 animation frames and /tmp
build logs remains documented.
