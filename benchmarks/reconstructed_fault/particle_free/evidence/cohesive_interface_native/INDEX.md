# Native fixed-Γ evidence index

Read [the report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_NATIVE_REPORT.md)
and [discrete formulation](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_NATIVE_FORMULATION.md)
for scope and definitions. This is an isolated native component experiment;
production/BP3 and the old checkpoint owners are not replaced.

| Evidence | Meaning |
|---|---|
| `native_response.png` | Intact onset, mature velocity change/hold/reversal, nonuniform solved normal pressure |
| `representation_error.png` | Native FE resolution versus sharp Q1 mixed-layer response; fixed and half width |
| `stability_and_work.png` | Corrected local curvature and physical-work remainders, separately from BE closure |
| `frozen_native_fields.png` | Physical shear/pressure on an independent frozen nonuniform probe, not an evolved specimen |
| `summary.json`, `native_reference_histories.csv` | Independent returns, energy errors, restart comparisons |
| `operator_comparison.csv` | Same-space sharp Fourier reference and finite-width differences |
| `runs.json` | Inputs, exit codes and sequential wall times; 180 s per-run bound |
| `provenance.json` | Protected/source/binary hashes and environment |
| `restart_negative.json` | Changed-width checkpoint rejected before publication |
| `development/` | Original unstable nodal mode and oversized near-zero FD increments; retained failures |

Each case directory contains `operators.csv` (Pa/m), `operator_checks.csv`,
`operator_arithmetic.csv`, and independently evaluated `field_checks.csv`.
`frozen_fields.csv` samples one actual Gauss6 point per cell for visualization;
JxW is that **point weight**, not a whole-cell volume. Checks integrate all 36
points per cell. The trace includes zz; physical stress is distinct from memory.

`history.csv` records accepted steps, wall displacement, energy and work per
unit out-of-plane thickness (J/m), force residuals (Pa), publication counts and
incremental curvature (Pa/m). Fracture inventory includes initial damage; new
fracture, recoverable bond storage, Maxwell loss, friction, radiation and signed
numerical remainders are separate. `states.csv` uses the historical column name
`node` for the **Gauss-record index**; physical coordinates are in metres.
`nodal_history.csv` instead contains true periodic Q1 nodes, signed velocity and
reduced bulk-memory coefficients. All times are seconds; Theta and damage are
dimensionless. `lifecycle.csv` records native owner tests, not an assertion-based
substitute for physical response.

Restart is a branch: join the split prefix through `restart_loaded.csv` with
`*-resume.csv`, not with the uncheckpointed tail or duplicate output. The latter
is retained separately. A final roundoff-sized mature step is retained.

Full simulator checkpoints, generated parameter files/logs, and the preserved
failed-source archive `nodal-quadrature-failed.zip` remain under
`build-particle-free/cohesive-interface-native/`. The compact package includes
inputs, final source, analysis dependencies and numerical evidence; it excludes
host executables, solver archives and duplicate image formats.
