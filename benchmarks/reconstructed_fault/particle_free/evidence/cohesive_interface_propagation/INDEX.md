# Integrated propagation evidence

See [the concise scientific report](REPORT.md). Reproduce with the maintained
`benchmarks/reconstructed_fault/particle_free/cohesive_interface_propagation.py`.
Native build targets are `cohesive_interface_path` and
`cohesive_interface_turn_solver` in the existing benchmark CMake project.
`demo` reuses verified caches, runs qualification before evolution, resumes
atomic accepted checkpoints, then performs checks/reporting/verification.
`run --resume --ranks 2` resumes only an unfinished main trajectory;
`report` regenerates all figures and tables without advancing any history.
Keep the original formed-turn source checkpoints and their native caches.

| Evidence | Meaning |
|---|---|
| [specimen.png](specimen.png) | Physical walls, loading history and actual fault |
| [propagation.png](propagation.png), [propagation.gif](propagation.gif) | Actual damage and signed slip-rate vectors, connected fronts, temporary tail |
| [fronts.png](fronts.png), [mature-events.csv](mature-events.csv) | Developed front entry and physical activation brackets |
| [mature-material.png](mature-material.png), [main-tracked-material.csv](main-tracked-material.csv) | Same material identities through decohesion, RSF sliding and state evolution |
| [energy.png](energy.png), [energy-windows.csv](energy-windows.csv) | Endpoint/physical work, dissipation, storage and separate numerical remainders |
| [uncertainty.png](uncertainty.png), [focused-uncertainty.csv](focused-uncertainty.csv) | Focused timestep and future-material spacing controls |
| [pressure-impact.json](pressure-impact.json), [pressure-impact-profiles.csv](pressure-impact-profiles.csv) | Fitted common-field normal operator and full-current-pressure response sensitivity |
| [parallel-qualification.json](parallel-qualification.json), [parallel-cost.csv](parallel-cost.csv) | Native 1/2/4-rank correctness and measured local scaling |
| [native-export-costs.csv](native-export-costs.csv) | Setup/application phases and per-rank peak RSS for completed kernels |
| [route-check.json](route-check.json), [mature-lifecycle.json](mature-lifecycle.json), [publication-accounting.csv](publication-accounting.csv) | Actual transaction/retry/restart and once-only advancement |
| [results.json](results.json), [verification.json](verification.json), [manifest.json](manifest.json) | Compact results, separate scientific/engineering gates and provenance |

`main-history.csv` has every accepted positive step; `*-profiles.csv` samples
every fourth accepted step plus events. `*-tracked-material.csv` retains every
accepted step for the first 0.1875 m of late-turn material. `*-material-ledger.csv`
records persistent IDs, admission, growth/full brackets and final histories.
All coordinates are physical metres, time seconds, pressure/stress Pa, signed
jump metres, velocity m/s; energies J/m (unit out-of-plane thickness).
`theta` is normalized age; physical age is theta*Dc/V0. Cohort averages use
actual surface Gauss weights (metres), not point counts. `intact_tied` marks
zero-damage constraints: their stored zero cohesive/friction return is not a
pointwise intact traction. Fronts are connected Gauss-record thresholds
a=.05, .5, exactly 1; lower/upper spatial and temporal brackets are retained.
No plotting interpolation creates a constitutive state. Cell-centred bulk
stress and raw sided tractions are not thereby qualified as point stresses.

Complete solver caches, all accepted/checkpoint journals, failed exports,
executed source versions and logs remain separately in
`build-particle-free/cohesive-interface-propagation`. No production checkpoints
are converted. `parallel-v1/` and `parallel-v2/` retain superseded performance
measurements; the final parallel result must not be represented as MPI speedup.
Compact initial/final accepted states are also retained here. Their kernel
paths refer to the preserved native caches; those caches are required to
restart. `execution-provenance.json` identifies all executed source versions.
