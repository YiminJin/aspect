# Boundary mechanism and longer finite-strip native experiment

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_BOUNDARY_REPORT.md) ·
[selection and budgets](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_BOUNDARY_SELECTION.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.

[Compact evidence/source archive](../cohesive_interface_boundary.zip) ·
[archive checksum](../cohesive_interface_boundary.zip.sha256) ·
[file manifest](manifest.json) · [changed files](changed-files.json).
The archive excludes raw solver directories and checkpoints, which remain separately
preserved. It is a reviewable evidence/source snapshot, not a standalone ASPECT build.

The diagnosis, reflected control, longer native run and focused half-step
comparison are complete. The refined result reaches 2.58 s with **3.2135 m**
left full advance; all declared temporal/work targets pass. This is a **finite-height periodic
fault specimen**, not an isolated distant-boundary fault. Surface damage is a;
bulk φ remains provisional geometry. No production/BP3 path or law changes.

## Primary figures and comparisons

- `boundary-mechanism.png`: long/mean modes, period-change ramp at identical
  memory, and wall-distance/centred reference controls at equal background stress.
- `native-reflection.png`: actual native pressure, damage and slip; the reflected
  geometry swaps the developed flank without prescribing friction pressure.
- `longer-histories.png`, `longer-profiles.png`: common-time baseline/half-step
  evolution, fully decohered advance versus process-zone growth and signed slip.
- `longer-pressure.png`: actual native Gauss6-weighted bulk normal-stress maps.
- `specimen.png`, `longer-energy-work.png`: physical boundaries/loading and
  separate recoverable storage, fracture inventory and physical dissipation.
- `longer-space-time.png`, `longer-evolution.gif`: accepted damage and slip-rate
  fields/profiles; no interpolation of constitutive histories between states.
- `temporal-comparison.csv`, `temporal-fields.csv`, `continuation-verification.json`:
  absolute/relative common-time errors, full tensor/pressure differences, raw
  energy screens and qualification targets.
- `summary.csv`: eight common-time baseline/half-step records for a compact
  comparison; `cost-summary.json` lists measured native wall/setup costs and RSS.
- `longer-history.csv`, `event-half-history.csv`, `selected-profiles.csv`: compact
  actual histories and common-time Gauss profiles. Names contain units.
- `longer-physical-cells.csv`, `animation-profiles.csv`: precisely the weighted
  bulk cell means and accepted Gauss frames used by the maps/GIF.
- `fronts-versus-loading.csv`, `front-secants.csv`: full fronts against prescribed
  displacement and model-time kinematics. These do not qualify physical rupture
  speed; the bulk has no inertia.

## Boundary mechanism and independent references

- `same-memory-period.csv`, `same-memory-period-profiles.csv`: native period
  differences at identical m, their analytic long-wave prediction and residuals.
- `operator-evolution-decomposition.csv`: exact separation of operator change
  from the later changed memory. Fixed-state friction pressure terms use the
  unchanged reference a, V and Θ and are diagnostic partial variations only.
- `operator-evolution-regions.csv`: the same quantities tested with common Q1
  weak-load functions: one at interior patch nodes, zero at endpoints/outside.
  Tests are [0.5,2.5], [5.5,7.5], [2.5,5.5] m with integral 1.9375/1.9375/2.9375 m.
  Full weak equations include tied reactions; these are not assertions of a free
  reduced direction across intact ties.
- `boundary-low-modes.csv`, `native-analytical-modes.csv`, `boundary-reference.json`:
  zero/long modes, independent 50-digit limits, Q1 alias tails and native symbols.
- `frozen-boundaries.csv`: identical Q1 memory and background Maxwell stress for
  several reference geometries. Changed H scales boundary displacement with H;
  histories are not evolved into these hypothetical new constitutive states.
- `analytical-L16-symbols.csv`, `frozen-mechanical-errors.csv`,
  `frozen-mechanical-profiles.csv`: exact Q1 Galerkin strip versus native bulk FE
  mechanics at the same actual final memory. This separates mechanical error
  from surface-damage spatial error; it is not full spatial convergence.
- `reflection-verification.json`: native proper-rotation identities, unchanged
  history comparison allowances and independent full-bulk stress/pressure checks.

`artifact-verification.json` verifies readable images/GIF, finite CSV data and
declared checks. `preservation.json` rechecks protected production/BP3 source
and prior evidence hashes. `source-entry.patch` compares against the preserved
task-entry source, rather than misattributing earlier local work to this task.

## Ownership, inputs and computational limits

`*-run.json` records wall times, observed RSS, parameter/DSO hashes and stops.
`inputs/` contains each actual parameter file, experimental environment, validation
log and native stdout. `native-checks/` retains operator, derivative, contact,
rejection/publication and restart checks; `trajectory-provenance.json` defines
the accepted-branch merge and lists separately retained checkpoint files.
`longer-config-mismatch` preserves the pre-advancement `16` versus `16.0` string
failure. `longer` preserves the 1500 s wall stop (accepted 2.57625 s).
`longer-completion` continues from valid 2.574375 s, requiring bitwise overlap.
`post-hold-extension-rejected` verifies that extending a load plan after its
accepted hold is rejected before publication. `longer-verification.json` records
the unchanged past prefix, endpoint roundoff and completion overlap.

The links to native run directories contain the separate full solver records:
K/N matrices, all six Gauss histories, nodal memory, physical bulk samples, raw
work/check files and native rolling checkpoints. They are not duplicated in the
compact evidence package. Durable pre-event/pre-hold/hold checkpoints are under
`build-particle-free/cohesive-interface-boundary/preserved/`.

At restart, the host's already-pending first interval remains unchanged. Subsequent
replay intervals are halved. Logical histories retain the latest checkpoint
branch and exclude overwritten overlap from sums; raw files remain intact.
All publication counters must advance once per accepted state. Timestep/source/
weak/derivative/energy/contact tolerances are unchanged.

Commands, sequential native runs:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_boundary.py
cmake -S benchmarks/reconstructed_fault/particle_free \
  -B build-particle-free/cohesive-interface-boundary/build \
  -DAspect_DIR="$PWD/build-particle-free/native"
cmake --build build-particle-free/cohesive-interface-boundary/build \
  --target cohesive_interface_native -j2
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_boundary.py reflection
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_cohesive_interface_boundary.py reflection
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_boundary.py longer
# Only after the recorded administrative wall stop:
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_boundary.py complete
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_cohesive_interface_boundary.py longer
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_boundary.py reject-past-loading
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_boundary.py replay
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_cohesive_interface_boundary.py compare
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_boundary.py reference
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_boundary.py reflection
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_boundary.py final
# Validate/package completed results only; this launches no solver:
python3 benchmarks/reconstructed_fault/particle_free/package_cohesive_interface_boundary.py
```

Wrappers validate inputs and launch ASPECT; C++ computes all accepted physical
states. Existing outputs are protected against accidental overwrite. All physical
energies are J/m of thickness, Gc is J/m², traction is Pa, jump is m and Θ is
dimensionless. Bulk normal maps show p0+p′−τyy on horizontal planes. Tensor norms
use xx²+yy²+zz²+2xy² and actual quadrature weights. Scalar energy/field differences
use the declared reference as a normalization scale, not as an accuracy oracle.

Preserve the historical 65.4% early spatial failure, endpoint/branch limitations,
old domain-sensitivity failures and coarse-timestep energy-screen failure. No
new full spatial qualification or physical rupture-speed claim follows here.
