# Native coupled evolution and straight propagation evidence

Start with [the report](COHESIVE_INTERFACE_EVOLUTION_REPORT.md) and the
[selection](COHESIVE_INTERFACE_EVOLUTION_SELECTION.md).

The straight path, mature seed, finite specimen and loading were imposed. Native
C++ computed mixed mechanical equilibrium, signed jump, irreversible surface
damage a and RSF histories; Python prepared inputs/launched jobs and analyzed
outputs. The initial mature seed has 3 J/m pre-existing fracture inventory.
Bulk phi is provisional geometry and is not a second fracture reservoir.

## Figures

- [specimen.png](specimen.png): full physical domain and left-tip mesh close-up,
  seed, candidate ends and buffer. Colors are actual surface-Gauss damage a.
- [evolution.gif](evolution.gif): accepted damage, signed physical jump and slip
  velocity on the whole installed Γ. Shading marks the original mature seed.
  No interpolated-age states or synthetic intervening frames are evaluated.
- [comparison.png](comparison.png): full-decohesion advance, growth excluding the
  initial seed, reaction/displacement and physical-work error. The early inset
  retains the 65.4% spatial discrepancy; front shading is Gauss sampling width.
- [profiles.png](profiles.png): process zone, developed damage and sliding at
  common times. Surface slip direction and propagation direction are distinct.
- [energy.png](energy.png): storage, fracture, friction, radiation and Maxwell
  loss, physical work, signed/absolute physical error and the separate BE ledger.
- [physical-stress.png](physical-stress.png),
  [physical-profiles.png](physical-profiles.png): actual native bulk τxy and
  solved p′, on identical scales. Background p0=10 kPa is separate. Maps use
  weighted cell means; extrema use every unaveraged Gauss6 point. Line means
  use .025 m physical strips immediately below/above Γ. These are not projected
  weak tractions, and not auxiliary-history plots.

The result is developed connected advance on a prescribed straight path. It
still lacks qualified early onset/speed, distant-boundary independence, free-path
selection and production propagation integration. Candidate-buffer stops remain
stops; the surface was not extended again to finish a hold.

## Compact data

- `growth-verification.json`, `criteria.json`: measured errors, gates and scope.
- `comparison.csv`: growth-relative differences at common times; .125 m physical
  box averages of actual Gauss histories, with initial seed subtracted.
- `straight-*-compact.csv`: every physical accepted time, reaction, growth,
  fronts, energy and numerical checks. Coarse hold continues to 2.48 s; use
  common 2.47 s rows for spatial comparisons.
- `front-reference.csv`: independent continuous Q1 jump threshold δ=2Gc/peak.
  All new-left nodal increments are nonnegative. No frictional state is
  interpolated. Complete newly saturated cell lengths distinguish genuine full
  advance from the first activated sample. Gauss brackets remain in histories.
- `milestones.csv`: accepted-step brackets, including diagnostic first a>1e−8.
  The threshold only labels events; it never changes the law. Precise onset is
  spatially unqualified.
- `selected-profiles.csv`: actual Gauss coordinates, arclength weights and
  histories. `cohesive_q_Pa` is bond traction on damaged points and the bounded
  tie reaction when a=0. Ties carry no fracture/sliding work. Weak shear and
  current normal pressure are labeled separately.
- `loading-hold.csv`: separate loading/hold work and raw absolute/signed error.
  **Baseline hold screens fail; half-step hold screens pass.**
- `physical-cells.csv`, `physical-profiles.csv`, `physical-field-checks.csv/json`:
  bulk stress/pressure averages and unaveraged extrema/storage checks.
- `native-qualification.json`, `growth-lifecycle.json`, `final-restart.json`,
  `legacy-default.json`: reference agreement, actual injected rejection/cutback,
  matching-step bitwise restart and protected default-route comparison.
- `*-run.json`, `run-manifest.json`, `preservation.json`, `SHA256SUMS`: cost,
  provenance and preservation. Known failed development attempts are retained
  and explained; intended buffer stops have nonzero process exits.

Relative response differences use the 128×80 Δt=.0025 trajectory as denominator; window sliding uncertainty uses the half-step loss.

Times are seconds, coordinates/jump metres, velocity m/s, stresses/pressure Pa,
energies J/m out-of-plane thickness, reaction N/m, integrated jump m². Damage a
and normalized age Θ are dimensionless. Surface weights have units m; bulk
weights m². There are 3 surface Gauss histories per cell; quadrature point counts
are not lengths or cell counts. Bulk Gauss3 has 9 points/cell; Gauss6 has 36.

## Reproduction and native ownership

Build the existing targets in a fresh build folder with the retained ASPECT
build, e.g.:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-evolution/build -DAspect_DIR="$PWD/build-particle-free/native"
cmake --build build-particle-free/cohesive-interface-evolution/build --target cohesive_interface_native cohesive_interface_kink_owner -j2
```

The executed bounded stages are available as:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_evolution.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_growth.py pilot
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_growth.py long-coarse
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_growth.py hold-coarse
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_growth.py long
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_growth.py half-window
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_growth.py lifecycle
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_growth.py final-restart
```

The two longer coarse stages intentionally stop at the buffer; an assertion in
the launch helper flags their nonzero exit for review. `guard-replay` checks the
specific expected message and preserves its full trial and last accepted state.
Use `legacy`, `fields` and the named analysis/plot scripts for those checks.
The drivers refuse to overwrite existing fresh runs; choose new case/output
names before repeating. The sources/parameter files in `reproduction/` and raw
solver archive retain the actual executed settings and source revision.

A direct native launch uses the generated `.prm`, `CI_SIGNED_SOLVER=1` and its
fixed geometry/loading environment from the driver. Native C++ advances every
accepted state; no Python solver is used in the propagation runs. The fitted
owner uses static operator/initial-state files, not a solution sequence.

ASPECT postprocessor serialization owns the histories and physical clock. The
preserved pre-event checkpoint is at
`build-particle-free/cohesive-interface-evolution/preserved/fine-prefix/`.
Checkpoint scheduling precedes the postprocessor: restored physical time is
2.2975 s, then the 2.30 s trial is recomputed. Merge diagnostics using that
restored physical clock; do not count ahead-of-checkpoint rows twice. Final-source
restart reproduces all Gauss, nodal/bulk and physical history columns bitwise.

Full per-Gauss histories, physical quadrature fields, failed trials, selected
native checkpoints and logs are stored separately in
`cohesive_interface_evolution_solver.zip` and the work/output directories. The
main `cohesive_interface_evolution.zip` contains the compact figures/data,
report and useful sources. No production/BP3 archive is replaced.
