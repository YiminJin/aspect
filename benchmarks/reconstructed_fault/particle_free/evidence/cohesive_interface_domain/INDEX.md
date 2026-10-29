# Horizontal enlarged-domain control

Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local work.
See the [selection](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_DOMAIN_SELECTION.md)
and [report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_DOMAIN_REPORT.md).
[Compact archive](../cohesive_interface_domain.zip), with adjacent SHA256 file.
The archive contains the report, figures/GIF, compact data, inputs and relevant
experimental sources; full solver output/checkpoints remain separately below.

Result: the common interval completed, including .02 s displacement hold.
Growth/slip, tensor and pressure domain-sensitivity targets fail; raw energy,
rejection/restart, once-only publication and added-record inactivity pass.

The 8×1 and 16×1 m strips have the same physical mature seed [2.5,5.5] m at
y=.4 m, Δx=.0625 m, Δy=.0125 m, physical law and loading. The large native
box [0,16] is translated by −4 m for all physical comparisons. No history is
interpolated. Surface damage is **a**, not bulk phase φ; the latter remains
provisional geometry in this exact-jump component. Units are per metre of
out-of-plane thickness.

Figures:

- `specimen.png`: physical domain, seed, tied candidate and wall motion.
- `histories.png`: damage, fully decohered advance, jump, reaction deficit,
  bulk excess and raw work error, with an early-growth inset.
- `profiles.png`: accepted Gauss-owned a and signed jump at common times.
- `physical-stress.png`: native full-bulk physical τxy cell means and difference.
- `tip-fields.png`: equal-aspect local physical shear and normal pressure.
- `derivative-reference.png`: preserved double failure versus the extended
  reference at identical perturbations and unchanged gate.
- `evolution.gif`: accepted surface histories; added records are visible.

Data and checks:

- `summary.csv`: compact common-endpoint comparison; `changed-files.json`
  identifies this task's code/docs and the evidence directory.
- `period-{8,16}-history.csv`: all accepted times, front brackets, integrated
  damage/jump, energy and reaction; initial seed fracture energy is 3 J/m and
  excluded from **new** fracture energy.
- `profiles.csv`: actual surface Gauss samples, coordinates, weights, signed
  jump, dimensionless Θ, velocity and current-pressure traction.
- `comparison.csv`, `surface-comparison.csv`: scalar functional and common-Γ
  weighted differences; normalize damage by **growth**, not the initial seed.
- `physical-comparison.csv`: Gauss6 full-plane-strain tensor and pressure norms
  on common [0,8]×[0,1] and tip [1,3]×[.2,.6] boxes.
- `physical-cells.csv`: actual Gauss6-weighted bulk cell means for plotting.
- `bulk-global.csv`, `operator-comparison.csv`: pressure-gauge/contact/remote
  checks and identical-memory common-block responses before evolution differences.
- `temporal-field-scale.csv`: earlier Δt=.0025 versus .00125 bulk-field changes
  at the same physical points. This gives an uncertainty scale, not a new sweep.
- `verification.json`: raw energy screens, residuals, derivatives, once-only
  publication and all-step added-record inactivity.
- `reference-identity.json`: fresh 8 m versus archived accepted trajectory.
- `restart-overlap.json`: replayed accepted physical columns versus the retained
  first enlarged attempt, before its derivative stop.
- `derivative-diagnosis.json`, `derivative-MP50.csv`: independent 50-digit local
  derivative reevaluation at saved failing-state coefficients. Original double
  errors remain in the raw output and first-attempt archive.
- `period-{8,16}-run.json`: timing, observed child-process RSS, inputs, DSO hash,
  exit status, preserved pre-event checkpoint and resource limits.
- `entry-protection.json`, `preservation.json`: protected source/BP3/archive hashes.

`period-8` and `period-16` link to native solver outputs under
`build-particle-free/cohesive-interface-domain/runs/`. Full Gauss histories,
Gauss6 bulk snapshots, operators, logs, failed trials if any, and rolling native
checkpoints stay there. Pre-event checkpoints are independently preserved under
`build-particle-free/cohesive-interface-domain/preserved/`. Compact packages do
not follow these links or duplicate full solver archives.

Definitions: stress norm squared is xx²+yy²+zz²+2xy², applied to **tensor
differences**. Tensor relative L2 is the square root of the quadrature-weighted
error norm divided by the reference norm. Pressure error is weighted RMS.
Gauss front coordinates are accompanied by adjacent-point brackets. The
continuous left full front uses the native Q1 jump threshold 2Gc/peak, after
checking monotone increments there. It is not a first-activated-node measure.

Energy: raw trapezoidal boundary work minus changes of bulk/bond storage,
new fracture, friction, radiation and Maxwell loss is retained step by step;
the report uses both its signed sum and sum of absolute values. Extensive
background differences are reported separately using an independently advanced
homogeneous Maxwell strip on the same accepted time grid. Excess quantities
never replace the raw physical-energy acceptance screen.

Commands (run serial timing cases sequentially; existing output is protected):

```sh
cmake -S benchmarks/reconstructed_fault/particle_free \
  -B build-particle-free/cohesive-interface-domain/build \
  -DAspect_DIR="$PWD/build-particle-free/native"
cmake --build build-particle-free/cohesive-interface-domain/build \
  --target cohesive_interface_native -j2
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_domain.py reference
python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_domain.py reference-check
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_domain.py enlarged
# The original double FD check stops at 1.9425 s. Preserve it, then use only
# the opt-in extended-precision diagnostic, with unchanged physical controls:
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_domain.py enlarged-resume
python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_domain.py
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_domain_derivative.py
python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_domain.py
```

The wrapper validates parameters and launches ASPECT; the C++ component solves
the physical evolution. Native `Time stepping / Function` sets Δt=.0025 s
through 2.30 s and .00125 s afterward, with loading to 2.45 s and hold to 2.47 s.
No production solver, point law, pressure treatment or tolerance is changed.
