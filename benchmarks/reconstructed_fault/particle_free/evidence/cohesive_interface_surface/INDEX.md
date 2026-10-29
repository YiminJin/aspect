# Surface/jump resolution: developed growth supported, onset still sensitive

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_SURFACE_REPORT.md) ·
[declared scope](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_SURFACE_SELECTION.md) ·
[compact archive](../cohesive_interface_surface.zip) · [checksum](../cohesive_interface_surface.zip.sha256).
Revision f1f9f29cf40c32d1ddcea000b724375b64853532 plus retained local work.

Fresh 256/512 Q1 surfaces, exact mixed-strip bulk, Gauss3 histories, fixed
finite-period specimen, current-pressure cohesive–RSF, unchanged calibration.
Loading to 2.55 s and hold to 2.58; focused fine temporal replay from 2.42375.
No native bulk rebuild or production/BP3 change.

- `comparison.png`: full advance, fracture inventory, right flank and slip.
- `surface-profiles.png`: common-scale damage, physical jump, current normal
  traction and work-adjoint shear load at 2.58. Grey is the initial mature seed.
- `space-time-errors.png`: separate spatial and temporal differences on common
  physical .125 m boxes. Late differences are small; two meshes establish no order.
- `early-response.png`, `onset-brackets.csv`: first admission shifts 72.5 ms.
  Early onset remains unqualified, including its separate temporal uncertainty.
- `surface-evolution.gif`, `animation-times.json`: actual accepted fine states;
  no interpolation of constitutive histories, no physical rupture-speed claim.
- `summary.csv`, `profiles.csv`, `comparisons.csv`, `verification.json`: common-time
  physical fields/functionals, absolute/relative differences and declared targets.
- `representation.csv`: independent best coarse Q1 L² representation of the
  computed fine linear jump. Not used to transfer or initialize histories.
- `native-mechanics-differences.csv`: native versus analytical bulk at identical
  256-cell surface/time sequence, separated from surface refinement.
- `energy.csv`: raw physical-work errors against new fracture and sliding loss,
  event and hold separately. No background correction; initial seed is 3 J/m.
- `qualification.json`: preserved default reference and three frozen native-step
  comparisons; complete pressure-coupled derivative checks.
- `operator-verification.json`: mean modes and actual alias300→600 sensitivity.
- `lifecycle-verification.json`: all history NPZ round trips, unchanged rejected
  state, counter and reproduced/recovered hold predictor stop.
- `surface-*-history.csv`, `surface-*-run.json`: accepted evolution histories,
  physical checks, costs and original stopped-attempt accounting.
- `manifest.json`, `checksums.sha256`, `preservation.json`, `changed-files.json`:
  source/data identity, protected entry hashes and task changes.

Units: coordinates/jump in m, velocity m/s, pressure/load Pa, energy J/m of
out-of-plane thickness, Gc J/m²; age Θ and surface damage a dimensionless.
Bulk φ is provisional and is not evolved. Tractions are the established Q1
projected coupling quantities, not claimed raw point-trace accuracy.

Damage/age comparison averages actual Gauss3 histories over identical .125 m
physical boxes. A Q1 contour is reconstructed from critical jump only after
checking its equality with stored Gauss damage. Initial jump/seed-inventory
representation differences are exactly zero. No phase histories are remapped.

Full reference checkpoints and preserved failure logs remain separately under
`build-particle-free/cohesive-interface-surface/`. The archive contains compact
data, useful selected checkpoints and source dependencies; it requires the
existing checkout and Python NumPy/SciPy/mpmath/Matplotlib/Pillow environment.
The previous native raw output and protected BP3 data remain in their original
locations and archives.

Recorded commands, sequential, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

```sh
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_surface.py qualify
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_surface.py 256
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_surface.py 512
# Preserved hold predictor stop; after its narrowly scoped guess recovery:
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_surface.py 512-resume
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_surface.py 512-half
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_surface.py check-lifecycle
python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_surface.py
python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_surface.py operators
python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_surface.py
python3 benchmarks/reconstructed_fault/particle_free/package_cohesive_interface_surface.py
```

The final runner retries that initial guess automatically. Reproducing the
historical stop uses `check-lifecycle`; it preserves the same incoming state and
complete signed acceptance checks. Existing run folders are protected against
accidental overwrite. No thresholds or tolerances changed after measurement.
