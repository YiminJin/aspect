> Artifact availability after 2026-10-30 cleanup: original quantitative and surface histories remain; some older bulk-animation frames were pruned. See the explicit [cleanup limitation](../../../../../doc/reconstructed_fault/particle_free/CLEANUP_REPORT.md#legacy-animation-data-loss). The current RSF checkpoint and original BP3 server data are unaffected.

# Native quasi-dynamic RSF instability: review evidence

Scientific report: `doc/reconstructed_fault/particle_free/COHESIVE_RSF_INSTABILITY_REPORT.md`.
[Comparison](review/comparison.png) · [GIF](review/evolution.gif) ·
[Temporal uncertainty](review/uncertainty.png) · [Energy](review/energy.png) ·
[Controller floor](controller-floor.png) · [Specimen](specimen.png).

All quantities are SI; energies are per metre of out-of-plane thickness.
Bulk `Txy_Pa` export is reconstructed physical shear stress, despite its historical
column name. Damage `a` lives at surface Gauss points, not in a bulk phase field.

- `specimen.png`: domain, boundary loading, initial mature patch and tied tail.
- `review/comparison.png`: feedback control and focused temporal replay.
- `review/energy.png`: physical work/storage/dissipation and numerical remainders.
- `review/evolution.gif`, `review/initial.png`, `review/final.png`: actual native
  physical-stress and surface histories, using fixed scales. Frame spacing is
  nonuniform in physical time; read the printed times. This is not a wave movie.
- `review/indexed-histories.csv`, `final-profiles.csv`, `temporal-common-times.csv`,
  `energy-windows.csv`, `comparison.json`: plotted values and comparison metrics.
- `stability-fine/`: independent current-pressure/Maxwell/aging/radiation spectrum
  from native mixed operators; finite-difference and scalar characteristic checks.
- `spatial-loads.png`, `loads-at-budget/`: identical physical surface memory through
  two native bulk approximations; Gaussian Q1 test functions with sigma=.375 m.
  No diagnostic response is fed into accepted history. Raw weak-load failures stay.
- `initial-control-identity.json`, `retry.json`, `restart.json`, `resume-hook.json`,
  `event-restart.json`, `verification.json`: physical identity and lifecycle evidence.
- `runs.json`, `provenance.json`: measured execution, inputs and source/binary identity.
- `SELECTION.md`: original pre-run criteria and retained first-step arithmetic failure.

The primary trajectory stops at its 600-step computational cap during acceleration.
It has small new cohesive damage, no fully decohered-front advance. A temporary
zero-damage reserve is not counted as growth. Point stresses/tractions remain
spatially unqualified. No final pulse maximum, arrest, rupture speed or distant-
boundary interpretation is claimed. Full native checkpoints and failed runs stay
separately under `build-particle-free/cohesive-rsf-instability/`.

The tighter replay stops at its computational timestep floor,39.43 microseconds
before the target. `rate-floor.json`, `rate-floor.csv`, and `floor-trials.csv`
preserve the exact reproduced pre-publication failure and independent decimal
reevaluation. `half-incoming-identity.json` verifies its shared accepted state.
`verification.json` includes all unchanged checkpoint files and the retained
portable regression. The1% state comparison and fracture-relative energy screen
remain failed. The GIF is the primary trajectory only; earlier replay curves in
the comparison share the accepted prefix and are not a second fine trajectory.
