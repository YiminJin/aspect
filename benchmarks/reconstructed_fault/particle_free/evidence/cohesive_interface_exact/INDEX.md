# Exact-jump fixed-surface evidence

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_EXACT_REPORT.md)
contains formulas, scope, results, commands and changed files.

| Figure/data | Meaning |
|---|---|
| `operators_and_traces.png` | Same-space sharp reference errors and physical two-side projected tractions |
| `response_and_energy.png` | Intact admission, mature signed reversal, nonuniform pressure and physical energy errors |
| `frozen_fields.png` | Full-domain physical shear/pressure for identical nonuniform memory; difference from source-only |
| `operator_comparison.csv` | 1 m mode errors, pressure coupling and two-side mismatch |
| `shorter_modes_diagnostic.csv` | Remaining wavelength-dependent resolution error, including .125 m failure |
| `frozen_trace_profiles.csv` | Plotted physical traces, coordinates in m and stress in Pa |
| `native_reference_histories.csv` | Accepted responses and independent return comparison |
| `summary.json` | Quantified checks, bitwise restart and energy errors |
| `runs-*.json` | Exact input choices, exit codes and sequential wall times |
| `restart_negative.json` | Wrong representation restart rejected, accepted outputs unchanged |
| `provenance.json` | Protected source/evidence checks, source and binary hashes |

Each case contains native operator/field/trace checks. `trace_operators.csv` stores
physical unit-jump response matrices for lower/upper shear and normal-pressure
traces (Pa/m). `operators.csv` contains weak energy stiffness K and mean projected
normal-pressure coupling N, with surface row weights in m. `trace_checks.csv`
separates exact displacement-jump identity, surface quadrature, whole-matrix trace
mismatch and work. The latter mismatch includes unresolved high modes.

`states.csv` has surface Gauss records (the legacy `node` column is the Gauss
record index). Damage/Theta are dimensionless; jumps/memories are m, V is signed
m/s, pressures/tractions Pa. `nodal_history.csv` instead stores actual surface
nodes and signed geometric velocity. `history.csv` stores accepted time, dt,
publication count, work/storage/loss per out-of-plane thickness (J/m), residuals
and local curvature. Fracture **increments** exclude the initial inventory.
Do not identify exact BE ledger closure with physical-work accuracy.

`frozen_fields.csv` samples one actual bulk Gauss6 point per cell for compact
plotting; JxW is that point's area weight, not the cell area. All 36 bulk points
are used for checks. In exact mode the old band-normal check is explicitly
inapplicable; normal verification is in the two-trace files. The source-only
control remains labeled separately. Field plots are probes, not evolving cracks.

Restart joins the split prefix up to `restart_loaded.csv` with `*-resume.csv`.
An uncheckpointed split tail is not counted twice. Both branches are retained.
The final roundoff-sized mature step is retained. Full checkpoints/build logs,
original edited fixture and unsigned-derivative failure source/log remain under
`build-particle-free/cohesive-interface-exact/`. No production executable is
included or replaced. The package is an overlay for this worktree, not a full
ASPECT distribution; the earlier native package supplies its retained controls.
