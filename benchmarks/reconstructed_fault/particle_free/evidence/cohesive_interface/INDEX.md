# Cohesive compliance / geometric-interface RSF evidence

Start with [the report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_REPORT.md)
and [the formulation](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_FORMULATION.md).
The experiment changes the constitutive model; these are independent Python
references, not production ASPECT results. Surface damage **a is not bulk φ**.
No native/BP3 file, checkpoint or prior evidence was modified.

## Pictures and plain-language captions

| File | What was imposed, calculated and not tested |
|---|---|
| [intact_formation.png](intact_formation.png) | A tied homogeneous Maxwell block is sheared. Fracture compliance admits at the calibrated peak and dissipates Gc once. Three timesteps test onset/work; no geometric propagation is implied. |
| [mature_interface.png](mature_interface.png) | A fully broken plane slides through a velocity step, hold and reversal. One geometric speed/age determines RSF. Bond storage stays exactly zero; this is not a fitted cross-band reference. |
| [seeded_tip_profiles.png](seeded_tip_profiles.png) | A mature interval is surrounded by tied material on a candidate plane. Loading generates small connected damage and a pressure asymmetry. Curves are actual physical interface tractions/jumps, not auxiliary T. Original tips are dotted; different Δt values are labelled. |
| [seeded_tip_history.png](seeded_tip_history.png) | Low-damage fronts extend; high-damage fronts do not. Initial .25 J/m fracture inventory is excluded from added-fracture plots. Early first-cell onset and peak remain spatially sensitive. |
| [mixed_interface_symbol.png](mixed_interface_symbol.png) | Exact mixed layer response, including pressure and state, has decaying short waves at the tested mature sliding state. This is not a general formation-stability certificate. |
| [seeded_tip.gif](seeded_tip.gif) | Whole periodic specimen plus tip profiles, fixed scales throughout. Surface colour is a∈[0,1], dark intact/light mature; no invented diffuse phase field. Displacement hold starts at .30 s. |

## Data and definitions

- `point_history.csv`: common physical times every .01 s, formation/mature runs,
  state, slip, traction and cumulative separated work/storage. `point_comparison.csv`
  uses matching sample times; onset brackets use every accepted step in JSON.
- `point_summary.json`: onset, interface identity, FD refinement, mapping and
  owner lifecycle results. `intact_admission.csv`: exact one-sided force.
- `direction_admission.csv`: closed candidate-plane selection, preferred material
  direction, shear sign and basis covariance. `admission_summary.json`: work
  authorization and mixed symbol summary. No direction is picked from an
  unresolved tie without a preference.
- `history_mapping.json`: all old T/A/K components, retained residuals, physical
  stress, coherent reference and explicit switch work. `reference_checkpoint.json`
  is an independent owner example, **not a native checkpoint**.
- `kernel_checks.csv`: volume strain energy vs independent interface work,
  Gauss24/48 and actual-pressure identity. The normal lift check is separately
  normalized; it does not equate diffuse and sharp mechanical operators.
- `mature_mixed_symbol.csv`: along-interface wavenumber k, wavelength 2π/k,
  tangential/normal symbols and the two state/Maxwell eigenvalues. Negative real
  parts mean decay. Physical units are in column names.
- `tip_history_n*_dt*.csv`: every accepted step; integral/energy weights ds=L/N.
  `integrated_new_damage_m`=∫(a−a_initial)ds; `integrated_slip_m2`=∫δ ds;
  mean shear=(1/L)∫s ds. Absolute physical-work remainder is a sum of per-step
  absolute errors, not merely a signed endpoint cancellation.
- `tip_profiles_n*_dt*.csv`: native surface-cell centers, approximately .01 s
  snapshots, plus the first step; no interpolated constitutive histories.
  `initial_damage` is the exact aligned mature seed. Surface damage is dimensionless,
  δ is m, V is m/s, Θ is dimensionless (physical aging time=Dc Θ/V0).
- `tip_summary*.json`: timings, residuals, energy, lifecycle, incremental tangent
  and first-new-cell brackets. `tip_comparison.csv` / `tip_refinement.csv` retain
  failed coarse results as well as the focused refinement. For integrated
  quantities `relative_error` uses max fine-history amplitude. For front positions
  its denominator is the declared two-fine-cell allowance, **not front magnitude**.
- `tip_checkpoint*.json`: persistent independent owners at .198/.199 s, prior to
  the lifecycle trial; all required history arrays and publication count are kept.
  Restart equality uses the actual serialized file. No dense tangent is stored.
- `provenance.json`, `SHA256SUMS`: baseline, environment, commands, protected-file
  verification, source hashes and artifact hashes.

Fronts are edges of the seed-connected discrete component at a=.01,.05,.5,.95.
Disconnected activation is not called extension; the remote diagnostic is
max new a for |x|>.5 m. No smoothing, alternate averaging kernel or geometric
threshold changes the solve. N=128/256/512 have Δs=.015625/.0078125/.00390625 m.
The normal direction is solved analytically in the mixed Fourier layer problem;
there is no normal-mesh convergence claim. Formation and mature point energies
are J/m²; seeded energies are J/m of out-of-plane extent. Initial fracture energy
is inventoried once. Trapezoidal work, backward-Euler closure, bulk BE and signed
bond endpoint remainders must not be conflated.
