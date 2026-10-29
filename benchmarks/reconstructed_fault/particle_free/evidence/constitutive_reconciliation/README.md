# Bounded constitutive reconciliation evidence

Read `doc/reconstructed_fault/particle_free/CONSTITUTIVE_RECONCILIATION_REPORT.md`
and its formulation in the repository-relative archive. The outcome separates
**smooth-branch kinetic regularity** from **unresolved global continuation**.
There is no native integration, specimen, MPI, server or propagation run.

From the repository root, reproduce the compact data and PNGs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/check_feng_rsf.py
MPLCONFIGDIR=/tmp/feng-rsf-matplotlib OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_feng_rsf.py
```

Python/NumPy/SciPy/Matplotlib/mpmath versions are in environment.json. The
unchanged feng_direction.py helper and original archived-point-inputs.csv
are included at their original paths. No original solver build is needed.
These commands overwrite only this reference's generated data/figures.
Final bounded executions used run_logged.py with unique labels; final and
failed verification logs are included. Timing runs did not overlap.

Data dictionary (SI units unless identified otherwise):

| Files | Meaning |
|---|---|
| endpoint-formulas.csv | Literal Feng angle formula vs all stationary branches and independent roots. Angle is plane **normal** from +x, degrees. C1=0 is reported undefined, not assigned an arbitrary angle. Global margin and stationarity are distinct Pa quantities. |
| printed-sign-control.csv | One synthetic unloading state separates printed physical-trial-shear sign from relative shear. Positive phase force means obstacle-compatible, not proven reachable. |
| continuous-viability.csv | First-activation directional Fdot in Pa/s, with Y and physical obstacle force in Pa. Both conjugate choices are checked separately. No mixed flow is substituted. |
| loading-summary.csv, loading-histories.csv | Fresh initial pre-damage and zero-history shear ramps, 32/128/512/2048 steps. Post-failure paper paths are **diagnostic inadmissible paths**. First rejection is preserved. T/A/K recurrences are not reset. Histories sample every eighth step for the finest path; the summary covers every step. |
| reachability.json | Distinguishes reached first-activation states, diagnostic paths and synthetic reversed histories. No claim that a phase-active trajectory produced the latter. |
| mapping.csv | AT1 density normalization, coherent integrated velocity m/s and friction/radiation power W/m²; phase-profile support sizes are physical. This is a kinematic/power check, not an equilibrated strip solve. |
| interface-histories.csv, interface-timesteps.csv | Separate fixed-plane coherent velocity steps, reversal and stop; exact segmentwise aging vs BE. State Theta is dimensionless, pressure/traction Pa, slip m, energy J/m². Pressure is prescribed here. |
| kinetic-states.csv | Frozen synthetic kinetic states: archived tensor *shape* is retained but stress magnitude is moved to the new kinetic surface. tau/Q Pa, w s^-1, chi m^-1. Exact maturity is an analytic specialization, not a phi=1 native API call. |
| kinetic-angles.csv, kinetic-summary.csv | Complete coupled mechanical/state spectrum vs wave normal. Angles are perturbation-wave normals, not flow/crack angles; rates s^-1. Both one-sided conjugate limits and strict noncoaxial states are retained. No simultaneous conjugate sum. |
| wavelengths.csv | Independent uneliminated mixed generalized eigenvalues vs exact reduction, including oblique earlier failure angles. k ell is a reporting coordinate, not a new slip length. Wavelength=2 pi ell/(k ell), m. No spatial operator was added. |
| pressure-lag-timesteps.csv | Coupled and previous-pressure discrete amplification, including the old-state forcing. Same wave normal in each comparison; continuous reference is evaluated at that same angle. |
| point-histories.csv, point-timesteps.csv | Fresh mature point under prescribed strain/pressure, shear reversal at .15 s; accepted states only. Full T/A/K identities checked. V is nonnegative speed; Wxy supplies the signed shear component. Energy densities J/m³, radiation separate from Maxwell and BE loss. |
| lagged-endpoint-failures.csv | Rejected trial and last accepted times, original error/allowance, independent 80-digit global margins. These evaluate the *lagged* objective, not an imposed current-pressure test on an intentionally lagged residual. |
| response-comparison.csv | Same physical times, full-tensor stress relative L2, speed relative L2 and work differences. Uniform time weights cancel. tau_yy=-tau_xx and tau_zz=0 here, so the full tensor factor sqrt(2) cancels in the ratio. |
| kinetic-checks.json, decision.json | Complete-Jacobian FD, full mixed QZ and discrete block checks; separate regularity/continuation decisions. |
| sources.json, parameters.json, preservation.json | Primary sources, explicit assumptions/units, protected-file hashes and unchanged index tails. |

The candidate uses positive direct effect and radiation, not a spatial
regularizer. No gradient of accumulated slip, reference band width, phase
growth mask, tolerance change, conjugate averaging or plane lock is introduced.
Intact state birth, noncoherent interface interpretation, global tie dynamics
and finite-wavelength/propagation qualification remain open. Prior production
and scientific evidence is preserved.

Reference verification corrections: the first mixed-QZ comparison had a
4.66e-10 error from disparate algebraic variable units. Invertible equilibration
of BOTH pencil matrices reduces it below 1e-10 without changing eigenvalues.
A subsequent diagnostic divided roundoff in a zero derivative column by an
arbitrary 1e-12 denominator; the corrected comparison measures the complete
Jacobian norm, including zero columns. The original errors/logs are retained.
Scientific endpoint failures were not repaired or accepted.
