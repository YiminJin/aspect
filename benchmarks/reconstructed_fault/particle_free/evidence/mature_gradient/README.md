# Mature sliding compatibility evidence

Read MATURE_GRADIENT_REPORT.md and MATURE_GRADIENT_FORMULATION.md. The independent
strip exactly solves 1-D mixed mechanical equilibrium with the qualified C++
normal-map projection. It is not a prescribed-strain run, new native backend,
full RSF implementation or phase-evolution qualification.

- history.csv: accepted states at every time; 1/m² energies are per fault area.
  delta_m is 2 integral K:S dr; V_m_s is 2 integral W:S dr. Boundary velocity
  includes additional elastic/Maxwell deformation. Rgradient/Rcohesive/Rfriction
  are power/V averages ONLY where V>1e-12 m/s; their stored zero entries before
  sliding are undefined ratios and are omitted from resistance plots.
- profiles.csv: actual physical nodal coordinates, positive constitutive weights,
  fixed phi/h, K/A/Q/X, physical stress/pressure and engineering strain/rate.
  Snapshots are at .25/.5/1/1.5/2 s. Phase never evolves. No fields are used to
  reinitialize a different constitutive state.
- final.csv and matched-slip.csv: final states or explicitly labeled linear
  interpolation between accepted monotonic-slip states. Cumulative energies
  integrate powers with each state's dt. All energy columns use J/m².
- sensitivity.csv: common times and physical P1 fields; exact Gauss3 integration
  of squared linear differences on the union of mesh knots. No averaging kernel.
- frozen-shape.csv: normalized rho and analytic Hgrad=G ell_s² integral rho'^2;
  extrapolated 5% crossing is a frozen-shape diagnostic, not an evolved run.
- checks.json and verification.json: original residual/action/energy bounds,
  initial phase representation error and mechanical/local references.
- PNGs: traction–plastic-slip, physical cross-band profiles and separate energies.
  No GIF is needed because phase is fixed and late flow shapes mostly scale.

Units: r,delta in m; V in m/s; K,A,L dimensionless; W in 1/s; all stress/force
components in Pa; surface energies J/m² and powers W/m²; ell_s in m. Physical
sigma=tau-p Id; shear sigma_xy=tau_xy. Both conjugate planes remain available.
The 995.037 Pa local plateau reflects the existing free-plane Coulomb law;
mu*p0=1000 Pa is the separate prescribed-aligned-plane reference, not recalibration.
Onset brackets use accepted V>1e-12 m/s, not an exact activation time.

Primary literature links and differences are in the formulation. No native
production source, coefficient, history checkpoint or BP3 default was changed.
All original artifacts and earlier failures remain separately preserved.
