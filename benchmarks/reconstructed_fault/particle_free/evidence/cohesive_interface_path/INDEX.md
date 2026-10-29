# Causal successive-extension experiment

[Scientific report](../../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_PATH_REPORT.md) · [formulation and retained corrections](../../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_PATH_SELECTION.md) · [commands](COMMANDS.md).

The new result is an isolated native mixed-mechanics/C++ evolution experiment,
with a Python accepted-state archive/controller. It does not replace the ASPECT
production geometry/history/checkpoint owners. BP3 and previous evidence remain
protected. Quantitative bulk spatial targets fail; the selected path is a
hypothesis, not a qualified physical propagation law.

Start with these figures:

- [Specimen and accepted-state sampling](specimen-and-selection.png): the whole
  finite confined specimen, old seed and prospective tied reserve.
- [Angle conventions](angular-conventions.png): principal compression,
  conjugate maximum-shear planes, diagnostic Coulomb planes and actual fronts.
- [Successive extensions](successive-extensions.png): physical stress, damage,
  amplified signed slip and unfiltered contact diagnostics.
- [Focused controls](path-controls.png): common-time actual fronts, not just
  candidate labels.
- [Growth and physical work](growth-and-work.png) and
  [numerical corrections](numerical-controls.png).
- [Animation](successive-deflection.gif): accepted states on common physical
  times. Playback speed is not a rupture-speed measurement.

Primary compact data:

| File | Meaning |
|---|---|
| `parameters.csv` | Physical calibration, specimen and added length R |
| `comparison.csv`, `focused-comparisons.csv` | Common-time fields/functionals and absolute/relative numerical differences |
| `geometry-events.csv` | Every accepted zero-time insertion, accepted stress directions, history and energy preservation |
| `physical-fronts.csv` | Actual connected a=.05, .5 and 1 fronts and tangents |
| `plotted-profiles.csv` | Gauss coordinates, physical weights, a, signed slip/rate, age and contact pressures |
| `initialization.csv` | Fresh continuum seed representation and separate initial energy inventory |
| `energy-accounting.csv`, `*-history.csv` | Work, storage, fracture, Maxwell/friction/radiation and absolute per-step physical remainders |
| `contact-and-residuals.csv` | Accepted weak residuals, gaps, tangent diagnostic and retained raw paired traction mismatch |
| `analytic-checks.json`, `lifecycle.json` | Angular/virgin-energy identities; actual failed native trial, retry, once-only publication and restart |
| `weak-normal-qualification.json`, `frozen-pressure-*.csv` | Independent full-pressure tangent/bulk-work checks and matched fitted-reference comparison |
| `interpolation-consistency.*`, `independent-interpolation-ODE.csv`, `angular-sampling.csv` | Numerical path interpolation versus the independent physical ODE and full-circle angular sampling |
| `assessment.json` | Predeclared targets, including failures |
| `retained-v1-comparisons.csv` | Superseded C2/point-pressure results; original files remain unchanged |

`work64`, `work96`, `work-step96`, `work-q3`, `work-time96` and
`work-surface96` are the final fresh variational-contact controls. `work-time96`
is a late-window replay from its own verified incoming state; its error is
normalized by growth during that window. Other labels retain the original
point-pressure or interpolation experiments and must not be pooled as one
trajectory.

All distances are metres, times seconds, stresses Pa (plots use kPa), slip m,
velocity m/s, surface energy J/m² and integrated energies J/m of out-of-plane
thickness. Angle is tangent counterclockwise from +x, planes modulo 180°.
The reported flow angle exists only where |V|>1e-9 m/s, a plotting convention.
Front coordinates use the actual Q1 trace only after checking its damage
relation against every new Gauss history; no constitutive record is remapped.
There is no new fully decohered front until a=1 reaches new material. The
original endpoint at (6,0) is the zero-advance baseline, not an initially mature
front. Dashed admitted geometry is not fracture advance.

`checkpoints/` contains selected isolated-owner states; `operators/` contains
compact native operators and actual bulk physical fields. Large raw basis
tables stay in `build-particle-free/cohesive-interface-path/`; their hashes and
sizes are in `raw-kernel-index.json`. Historical diagnostics and source snapshots
are retained under `diagnostics/`. `changed-files.json`, `preservation.json`,
`manifest.json` and `checksums.sha256` describe task changes and provenance.
The adjacent ZIP is a compact matching-worktree source/evidence snapshot,
not a standalone ASPECT distribution or production restart package.
