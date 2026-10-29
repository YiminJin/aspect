# Coulomb mechanical localization audit

Start with the portable `COULOMB_LOCALIZATION_REPORT.md` and
`COULOMB_LOCALIZATION_FORMULATION.md` beside this README, or the corresponding
worktree documents. Exact worktree sources are also retained under `context/`.
**Mechanical prerequisite fails; phase evolution remains
disabled.** No regularized/dilatant law, larger run or tolerance change is made.

`mechanical-symbol.png` shows full mixed rank loss, phase range, same-incoming
state timestep probes and formal free-phase high-frequency behavior.
`native-refinement.png` shows actual weighted quadrature slip values,
common-physical-kernel profiles, mechanical conditioning and broad slip widths.
The short runs do not demonstrate a collapsing band; this limitation is
separate from the demonstrated continuum mechanical loss of ellipticity.

## Compact data

- `point-states.csv`: actual stateless return and full strain/pressure/phase
  derivative columns for 960 states. Stress in Pa, dt in s, angles in radians.
  Columns d0/d1/d2 are unit epsilon_xx/epsilon_yy/engineering gamma_xy;
  d3 is delta p=G; d4 is delta phi=1. Stress outputs are divided by G=1e6 Pa;
  driving outputs by Kf=8 Pa. Full plane strain includes zz and two xy terms.
- `point-symbols.csv`, `phase-symbol-summary.csv`, `point-timestep.csv`:
  actual algorithmic and independent continuous-rate minima. `active` bitmask
  0/1/2/3 means stick/plus/minus/both. At phi=0 finite Q can label active
  yield constraints while physical A/W are zero. Single-system analytic
  threshold is in `analytic-critical.json`. Mirrors and constant-R controls
  are retained. The phase plot displays the developed-damage portion; the
  CSV retains the intact/low-phase controls too.
- `angular-symbol.csv`, `mixed-null-modes.csv`: pressure-eliminated coefficient,
  fixed-pressure comparison, determinant and actual 3x3 null vectors. Wave
  normals and band axes are not flow-plane or crack-front angles.
- `coupled-symbol.csv`: formal free-phase elimination at one selected normal.
  Actual strict-obstacle phase directions are zero; this is not a native
  phase-active result or a nonsymmetric stability certificate.
- `native-histories.csv`, `native-final-comparison.csv`, `native-cases.json`:
  physical-time summaries, exact zero phase increment, unsmoothed widths,
  mechanical/source/energy checks and actual exit status. Negative-symbol
  fractions use actual JxW area, not point counts; local native normal scan
  has 721 angles. Cutback histories have four new publications9–12, not a
  fresh twelve-step initialization. The two full-step fine failures remain
  at .1 s; do not compare those final states as if they reached .125 s.
- `native-slip-maps.csv`: actual Gauss6 values at .125 s for the three fresh
  half-step meshes. Equal color/physical scales. Equivalent bulk slip rate
  `sqrt(2 W:W)` has units1/s, not surface slip velocity. JxW is physical m².
- `normal-profiles.csv`: quadrature-weighted Gaussian averages of that scalar
  rate in coordinate `s=x cos(theta)+y sin(theta)`, theta=2.913739130895119 rad,
  physical sigma=ell/2=.0234375 m. Normalize by kernel mass (m²); blank values
  mean mass<1e-12 m² and are excluded from figures. No constitutive history
  is replayed on an interpolated/averaged state. Width measurements do not
  use this smoothing: width=(integral w)^2/(.25 m integral w²).
- `conditioning.csv`: singular values of the actual nonsymmetric native
  constrained mixed matrix, Riesz-normalized with same-mesh elastic H1
  velocity and pressure L2 metric; no artificial constrained rows. Both
  pressure mean constraint and independent FE reference use C^T J C once.
  Individual conditioning JSONs give the measured postprocessing durations.
- `temporal.csv`: full-tensor W L² differences at the same physical time,
  normalized by the half-step W norm. These9.60%/11.92% differences do not
  establish temporal convergence.
- `execution-summary.csv`: command wall time, actual exit code, accepted
  interval count, assembly/linear/history/output timers and maximum-rank
  process peak RSS in KiB. Timers cover accepted rows; failed work/startup,
  checkpoint and diagnostic overhead need not add to the command wall time.
- `independent-checks.json`, `independent-assembly.json`, `native-gates.json`:
  errors, predeclared bounds, constant-R, same-parent cutback and preservation
  checks. BE ledger precision is not physical-energy accuracy qualification.

## Reproduction and provenance

`context/` retains changed sources, unchanged constitutive/fixture dependencies,
native parameter inputs and actual build/launch records. `entry-delta.patch`
is relative to the preserved task-entry files, not the old git HEAD alone.
`changed-files.json` and `provenance.json` carry exact hashes. `logged/`
contains commands, durations and original failures. The temporary plotting
failure was an empty low-mass profile-tail conversion, corrected by omitting
the explicitly undefined entries. Do not delete or reinterpret that log.

For independent point replay, build CMake target `coulomb_localization` in
`benchmarks/reconstructed_fault/particle_free/bulk_cohesive`, and pass one CSV
output path. The Python analysis scripts use numpy/scipy/matplotlib and the
worktree's `build-particle-free/coulomb-localization` and evidence paths.
Native input/environment/commands are recorded verbatim; run with the existing
bounded `run_logged.py` wrapper, four ranks and one thread. Individual native
timeouts are240s. Geometry, coefficients, initialization and load are identical
across meshes. Do not enable phase evolution as a reproduction step.

Full native owner state, point derivatives, sparse matrices, accepted/rejected
logs and checkpoints remain under `build-particle-free/coulomb-localization/runs/`;
they are intentionally not copied into this compact package. Recomputing all
native matrix/profile checks requires those retained outputs. No checkpoint
has been converted, reset or silently replaced. Original P4/P5c/BP3 evidence
and `strict_reference_match_passed=false` remain protected.
