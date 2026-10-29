# Prescribed oblique admission evidence

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_ADMISSION_REPORT.md)
· [selection/equations](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_ADMISSION_SELECTION.md).

## Figures

- [Specimen](specimen.png): fixed fitted block, old tied segment, prespecified
  26.565° append, affine axial loading and background confinement.
- [Profiles](profiles.png): surface damage, signed geometric slip/rate and current
  normal pressure at common times. Markers are Gauss-owned histories; their
  integration uses arclength weights, not point counts.
- [Plane traction](plane_traction.png): actual physical FE plane shear versus
  weak Gauss resistance and cohesive traction. Intact multipliers can be
  nonunique; they are not interchangeable with pointwise physical stress.
- [Temporal response and energy](temporal_energy.png): event inset, common-time
  damage, signed and absolute work remainders, new fracture and sliding loss.
- [Physical Cartesian shear stress](physical_stress.png): identical scales
  before/after admission. Zero Cartesian tau_xy before activation does not mean
  zero oblique-plane shear: t·tau n=−1.6 G gamma_memory under axial loading.
- [Computed-state animation](admission.gif): accepted fine states only, stress
  and surface damage; no interpolation of histories or validated rupture speed.

**Caption.** The new segment was prescribed, inserted while it could remain tied,
and subsequently loaded through its intact strength. Damage and slip were solved
with the unchanged law and existing histories. The entire append activates at
small damage; no fully decohered region advances. The comparison supports the
admission/lifecycle components, while sliding-scale physical energy accuracy,
free-path selection and the earlier spatial failure remain unresolved.

## Indexed data

| File | Definition |
|---|---|
| `histories.csv` | both timestep schedules; time/dt s, strain dimensionless, energies J/m, weak force Pa, damage and integrated slip, separate BE/physical remainders |
| `temporal_comparison.csv` | comparisons at identical times; damage L2 uses physical Gauss arclength; denominators are measured growth, not a large initial seed |
| `profiles.csv` | physical x/y m, arclength weights m, a, signed jump m, signed V m/s, Theta, sided-mean normal pressure/traction Pa; tied/installed/full-decohesion flags |
| `physical_fields.csv` | physical Gauss6 area weights m², full plane-strain tau_xx/yy/zz/xy and solved p' Pa |
| `verification.json` | measured temporal and energy screens, analytic tied-event control and native validation summary |
| `bulk_energy_diagnostic.csv/json` | homogeneous Maxwell block with no fracture interface on the same BE schedule; diagnostic, never subtracted from acceptance error |
| `coarse/fine-summary.json` | discrete event brackets, birth certificate, timing and publication counts |
| `coarse/fine-history.csv`, `*-states.json` | all accepted reference steps and every persistent history; source state from each mesh is not remapped |
| `*-preactivation.json`, `*-activation-trial.json` | immutable incoming state and first physical damage candidate for retry/restart |
| `preinstalled-identity.json` | entire physical trajectory agrees bitwise at the same timestep |
| `partial-tie-check.json` | independent general Gauss nullspace and derivative check |
| `native-owner.json`, `native-*/*` | three actual native publications, rejection/duplicate guards and ASPECT checkpoint identity across activation |
| `default-regression.json` | prior native kink owner unchanged with admission replay disabled |
| `parameters.csv`, `provenance.json`, `preservation.json`, `changed-files.json` | calibration, source/input hashes, preserved production evidence and task changes |

The full append has arclength .2795084972 m. Plotted damage thresholds are .01
and .02; a=1 denotes full decohesion and is never reached. No additional growth mask is
used on the represented surface; bulk phi and other paths are not evolved. Surface a is distinct from provisional geometric bulk phi. Angles describe
the prescribed interface; no crack direction is selected from the plotted stress.
The initial bulk prestress energy is 4.5 J/m; new fracture starts from zero.

The physical model interval is .2–.66 s with a hold starting exactly at .65 s.
The numerical tied-loss bracket is ≤1e−8 s, but timestep differences and the
restriction to prescribed planes remain. The native host's .001 s scheduler
increments exercise checkpoint callbacks only; physical increments are read from
the verified .5814467–.595 s activation sequence.

## Reproduction

Reuse the retained native axial kernel under
`build-particle-free/cohesive-interface-kink/native-16-axial/`. Its source and
output hashes are indexed in provenance. The preceding kink runner can regenerate
that kernel; it preserves earlier directories rather than overwriting evidence.
The actual mixed matrices, Gauss6 response and raw checkpoints remain separately
under `build-particle-free/`.

```sh
cmake -S benchmarks/reconstructed_fault/particle_free \
  -B build-particle-free/cohesive-interface-admission/build \
  -DAspect_DIR="$PWD/build-particle-free/native"
cmake --build build-particle-free/cohesive-interface-admission/build \
  --target cohesive_interface_kink_owner -j2
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_admission.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_admission_owner.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_admission.py
MPLCONFIGDIR=/tmp/admission-mpl OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_admission.py
```

Native invocations have 120 s timeouts and are sequential. No production or
server job is launched. The archive is a source/evidence overlay for this
worktree, not a standalone ASPECT distribution. The new native mode is selected
only by the fixture environment; production parameters/defaults are unchanged.

The solver's first development attempt asked an exact tied certificate to
balance an unequilibrated post-activation predictor; the LP correctly reported
infeasibility. The predictor now uses a least-squares multiplier estimate, while
final bounded certification and tolerances are unchanged. This was a solver
initialization issue, not a relaxation of physical admission.
