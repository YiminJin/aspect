## Later comparison-figure revision

The working `comparison.png` now includes an early-time inset, absolute and
relative integrated-growth differences and a common 2.42 s endpoint. The added
`comparison_growth_differences.csv` contains the plotted differences. The original
65.4% spatial failure and all solver data are unchanged. The historical source/
evidence ZIP retains the original figure; see the subsequent kink evidence for
the revised plotting source and figure provenance.

# Finite insertion and localized growth evidence

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_INSERTION_REPORT.md)
· [equations/ownership](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_INSERTION_FORMULATION.md)
· [declared selection](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_INSERTION_SELECTION.md).

## View first

- [Specimen](specimen.png): physical domain, initial mature seed and finite tied candidate.
- [Loaded transfer](transfer.png): actual physical stress before/after insertion and retained jump at .28 s. Bulk stresses are sampled on the nearest exported Gauss row, not an interpolated new constitutive state. Shading marks the added tied records.
- [Growth profiles](evolution.png): surface damage a, physical jump and projected weak shear. Dashed lines mark the seed; dotted lines mark finite endpoints. Projected weak stress can locally exceed the intact multiplier bound; these are different objects.
- [Physical bulk stress](physical_stress.png): Gauss6-weighted cell means, identical colour scale and physical aspect ratio. The solid white line is the **initial** seed, dotted line the installed candidate, not the current a=1 front.
- [Controls](comparison.png) and [energy](energy.png): matched spatial/time responses and physical versus BE accounting.
- [Animation](evolution.gif): finest trajectory, fixed axes. The strip above the damage curve is grey for tied records, orange for the cohesive zone, blue for a=1 and blank outside the installed interface.

**Caption:** A finite straight candidate and a mature seed were imposed. Mechanics,
current normal pressure, interface jump, irreversible surface damage and RSF age
were solved. Added tied records preserve the loaded state; subsequent work pays
for actual fracture and sliding. The process zone expands on both sides, while
full decohesion advances on the left. No free direction, bulk phi evolution,
opening, MPI or production/BP3 propagation is tested.

## Compact numerical data

| File | Meaning |
|---|---|
| `parameters.csv` | Unchanged material calibration with units |
| `verification_summary.json` | Measured errors and explicitly retained failed criteria |
| `comparison_table.csv` | One table at common t=2.42 s; Δx, Δy and physical inventories |
| `physical_histories.csv` | Every accepted time; displacement, growth, fronts, reaction, separate storage/loss/work, curvature and absolute/signed energy remainders |
| `comparison.csv` | Common-time spatial and temporal differences; growth-relative norms, not normalization by initial seed |
| `profiles.csv` | Gauss3 histories at selected times; x in m, quadrature weights in m, slip in m, rate in m/s, tractions in Pa, dimensionless Theta |
| `animation_frames.csv` | Actual exported finest states used by the GIF; time and imposed displacement included |
| `bulk_fields.csv` | Gauss6-weighted cell-mean full plane-strain stress and pressure; coordinates/areas and maximum in-plane shear sampled within each cell |
| `bulk_field_summary.json` | All-Gauss6 maxima and remote region x<1 or x>7 at two saved times |
| `qualification_summary.json` | Bitwise transfer, independent weak feasibility, cache identity, buffer and actual restart comparisons |
| `growth_summary.json` | Event brackets, accuracy/work metrics and native checks; the coarse case ends at its buffer guard |
| `curvature_mode.csv`, `curvature_diagnosis.json` | Negative unrestricted release mode, intact constraints and separate bulk/bond/contact/pressure contributions |
| `specimen_estimate.csv/json` | Elastic selection estimate using approximate constant friction; **not** the actual RSF trajectory |
| `restart_rejection.json`, `legacy_regression.json`, `preservation.json` | Rejected changed configuration/checkpoint hashes, prior-front identity and protected files |
| `runs-*.json`, `*-run.json` | Bounded commands/configurations and measured times |

The same .125 m physical box kernel, weighted by actual Gauss measures, defines
the cross-mesh L2 comparison. Thresholds are a≥.01/.05/.5 and exactly a=1.
Gauss fronts are sampled locations and carry subcell uncertainty. Continuous
`jump_threshold` fronts instead solve the actual Q1 jump δ=2Gc/peak; new-material
nodal rates are nonnegative in every retained step, so this reconstruction follows
the irreversible scalar law. It does not interpolate RSF age. The exact continuous
fracture/bond integrals split each Q1 cell at that threshold and retain the original
native Gauss3 inventory separately. Full-cell lengths require both jump endpoints
to exceed the threshold. Reaction is per out-of-plane length (N/m); work/storage
are J/m. The initial 3 J/m seed fracture inventory is excluded from added fracture.

The no-extension and different tied-buffer controls use the small 1 m × .125 m
strip; their coordinate scale must not be confused with the 8 m growth specimen.
Before/after insertion field CSVs and lifecycle/derivative/operator checks remain
in their case directories. Intact multipliers are nonunique and are not physical
stress histories. Gauss entries flagged `installed=0` are virgin-material/virtual
candidate diagnostics; their placeholder cohesive reaction is not a physical
traction. No phase-growth unknown is silently clamped outside an installed record.

## Reproduce and archive boundaries

Use the existing qualified host `build-particle-free/coulomb-native/aspect-release`
and build the existing `cohesive_interface_native` CMake target into
`build-particle-free/cohesive-interface-insertion/plugin-final` using the native
Aspect_DIR. See the report and `run_cohesive_interface_insertion.py` for selected
stages. A complete replay sequence is:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-insertion/plugin-final -DAspect_DIR="$PWD/build-particle-free/native"
cmake --build build-particle-free/cohesive-interface-insertion/plugin-final --target cohesive_interface_native -j2
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_insertion.py qualification
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_insertion.py final-checks
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_insertion.py buffers
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_insertion.py
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_insertion.py growth
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_insertion.py focused
python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_insertion.py
python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_insertion.py fields
python3 benchmarks/reconstructed_fault/particle_free/fields_cohesive_interface_insertion.py
python3 benchmarks/reconstructed_fault/particle_free/diagnose_cohesive_interface_insertion.py
python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_insertion.py
```

Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for all commands. The `growth`
stage explicitly expects the coarse buffer guard; a different failure is rejected.
`focused` runs the two fine controls to 2.42 s. Runs are serial/sequential, not server jobs. Environment settings are
fixture-only; production parameters/defaults are unchanged.

The ZIP is a source/evidence overlay, not an ASPECT distribution. Full solver
outputs, restart owners, long Gauss streams and full Gauss6 stress samples remain
under `build-particle-free/cohesive-interface-insertion/runs/`; compact exports
live here. `provenance.json` hashes those retained files and the source inputs.
The stopped coarse buffer case and the intentionally stopped uncached fine run
are preserved. No old milestone archive is overwritten.
