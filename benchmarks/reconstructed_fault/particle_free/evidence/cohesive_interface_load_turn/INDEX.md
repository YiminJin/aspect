# Boundary-driven causal extension evidence

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_LOAD_TURN_REPORT.md) ·
[declared experiment and amendments](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_LOAD_TURN_SELECTION.md).

Physical boundary strain turns gradually after growth begins. Native mixed
mechanics and the unchanged Maxwell/cohesive–RSF updates calculate subsequent
stress, damage and slip; existing geometry/histories are not rotated. The plots
show a stress-responsive damaged process zone but significant reserve/scheduling
dependence, retained birth failures and limited developed-front evidence.

## Figures

- [specimen.png](specimen.png): physical boundaries, old seed and separate R/w/S definitions.
- [geometry.png](geometry.png): equal-scale actual Gauss damage, signed slip and a=.05/.5/1 fronts at common .440 s. Grey records are tied; red rings are saturated.
- [reserve.png](reserve.png): actual tangent, damage and birth time at common .44425 s. Solid points require a≥.05; dashed curves include tied reserve. Direction commitment precedes damage.
- [response.png](response.png): imposed loading turn, accepted sampled preference, developed-front tangent and physical histories. Dots mark actual append decisions.
- [loads.png](loads.png): common-field weak shear/normal and raw sided-mean point shear. Displayed lumped shear is a visualization; quantitative work uses common physical virtual-jump tests.
- [energy.png](energy.png): independent physical work/loss and sum of absolute per-step remainders for fresh trajectories. Different endpoints are explicit; only turned-large completes hold.
- [evolution.gif](evolution.gif): equal-aspect accepted frames, damage and signed slip, sampled direction and developed front. Common physical times from short native replays of each own accepted state; all frames reproduce the original damage integral within 1e−11 m. No interpolated histories. Frame clocks are shown.

## Compact data and conventions

All lengths m, times s, stresses Pa, velocities m/s and energies J/m per unit
out-of-plane thickness. Damage a is dimensionless. Integrated slip is m².
Angles are in degrees counterclockwise from +x; imposed alpha rotates the
strain axes relative to E0. It is not an absolute crack or flow angle.

- `comparison.csv`, `common-times.csv`, `uncertainty.csv`: endpoints, stopping causes and common-time absolute/relative comparisons. Growth-since-turn uses each trajectory's actual .410 s field; restart suffixes reuse that exact accepted field.
- `births.csv`, `segment-response.csv`, `common-segments.csv`: committed birth time/preference/tangent and subsequent **actual damage**, with Gauss-weighted saturated measure rather than point counts. Common-segments uses .44425 s; segment-response uses each run's labeled endpoint.
- `common-shape.csv`: physical arclength, coordinates, birth clock, tangent, damage and slip at .44425 s. A short native replay from each own accepted state reproduces the original damage integral within 1e−11 m. No geometry insertion occurs in that replay window.
- `surface-history.csv`, `sampled-stress.csv`: selected accepted Gauss records and full plane-strain tensor samples. Physical deviatoric stress is tau; sigma=tau−(10000+pprime)Id. Principal directions are unchanged by that isotropic subtraction.
- `*-history.csv` / `*-partial.csv`: actual per-step clocks, two imposed coefficients, two affine Maxwell memories, storage, friction/radiation/Maxwell losses, physical work, exact BE ledger, weak residuals and front brackets. Partial means stopped, not successful completion. Half-step and half-quantum rows cover a suffix beginning .410 s; their energy sums exclude the common prefix.
- Reported fronts use `*_front_lower/upper_*` and `*_detached_Gauss` from owned irreversible damage. `new_damage_trace_identity` and the older `*_detached_cells` diagnostics are retained for traceability; current-slip reconstruction is not authoritative under unloading. Equal Gauss brackets do not resolve sub-grid front differences.
- `frozen-response.csv`, `frozen-profiles.csv`, `frozen-state.json`: same physical Gamma/jump/Maxwell field independently reconstructed in four bulk spaces. The R=.75 m stress disk is centred R ahead of the geometric endpoint, weighted by `(1-r²/R²)²`, using native physical quadrature. It does not test tip point stress.
- Frozen load tests are common physical Q1 virtual jumps `v_i=exp(-.5*((x_i-c)/sigma)^2)` on the same Gamma. Centres 4.5/6.4/7.15/7.65 m, sigma .55/.25/.3/.2 m. Shear is `-(Kc)_jump.v / integral(Qv ds)`; normal perturbation is weighted with the same `Qv`. Full-tensor error uses xx²+yy²+zz²+2xy² of the **difference**. Pressure perturbations remain separate.
- `initial-qualification.json`, `homogeneous-loading.csv`, `loaded-lifecycle.json`, `publication-accounting.csv`, `verification.json`: mathematical/native identity and lifecycle evidence; engineering checks do not assert that failed scientific gates passed.
- `native-cost.csv`: exact native mesh/master counts and kernel timings, including rejected proposals. Cache reuse prevents interpreting trajectory times as a performance scaling study.
- `native-changes.diff`, `entry-protection.json`, `manifest.json`: experimental native diff, task-entry preservation hashes and final checksums.

## Inputs, raw archives and commands

Full solver archives stay separately under
`build-particle-free/cohesive-interface-load-turn/`: each run has `run-input.json`,
initial/accepted JSON, snapshots, geometry events and preserved failures; kernels
contain input geometry, all operator tables, compressed physical field bases and
native metadata. Rejected-birth JSON stores both the proposed representation and
unchanged incoming accepted state. No archive conversion is performed.

Native build used Release, `/opt/openmpi/5.0.6/bin/mpic++`,
deal.II `/opt/dealii/9.6-local/lib/cmake/deal.II`:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-load-turn/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/openmpi/5.0.6/bin/mpic++ -Ddeal.II_DIR=/opt/dealii/9.6-local/lib/cmake/deal.II
cmake --build build-particle-free/cohesive-interface-load-turn/build --target cohesive_interface_path cohesive_interface_turn_solver -j2
```

All calculations use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; runs were sequential.
The driver is `benchmarks/reconstructed_fault/particle_free/cohesive_interface_load_turn.py`:

```sh
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_load_turn.py turn-large64 --reserve .75
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_load_turn.py turn-small64-reproduced
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_load_turn.py constant-small64 --control
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_load_turn.py constant-large64 --control --reserve .75
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_load_turn.py turn-small96 --n 96
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_load_turn.py turn-small-half64 --checkpoint build-particle-free/cohesive-interface-load-turn/turn-small64-reproduced/turn-start.json --dt .000125
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_load_turn.py turn-small-quantum64 --checkpoint build-particle-free/cohesive-interface-load-turn/turn-small64-reproduced/turn-start.json --extension .1875
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_load_turn.py lifecycle turn-small64-reproduced
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_load_turn.py frozen turn-small64-reproduced
python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_load_turn.py
python3 benchmarks/reconstructed_fault/particle_free/verify_cohesive_interface_load_turn.py
```

Existing run directories are deliberately protected from overwrite. Use new
labels for reruns; the stored own-prefix checkpoint is also contained in
`snapshots.json` at exactly .410 s. No production executable or BP3 input is changed.

## Changed files

Only the existing experimental native files `cohesive_interface_path.cc`,
`cohesive_interface_evolution.h` and `cohesive_interface_turn_solver.cc` change.
New experiment code is `cohesive_interface_load_turn.py` and the corresponding
`qualify_`, `check_`, `plot_` and `verify_` scripts in the same benchmark directory.
New report/selection files and prepended STATUS/SOURCE_AUDIT entries describe the
results. This evidence directory contains the compact deliverables; manifests
provide exact paths and hashes. No CMake or production/BP3 change is part of this task.
