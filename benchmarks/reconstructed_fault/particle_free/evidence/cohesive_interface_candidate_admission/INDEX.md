# Temporary candidates and atomic formation evidence

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_CANDIDATE_ADMISSION_REPORT.md) ·
[equations/ownership](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_CANDIDATE_ADMISSION_FORMULATION.md) ·
[declared controls](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_CANDIDATE_ADMISSION_SELECTION.md).

The native solver advances the loaded Maxwell/cohesive–RSF state under the
existing gradual physical loading turn. Temporary geometry carries zero
interface history; the generally nonzero bulk tensor is preserved. Directions
become persistent only when an accepted joint step creates material history
and accounts for cohesive/fracture energy. This qualifies bounded early
formation, not developed adaptive propagation or point-stress accuracy.

## Pictures

- [formation.png](formation.png): common t=.425 s, equal physical scales. Solid blue is persistent material, dashed grey is a zero-history candidate. Coloured dots are actual Gauss damage; purple arrows are signed slip. More grey tail does not create more material. No new a=1 advance occurs.
- [response-energy.png](response-energy.png): actual damage, accepted preferred directions/formation selections, absolute physical-work screen and total fracture inventory on material added after .410 s. Metadata adds no extra fracture charge. Coarse and coarse-cadence energy failures remain visible.
- [profiles.png](profiles.png): owned Gauss damage/slip and the current **variational** normal pressure; no smoothing or conversion to a point-traction reference.
- [formation.gif](formation.gif): both temporary extents at identical actual accepted clocks, including formation events. No history interpolation. Physical quantities coincide within roundoff while the dashed reserve differs.

## Data definitions

Units: time s, position/slip m, velocity m/s, stresses Pa, energy J/m per unit
out-of-plane thickness. Damage a and normalized age theta are dimensionless.
Integrated slip has units m². Counterclockwise tangent/preference angles are
degrees from +x; imposed alpha rotates the boundary strain axes relative to E0.

- `comparison.csv`, `uncertainty.csv`: measured growth (relative to each incoming physical state), integral/front comparisons, publication counts, energy screens and wall time. Cadence comparison is a .420–.425 s suffix; the other cases cover .410–.425 s.
- `*-history.csv`, `*-summary.json`: every actual physical step, native residual/contact/tangent, work, storage, dissipation, history and geometry diagnostics. `selected_preference_deg` is the incoming accepted-stress preference; new cells must use that incoming clock. `accepted_preferred_deg` samples the resulting formed endpoint.
- `admission_fracture_J_m` in those histories is the first physical damage increment on cells admitted in that step. It is **not** their total later fracture work, and tends to zero on timestep refinement. `new-material-storage.csv` gives total fracture/cohesive inventories on all material added after .410 s, including subsequent increments. No second registration charge is added.
- `surface-profiles.csv`: native coordinates, physical Gauss weights, history, velocity and variational normal reaction, with an explicit `persistent` flag. Zero temporary rows are not material records. Damage a=.05/.5/1 fronts use actual irreversible Gauss brackets, not current-slip inversion. Identical brackets do not establish sub-grid accuracy.
- `weak-loads.csv`: work-conjugate regional load tests on the persistent prefix. Node values are `exp(-.5*((x-centre)/sigma)^2)` at formed internal nodes, zero thereafter. Shear is `-(Kc)_jump.v / integral(Qv ds)`; normal perturbation uses that same physical test. The 2/8 controls have identical formed geometry/tests.
- `reserve-independence.json`: all six persistent histories, IDs, full represented stress/pressure and geometry comparison. Tensor norms use xx²+yy²+zz²+2xy² of the difference; pressure stays separate. Full Cauchy differences include the isotropic pressure difference.
- `inactive-candidates.csv`: actual zero-time replacement in the nonzero loaded bulk state, changed extent and a 15° changed cached direction, plus positive-time inactive controls. Any nonzero reference/memory, even 1e−30, rejects retirement.
- `initial-qualification.json`, `active-lifecycle.json`: independent constraint-space/derivative checks, actual native contact rejection, stale-direction predictor rejection, retry, duplicate and zero-cell commit rejection, loaded restart and changed-cache controls. Changed-cache tests force the same fresh selection; arbitrary changes to a stale numerical predictor between refresh clocks are not presumed exact.
- `promotion-inventory.json`: native whole-view energy equals the persistent inventory after physical admission; zero contribution remains in the temporary tail. The roundoff bound for two positive dot products is `2 gamma_(2N) sum|x_i y_i|`, gamma_k=k eps/(1−k eps), not a fitted tolerance.
- `overloaded-joint-admission.json`: the original pure-tied guard still rejects; positive-time joint admission accounts separately for new cohesive storage, fracture and friction/radiation work. Endpoint surface remainders and full physical-work uncertainty are retained.
- `candidate-completion.json`: a controlled .4165→.4175 s trial enlarges a one-cell temporary view and retries from the original history. Its physical endpoint equals direct two-cell padding. This is a numerical completion test, not a timestep-accuracy run.
- `sampling.json`: formed-endpoint R-disk fixed-Gauss8/12 comparison, independent of unused candidate cut partitions; not tip point-stress qualification.
- `onset-cadence.json`: the same cadence halving applied over .410–.41675 s, covering the first two admissions. Both clocks, formed geometry and all histories match; brackets are algorithmic resolution, not exact physical initiation times.
- `default-identity.json`: eight default native operator/trace/probe tables match the retained baseline bitwise.
- `physical-energy-gates.json`, `publication-accounting.csv`, `verification.json`, `native-cost.csv`: explicit scientific pass/fail screens, engineering invariants and measured setup cost. Engineering success does not mean the coarse physical-energy or historical accuracy failures passed.
- `native-changes.diff`, `entry-protection.json`, `manifest.json`: actual entry source diff, preservation hashes, final source/data/binary checksums.

## Ownership and raw archives

Full arrays/operators/checkpoints remain under
`build-particle-free/cohesive-interface-candidate-admission/`. A state carries a
computational `geometry` view, with only the `formed_arcs` prefix persistent.
`arc_metadata` marks candidate versus formed records. `record_ids` contains
**only persistent** Gauss identities. `formed_generation` and `publications`
count material transactions; `generation`/`candidate_revision` invalidate view
caches; `physical_publications` counts physical accepted steps once.

Each run stores initial/final/last-accepted JSON, physical snapshots, input,
and before/after-formation checkpoints. Native kernel folders contain the
exact geometry, K/Q/normal operators, physical response-column hashes,
compressed tensor-field bases, fixed endpoint samplers and native metadata.
The stale predictor and fresh accepted state are saved separately. The legacy
tied-rejection control operates on a PRIVATE COPY so it cannot overwrite
previous failure evidence. Legacy wholly virgin committed pieces are not
automatically converted or stripped of numerical history.

## Sources and commands

Existing experimental files changed: `cohesive_interface_evolution.h` (optional
exact Q1 masters), `cohesive_interface_turn_solver.cc` (opt-in C API),
`cohesive_interface_path.cc` (optional candidate-independent endpoint samplers),
and `cohesive_interface_load_turn.py` (optional library path only).
The new owner is `cohesive_interface_candidate_admission.py`, with `qualify_`,
`check_`, `analyze_` and `verify_` scripts. No constitutive source, production,
BP3, or CMake behavior changes in this task.

From the worktree root, use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for all
calculations; the reported trajectory timing runs were sequential. Native configuration:

```sh
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-candidate-admission/build -DAspect_DIR=$PWD/build-particle-free/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/openmpi/5.0.6/bin/mpic++ -Ddeal.II_DIR=/opt/dealii/9.6-local/lib/cmake/deal.II
cmake --build build-particle-free/cohesive-interface-candidate-admission/build --target cohesive_interface_path cohesive_interface_turn_solver -j2
python3 benchmarks/reconstructed_fault/particle_free/qualify_cohesive_interface_candidate_admission.py
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_candidate_admission.py candidates2
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_candidate_admission.py candidates8 --cells 8
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_candidate_admission.py half-step --dt .000125
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_candidate_admission.py cadence-half --cadence .00125 --checkpoint build-particle-free/cohesive-interface-candidate-admission/candidates2/verification-prefix.json
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_candidate_admission.py cadence-onset --cadence .00125 --end .41675
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_candidate_admission.py inactive
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_candidate_admission.py sampling
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_candidate_admission.py defaults
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_candidate_admission.py loaded
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_candidate_admission.py lifecycle
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_candidate_admission.py completion
python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_candidate_admission.py
python3 benchmarks/reconstructed_fault/particle_free/verify_cohesive_interface_candidate_admission.py
```

Run labels deliberately refuse overwrite. The suffix checkpoint is the own
accepted .420 s member of `candidates2/snapshots.json`, copied without changing
any physical data. A new rerun needs fresh labels/paths; existing evidence is
preserved. The initial projected-space diagnostic had an integer-versus-Boolean
indexing error in the test script; its log is retained and the corrected test
checks the actual tied rows, without changing any numerical tolerance.
