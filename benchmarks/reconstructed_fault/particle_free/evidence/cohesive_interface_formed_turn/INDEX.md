# Short full loading turn / future-material refinement

[Report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_FORMED_TURN_REPORT.md) ·
[predeclared controls](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_FORMED_TURN_SELECTION.md) ·
[retained admission equations](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_CANDIDATE_ADMISSION_FORMULATION.md).

[Compact review package](../cohesive_interface_formed_turn.zip) contains these
figures/data, the report/selection and four new scripts, using worktree-relative
paths. Existing native binaries and raw solver archives remain separate.

Both cases resume exactly the same verified .425 s physical state. That state's
formed geometry and histories are preserved. Only **future** material-cell
spacing differs, .09375 versus .046875 m. Together with the reused common
.410–.425 s history, they complete the physical loading turn at .440 s. This
is not a refinement of the already formed prefix, bulk mesh or phase length.

## Figures

- `geometry.png`, `turn.gif`: equal physical scales, actual accepted Gauss
  damage and signed slip vectors (×100 for readability). Original incoming
  material is grey, newly formed material blue, zero-history temporary geometry
  dashed. Dotted line marks the common incoming formed endpoint. Triangles,
  diamonds and red squares mark connected a=.05, .5 and 1 fronts, respectively.
  A newly drawn tail is not a crack advance. Frames use common actual accepted
  clocks without interpolating histories.
- `response.png`: incremental growth, process/developed fronts, accepted stress
  preference versus actual front tangent, physical work/loss and the unchanged
  1% energy screen. Angles describe different quantities; a formed direction
  cannot rotate to follow a later accepted stress.
- `profiles.png`: actual native Gauss damage, signed jump and **variational**
  normal contact pressure. Lines only connect plotted samples; no constitutive
  history is interpolated into another state.
- `physical-stress.png`: represented physical shear stress at bulk-cell centers
  on common scales. This diagnostic does not close the earlier point-traction
  accuracy failure.
- `frozen-contact.png`: independent Cartesian64/fitted64/fitted96 bulk
  reconstructions of exactly the same physical jump/Maxwell field. Raw sided
  tractions remain visible and are distinct from work-conjugate normal reaction.

## Indexed data and definitions

Coordinates/jump in m, time in s, rate in m/s, pressure/stress in Pa, energy in
J/m per unit out-of-plane thickness. Integrated signed jump has units m²;
integrated damage has units m. Angles are counterclockwise from physical +x.
Boundary α rotates the prescribed affine strain axes; it is not a fault angle.

- `coarse/fine-input.json`: source checksum, spacings, physical/numerical lengths
  and timestep. `*-history.csv`, `*-summary.json`: every accepted physical step,
  full native residual/contact certificates, energy/work, selection, transfer
  and publication diagnostics. `comparison.csv` separates old-material damage
  increments from damage on material admitted after .425 s.
- `spatial-difference.json`, `common-time-differences.csv`: absolute and relative
  physical response differences at matching times. Relative growth uses the
  measured increment beyond the common initial state, not the large seed.
  `comparison.csv` also reports damage/slip restricted to material newly formed
  after .425 s. `new-geometry-difference.csv` compares actual curves at common x;
  this is geometry evaluation only, not history interpolation.
- `surface-profiles.csv`: actual physical Gauss coordinates, weights, persistent
  flag, damage, jump, velocity and normal reaction; no synthetic common history.
  Front brackets use connected irreversible Gauss records. Matching brackets
  do not establish sub-Gauss position accuracy.
- `formation-events.csv`: actual incoming selection clock, committed endpoint,
  tangent, preferred direction and zero-time transfer defects. Whole new Q1
  cells commit; smaller cells change both future jump resolution and formation
  granularity, including the cellwise normal-work test. This is not a pure
  metadata-only perturbation.
- `physical-stress-centers.csv`: full embedded 3D deviatoric components and
  pressure perturbation at actual cell centers. Total Cauchy stress is
  sigma=tau−(10000 Pa+p′)Id; tau_xy is physical shear stress.
- `temporal-difference.json`, `temporal-half-*`: same own incoming .4375 s state,
  late .4375–.440 s suffix, one timestep halving. Compare suffix increments;
  this is not a full-trajectory temporal error estimate.
- `frozen-input.json`, `frozen-tensors.csv`, `frozen-loads.csv`,
  `frozen-profiles.csv`: common physical surface field and independently solved
  bulk approximations. The same Gaussian physical Q1 virtual jumps are used in
  all three bulk spaces. Shear is −(Kc)_jump·v / ∫Qv ds, and normal perturbation
  is ∫Qv (Nc) ds / ∫Qv ds. Centers/widths/masses are exported. R-disk tensor norms
  include zz and both xy contributions; physical confinement is not shifted.
  No diagnostic reconstruction is published as accepted history.
- `physical-*-region*.json`: native independent higher-order physical integrals
  of bulk storage, perturbation storage, pressure, divergence and pressure work.
- `frozen-assessment.json`, `frozen-reference-pair.csv`: sensitivity of the weak
  loads to fitted64/fitted96 reference uncertainty. The bend normal difference
  is 1.94% against fitted96 but 2.006% against fitted64: the retained 2% target
  is not robustly met, even though direction sampling and shear are supported.
- `*-lifecycle.json`: actual native contact rejection, immutable input,
  deterministic retry, duplicate prevention, coupled FD and rebuilt-context
  JSON restart. This is an experimental serial owner, not Simulator/MPI restart.
- `accuracy-gates.json`, `publication-accounting.csv`, `verification.json`:
  explicit scientific screens separately from ownership/engineering invariants.
  Earlier raw traction/ahead-normal failures remain open regardless of new gates.
- `native-cost.csv`, `entry-protection.json`, `manifest.json`: measured native
  setup time, protected-file hashes and final source/data/binary identities.

## Reproduction and raw checkpoints

Run from the worktree root with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
The new driver reuses the already qualified native executable/shared library
through `build-particle-free/cohesive-interface-formed-turn/build`; no native
source, production default or BP3 fixture changes. Each label refuses overwrite.
On a fresh reproduction directory, create the work/output directories and link
its `build` to the qualified `cohesive-interface-candidate-admission/build`;
retain the original source checkpoint and its referenced native kernel.

```sh
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_formed_turn.py coarse
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_formed_turn.py fine
python3 benchmarks/reconstructed_fault/particle_free/cohesive_interface_formed_turn.py coarse --label temporal-half --dt .0000625 --checkpoint build-particle-free/cohesive-interface-formed-turn/temporal-prefix.json
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_formed_turn.py lifecycle --mode coarse
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_formed_turn.py lifecycle --mode fine
python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_formed_turn.py frozen
python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_formed_turn.py
python3 benchmarks/reconstructed_fault/particle_free/verify_cohesive_interface_formed_turn.py
```

Raw operators, physical snapshots, before/after-admission checkpoints, inputs
and last accepted states remain under
`build-particle-free/cohesive-interface-formed-turn/`. `temporal-prefix.json`
is the unmodified .4375 s member of the coarse run's own snapshots. Preserve
these together with the original .425 s checkpoint for further continuation.
Reproduction of a new case requires a fresh label/path; this evidence is retained.

To fit the declared disk budget, only newly generated **unused cell-mean
diagnostic bases** are omitted after recording their exact hashes in each
kernel's `omitted-mean-diagnostic.json`. Center bases, all bulk/surface operators,
physical histories and inputs remain. Repeat that kernel's native export into
a new folder to recover its means. No old file/archive or history was deleted.
