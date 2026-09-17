# Modified BP3 long-run preparation

This records the original 200-km preparation. The subsequent
[cleanup and 300-km qualification](stage_K5_long_run_cleanup.md) supersedes
its current fixture/log/output choices without replacing this evidence.

Preparation and bounded verification complete. **No long run launched.**
The new fixture is qualified for a user-launched exploratory run on the
tested four-rank Release setup, not for first-event/recurrence accuracy.

## Deliberate scope

The 200 x 100-km research reference and its successful seven-step evidence are
preserved. The separately named long-run fixture removes junction-specific
bulk/fault refinement, uses adaptive convection/RSF timesteps, and keeps mature
C=0, the work measure, paired endpoints, immutable effective background,
split aging, GMG velocity preconditioning and all residual criteria unchanged.
The old executable, plugin, source diff and reference PRMs are preserved in
`long-run-preparation/reference/` before the particle layout change.

The spatial strengthening property is now supplied by selected-field refresh
in `initial composition`, not an extra BP3 particle plugin. In this branch
the property name comes from the explicit composition-to-property mapping:
`strengthening:initial strengthening[0]`. Theta still maps to `phase field fault
state[0]`; Maxwell components still map to `maxwell stress[0..2]`. Only the
strengthening slot refreshes at current particle positions after advection.
The initial-composition manager is retained by shared ownership for that
purpose. Empty refresh selection preserves the ordinary plugin behavior.
New checkpoints are required; no old particle-layout conversion is advertised.

## New fixture

`prepare_long_fixture.py` reuses the saved ordinary central tree from before
the junction bisection, applies the validated wide-box root mapping and retains
the graded side strips. It removes 6774 cells: 36194 remain. Removing only the
80 inserted midpoints leaves 1156 fault nodes, with approximately 100-m spacing.
All fault-crossed axis-aligned cells have width 97.65625 m, including 15–18 km
and 36–44 km. Ell=400 m gives ell/h=4.096, not a nucleation/peak-rate accuracy
qualification. Actual exported geometry confirms these target statistics.

The immutable rational prestress coefficients are restricted to retained
physical vertices, not recalibrated against a new mechanical solution. Their
denominator is part of the *background input*, not replaced by the new current
Ih. This is the same captured-initialization procedure with a coarser Q1
representation; its mesh-related initialization changes are reported separately.
The physical endpoint meshes are exactly unchanged within 2 km of both ends.
Completion rows are regenerated at new surface profile IDs/origins; the 28
nonzero endpoint integrals have unchanged origins and values. Only rigorously
interior merged-element profiles acquire new zero-completion rows. Inputs and
hashes are maintained in `fixtures/modified_bp3_long_run/`.

## Output and restart contract

The BP3 observer runs before native visualization/particles/reconstructed-fault
writers. A final dependent observer publishes the heavy-output slip reference
only after those writers return successfully. Background-thread output is off.
No particle evolution, statistics or history audit is gated. Heavy thresholds
are 0.1 m maximum *per-node change since output*, or one physical year; light
profiles use 0.01 m or 0.1 year and first onset/end/final state. Time triggers
can be disabled with zero. Initial and gracefully final accepted states are
always written. No thresholds constrain dt, and crossing multiple thresholds
produces one actual accepted state, never interpolated/fabricated states.

Native bulk `tau_xx`, `tau_yy`, `tau_xy` are **retained FE Maxwell-history
inputs**, not accepted current stress. Native particle `maxwell stress` is
the accepted committed tensor (supplied zero history at timestep zero).
Native bulk pressure is the physical stress perturbation Delta p. Fault CSV
q/sigma are consistent Q1 projections of accepted current **work-weighted**
tractions. This task does not add a second Maxwell evaluation for visualization.

Checkpoint interval is 1800 wall seconds, independently of graphics; per-step
checkpointing is disabled outside verification. This local branch hard-codes
three retained slots, so no unsupported count parameter is declared. Native
fault PVD history, cumulative slip, schedule clocks/references and event state
are serialized. Each checkpoint records its actual last accepted step/time
and a small metadata snapshot; it is not assumed to be the preceding step.
No automatic event-peak directory copies occur in the long-run mode.

Restart always creates a **new output branch**, copying the old directory and
restoring the selected checkpoint's CSV/PVD/Visit metadata prefix. Prior-run
evidence is untouched. This simple policy requires disk space for a copy;
unindexed later payloads may remain but are not interpreted as accepted output
of the resumed branch. Binary, plugin and physical fixture/configuration hashes
must match; rank count must match. Changing end time, wall budget and accepted
step cap is allowed. Reconstructed caches are rebuilt.

The explicitly authorized material fix recomputes normalization once after
restart to validate the inputs, requires frozen mature phase, fixed geometry,
composition-independent degradation and identical current/old phase values,
and compares cold versus persisted Ih within the existing quadrature/tail
budget. Only then does it restore persisted previous-Ih as current Ih, preserving
the exact frozen-history invariant. No previous history is overwritten with
new rounded values. Launcher hashes additionally reject changed physical inputs.

## Verification results

Evidence root: `benchmarks/reconstructed_fault/bp3/long-run-preparation/`.
All simulations below were four-rank Release/GMG, one thread per rank; no
physics or acceptance tolerances were changed. The new mesh has 1,286,054
bulk DoFs and 325,746 particles. No long simulation or parameter campaign ran.

| Case | Accepted states | Wall seconds | Peak child RSS KiB | Result |
|---|---:|---:|---:|---|
| `fresh-two` | 0,1,2 (adaptive) | 128.994 | 1,348,616 | convergence/history/native output pass |
| `prefix-one` | 0,1 (adaptive) | 58.651 | 1,336,752 | new-layout checkpoint pass |
| `resumed-two` | 2 from prefix checkpoint 02 | 97.674 | 1,379,916 | **all compared fields bitwise identical** |
| `same-mesh-one` | 0,1 (old mesh/saved clock) | 84.211 | 1,578,708 | **all compared fields bitwise identical** to `wide-gmg-seven-local4` |

Peak child RSS is not summed MPI job memory. The initial continuous reference
was built before adding the compact residual/history columns to
`accepted_steps.csv`; that intervening plugin edit changes only summary
diagnostics, not mechanics. The checkpoint prefix and resumed process use
the identical final plugin/binary. Source diffs and executable hashes are
captured separately for every launch. Strict numerical comparison against
the uninterrupted reference nevertheless gives exactly zero in every field.

The existing 1e-8 per-component field allowance is retained by
`check_long_run.py`; no near-zero stress denominator was replaced by the
50-MPa background. It checks bulk velocity/pressure/each compositional
component, stable-ID particle properties, committed stress/H, current
work-QP stress/source/strain, endpoint controls, weak loads and native fault
properties (Theta/C/Ih/background/slip). Initial/final geometry and frozen Ih
match exactly. Restart uses the identical adaptive next timestep, with no
reset or extra history update. Independent in-process history checks remain
mandatory before an accepted summary is appended.

The cold-Ih restart comparison was 4.44089e-16 relative; validated restoration
then preserves the persisted values exactly. Reconstructed geometry is
checked against the immutable input; no caches are deserialized. Launcher
hash/rank validation prevents using changed physical inputs or old layouts.
Changed-geometry/evolving-phase and cross-rank restart are **not qualified**.

| Adaptive accepted step | Physical time [s] | dt [s] | Newton / total Krylov | Final max normalized residual | Surface RMS [Pa] | max(dt V)/Dc |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 (artificial Maxwell interval separately 4e6) | 1 / 35 | 3.25784e-9 | 0.00451633 | 0 |
| 1 | 2666075.0400048387 | 2666075.0400048387 | 1 / 33 | 5.45075e-10 | 0.000180996 | 0.333312 |
| 2 | 5332320.5118842227 | 2666245.471879384 | 13 / 220 | 5.77371e-10 | 0.000905883 | 0.333331 |

All returned directions pass their fresh linear-residual check. At step 2 the
final fresh residual is 4.31731579e-13 against target 4.68846654e-13. Theta
reference error is 2.22044605e-16 on both real steps. The first Maxwell update
error is 3.06884e-8 Pa against a 4548.67-Pa scale, with zero incoming FE stress.
All 1156 nodes remain free. Step 2 develops min V=2.08698e-13 m/s and needs
13 Newton updates; this is retained as trajectory behavior, not hidden by
output changes or promoted into a long-time accuracy claim.

With deliberately small verification thresholds, continuous output is heavy
at steps 0/2 and lightweight-only at 1. Gracefully stopping the prefix at 1
forces its heavy state; the resumed branch then contains heavy 0/1/2.
Bulk/particle/fault PVD times agree, file targets exist, and profile indices
have unique ordered physical times. Native particle arrays include the
selected `initial strengthening`, initial state seed and committed Maxwell
components; native fault arrays include evolving Theta and cumulative slip.
The restart preserves numbering and previous slip references. The first
trial used both checkpoint time and step triggers; because the local time
trigger takes precedence, it produced only a final checkpoint. The explicit
step-based **test** now uses wall interval zero. Production remains 1800
wall seconds and zero step interval.

### Focused checks

- Output schedule standalone test: moving maximum location, threshold
  crossing, noncommitting/rejected candidate, final/time trigger, time disabled,
  and archive round-trip all pass.
- Actual selected-field plugin: initialization and deliberately displaced
  transition fractions 0.25/0.75 pass on four ranks; all unrelated properties
  are exactly retained and the probe restores particle location/properties.
- `particle_interpolator_cell_average`: passed (25.04 s).
- `particle_output_gnuplot`: expected-output comparison **failed** (9.30 s).
  This is pre-existing: the checked-in expectation includes `initial C_1`,
  while the branch's existing default selects particle-advected fields only
  and that fixture's composition is not particle-advected. Running the
  preserved pre-task binary reproduces the same missing field; all four new
  and old particle output files are identical after removing generation time.
  The expectation was not refreshed or the field-selection meaning changed.
- Core and plugin built with `-j4`; Python compilation, generated launcher
  syntax, plot from recorded profiles, parameter validation and scoped
  `git diff --check` pass. Full ASPECT suite not run.

### Mesh change versus refactoring

Actual export has 36,194 cells and 1615 fault-crossed cells. Crossed square
side is 97.65625 m in the top/bottom and 15–18, 35–37, 39.5–40.5 and
43–45 km windows; no special junction grading remains. The square's projected
extent in both tangent and normal directions is 133.400918 m; its centered
line chord in either direction is 112.763724 m. These geometric measures are
not additional grid spacings; the isotropic grid spacing used in ell/h is
97.65625 m. Fault elements are approximately 100 m. Far-field squares grade
through powers of two up to 25 km. This is not a resolution qualification.

On exactly retained physical fault nodes, new-mesh versus old-mesh initial
maximum differences are: V=5.12593e-13 m/s (0.05124% of old max V),
Theta=0, Q1 current weak shear=639.637 Pa, and Ih=274.599 m (2.11768% of
old maximum Ih). These mesh-dependent initialization changes are expected
to affect later trajectories and are **not** subtracted away. For the
explicitly lumped work measure, initial normal-traction change is at most
157.854 Pa, pressure 128.113 Pa and tau:N 53.3865 Pa. These are distinct
representations from the consistent-Q1 profile. In contrast, the fixed-mesh
property/output comparison has zero changes in every tested field.

The prior AMG-versus-GMG initial raw-stress discrepancy (up to 0.00110 Pa,
failing the old strict per-field comparison) remains documented in the GMG
report. No initial-stress tolerance was relaxed here and the new fixed-mesh
comparison uses GMG on both sides.

## Handoff and provenance

See `benchmarks/reconstructed_fault/bp3/LONG_RUN.md` for exact fresh/restart
commands, output definitions, optional event-relative plotting and resource
guidance. `run_long.py` prepares only by default; `--execute` is explicit.
`launch-preflight/` was generated and syntax-checked but **not executed**.
The old reference directory/configuration is unchanged apart from the
maintained field-selective particle interface; archived binaries/inputs
remain available for old investigations.

Base Git revision: `3335d3d26c298ff5aaeba77062b0a77c8d20f0b5`, plus preserved
working-tree changes. No commit/reset was performed. Qualified SHA-256:

- Release: `0b17a4d932f2ea6e96e372f6b908a9a24d4fe2517210ac709945709c77891d0b`
- Plugin: `c9fdb814959ff0f2953de32a1928561b0f9852202ce86ca169c388299dee7d21`

Input hashes are in `fixtures/modified_bp3_long_run/manifest.json` and every
`launch.json`. Exact verification commands (repository root):

```sh
python3 benchmarks/reconstructed_fault/bp3/run_long.py --output benchmarks/reconstructed_fault/bp3/long-run-preparation/prefix-one --purpose recurrence --end-years 1500 --wall-hours 0.5 --verify-through-step 1 --verify-output --execute
python3 benchmarks/reconstructed_fault/bp3/run_long.py --output benchmarks/reconstructed_fault/bp3/long-run-preparation/resumed-two --resume-from benchmarks/reconstructed_fault/bp3/long-run-preparation/prefix-one --checkpoint 2 --purpose recurrence --end-years 1500 --wall-hours 0.5 --verify-through-step 2 --verify-output --execute
python3 benchmarks/reconstructed_fault/bp3/check_long_run.py benchmarks/reconstructed_fault/bp3/long-run-preparation/fresh-two benchmarks/reconstructed_fault/bp3/long-run-preparation/resumed-two --steps 2
python3 benchmarks/reconstructed_fault/bp3/run_long.py --output benchmarks/reconstructed_fault/bp3/long-run-preparation/same-mesh-one --purpose recurrence --end-years 1500 --wall-hours 0.5 --verify-through-step 1 --verify-output --reference-mesh --saved-clock-check --execute
python3 benchmarks/reconstructed_fault/bp3/check_long_run.py benchmarks/reconstructed_fault/bp3/wide-gmg-seven-local4 benchmarks/reconstructed_fault/bp3/long-run-preparation/same-mesh-one --steps 0 1 --same-mesh --log fresh.log
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^(particle_interpolator_cell_average|particle_output_gnuplot)$' -j1
```

These paths already contain evidence; use new paths to repeat deliberately.
No automatic retries. Remaining limitations include fixed-mesh/four-rank
restart only, no earthquake/recurrence validation, unresolved coarse spatial
accuracy, and no new server scaling or broad solver/physics qualification.
