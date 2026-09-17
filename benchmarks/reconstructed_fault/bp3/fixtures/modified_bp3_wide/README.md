# Centered wide-box modified BP3

This is the fully frictional research fixture widened **laterally**, not
stretched: x=-50 to 150 km, y=0 to 100 km. Its center and the midpoint of the
60-degree through-box fault are both (50,50) km. The fault top/bottom points,
1236 vertices and their spacing are unchanged. The depth remains 100 km.

All 42,880 leaf cells inside the original box are preserved physically,
including the 48.828125-m minimum cell width. Each added 50-km lateral strip
uses 6.25-km cells in its first 12.5 km, 12.5-km cells in the next 12.5 km,
and 25-km cells in the final 25 km. Total: 42,968 cells. Added material has
the same constitutive model; coarsening changes mesh resolution, not material
properties. Ell remains 400 m, with the same particle density per cell.

`prepare_wide_fixture.py` maps the old physical quadtree onto the eight
50-km coarse roots (distributed Morton numbering) and adds the graded strips.
`target_cells.txt` is the resulting saved mesh; `manifest.json` records its
hash. The original fixture's fault, fixed prestress, endpoint-completion and
clock files are reused without content changes. Their physical near-fault
coordinates and transverse profile mesh remain unchanged. Surface properties
are initialized normally in a fresh run; this is not a state transfer from
the narrow box.

## Unchanged physics and clock

The parameter file includes `bp3_modified_fully_frictional.prm` and overrides
only Box width/origin/coarse repetitions, initial refinement counts and the
output path. It retains mature C=0, continuous Q1 V/Theta/slip, the corrected
work measure, both endpoint corrections, initial stress-perturbation procedure,
immutable background, frozen phase, true normal stress, and split aging.
The same rigid translations are applied at the **new** lateral boundaries;
top/bottom perturbation traction conditions and all solver tolerances remain
unchanged. The plugin now obtains side coordinates and source-continuation
box bounds from GeometryModel::Box instead of assuming x=0,100 km.

The launcher retains the artificial dt0=4e6 s and saved seven real steps to
132230424.76671731 s (4.190129 yr), with the same acceptance guards. Moving the
outer boundaries changes the finite-domain mechanical problem; no claim of
an identical trajectory is made.

From the repository root:

```sh
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3 -j4
python3 benchmarks/reconstructed_fault/bp3/run_research.py --configuration wide --prepare-only --output benchmarks/reconstructed_fault/bp3/wide-replay-prepared
```

Omit `--prepare-only` and choose a new output directory to run four ranks
with the same 2400-s cap and no automatic retry. Do not resume a narrow-box
checkpoint on this mesh. Restart/continuation qualification is not changed
by this geometry task.

## Verification performed

Plugin Release build passed. Four-rank preflight:

```sh
python3 benchmarks/reconstructed_fault/bp3/run_research.py --configuration wide --mesh-only --output benchmarks/reconstructed_fault/bp3/wide-box-mesh-local4-corrected
```

Passed in 17.19 s, reported peak child RSS 743708 KiB (not aggregate MPI
memory). Exported owned cells match the generated tree exactly: no additional
grading leaves, zero coordinate error, all original physical cells retained.
The actual reconstructed fault has 1236 vertices and zero coordinate error
against the unchanged fault input. See that directory's
`mesh_verification.json`, `mesh_cells_rank*.csv` and `run.log`.

The preflight deliberately raises `Intentional mesh-only stop before
mechanics`; ASPECT's nonzero exit is expected and checked by the launcher.
No linear/nonlinear convergence or history-update claim follows from it.
The earlier incorrect root-numbering preflight and MPI sandbox failure are
preserved in the other `wide-box-mesh-local4*` directories. Pre-edit plugin
sources/library and the rejected target mesh are in `wide-box-preserved/`.
No mechanical trajectory, restart test or broad test suite was run during
that mesh-only preparation task.

## Subsequent requested short comparison

`wide-seven-local4/` now contains the completed seven-step replay (721.060 s,
four ranks, 71 passing fresh linear checks). The saved narrow-box trajectory
was reused, not rerun. `compare_box_width.py` checks the common clock, split
state/slip updates, and native weak-traction lifecycle, and writes the full
0–40-km profiles and initialization-separated evolution differences.
See `doc/reconstructed_fault/bp3/stage_K5_wide_box_comparison.md` in the
repository for conclusions. The wide box is not established as a converged
finite-domain reference; no longer continuation was run.
