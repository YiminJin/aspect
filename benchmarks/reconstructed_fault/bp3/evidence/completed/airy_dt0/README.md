# Frozen-Airy dt0 comparison

One initialization-only 4e6 s case, compared with saved `../initialization04`
(1 s). Results and exact commands:
[report](../../../../doc/reconstructed_fault/bp3/stage_K5_airy_dt0_sensitivity.md).

`airy.cc` preserves the tested Airy architecture from
`../evidence/initialization04-bp3.cc`. `frozen_airy.csv` freezes its complete
baseline traction curve so changing dt0 does not also retune the Airy field.
The unfinished offset plugin in `../bp3.cc` is not loaded.

`run.py` refuses overwrite/retry; the saved attempt is complete. Rerun only
the cheap `test_analysis.py` / `analyze.py` to regenerate the comparison.
`comparison.json`, the log and resources JSON contain the numerical evidence
and recoverable source/binary hashes. No real BP3 timesteps were run.

## Mixed-FE indexing and distance-weighted interpolation follow-up

`cg_phase.prm`, `dg_phase.prm` and `dg_phase_two.prm` reuse the small production
phase-precision regression with the added independent vertex-map check. Build
`aspect.exe.release` in `build-pf-cpdi` and `phase_precision_tests.release` in
`uniform_shear/evolving/phase-floor/tests/build`, both with `-j4`.
Run these via `run_interpolation.py CASE` (120 s per regression). Saved tests
passed on one rank with CG/DG and on two ranks with DG; no overwrite is allowed.

`distance_initialization.prm` retains the frozen Airy, full smoke mesh, 4e6 s
interval and all controls from `dt4e6.prm`, changing only the interpolation
scheme to `distance weighted average` (default linear weights).
`run_interpolation.py distance_initialization` has a 900 s cap and stops at
timestep zero. `analyze_interpolation.py` compares the realized bulk velocity,
surface mechanics and resolved inputs with the saved cell-average case.
