# Bounded polyline quadrature verification

The geometric contract, including its explicit departure from the old
width-filtered point map, is in
`doc/reconstructed_fault/benchmarking/stage_K2_polyline_quadrature_addendum.md`.
This is an approved discrete-formulation revision, not a repair claiming that
the former point-volume source violated its specification.

- `voro-plane-cuts.log`: installed Voro++ temporary-copy/cut comparison.
- `captured_fault_geometry.csv`: actual reconstructed geometry from the
  formerly blocked `phase_field_fault_condensed_adiabatic` fixture.
- `build.log`, `plugin-build.log`: final Debug/Release builds with `-j4`.
- `unit-one.log`, `unit-two.log`: final geometry/nonlinear-safeguard tests;
  815 assertions in 18 cases per rank pass.
- `first-coupling.log`: first formerly blocked condensed test passes.
- `focused-tests.log`: selected 19 surface/coupled/lifecycle tests.
- `restart-isolated.log`: fresh sequential create/resume pair, 2/2 pass.
- `stage_i_isolated.log` / `.resources.json`: the unchanged dynamic-pressure
  fixture exhausts its nonlinear budget, despite exit zero under its default
  continue policy. This is not a converged test.

Current status and exact results are in
`doc/reconstructed_fault/benchmarking/stage_K2_polyline_quadrature_review.md`.
The polyline geometry/action checks pass, but the nonlinear verification gate
is unresolved. K1 and K2 replay inputs below were **not run**.

The three `.prm` files include the existing accepted benchmark inputs, changing
only output directory and end time (1 s). Run sequentially after the focused
integration gate, using `convergence/run_case.py --configuration Release`.
The runner saves executable/plugin/input hashes, exit status, elapsed time
and peak RSS. No broader refinement campaign is authorized here.

For corrected trajectory diagnostics, `surface_weak_*.csv` contains the
**actual frozen mechanical linearization's** integrated mass and weak loads.
Use these with `nonuniform/measure_case.py`; do not reconstruct that equation
from newly published parent stress or substitute a bulk normal-column average.
Raw published parent stress and FE history remain separate diagnostics.
The K2 family provisional omitted-fraction allowance is 1e-4 (originally
1e-6); the separate actual slip-normalization requirement remains 1e-4.
Neither the full I_h nor support widths are changed or renormalized.

The bulk-history transfer concern remains separate and unchanged. K2.2 and
Gate K2 are not established by these short replays.
