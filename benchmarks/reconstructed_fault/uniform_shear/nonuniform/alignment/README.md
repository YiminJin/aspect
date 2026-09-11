# K2.4 normal-grid parity sensitivity

Approved execution completed: bumped/control 151.164/131.842 s, both through
0/.5/1 s, all actual geometry and reused acceptance checks pass. See
`doc/reconstructed_fault/benchmarking/stage_K2_4_report.md`. `*-geometry.json`
records the realized cell-interior placement; `*-verification.json` records
nonlinear/fresh-linear, history, weak-balance, containment and normalization
checks. `comparison.json`, `comparison-*.csv`, `comparison.png` and
`endpoints.png` retain the signed matched-field and endpoint/interior evidence.
The reference remains provisional and Gate K2 unmet. No further run follows.

The preflight below is retained as a historical record. Execution was approved
after its review. Analysis can be repeated without running ASPECT using
`verify_case.py pilot`, `verify_case.py homogeneous`, then `compare_cases.py`.

`pilot.prm` and `homogeneous.prm` retain their accepted true-pressure 64x256
counterparts except Box X/Y repetitions=64/255, initial global refinement=0,
and new output directories. The physical fault remains at y=0 in the same
box; the normal cell size increases .3922% and y=0 lies in cell interiors.
No production mesh/mapping change is needed. This is bounded sensitivity
against provisional data, not post-convergence verification or a pure shift.

Static check (does not run ASPECT):

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/alignment/check_preflight.py
```

The checked syntax is only the simple syntax used by this fixture family.
Actual reconstructed geometry, initialization and convergence are untested.
Require preflight review before execution; no automatic runs or retries.
See `doc/reconstructed_fault/benchmarking/stage_K2_4_preparation.md` for the
exact changes, resource estimates, independent <=1e-4 containment and actual
normalization checks, and the planned signed constitutive/endpoint/interior
comparisons. Keep initial projections visible and stop on sensitivity to the
unresolved interior tau:N response. No 128x512 run is authorized.
