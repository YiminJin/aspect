# Structural-plane RSF: independent evidence

Read `doc/reconstructed_fault/particle_free/DIRECTIONAL_CONTINUATION_REPORT.md`
in the source tree/archive. This is an experimental closure, not native code.
Three PNGs show imposed choices and calculated responses. No production input,
checkpoint schema, BP3 or original failure evidence is changed.

Reproduce from the repository (or extracted archive) root, with Python 3,
NumPy, SciPy and Matplotlib. No native executable or MPI is needed:

```sh
OPENBLAS_NUM_THREADS=1 timeout 90s python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/check_structural_rsf.py
OPENBLAS_NUM_THREADS=1 timeout 240s python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/structural_rsf_band.py
MPLCONFIGDIR=/tmp/mpl-directional-continuation OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_structural_rsf.py
```

- `point_history.csv`: full tensor components, signed local speed, retained
  memories/state, actual plane pressure, other-plane diagnostic and energy;
  phase is fixed. Plane-normal angles are degrees counterclockwise from +x.
  Normal sign is immaterial, flow speed is signed on t=(ny,-nx).
- `point_timestep.csv`: common-time full tensor norms; xy counted twice.
  Branch 0/1 are geometry preferences; branch 2 is nonproportional loading.
- `formation_history.csv`: actual loaded conditional-admission continuations,
  not fresh-history substitutes. No phase creation is simulated.
- `point_summary.json`, `point_checkpoint.json`: point gates and serialized
  fixture state. These are not native ASPECT checkpoints.
- `band_history.csv`: every accepted state of the independent mixed reduction,
  with units in column names. `slip_m` is integral 2Kxy dy; it excludes elastic
  boundary motion. `work_J_m2` is cumulative endpoint reaction work.
  `BE_J_m2` is a numerical remainder, never physical dissipation.
- `band_profiles.csv`: native mesh-node fields at .10/.15/.20/.25/.30 s.
  `flow_rate_s_inv=chi*signed_local_V_m_s`; pressure p=p0+p', with p0=10000 Pa.
  Fixed phase and normalized chi use the same nodal mass measure in each run.
- `band_comparison.csv`: max absolute differences normalized by reference
  maximum on the common-time .15–.30 s interval. Initial spatial runs and the
  failed first temporal pair are retained; `matched_spatial` is the verified
  common timestep. Width=(integral |flow_rate|dy)^2/integral flow_rate²dy.
- `band_summary.json`: the original strict all-checks `passed=false` is
  preserved. Selected temporal and spatial magnitude checks pass; work's tiny
  spatial differences are not monotone. Do not silently turn that into a pass.
- `band_endpoints.csv`: compact final physics/energy table on the matched step.
- `parameters.csv`, `provenance.json`, `preservation.json`: inputs, source and
  environment identifiers, and protected-file verification.

Point strain loading is engineering shear ±.02/s with reversal at .15 s;
p=10000+500 sin(2pi t/.3) Pa. The nonproportional control adds Dxx=-Dyy=.004/s
from .10 to .20 s. Band displacement is .001 t m to .25 s, then .00025 m to
.30 s. Each spatial mesh starts freshly from the same continuum seed. No
phase masks, arbitrary slip widths, gradient hardening, or pressure lag.

Numerical differences, physical symmetry alternatives, and unsupported
constitutive/history states are identified separately in the report. This
short weak-sliding transect does not establish sustained state weakening,
oblique mechanical stability, new-plane birth, or propagation readiness.
