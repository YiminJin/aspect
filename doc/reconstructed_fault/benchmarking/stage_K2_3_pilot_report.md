# K2.3 bounded true-normal-stress feasibility pair

## Decision

The bumped pilot and matched homogeneous control both genuinely converge at
0, 0.5 and 1 s with the approved top normal traction. This establishes bounded
feasibility, not a converged K2 reference or Gate K2. The bumped run produces
a detectable but small along-fault normal-stress response compared with the
homogeneous control; coarse discretization error in that response is unmeasured.
No extra asymmetry, pressure shift, solver change or convergence run was used.

Baseline `8ace2d1b8` retains continuous Q2 ADD/count stress transfer, the accepted
history timeline and lookup optimizations. Those investigations were not reopened.
K2.2's reference remains provisional. Stop for review before further testing.

## Fixture, pressure and execution boundary

Artifacts below are under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/`.
Both cases use 32x128 cells, 17 fault vertices, fixed ell=0.15625 m and maximum
dt=0.5 s. The realized accepted sequence is exactly 0/0.5/1 s, with two real
updates. The pilot retains the compact 5% initial Theta bump; `homogeneous.prm`
removes only that bump. The initial phase CSVs are byte-identical between cases,
and initial C and I_h values agree exactly.

The explicit physical boundary change is top total normal traction -1000 Pa,
top prescribed tangential velocity, both bottom velocity components, and x
periodicity. The former top u_y=0 condition is removed. `Pressure normalization
= no`: no pressure offset is applied. Physical compression-positive stress is
sigma_n=p_FE-tau:N in Pa; solver pressure perturbations are multiplied by the
existing pressure scaling before G evaluation. Both runs report pressure
quotient=0 throughout. Top traction establishes the pressure datum; this is
not the zero-mean pressure convention of the earlier prescribed-pressure runs.

Prescribed velocity traces reconstructed from exact cell Q2 samples agree to
at most 4.07e-20 m/s. Bumped top u_y reaches about +/-2.574e-10 m/s; its net
flux magnitude remains below 9.30e-21 m2/s. The homogeneous top u_y is below
5.57e-19 m/s. Thus the free normal boundary has not inadvertently retained the
old pointwise impermeability constraint. The natural traction is imposed by
the existing weak boundary assembly; no separate pointwise traction-exactness
claim is made from this coarse FE solution.

## Actual constitutive normal stress, not bulk-column stress

An opt-in diagnostic in `surface_system.cc` records the **existing point
response during Jacobian assembly**, before history publication. It recovers
sigma_n from the response's mu*sigma_n divided by the positive mu, and tau:N
as p-sigma_n. It does not reevaluate a Maxwell update. Bulk inputs use the
actual parent-P0 samples and frozen particle history; all surface data and
basis weights follow the existing domain quadrature.

Rank-local first moments, admissible weights and extrema are written to
`constitutive_normal_<step>_rank0.csv`; each new linearization replaces that
step's diagnostic. Only a final converged solve makes the snapshot accepted
evidence. The one-rank analysis verifies the weights against the production
consistent mass row sums and obtains along-fault fields with M^-1 times the
recorded loads. Raw extrema below are over constitutive samples, **not**
extrema of the projected field or raw bulk-column averages.

### Bumped pilot, dimensional values

| Time (s) | p mean / raw range (Pa) | sigma_n mean / raw range (Pa) | tau:N mean / raw range (Pa) |
| --- | --- | --- | --- |
| 0 | 1000.000000000 / [991.653616, 1008.346383] | 1000.000000000 / [987.412965, 1012.587033] | -8.44e-13 / [-4.778731, 4.778734] |
| 0.5 | 1000.000005321 / [996.617670, 1003.381415] | 1000.000007980 / [994.957291, 1005.042911] | -2.65910e-6 / [-1.899083, 1.898632] |
| 1 | 1000.000035536 / [994.386207, 1005.607740] | 1000.000051798 / [991.592597, 1008.402794] | -1.62626e-5 / [-3.190285, 3.187495] |

Means use the full admitted-domain integration measure. Along-fault Q1
sigma_n anomaly RMS is 3.82880e-5, 4.45667e-5 and 2.18936e-3 Pa respectively.
At 1 s its projected range is [999.997842593, 1000.006338665] Pa. The much
larger raw extrema are not reported as a net along-fault feedback amplitude.

The homogeneous control has sigma_n=1000 Pa to about 3.5e-8 Pa in raw samples
over the whole trajectory. Its along-fault sigma_n anomaly RMS is at most
1.56e-11 Pa; p is 1000 Pa and tau:N zero to corresponding numerical accuracy.
Full p/sigma_n/tau:N statistics for both cases are in their verification JSON.

### Matched bump-minus-homogeneous along-fault fields

| Time (s) | delta p RMS (Pa) | delta sigma_n RMS (Pa) | delta tau:N RMS (Pa) | delta V RMS (m/s) | delta slip RMS (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.64830e-6 | 3.82880e-5 | 3.26397e-5 | 8.70086e-7 | 0 |
| 0.5 | 2.80543e-5 | 4.52755e-5 | 2.02513e-5 | 1.41979e-6 | 7.09896e-7 |
| 1 | 1.01129e-3 | 2.18997e-3 | 3.16596e-3 | 9.47958e-7 | 1.18239e-6 |

At 1 s delta sigma_n has mean 5.17983e-5 Pa and range
[-0.002157407, 0.006338665] Pa. Its mean-removed RMS is 0.002189361 Pa, versus
1.47e-11 Pa in the homogeneous field. This detects nonzero response in the
coarse discrete problem; it does not demonstrate spatially resolved continuum
normal feedback. The initial Theta perturbation also directly changes mu, so
the full delta V/slip cannot be attributed solely to delta sigma_n.

Initial projected Theta is [199.999024405, 210.523585777] s, mean 201.875 s;
the control is 200 s to roundoff. The overshoot in the realized projection is
reported, not reset. At 1 s delta Theta mean/RMS is 1.08946/2.24557 s.
Initial differences are part of the coupled trajectory, not subtracted away
as if they had no physical effect.

## Slip, actual surface balance and active/free nodes

| Time (s) | Bumped V mean / range (m/s) | Control V mean (m/s) | Bumped mean slip (m) |
| --- | --- | ---: | ---: |
| 0 | 3.148085961e-4 / [3.124076205e-4, 3.154610654e-4] | 3.148635298e-4 | 0 |
| 0.5 | 8.397184021e-4 / [8.358752401e-4, 8.406345708e-4] | 8.399074636e-4 | 4.198592010e-4 |
| 1 | 2.746199202e-4 / [2.720020183e-4, 2.752917444e-4] | 2.746570737e-4 | 5.571691611e-4 |

All 17 vertices are far above V_min=1e-12 m/s at every accepted state in both
runs: **0 active / 17 free**. The production active-set rule can activate only
at-bound nodes, so no active inequality is hiding a nonzero free residual.
This pilot does not exercise active/free switching.

The saved `surface_weak_<step>.csv` contains actual pre-publication weak shear,
cohesive, friction and damping loads plus F. Their sum closes to 1e-10 in weak
coefficients, and the independent M^-1 residual norm agrees with the logged
free-surface norm to 1e-9 Pa. They are not reconstructed from bulk columns or
newly committed particle stress.

| Time (s) | Bumped mean q / evaluated C / friction / damping (Pa) | Bumped F RMS (Pa) | Control F RMS (Pa) |
| --- | --- | ---: | ---: |
| 0 | 1045.048556 / 318.202669 / 695.365027 / 31.480860 | 7.12142e-8 | 6.12035e-8 |
| 0.5 | 1124.896736 / 321.032259 / 719.892637 / 83.971840 | 6.17004e-9 | 5.99080e-9 |
| 1 | 1034.692553 / 320.698168 / 686.532392 / 27.461992 | 1.09556e-6 | 1.05242e-6 |

Evaluated initial cohesive response and retained initial C are distinct:
retained C_0=318.747655768 Pa, unchanged in both cases. Sampled CSVs preserve
separate `C` and `C_evaluated` columns rather than overwriting that distinction.

## Convergence, histories and allowances

Both runs have three final nonlinear acceptances and 17 freshly checked linear
directions (9/4/4 per step). Maximum fresh/target ratio is 0.984779 in the
pilot and 0.795371 in the control. No pressure complement is applied and no
Armijo exhaustion occurs. Final bulk residuals are 1.74e-12--6.64e-12, below
their fixed mixed bulk targets (about 1.59e-5 at initialization and 1e-7 at
real steps). Surface residuals in the table are below 1e-8 times their fixed
surface scales, about 952/340/789 Pa. The positive benchmark postprocessor
also explicitly requires successful final convergence, not merely exit zero.

The phase field remains bit-identical within each trajectory and H is retained
exactly by stable particle ID. Initial particle stress remains (0,0,1500) Pa;
there is no physical stress or Theta update through the numerical 2-s Maxwell
initialization interval. Real-step committed particle stresses change by up to
377.031/91.341 Pa (pilot), 375.784/90.537 Pa (control). Theta's real-step update
matches the independent constant-V aging-law exponential evaluation exactly
at the exported precision. Existing C/stress commit and rollback code is not
modified; the accepted transfer/timeline audit is reused, not repeated.

Each case measures all 33 initial FE transverse profile columns and all 96
bulk-QP normal columns at all three accepted times. Profiles are unchanged,
so the same independently measured omitted fraction applies throughout:
**5.830273657e-5**, below the explicitly approved pilot-only 1e-4 allowance
(original target 1e-6). Actual slip-normalization error is at most
**5.236487061e-5**, separately below 1e-4, locally and globally in both runs.
Global normalization is approximately 0.9999476351294. No tail was removed
from I_h, no support was widened, and same-support feasibility does not prove
full-profile equivalence.

## Artifacts, commands and budget

Builds passed with `-j4`:

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build --target uniform_shear.release -j4
```

For each `CASE=pilot` then `CASE=homogeneous`, from repository root:

```sh
ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC=1 python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/CASE.prm --configuration Release --timeout 120
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/true-pressure/verify_pilot.py CASE
```

`verify_pilot.py compare` produces the matched differences. The script was named
`analyze.py` during the two original checks, then renamed to avoid shadowing
the existing shared `analyze` module when imported. Its numerical checks are
unchanged. Case verification logs,
JSONs, run logs and resource/provenance JSONs remain in the artifact directory.
Measured wall/RSS: pilot **39.184 s / 615768 KiB**; homogeneous **36.589 s /
557076 KiB**. Analysis took 1.308/1.315 s plus less than 1 s for comparison.
No run exceeded 120 s; total simulation/analysis execution was under 80 s,
well below 600 s. No timeout retry or additional simulation was launched.

ParaView entrypoints for **each** case are `solution.pvd`,
`reconstructed_faults.pvd` and `particles.pvd` in its output directory. Bulk
stress compositions show old-history inputs; particle stress arrays show
terminal commits (supplied stress at zero). Along-fault actual constitutive
and weak profiles are `pilot-surface-{0,1,2}.csv` and
`homogeneous-surface-{0,1,2}.csv`; matched responses are
`bump-minus-homogeneous-{0,1,2}.csv` with `comparison.json`.

Tested executable SHA-256:
`abd1a3b8ec4cac3fd025f2ce9ee727abdaef4888d5507e859ce3a1915fb9f2a9`.
Benchmark plugin SHA-256:
`eb42df87be0f9665beb0888d21fea2923ae2292274723e51f8cbfa3188ee90a9`.
These include the uncommitted opt-in diagnostic and the benchmark-only fix
to sample actual initial pressure instead of hard-coding zero in the old
particle-center initial-traction export. That legacy export is not used as
the domain-integrated surface balance. No production constitutive equation,
surface quadrature, solver criterion or history update was changed.

No MPI true-pressure pilot, restart replay, orientation variant, refined
normal-feedback study or full test suite was run. They are not claimed as
verified here. No further run is proposed automatically: review this small
discrete normal-feedback signal and the retained K2.2 uncertainty first.
