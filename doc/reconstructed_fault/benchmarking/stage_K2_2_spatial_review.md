# K2.2 spatial refinement: endpoint-traction review gate

2026-09-09. **Spatial runs completed; temporal runs held for review. K2.2 is
not complete and Gate K2 has not passed.** The requested stop condition was
triggered by a late-time plateau in the nonuniform actual surface traction.
No production correction, support change, smoothing or tolerance change was
made. K2.1 remains an accepted feasibility pilot.

## Scope, revised allowance and reference

The original omitted-profile-fraction target was **1e-6**. The user explicitly
approved **provisional 1e-4** for this fixed-profile, prescribed-normal-stress
K2.1/K2.2 fixture family. This is a separate revision, not inheritance of the
K1 exception. The independent **1e-4 actual slip-normalization requirement**
is unchanged. Neither allowance is transferred to true-normal-stress or
evolving-profile tests.

The three full coupled runs have bulk meshes 32x128, 64x256 and 128x512, fault
spacing 1/64, 1/128 and 1/256 m, and dt=0.5 s through 2 s. The 64 run is the
accepted pilot; only its initialization was replayed to add evaluated initial
particle traction. Compared exported fields are bit-identical at t=0.
The accepted loading and physical initializer are unchanged. In particular:

- ell=0.15625 m, measured association half-width=0.3088215939070757 m;
- full independently checked I_h remains in the cohesive law;
- the converged initial Q1 phase field is frozen by phase-field-only
  benchmark constraints; mechanics remains unconstrained by that fixture;
- retained Theta0, H0, Maxwell stress0 and C0 follow the approved initialization
  rules, with a 2 s artificial Maxwell response interval at t=0;
- real histories advance once per accepted step; they are not reset from a
  scalar reference or from another resolution;
- sigma_n=1000 Pa is prescribed, not dynamic bulk pressure.

The comparison reference is the **full nonuniform coupled 128x512 solve at
the same support and dt**. It is not an independent scalar solve at each
fault vertex and is not yet a time-converged reference. Same-support
convergence does not establish equivalence to the untruncated profile.
Omitted tails remain a distinct fixed-support approximation.

## Completed runs and resources

All new simulations used one rank. Prospective estimates were recorded in
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/refinement/README.md`
before running: coarse 100--250 s / 0.5 GiB, diagnostic initialization
100--250 s / 1 GiB, fine 1700--3500 s / 4--6 GiB.

| Case | Accepted states | Measured wall time | Peak RSS |
|---|---:|---:|---:|
| 32x128, dt=.5 | 0, .5, 1, 1.5, 2 s | 72.69 s | 401224 KiB |
| 64x256 reused pilot | same five states | 414.14 s | 933372 KiB |
| 64x256 diagnostic replay | t=0 only | 104.47 s | 927272 KiB |
| 128x512, dt=.5 | same five states | 2412.76 s | 3071536 KiB |

The prepared dt=.25/.125 inputs have **not run**. The original prospective
budgets were 2500--5000 / 4500--8500 s with 4--6 GiB reserved. The measured
fine run uses 2.93 GiB; these conservative budgets remain adequate planning
figures, not measurements or authorization to bypass the review gate.

Production executable SHA256 is
`27a2defaf8fa44492d33c22235927cdf2a60cb4ed1c35e722c72f9345d9b61b4`.
The diagnostic plugin SHA256 is
`2132bae10bf47dcb3b6a7778a86670dd70dc5bdfa3663ee83f4d968f3a374ec4`.
Per-run `*.resources.json` records executable, parameter and plugin hashes,
working directory, command and exit status. The production baseline remains
commit `a24c3623108e99240997141121a9d167fa119042`.

## Allowance and lifecycle measurements

The independent Q1 normal-profile integral is evaluated at all 33/65/129
phase-field x columns. The saved phase arrays are identical at every accepted
time within each run. Actual integrated slip is measured at all 96/192/384
bulk quadrature x columns and globally at **all five accepted times**, not
only at the center or initialization.

| Bulk nx | Maximum omitted fraction | Maximum actual slip-normalization error | Full I_h (m) |
|---:|---:|---:|---:|
| 32 | 5.830274e-5 | 5.236487e-5 | 108.098095072 |
| 64 | 5.913959e-5 | 6.006427e-5 | 108.144833844 |
| 128 | 5.897277e-5 | 5.771154e-5 | 108.126657883 |

Both separate 1e-4 requirements pass over these measured locations/times.
No assertion of unmeasured continuous-space maxima is made. At fine resolution,
production I_h differs from the independent integral by at most 4.986e-11
relative. Prescribed boundary velocity errors are at most 4.066e-20 m/s across
the runs. The real-step exponential-aging Theta update agrees within
1.422e-14 s; Theta0 is retained exactly. Fine particle-domain area remains
within 5.19e-9 m2 of 0.25 m2, with positive sampled domain volumes.
These checks do not assert periodic particle-domain behavior from absence of
particle wrapping.

## Actual particle traction and weak balance

The benchmark-only export added to `uniform_shear.cc` evaluates the production
point response at the actual cached particle positions at t=0. It uses the
accepted strain, initialized old particle stress and production Maxwell
coefficients. Initial retained tau0 is deliberately not mistaken for evaluated
tau at t=0. At real steps, the existing particle export contains the actual
accepted stress published by production history commit.

`nonuniform/measure_case.py` contracts those stresses with the measured slip
tensor and independently evaluates the homogeneous fixed-profile cohesive
law and friction using **previous** C/Theta and accepted V:

    F_p = q_p - C_p - mu(V_p,Theta_previous,p)*1000 - eta_d*V_p.
    r_i = sum_p volume_p N_i(xi_p) F_p,
    M_ij = sum_p volume_p N_i(xi_p) N_j(xi_p).

The diagnostic RMS is sqrt(r^T M^-1 r / sum_p volume_p), the production
strong-residual norm. Owned particle volumes, associations and Q1 coordinates
are used, not normal-column bulk traction averages. The benchmark's known
homogeneous/fixed-profile restrictions are essential to this independent
formula; it is not a new general constitutive implementation.

For every accepted state at all three resolutions, this RMS matches the
production final RMS within **7.78e-14 Pa**. The initial independent F also
matches the production point F under its 1e-9 Pa verification bound.
Raw particle F and q are exported separately. The consistent Q1 representation
of q is a diagnostic of the weak surface equation, not smoothed raw stress or
an input fed back into mechanics. At fine t=2, raw F RMS is 1.94789 Pa while
the weak strong-residual RMS is 4.80537e-10 Pa: the method enforces the weak
equation, not F_p=0 at every particle.

The fine run passes all 25 recorded fresh linear-residual checks. At 2 s its
bulk residual is 3.38285e-12 (fixed scale 19.98085) and surface residual is
4.80537e-10 Pa (fixed scale 166.60954 Pa). The observed traction differences
are not attributable to an unfinished surface solve.

## Initialization versus mechanical differences

The analytic supplied state peaks at 210 s. The realized Q1 projection
converges toward it; it is not replaced by the analytic state after projection.
The RMS below integrates the difference to the analytic compact bump along
the fault, using eight-point Gauss quadrature on each surface interval.

| nx | Realized Theta0 min/max (s) | Projection RMS (s) | Retained C0 mean (Pa) | Initial phi center |
|---:|---|---:|---:|---:|
| 32 | 199.998632 / 210.542078 | 0.0973397 | 318.747656 | 0.599910323 |
| 64 | 199.999861 / 210.132029 | 0.0212810 | 317.513763 | 0.599764017 |
| 128 | 199.999991 / 210.032687 | 0.00514090 | 317.245179 | 0.599711051 |

The total initial phase-profile RMS differences to the fine run are
5.81723e-5 (32) and 6.03052e-5 (64): this is **not monotone convergence**.
I_h is also nonmonotone. The initializer retains the previously audited H
activation substitution at phi_hat=0.1; see
`stage_K1_cpdi_cause_and_correction_plan.md`. That is an established source of
differences from the unmodified analytic stationary profile, but these runs
do not isolate its contribution to the nonmonotonic spatial error. No
initialization parameter or history was changed to conceal this limitation.

Total late-time C and stress errors retain large mean offsets from C0.
Subtracting the along-fault mean isolates nonuniform error; it does **not**
produce a mechanically identical-initial-state comparison. No claim that all
remaining error is purely mechanical is made.

The following RMS differences are to the full 128 run at common physical
coordinates. Surface norms integrate exact Q1 differences; bulk values use
native FE cell polynomials at the reference's raw quadrature points. Bulk
stress is recomputed from raw FE strain and frozen history with the actual
Maxwell decomposition, not by smoothing stresses. Bulk maxima below are
sampled maxima, not continuous Linfinity bounds.

| Quantity | 32 vs 128, t=0 | 64 vs 128, t=0 | 32 vs 128, t=2 | 64 vs 128, t=2 |
|---|---:|---:|---:|---:|
| V total RMS (m/s) | 6.503e-7 | 1.140e-7 | 4.041e-8 | 7.800e-9 |
| V anomaly RMS (m/s) | 3.079e-8 | 6.473e-9 | 2.036e-8 | 5.273e-9 |
| Theta total RMS (s) | .097211 | .020649 | .156674 | .028350 |
| C retained total RMS (Pa) | 1.50248 | .268584 | 1.46119 | .259868 |
| Cumulative slip total RMS (m) | 0 | 0 | 1.454e-6 | 2.563e-7 |
| Raw bulk q total RMS (Pa) | 1.46125 | .283791 | 1.46908 | .262715 |
| Raw bulk q sampled max error (Pa) | 3.75250 | .916887 | 2.43895 | .577503 |
| Raw bulk q anomaly RMS (Pa) | .010188 | .001414 | .078768 | .034028 |
| Physical bulk p RMS (Pa) | .016020 | .002373 | .072670 | .023072 |

Every accepted time, including the larger first-step V difference, is retained
in `space32-vs128.json` and `space64-vs128.json`. Bulk anomaly subtracts the
weighted along-x mean separately at each y; surface anomaly subtracts the
arclength mean. These definitions do not equate normal averages with surface
traction. Raw fine particle q ranges from 1039.86 to 1055.47 Pa at t=0 and
989.07 to 1005.27 Pa at t=2; these spreads are not the weak-form residual.

## Stop condition: nonuniform endpoint traction

Adjacent-grid RMS differences in the **actual particle/Q1 surface traction
anomaly**, in Pa:

| Time (s) | 32 minus 64 | 64 minus 128 |
|---:|---:|---:|
| 0 | 9.40976e-4 | 5.81422e-5 |
| .5 | 4.70446e-4 | 2.71228e-5 |
| 1 | 7.13064e-4 | 4.92231e-4 |
| 1.5 | 8.70746e-4 | 6.78927e-4 |
| 2 | 1.00656e-3 | 9.71051e-4 |

At 2 s the ratio is 0.965, unlike the initial ratio 0.0618. The fine traction
anomaly has RMS 0.0394323 Pa; the late adjacent difference is about **2.46%**
of that nonuniform signal, despite being small relative to total traction.
It cannot be dismissed by a total-field plot or a small nonlinear residual.

The saved-data component audit localizes the 64--128 difference:

- Inside the initial bump window (0.0625,0.1875) m: 6.42415e-5 Pa RMS.
- Outside that window: 1.37177e-3 Pa RMS, concentrated near the endpoints.
- Maximum absolute anomaly difference: 0.00729693 Pa at s=0; the opposite
  endpoint has a comparable error. The adjacent coarser maximum is 0.00272011
  Pa, so even this sampled surface maximum does not improve.
- Adjacent fine-grid component anomaly RMS values are C: 1.22067e-4 Pa,
  friction: 8.28754e-4 Pa, radiation: 5.27296e-4 Pa; their signed sum gives
  the q difference. The remaining F difference is only 1.27436e-12 Pa RMS.

This establishes a realized endpoint response difference, not an error from
using a bulk average. It **does not establish its cause**. Endpoint sampling,
particle-history transfer and coupled spatial error remain to be distinguished;
no production defect is asserted. The initial phase-profile nonmonotonicity
is also not fully separated from subsequent mechanical errors.

For context, at t=2 the V-anomaly RMS is 5.7364e-7 / 5.71294e-7 / 5.71256e-7
m/s on the three meshes. The fine V range is [1.2088218e-4,1.2281033e-4] m/s.
Thus the central nonuniform response is resolved substantially better than the
endpoint traction. Mean subtraction alone is not proof of nonlocal response;
the paired homogeneous 64-run comparison from the accepted K2.1 report remains
the valid measured outside-window response evidence, not a K2.2 reference.

**Review requested before temporal runs.** A bounded next diagnostic would
compare saved endpoint particle volumes/associations, actual old particle
stress versus its FE transfer, and the endpoint weak traction moments across
these same runs. If those data cannot distinguish sampling/history transfer
from underresolution, propose one isolated diagnostic replay before any
production correction. Do not widen support, identify fault endpoint DoFs,
smooth stress, revise an accuracy threshold or run the larger campaign by
default. The prepared time refinements and all true-normal-stress tests remain
unrun.

## Reproduction, tests and artifacts

From repository root, with B=benchmarks/reconstructed_fault/uniform_shear:

```sh
B=benchmarks/reconstructed_fault/uniform_shear
cmake --build "$B/diagnostics/pilot-build" -j4
python3 "$B/convergence/run_case.py" "$B/nonuniform/refinement/space32.prm" --configuration Release --timeout 1200
python3 "$B/convergence/run_case.py" "$B/nonuniform/refinement/initial64.prm" --configuration Release --timeout 1200
python3 "$B/convergence/run_case.py" "$B/nonuniform/refinement/space128.prm" --configuration Release --timeout 5000
python3 "$B/nonuniform/measure_case.py" "$B/nonuniform/refinement/space32" "$B/nonuniform/refinement/space32.log"
python3 "$B/nonuniform/measure_case.py" "$B/nonuniform/output" "$B/nonuniform/pilot.log" --initial-traction "$B/nonuniform/refinement/initial64/particle_traction_initial_0.csv"
python3 "$B/nonuniform/measure_case.py" "$B/nonuniform/refinement/space128" "$B/nonuniform/refinement/space128.log"
python3 "$B/nonuniform/compare_cases.py" "$B/nonuniform/refinement/space32" "$B/nonuniform/refinement/space128" "$B/nonuniform/refinement/space32-measurements" "$B/nonuniform/refinement/space128-measurements"
python3 "$B/nonuniform/compare_cases.py" "$B/nonuniform/output" "$B/nonuniform/refinement/space128" "$B/nonuniform/output-measurements" "$B/nonuniform/refinement/space128-measurements"
python3 "$B/nonuniform/audit_spatial.py"
python3 -m unittest discover -s "$B/nonuniform" -p test_diagnostics.py -v
python3 -m unittest discover -s "$B" -p 'test_*.py' -v
```

Exact executed ASPECT commands and run limits/results are retained in the
runner logs/resources; the commands above reproduce the cases. The plugin
build completed with -j4. Two new diagnostic unit tests pass (exact Q2
sampling and particle-volume Q1 affine reproduction); 15 existing independent
reference/analysis unit tests pass. All three full-run measurement scripts
pass. The t=0 diagnostic replay matches the prior pilot fields exactly. No
full ASPECT suite or new MPI/restart campaign was run: production is unchanged
and its accepted K1 evidence is reused. The new initial export has been
exercised on one rank only.

Under `$B/nonuniform/refinement/`:

- `space32-measurements/report.json`, `space128-measurements/report.json` and
  the reused pilot's `../output-measurements/report.json`: full-time allowances,
  lifecycle and actual balance measurements.
- `*-measurements/particle_balance_*.csv`: raw actual particle traction/F;
  `surface_balance_*.csv`: separate weak-Q1 q/C/friction/radiation/F histories.
- `space32-vs128.json`, `space64-vs128.json`, `space32-vs64.json`:
  full accepted-time error tables. The first two include exact surface vertex
  maxima; the older adjacent comparison's maxima are Gauss-sampled (its RMS
  is valid). `spatial_audit.json` supplies exact surface-vertex adjacent maxima.
- `spatial_audit.json`, `spatial_surface_audit.png`: initial projection,
  realized response and endpoint/component localization.
- `space128-residuals.json`: fresh linear and final nonlinear checks.
- `diagnostic-replay-comparison.json`, `diagnostic-unit-tests.log`,
  `reference-unit-tests.log`: verification results.
- `space32/solution.pvd`, `space128/solution.pvd` and each corresponding
  `reconstructed_faults.pvd`: ParaView-readable bulk/fault trajectories;
  native `bulk_*.csv`, `phase_*.csv`, `particles_*.csv` preserve raw data.

Raw run/build files remain local; compact input, analysis and review files
are the reproducible record. No changes were made to source/, include/, or
the production equations/APIs. Changed C++ is limited to the benchmark export.
