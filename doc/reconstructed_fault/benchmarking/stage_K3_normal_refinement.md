# K3 first normal-resolution refinement

## Pre-execution scope and resource estimate

The accepted 32x128 smoke is retained without rerunning it. One 32x256 run is
authorized, on one rank in Release, through initialization and two real steps.
The existing Box `subdivided_hyper_rectangle` uses the configured repetitions,
so only `Geometry model/Box/Y repetitions` changes, **4 -> 8**. Refinement level
5, X repetitions=1 and structural spacing=.015625 m remain unchanged.

| Quantity | Accepted baseline | Prepared normal refinement |
|---|---:|---:|
| Bulk cells | 32x128 | 32x256 |
| h_x (m) | .0078125 | .0078125 |
| h_y (m) | .0078125 | .00390625 |
| ell (m) | .15625 | .15625 |
| ell/h_y | 20 | 40 |
| Fault cells/nodes | 16/17 | 16/17 expected and checked |
| Particles per cell | 3x3 | 3x3 |
| Initial particles | 36864 | 73728 |

Same box, physical fault at y=0 on cell faces, loading/ramp, full I_h,
support, initialization semantics, continuous stress transfer, quadrature,
tolerances and timestep rules. The refinement changes the initialization's
representation; do not reset the independent reference from production data.
Unchanged CFL rules can shorten dt2 as h_y halves. This is not a separate
temporal-refinement experiment: use the actual accepted sequence for each
independent reference and compare errors/feedback at the corresponding steps,
without treating differing final times as a pure spatial comparison.

Estimate before execution: **90--150 s, 1.0--1.4 GiB**, extrapolated from the
43.412-s/585-MiB baseline with twice the bulk/particle count and allowance for
anisotropic cells. Retain the existing **180-s process-group cap**, no retry.
No new plugin build or performance study is needed. The existing plugin's
per-state gates stop on failed convergence, admissibility, geometry, support
or homogeneity. A normalization error above 1e-4 triggers review, not a changed
criterion, support width or further level. No MPI/temporal/further spatial run.

The refined overlay is `evolving/normal256.prm`. The case runner and comparison
script accept a case name so they cannot overwrite the baseline; independent
reference particle/CFL estimates read the resolved mesh counts rather than
assuming the original 128 normal cells. Exact-accepted-time integration still
uses only exported time/dt/U, with independently initialized histories.

## Result: normalization review gate reached; no further run

The one authorized refined run took **87.979 s, 930692 KiB peak RSS
(908.9 MiB)**. It completed phase and mechanical solves at both real steps,
then the unchanged benchmark guard stopped it with exit status 1:
**step-2 actual supported normalization error = 1.5269755e-4 > 1e-4**.
Omitted h fraction remains 6.1716886e-5. The saved last state is mechanically
converged and committed, but **not a passing benchmark state**. The comparison
records `complete_smoke=false`. No retry, support change, tolerance change,
additional mesh level or production-code change was made.

Exported mesh/profile coordinates confirm 32x256 cells. All 17 fault nodes,
their coordinates and support widths match the baseline and remain unchanged
during the run. Resolved parameters differ only in Y repetitions and output
directory. The original 32x128 ASPECT smoke was not rerun; its saved data were
reprocessed to provide matching profile/increment exports.

### Exact time sequences and independent reference

| Mesh | dt1 (s) | dt2 (s) | Final time (s) |
|---|---:|---:|---:|
| 32x128 | 2 | 1.712347940221483 | 3.712347940221483 |
| 32x256 | 2 | .8560969314396136 | 2.8560969314396134 |

The unchanged CFL rule shortens the second step. Each primary reference uses
its own exact exported time/dt/loading sequence and initializes histories once,
independently of ASPECT; no later production history resets are used. Thus
final mechanical fields are not compared as if they shared a physical time.
The second phase solve in both references uses the same H1 from the common
first step, so phi2 and I2 reference values coincide.

The refined-sequence independent reference itself predicts a **1.1166385e-4**
supported-normalization error, with h omission 6.1672775e-5 and history integral
-1.1453738e-7 m/s. The baseline-sequence value was 8.6927246e-5. A shorter
Maxwell interval changes the history correction even with the same phi2 input;
this independently establishes a support-budget failure for the new sequence,
not merely an ASPECT endpoint effect. It does not explain every production/
reference difference. No new stopping rule or support policy is proposed here.

### Support measured at every exported state

| Step | Baseline omission | Baseline total normalization error | Refined omission | Refined total normalization error |
|---|---:|---:|---:|---:|
| 0 | 5.830274e-5 | 5.236487e-5 | 5.913959e-5 | 6.006423e-5 |
| 1 | 5.831816e-5 | 5.248313e-5 | 5.915475e-5 | 6.019199e-5 |
| 2 | 6.099980e-5 | 9.566588e-5 | 6.171689e-5 | **1.526976e-4, fails** |

Actual refined signed supported integrals, in m/s (ranges over sampled x):

| Step | Instantaneous | History | Total |
|---|---|---|---|
| 0 | [3.1538026844373e-4, 3.1538026844382e-4] | [-2.54e-19, 1.53e-19] | [3.1538026844373e-4, 3.1538026844382e-4] |
| 1 | [2.3204340900571e-3, 2.3204340903658e-3] | [-2.6084344e-10, -2.6084326e-10] | [2.3204338292138e-3, 2.3204338295224e-3] |
| 2 | [2.291231495714e-3, 2.291285356248e-3] | [-2.018284676e-7, -5.273249702e-8] | [2.291029667246e-3, 2.291214022006e-3] |

Extrema in different columns need not occur at the same x. Full paired values
are in `normal256/crack_integrals_{0,1,2}.csv`. At step 2 the interior error is
also about 1.12e-4; the endpoint maximum is not the sole gate violation.
The evolving-profile finding remains: omitted h alone does not certify the
history-corrected crack-strain normalization. Full I_h remains unchanged.

### Transverse profiles and feedback

Errors below compare each mesh with its independent reference sampled at its
own profile coordinates, without subtracting the physical influence of initial
projection differences.

| Diagnostic | 32x128 | 32x256 |
|---|---:|---:|
| Max absolute phi0 profile error | 2.074680e-4 | 7.839128e-5 |
| Max absolute phi2 profile error | 2.136524e-4 | 6.130321e-5 |
| Max absolute H1-H0 profile error (Pa) | .0117000 | .0100062 |
| Max absolute phi2-phi1 profile error | 4.424282e-5 | 1.844981e-5 |
| Max phi2-phi1 | .0009900973 | .0009373476 |
| Mean I2-I1 (m); reference .1518550775 | .1592259943 | .1490042890 |
| Max H0 profile error against interpolated reference samples (Pa) | .0230953 | .0258859 |
| Direct initial-H formula error at particle positions (Pa) | .00630028 | .00603936 |

The full phi/H profiles and their increments are preserved in each case's
`comparison_phase_{0,1,2}.csv` and `comparison_H_{0,1,2}.csv`; H rows use stable
initial particle ordinates and volume-weighted current means. The H comparison
interpolates the reference's Gaussian sample data, not an analytic H field.
The separate direct initial-formula audit exposes this representation effect.
The total H profile metric does **not** demonstrate decreasing error, and no
full H convergence claim is made. H1 increment sampled maxima (3.85048 versus
3.71959 Pa) also use different particle rows near the activation transition;
they are not same-location errors. Matched-location increment errors are shown
above. H2-H1 is zero in the refined run and its reference under the maximum
rule; the baseline's longer step produces a further increment.

![Saved profile and normalization comparison](../../../benchmarks/reconstructed_fault/uniform_shear/evolving/normal256-comparison.png)

### Mechanics and histories

Refined surface means versus the primary exact-time reference:

| Step | V / reference (m/s) | Theta / reference (s) | C / reference (Pa) | I_h / reference (m) | Slip / reference (m) |
|---|---|---|---|---|---|
| 0 | .000315399213 / .000315270817 | 200 / 200 | 317.513763 / 317.742650 | 108.144834 / 108.134923 | 0 / 0 |
| 1 | .002320573509 / .002320272775 | 2.35609624 / 2.35730983 | 353.713457 / 353.939029 | 108.145699 / 108.134923 | .004641147019 / .004640545549 |
| 2 (support fails) | .002291420705 / .002291156390 | .706360961 / .706636013 | 368.252662 / 368.465969 | 108.294703 / 108.286778 | .006602825253 / .006601997504 |

Raw evaluated q means at steps 0/1/2 are 1043.816504/1330.829683/1284.475279 Pa.
Raw q error RMS/max pairs against the independent response are
.277851/.689083, 1.318429/4.025459, and .578459/1.783275 Pa. The corresponding
baseline pairs were 1.089703/3.187990, 5.145764/17.792656 and 4.239303/14.934264
Pa; the final pair is at a different physical time. These raw errors are not
substituted for the actual weak surface-balance measurements.

All **16 fresh linear residual checks pass** (largest fresh/target ratio
.989433). All 17 surface nodes remain free, none active. Weak surface RMS
residuals at steps 0/1/2 are 5.9240e-8/1.8017e-11/9.4468e-12 Pa. Final bulk
residual is 6.19755e-11 against target 2.08201e-8. Phase-entry probes at step 2
give R(phi1;H0)=6.98034e-5, R(phi1;H1)=.0014114404 and
R(phi2;H1)=1.44532e-14 (relative 1.02400e-11). Stable-ID preceding-history
handoff and normal/exceptional probe restoration pass. Independent discrete
Theta-update discrepancies are <=4.45e-16 s. Initial retained histories are
not reinitialized or physically evolved through timestep zero.

There are 0/448 periodic crossings at real steps 1/2. Final along-fault full
ranges, including seam/endpoints, are H=5.78e-10 Pa, phi=2.25621e-6,
I_h=.000512478 m, C=.00128408 Pa and V=5.51636e-8 m/s. These pass the existing
homogeneity gate; the largest ratio to the predeclared feedback scales is
about .34%. Maximum phi is .599845139 < .8. Geometry never changes. The support
failure is kept separate from these passing diagnostics.

## Commands, artifacts and disposition

All paths below are relative to
`benchmarks/reconstructed_fault/uniform_shear/evolving/`:

```
python3 test_reference.py -v
python3 run_smoke.py --case normal256
OPENBLAS_NUM_THREADS=1 timeout 120 python3 reference.py --parameters normal256/parameters.prm --ramp-peak .00225 --accepted-times normal256 --output normal256-reference
OPENBLAS_NUM_THREADS=1 timeout 120 python3 compare_smoke.py --case normal256
OPENBLAS_NUM_THREADS=1 timeout 120 python3 compare_smoke.py --case smoke
OPENBLAS_NUM_THREADS=1 timeout 120 python3 plot_normal_refinement.py
```

The runner was invoked from the repository root; commands using relative data
paths above express the equivalent evolving-directory invocation. Seven cheap
reference tests pass (0.420 s, `normal256-reference-tests.log`). The single
ASPECT run exits 1 on the intended review gate, not numerical nonconvergence.
Reference and read-only comparisons finish successfully; the latter explicitly
preserve the failed support status. No MPI or broad tests were run.

`normal256.resources.json` records the actual command, revision
`dc1a96d3f72dce123393c90ad025e729885e8ee9`, executable/plugin/fixture hashes and
resources; `normal256.log` preserves the stop. `normal256-comparison.json`,
`normal256-reference/`, `normal256/guard_*.json`, the CSV profiles and
`normal256-initial-H-audit.json` preserve the numerical evidence.

Changes in this action are confined to the new fixture, case-aware runner and
checker, reference mesh metadata, read-only comparison/plot scripts and this
record/README/progress update. The existing production binary and plugin were
reused without rebuild. Earlier working-tree changes were preserved.

**Disposition: stop for review.** The requested refinement action is complete,
but K3 convergence is not established. The support/normalization gate prevents
advancing to any further resolution or temporal study. No adjustment is made
to the criterion, support, full I_h, loading, initialization or equations.
