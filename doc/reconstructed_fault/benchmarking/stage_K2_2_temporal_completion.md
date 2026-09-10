# K2.2 temporal completion after the accepted performance baseline

The accepted performance baseline is preserved and recoverable. Only the two
missing approved temporal cases were completed; the spatial sequence and
valid temporal prefix evidence were reused. No production code, equations,
initialization, support, quadrature or acceptance criteria changed. Both new
trajectories genuinely converge through 2 s and pass the separate containment
and actual-normalization requirements. Total-field errors contract, but the
mean-removed traction comparison approaches a near plateau. Signed-term
accounting identifies cancellation, not a validated production defect or an
asymptotically resolved temporal reference. The approved sequence is complete;
the K2.2 reference remains provisional. All runs are stopped/completed, and
K2.3 has not begun. Further expensive cases or corrections require review.

## Scope and recoverability

The exact lookup reuse and mapping-aware Cartesian early rejection are closed
as performance work. No production source, numerical equation, initial-history
semantics, domain quadrature, support, full I_h, tolerance or iteration budget
is changed in this task. No additional cache/integration redesign is included.

The tested dirty-tree baseline is recoverable from
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/performance/accepted-cartesian-baseline/`:
source archive, binary working-tree patch, executable, plugin and per-file
manifest. The parent commit is `a24c3623108e99240997141121a9d167fa119042`;
that commit alone does **not** identify the tested dirty tree. The source
archive SHA256 is
`2adcaac8bf9af0b7d019eed8d04bbf37b260d9e50fa671651c5d66a6cd11825a`,
and the executable SHA256 is
`e2451b97ad5a76ba431967c4938bccb5b94cf910ff1182f9bd830e7666f1b453`.
`record_performance_baseline.py` records the snapshot without resetting or
committing unrelated changes. The accepted Cartesian tests and measurements
remain in `nonuniform/performance/cartesian-review.md` and its linked data.

The completed new-domain-rule 32/64/128 spatial study and dt=.5-s trajectory
are reused from `stage_K2_2_domain_convergence.md`; old point-rule data are not
part of this comparison. Only the already-approved .25/.125-s trajectories
need completion. There are no executable checkpoints in their interrupted
outputs, so output-only wrappers replay them and verify the accepted overlap
byte-for-byte. This is not a repeat of a completed study.

## Bounded interpretation of the saved t=1-s reversal

The previously unresolved quantity was the mean-removed **actual weak surface
traction**, represented by M^{-1}Q. Its successive temporal RMS differences
at t=1 s increase from 2.810874e-4 to 6.537555e-4 Pa, despite decreasing
total-field differences. The following accounting uses actual saved weak
loads, mass matrices and accepted histories, not a normal-column average or
independent per-vertex K1 solves.

| t=1-s anomaly RMS difference | .5--.25 s | .25--.125 s |
|---|---:|---:|
| Friction contribution (Pa) | .01853517 | .01012764 |
| Radiation contribution (Pa) | .01805965 | .009294129 |
| Friction + radiation (Pa) | .0004765742 | .0009087636 |
| Evaluated cohesive contribution (Pa) | .0002664949 | .0003424365 |
| V (m/s) | 1.805965e-7 | 9.294129e-8 |
| Theta (s) | .04509369 | .02246971 |

Friction/radiation difference correlations are -.999998508 and -.999303530.
Each large term contracts, while the small, nearly cancelling remainder need
not contract at these timestep levels. The saved weak-traction decomposition
closes to approximately 3e-11 Pa.

Slip admits an exact accounting into coarse-endpoint solution differences
and the finer trajectory's time/load quadrature difference. Their respective
anomaly RMS values decrease from 1.602636e-7 to 1.101085e-7 m and from
1.357360e-7 to 7.432671e-8 m, while their sum increases from 2.863330e-8 to
3.692237e-8 m (closure below 1e-19 m). Cohesive recurrence accounting driven
by accepted V reproduces the corresponding trend; its remaining projection
contribution is 8.78e-7/1.65e-6 Pa and is reported separately. This accounting
is not a replacement mechanical reference or a reset of reference histories.

Raw bulk stress has the same important distinction. Its t=1-s total RMS
difference decreases from 21.79248 to 9.593347 Pa, but its anomaly difference
increases from .01785518 to .02792598 Pa. The strain/slip/frozen-FE-history
anomaly contributions individually decrease (.4690/.5943/.2256 Pa to
.1789/.2266/.06425 Pa); their signed sum closes to 1.71e-13/6.46e-14 Pa.
This is measured cancellation, not evidence for a new history-transfer fix.
The published-versus-constrained FE-history concern remains separate.

These diagnostics explain the reversal without changing the criterion. They
do **not** prove that the finest timestep is an asymptotically resolved
reference. Complete common-time evidence through 2 s is still required for
that assessment.

Artifacts under `nonuniform/domain-convergence/`:
`temporal-cancellation.json`, `temporal-saved-bulk.json`, and
`temporal-bulk-decomposition.json`. The missing saved-state normalization
checks pass at 384 native bulk-QP columns per state, with maximum error
5.771154e-5. The independently measured omitted fraction is 5.89728e-5.
Both separate 1e-4 fixture-family requirements are retained; same-support
convergence is not full-profile equivalence. Bounded diagnostics took about
60 s aggregate; no all-profile or separate performance campaign was run.

## Completion execution record

The prospective resource bounds and exact wrappers are in
`nonuniform/domain-convergence-completion/README.md`. The approved .25/.125-s
case caps remain 9000/14400 s, with no solver-budget change. Distinct physical
cores avoid the previous CPU-placement collision. Both use the accepted
Release binary, `ASPECT_FAULT_PERFORMANCE=1`, and `convergence/run_case.py`
for elapsed time, peak RSS and binary hashes. Completed prefix checks are
saved incrementally in `domain-convergence-completion/prefix-checks.json`.

Both approved cases are complete. No K2.3 testing was performed.

Commands from the repository root:

```sh
completion_dir=benchmarks/reconstructed_fault/uniform_shear/nonuniform/domain-convergence-completion
ASPECT_FAULT_PERFORMANCE=1 taskset -c 0 python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py "$completion_dir/time025.prm" --configuration Release --timeout 9000
ASPECT_FAULT_PERFORMANCE=1 taskset -c 1 python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py "$completion_dir/time0125.prm" --configuration Release --timeout 14400
timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/check_temporal_prefixes.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-k2-temporal-mpl taskset -c 3 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/complete_temporal_measurements.py time025
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-k2-temporal-mpl taskset -c 3 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/complete_temporal_measurements.py time0125
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-k2-temporal-mpl taskset -c 3 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/audit_convergence.py temporal benchmarks/reconstructed_fault/uniform_shear/nonuniform/domain-convergence/space128 "$completion_dir/time025" "$completion_dir/time0125" --output "$completion_dir/temporal-audit.json"
OPENBLAS_NUM_THREADS=1 taskset -c 0 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/measure_saved_temporal_bulk.py --completed
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-k2-temporal-mpl taskset -c 3 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/diagnose_temporal_cancellation.py --completed
```

The prefix comparison is complete: 73 quarter-second and 82 eighth-second
CSV files are byte-identical, covering all 17 previously saved accepted states.
The quarter-second run finishes in 1750.999 s, peak RSS 5784664 KiB, with
all nine final nonlinear checks and 41 fresh linear checks passing. Its
19.062-s postprocessing check verifies both 1e-4 allowances at every accepted
time, fixed geometry/profile, actual weak balance, prescribed boundary traces
and the independent exponential Theta update. Maximum boundary error is
3.39e-20 m/s and maximum Theta-update error is 1.42e-14 s.

The eighth-second case completes in 2908.272 s, peak RSS 5786604 KiB, with
17 final nonlinear and 70 fresh linear checks passing. Its measurement pass
takes 33.925 s; maximum boundary error is 4.74e-20 m/s and the independently
evaluated Theta update is identical in floating point. The completed raw-bulk
comparison takes 15.129 s, reusing all valid saved common-time comparisons.
The final surface accounting takes .515 s. All bounded analysis invocations
finish below 120 s and their aggregate execution remains below 600 s; the
simulation runs use their separate approved case budgets. No build, new MPI
campaign, or broad regression suite was needed: no production source changed,
and the accepted baseline's focused MPI/lifecycle evidence is retained.

## Completed temporal evidence

All three temporal initializations are byte-identical, including the phase
profile and retained histories. Realized initial Theta has RMS projection
error .005140822 s, range 199.9999922--210.0317795 s; initial mean C is
317.2451785 Pa. Initial histories are never reset from a later output.
The spatial initial-projection differences remain in the prior spatial report;
they are not reinterpreted as temporal or purely mechanical errors.

All 31 accepted states across the three timestep levels meet both final
nonlinear criteria and all 136 returned linear checks pass. Independent
surface strong-residual reconstruction agrees with the production measure.
The unchanged fixed profile has independent full I_h=108.1266579 m and
omitted fraction 5.897277e-5. Worst actual local normalization error is
5.771154e-5; global normalization also passes at every accepted time. There
are 384 native bulk-QP columns for normalization and 129 independent profile
columns. Identical geometry/profile hashes permit reuse of the independent
profile integral, not alteration of its value.

Errors below are adjacent fully coupled numerical solutions, not errors
against independent per-vertex scalar solves. Surface norms use exact Q1
arclength integration; bulk differences use the same native Q2 Gauss points.

| t=2-s field | Total RMS, .5--.25 | Total RMS, .25--.125 | Mean-removed RMS, .5--.25 | Mean-removed RMS, .25--.125 |
|---|---:|---:|---:|---:|
| Actual weak q (Pa) | 4.622582 | 1.489934 | .001056798 | .001035198 |
| V (m/s) | 1.327262e-5 | 3.951740e-6 | 3.757638e-8 | 1.060914e-8 |
| Theta (s) | .3634154 | .09840193 | .01433279 | .008226690 |
| Retained C (Pa) | .03132915 | .008079259 | .0006575077 | .0005462072 |
| Slip (m) | 4.071621e-6 | 1.234864e-6 | 7.127410e-8 | 5.927590e-8 |
| Raw bulk tau_xy (Pa) | 4.622637 | 1.490501 | .04943618 | .04429940 |
| Raw bulk pressure (Pa) | .06813763 | .05934744 | .06813763 | .05934744 |

Frictional normal stress remains prescribed at 1000 Pa, distinct from the
reported raw bulk pressure. Finest raw-stress anomaly RMS is 1.439348 Pa,
with sampled maximum 11.22309 Pa; it is not smoothed. The finest-pair raw
stress total/anomaly maximum differences are 1.870515/.3806676 Pa.

| t (s) | q anomaly RMS .5--.25 (Pa) | q anomaly RMS .25--.125 (Pa) | Raw stress anomaly RMS .5--.25 (Pa) | Raw stress anomaly RMS .25--.125 (Pa) |
|---:|---:|---:|---:|---:|
| .5 | .0005186217 | .00006686385 | .01922443 | .004285361 |
| 1 | .0002810874 | .0006537555 | .01785518 | .02792598 |
| 1.5 | .0008354784 | .0009622834 | .03959880 | .04065842 |
| 2 | .001056798 | .001035198 | .04943618 | .04429940 |

At t=2 s the q-anomaly difference at left/center/right is
(-.00135936, .00170572, -.00133239) Pa for .5--.25 and
(-.00137608, .00162508, -.00132386) Pa for .25--.125.
Outside-bump RMS barely decreases (.00100510 -> .000995472 Pa), as does
inside-bump RMS (.00110608 -> .00107345 Pa). The finest actual q values are
987.024136/987.130653/987.024037 Pa at left/center/right, with mean
987.074259 Pa and anomaly RMS .03733816 Pa. Thus the finest-pair anomaly
difference is 2.77% of that signal, not hidden by the approximately 987-Pa
total field. q's maximum mirror difference increases from 2.02e-5 to
4.71e-5 to 9.94e-5 Pa; no mirror symmetry is imposed on advected shear history.

### Interpretation and review boundary

At t=2 s the friction/damping anomaly differences decrease from
.00517860/.00375764 Pa to .00236703/.00106091 Pa, but their correlation
changes from -.99220 to -.92778. Their sum barely decreases
(.00152403 -> .00143828 Pa), before addition of the cohesive contribution.
The complete weak-load identity still closes within 3.1e-11 Pa. Slip's exact
endpoint/time-load accounting closes within 1.1e-19 m; both components
decrease, but their near cancellation leaves the slower-converging remainder.
Cohesive accounting has a separately retained 2.07e-6/3.79e-6-Pa projection
remainder. This is far smaller than the observed traction difference and
does not establish a history-transfer correction.

The spatial sequence supports the chosen fine mesh; total temporal fields
and V/Theta anomalies improve. However, the available levels do **not**
establish an asymptotic temporal reference for actual nonuniform traction,
raw stress or its endpoint response. Cancellation explains the measured
signed balance but does not prove that the near plateau will disappear at
smaller dt, or exclude a cumulative fixed-mesh/history-transfer limitation.
No algebraic-convergence or allowance failure was found. No production
correction, fitted criterion, or additional expensive case is justified by
this evidence alone. Stop here for review; the separate bulk-history
transfer concern remains a separate task. K2.3 is untested and Gate K2 is unmet.

## Performance record and artifacts

Opt-in measurements during the necessary runs (timers are nested; do not sum
inclusive parents and children):

| Operation | .25-s case calls / seconds | .125-s case calls / seconds |
|---|---:|---:|
| Domains/CPDI | 10 / 159 | 18 / 282 |
| I_h preparation | 9 / 250 | 17 / 408 |
| Surface R/K total | 91 / 193 | 157 / 344 |
| G actions | 659 / 73.9 | 1063 / 126 |
| B actions | 659 / 183 | 1063 / 306 |
| Geometry-cache build | 9 / 14.2 | 17 / 24.5 |
| Cache validation | 110 / .405 | 192 / .666 |

MappingCartesian rejection is eligible in both runs; each preparation rejects
10560 requests. Cold preparation is 69.63/66.24 s; later preparations reuse
3505 lookup batches with zero rebuilds. Warm preparation ranges are
21.18--24.35 s (.25-s case) and 20.22--22.35 s (.125-s case).
Detailed call counts, integration-point counts and all component timers are
retained in the logs. No new performance claim is inferred by comparing these
whole-run timings with previously contended/interrupted runs.

Under `nonuniform/domain-convergence-completion/`, `temporal-audit.json/.png`
contains every common-time surface comparison and initial/error accounting;
`temporal-bulk.json` contains native raw velocity/pressure/stress comparisons;
`temporal-cancellation.json` contains the signed contributions. Each case has
its input wrapper, complete log, resource/provenance JSON, measurement report,
sampled CSVs and ParaView-readable `solution/` and `reconstructed_faults/`
outputs. Scripts added/extended here are benchmark-only diagnostics and
snapshot/prefix recording; the existing production tree is preserved.
