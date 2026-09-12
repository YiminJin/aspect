# K3 phase precision audit and mixed residual criterion

## Frozen original-solver audit

The audit uses the exact 32x256/fault32, dt=.375 problem and original binary
from `stage_K3_periodic_convergence.md`. A benchmark-only hook mirrors the
unchanged Newton/line-search steps on private vectors, calls production
constrained residual/Jacobian assembly and CG, and preserves the original
Jacobian during trial evaluations. It compares step 1 (.375 s, passing) with
step 2 (.75 s, failing). Live solution/history/solver fingerprints are unchanged.
The terminal completion exception stops before step-2 production mechanics;
it is not a claimed convergence result. Runtime 79.791 s, peak 1,127,312 KiB.

The private trajectory reproduces the original iteration logs. At the failed
plateau the base residual is 1.3829684e-14 and fresh constrained linear residual
is approximately 1.1e-21. Thus the computed direction solves the assembled
Jacobian accurately. Trial evaluations use the actually represented change,
including periodic constraints, not simply alpha times the nominal direction.

| Frozen state / full step | Passing solve, final update | Failed solve, plateau |
| --- | ---: | ---: |
| Production trial residual | 1.535983e-14 | 1.425446e-14 |
| Extended-precision residual of represented trial | 1.521478e-14 | 1.424094e-14 |
| Extended-precision trial without double rounding of update | 3.001651e-15 | 3.120588e-15 |
| Residual effect of rounding the updated nodal state | 1.490994e-14 | 1.406538e-14 |
| Fresh J action versus independently evaluated action | 2.4354e-24 | 3.7398e-30 |

At the failed state reaction and gradient vector norms are both .175920149,
while their sum is about 1.39e-14: a cancellation ratio of 2.53e13. The
production-versus-extended residual difference is 3.12e-15. The larger
limitation is nodal representation: 3450 of 8224 free DoFs do not change under
the nominal full Newton step. Its represented action cannot cancel the base
residual even though the unrounded direction satisfies the linear equation.
Higher-precision summation alone does not remove this floor.

The alpha=1, .5 and .125 production trial residuals are 1.42545e-14,
1.38575e-14 and 1.38295e-14. The predicted-versus-fresh residual-change error
is 1.67e-15 at alpha=1, falling to 1.18e-18 under extended-precision diagnostic
evaluation. Amplifying the tiny plateau direction by 1e6 yields a represented
action 1.38297e-8 and an extended action disagreement of 4.46e-24; extended
affine error is 2.82e-18. Both signs of 1e4/1e6 and ordinary step lengths are
exported. These checks support the Jacobian and distinguish the numerical
floor from inaccurate CG or a different phase equation.

Extended precision is offline only, using exported actual CPDI weights,
gradients, parent H/measure, phase values and constraints. The known polygon
orientation convention is retained. No geometry, equation or update was
modified to produce this comparison.

## Authorized stopping correction

The criterion and positive term-scale formula are documented in section 25 of
`current_design.md` and `specification.tex`. Use
`||R|| <= max(1e-8 ||R_entry||, 8 epsilon_double ||S_entry||)` for this fixture.
Both terms are fixed for the entire solve. Keep the configured relative target,
iteration budget, phase line-search fallback, degradation domain and equations.
No update-size or stagnation condition grants convergence.

The independently evaluated scale at the failed entry is 738.433 in discrete
weak-load units, giving an allowance about 1.31172e-12. It is deliberately a
conservative arithmetic estimate, not a fit to the 1.4e-14 plateau. The
unexpanded reaction/gradient norms alone would miss cancellation while
interpolating gradients of nearby O(1) nodal values. The scalar reaction
derivative term also captures sensitivity to represented phi. Scale and MPI
ownership verification, rejection of materially larger residuals, and the
unchanged K3 replay are required before claiming the correction verified.

## Evidence and harness corrections

Artifacts live under `benchmarks/reconstructed_fault/uniform_shear/evolving/phase-floor/`.
`output/audit{1,2}_iterations.csv`, nodal/trial CSVs, CPDI coefficient and
constraint CSVs, `analysis.json`, and `extended_*.npz` preserve raw and derived
evidence. `analyze.py` repeats only offline evaluation. `diagnostic.cc` uses
the existing test-access boundary; production code was unchanged during audit.

Two harness failures are retained, not silently discarded: first, a private
Jacobian needed complete block layout/`collect_sizes()` for constrained matrix
assembly; second, amplification intended for tiny plateau directions was
initially applied to a large first correction and left the admissible branch.
The diagnostic now amplifies only tiny corrections. These are diagnostic-only
fixes, not production fixes or altered phase line searches.

## Verification and resumed trajectory

Both Debug and Release executables were rebuilt with `-j4`. Production changes
are limited to the private phase assembly's optional scale output and the
nonlinear convergence test in `source/simulator/phase_field.cc`, its declaration
in `include/aspect/phase_field.h`, and parameter documentation. Tests/benchmark
probes read the same scale; no material, particle-domain, surface, Stokes,
history, loading or support implementation is changed in this correction.

### Independent focused checks

`tests/phase_field_precision.cc` exercises production residual assembly and
Newton on private fields with H=Hc and phase values 0, 1e-18 and 1e-6 on one
rank and a normal-refined mesh. Two ranks additionally test phi=1e-13. The
residual with/without scale calculation is identical. Zero/tiny inputs meet
the same criterion without changing their state; larger inputs require a
solve and pass a fresh residual check. All live-field/history fingerprints are
preserved.

| Check | Result |
| --- | --- |
| One-rank 4x16 precision test | Pass, 1.517 s |
| One-rank 4x32 precision test | Pass, 1.566 s |
| Two-rank 4x16 precision test | Pass, 2.067 s |
| `[phase_field_domain],Stage-I*` Release units | 61 assertions / 11 cases pass |
| `phase_field_fault_stage_i_rollback` | Pass, 61.34 s |
| `phase_field_nonlinear_exhaustion` | Pass, 38.27 s |
| Python smoke/phase-criterion guards | 2 tests pass, 5.772 s |

For the exactly intact state, the independent scale is `2 E` times the exact
lumped Q1 mass. With nx=4, periodic x and natural y boundaries, its norm is
`2 E * .25 * sqrt((ny-.5)/4)/ny`. Production allowances 6.993523870246154e-15
(ny=16) and 4.984889017266270e-15 (ny=32) match this formula to roundoff;
one/two-rank agreement is also roundoff. This explicitly verifies dimensional
weak-load scaling rather than a mesh-independent absolute constant.

The small-but-material residual 4.9212553e-11 is more than 7,000 times its
6.9935239e-15 allowance and is not accepted at entry; Newton reduces it to
3.62e-26. The much larger test residual 4.92033e-4 is also reduced. The
deliberately insufficient one-iteration production phase budget still fails
and aborts before reconstruction. The rollback fixture still verifies
restoration after an accepted mechanical update. No expected output was
loosened. No complete ASPECT suite was run.

The first MPI test exposed a test-only owned-minus-ghosted vector comparison;
copying the state into the owned layout before subtraction fixes it. Its
failed log/output are retained under `two-before-owned-comparison-fix*`.
This was not a production assembly or convergence failure.

### Completed K3 replay

`spatial0375_n256_f32_periodic_floor.prm` inherits the interrupted fixture;
only output and rebuilt plugin paths change. A fresh initialization is used
because the failed phase iterate is not a committed restart state. The old
failed run is preserved. Runtime is **237.817 s**, peak RSS **1,161,548 KiB
(1.108 GiB)**, below the 600-s cap.

Initialization plus eight real .375-s steps through 3 s pass all nine state
guards and **37 fresh coupled-linear checks** (maximum fresh/target .990924).
Maximum omitted fraction / actual complete normalization error are
**5.919417e-5 / 6.113454e-5**, below the unchanged 1e-4 limits. Maximum surface
balance RMS is 5.78457e-6 Pa. Stable-ID history handoff, Theta update (maximum
error 7.11e-15 s), geometry invariance, phase admissibility and whole-fault/seam
homogeneity pass at every accepted state.

| Step / time | Entry residual | Fresh exit residual | Exit relative | Fixed absolute target |
| --- | ---: | ---: | ---: | ---: |
| 1 / .375 s | 3.086469e-6 | 1.644909e-14 | 5.329421e-9 | 1.350199e-12 |
| 2 / .75 s | 5.692113e-7 | 6.466376e-14 | 1.136024e-7 | 1.311721e-12 |
| 3 / 1.125 s | 3.066122e-6 | 1.092904e-12 | 3.564452e-7 | 1.300184e-12 |
| 4 / 1.5 s | 1.545989e-6 | 3.083517e-13 | 1.994527e-7 | 1.294015e-12 |
| 5 / 1.875 s | 2.861584e-6 | 1.497626e-14 | 5.233557e-9 | 1.285150e-12 |
| 6 / 2.25 s | 2.119363e-5 | 1.427423e-14 | 6.735151e-10 | 1.274519e-12 |
| 7 / 2.625 s | 1.887079e-5 | 1.401827e-14 | 7.428555e-10 | 1.264203e-12 |
| 8 / 3 s | 8.483014e-6 | 1.436808e-14 | 1.693747e-9 | 1.261629e-12 |

The absolute branch is visible: steps 2--4 do not satisfy the pure relative
1e-8 test, and are not reported as doing so. Their fresh residuals meet the
independently verified, frozen term-scale allowance. In particular this is
not convergence inferred from exit zero or from a stagnation label.

Initial H and all Maxwell history components are exactly unchanged versus
the old failed run. Initial phi differs by at most 1.11e-16; initial I_h by
4.26e-14 m. At the previously accepted first step, phi differs by at most
1.11e-16 and raw particle stresses by at most 4.02e-10 Pa. These differences
are negligible against the K3 profile/stress errors below. At the audited
plateau the maximum nominal Newton update is 7.85e-17 and represented nodal
change is one double spacing, 1.11e-16. This places the diagnosed limitation
many orders below the intended ~2e-5 phase feedback. The allowance is still an
estimate of arithmetic resolution, not a rigorous global solution-error bound.

### Resumed spatial evidence, not a Gate-K3 claim

The existing independent reference is reused after checking every accepted
time/dt/loading against its exact sequence. It retains its independently
initialized histories throughout. The passing corrected-periodic 32x128
baseline is not rerun. At common time 3 s:

| Quantity/error | 32x128/fault32 | 32x256/fault32 |
| --- | ---: | ---: |
| Maximum total phi error | 2.172693e-4 | 7.699359e-5 |
| Raw stress RMS/max error [Pa] | 1.32702 / 4.10665 | .330944 / .869520 |
| Last-step H increment profile error [Pa] | 3.880361e-4 | 1.387415e-4 |
| Last-step phi increment profile error | 1.769252e-6 | 8.749470e-7 |
| Cumulative phi increment profile error | 9.801340e-6 | 5.229569e-6 |
| Cumulative H increment profile error [Pa] | .00207514 | .00164572 |
| I_h cumulative increment [m] | .00561957 | .00435724 |
| I_h increment error versus .003304695-m reference [m] | .00231488 | .00105255 |

Total and feedback errors contract, but the fine I_h feedback increment still
differs from the reference by approximately 31.85%. Initial H-profile errors
remain .0230953/.0258859 Pa and are not evolved increments or removed physical
influences. The fine final values are V=.00224336149 m/s, Theta=1.58273380 s,
C=355.402560 Pa, I_h=108.14919108 m and slip=.00517984654 m. Corresponding
reference values are .00224318592, 1.58347820, 355.626770, 108.13822722 and
.00517921926. Full transverse profile/increment CSVs are preserved.

The nonlinear obstacle is resolved and the interrupted normal-resolution run
is complete. **K3 convergence is not yet established.** The prior independent
finding that smaller timesteps suppress feedback under the specified H maximum
rule is unchanged. No normal512 or production temporal sequence is run as part
of this audit/correction/recovery task; further convergence work must retain
that distinction and the unchanged physical criteria.

`verification.json` records analytic, mesh and MPI checks. `replay-summary.json`
records all fresh phase targets, coupled checks, initial/prefix differences and
spatial comparisons. Exact executable/plugin/source SHA256 values are in
`spatial0375_n256_f32_periodic_floor.resources.json`; the tested Release binary
is `61261bbaa827dd4e368f3879aff8cc078e96fc67684e604e898c15605d84c7f5`.
Changes remain uncommitted; prior worktree changes and failed evidence remain.
