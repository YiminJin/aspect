# K3 common-timestep spatial discrimination

## Review conclusion: (2), surface I_h representation limits the comparison

The matched .375-s runs remove the earlier CFL/history ambiguity. Both reach
t=1.125 s with converged phase/mechanics and correctly retained histories, then
fail the unchanged normalization guard: 1.8871e-4 on 32x128 and 2.2735e-4 on
32x256. The finer normal mesh improves transverse phase/raw stress errors but
does not reduce the endpoint normalization defect.

The dominant measured error is **not an excessive omitted tail**. Small
along-fault variations in the actual column I_h are not resolved by its
16-element surface Q1 representation; the time-discrete history term amplifies
the current/previous representation mismatch. Independent integration
reproduces the implemented three-point projection, ruling out an adaptive
normal-integration error as the cause. Better quadrature on the same Q1 space
still fails. Offline bulk-aligned Q1 representation reduces the full-profile
defect below 1e-9 while the unchanged supported tail remains near 6e-5.

No equations, production algorithms, support policy or acceptance threshold
changed. Both failures and their valid spatial evidence are preserved. Stop
for a targeted surface-resolution/normalization-consistency test/design
decision; no 32x512 or temporal campaign follows. K3 is not converged.

## Pre-run question and design

This follows the user's broader Stage-K execution authorization. The earlier
32x256 support failure remains nonpassing and untouched. No production
algorithm, equation, loading, initialization, support policy, full I_h,
pressure treatment, quadrature, endpoint topology or solver tolerance changes.

Competing explanations to distinguish: (a) the earlier normalization/feedback
difference was largely a consequence of mesh-dependent timestep history;
(b) at common time/steps, the transverse phase/history response improves with
normal resolution; (c) a spatial component or seam/transfer effect remains
comparable to the intended feedback and limits the 1-D interpretation.

Use eight real steps of **.375 s through 3 s** on 32x128 and 32x256, keeping
the same 16 fault elements and tangential discretization. This cap also leaves
room below the estimated .428-s CFL restriction of a possible 32x512 level,
unlike .5 s. Actual accepted times are checked, not assumed. Artificial initial
time step remains 2 s and preserves the approved retained histories. Only
maximum real/first timestep, final time/step count, output directory, and normal
Box repetitions differ from the accepted smoke. Reference mechanics uses
identical physical loading at every common time.

An additional independent .375-s preflight (same 2048-cell/six-point reference)
passes all states: maximum total supported normalization **5.990494e-5**;
final cumulative feedback H=.352011991 Pa, phi=2.052559e-5,
I_h=.00330469510 m. This retains a measurable finite-step feedback, unlike the
.125-s reference's inactive H maximum. It is not a claim of temporal
convergence. The existing seam comparability rule uses these smaller pre-run
reference signals, not the larger original two-step smoke values. Thresholds
1e-4 for containment and complete normalization, and phi<.8, are unchanged.

Resource estimates before execution: **2--4 minutes for 32x128, 4--7 minutes
for 32x256**, roughly .6/1.0 GiB peak memory. Run sequentially with a 600-s
process-group cap per invocation and retained failure artifacts. A third level
is useful only if it can clarify an observed spatial trend; it is not launched
automatically for report completeness. No full convergence product is planned.

Benchmark-local changes generalize the runner's case/cap arguments, parameter/
geometry guards, exact-time reference reader and comparison from two to an
arbitrary contiguous number of exported steps. Existing reference tests pass
(7 tests, .509 s). Production executable and plugin are reused without rebuild.
The selected reference's independent spatial-noise check and actual production
results follow below; no successful-run claim is inferred from exit status alone.

## Executed evidence and common-time comparison

| Case | Accepted benchmark times (s) | Mechanically converged but guard-failed time | Wall time | Peak RSS |
|---|---|---|---:|---:|
| 32x128 | 0, .375, .75 | 1.125 s | 54.469 s | 598128 KiB (584 MiB) |
| 32x256 | 0, .375, .75 | 1.125 s | 115.075 s | 999148 KiB (976 MiB) |

Both invocations exit 1 on `supported_normalization=false`, not solver
nonconvergence. Aggregate ASPECT execution is 169.544 s. No failed case was
retried or overwritten. The intended eight-step trajectories were not
completed; both comparison JSONs explicitly record `complete_smoke=false`.
The exported mesh and parameters confirm unchanged tangential/fault resolution,
physical geometry and support, with precisely the predeclared parameter changes.

Each primary independent reference was rerun from its own continuum
initialization with the exact exported 0/.375/.75/1.125-s time/loading sequence.
Neither reference is reset from production H/C/stress. On this prefix the
reference H, phi and I_h are stationary; H first increases at 1.5 s in the
planned sequence. All production particle H values are also exactly unchanged
through the failed state. These new runs therefore do not constitute completed
evolving-feedback verifications.

| Time (s) | 128 h omission | 128 actual normalization | 256 h omission | 256 actual normalization |
|---:|---:|---:|---:|---:|
| 0 | 5.830274e-5 | 5.236487e-5 | 5.913959e-5 | 6.006423e-5 |
| .375 | 5.831816e-5 | 5.335760e-5 | 5.915475e-5 | 6.113454e-5 |
| .75 | 5.832140e-5 | 7.948384e-5 | 5.915810e-5 | 9.336290e-5 |
| 1.125 | 5.834184e-5 | **1.887058e-4** | 5.917923e-5 | **2.273543e-4** |

The reference normalization on this stationary prefix is 5.909980e-5.
Actual instantaneous/history/total signed integrals at every state and every
bulk quadrature column are preserved in each case's `crack_integrals_*.csv`;
the range summaries remain in `guard_*.json`. At the worst final column:

| Integral (m/s) | 32x128 | 32x256 |
|---|---:|---:|
| Local V | .001230467488073 | .001230568489124 |
| Actual instantaneous | .001230400500556 | .001230491433075 |
| Actual history | -1.652087794e-7 | -2.027189639e-7 |
| Actual total | .001230235291776 | .001230288714111 |

No full-I_h or tail renormalization is used in these measurements.

### Spatial improvements remain valid, with initialization errors visible

| Metric | 32x128 | 32x256 |
|---|---:|---:|
| Max phi0 error against primary reference | 2.074680e-4 | 7.839128e-5 |
| Max phi error at 1.125 s | 2.102788e-4 | 7.912859e-5 |
| Raw q RMS error at t=0 (Pa) | 1.089703 | .277851 |
| Raw q RMS error at 1.125 s (Pa) | 1.083978 | .263889 |
| Mean V at 1.125 s (m/s) | .001230504612 | .001230612157 |
| Mean Theta at 1.125 s (s) | 53.12897645 | 53.06817886 |
| Mean C at 1.125 s (Pa) | 327.4804185 | 326.2658672 |
| Mean I_h at 1.125 s (m) | 108.09913835 | 108.14579799 |
| Accumulated slip at 1.125 s (m) | .001337424007 | .001338580377 |

The common reference at 1.125 s gives V=.001230485533 m/s,
Theta=53.08540531 s, C=326.49322537 Pa, I_h=108.13492252 m and
slip=.001338252908 m. Initial C values are 318.7476558/317.5137627 Pa versus
317.7426503 Pa independently; those differences are retained, not removed by
resetting histories. Full transverse phi/H profiles and increment comparisons
are in both cases' `comparison_phase_*.csv` and `comparison_H_*.csv`.

A 4096-cell/eight-point independent check of the full .375-s reference sequence
finds at most 6.431e-7 absolute phi-profile difference from the 2048-cell
reference, 4.730e-10 difference in cumulative phi feedback, 4.036e-7 relative
I_h difference, and 1.007e-9 absolute normalization difference. This is
reference discretization evidence, not a production spatial pass.

## Localization identity audit: tails versus projection versus bulk quadrature

At each actual bulk column x, independently integrate the saved current and
previous Q1 phase profiles using 12-point normal Gauss quadrature split at all
bulk cells and exact support boundaries. Use the saved current/old surface
I_h and old C, not already-committed C as the previous state. Evaluating the
same expressions at the actual production QPs reproduces chi within 1.51e-14
and history within 1.87e-15 m/s. This checks both the diagnostic formula and
the actual previous FE phase input before interpreting the discrepancy.

Let J_k(x) be the independently integrated full profile, and hat I_k(x) the
interpolated surface Q1 value. Define e_k=J_k/hat I_k-1. Then the full integral
of the implemented localization minus V is exactly

    V e_k + beta C_(k-1)/kappa * hat I_(k-1) * (e_k-e_(k-1)).

This is an error-accounting identity, not a changed constitutive equation.
The zero-integral history identity requires the column and represented I_h to
agree at each x; a global consistent projection does not guarantee that.
At the worst endpoint column x=.249119518239 m, e_k is -2.0431/-2.5172 ppm and
e_(k-1) is -.28937/-.35679 ppm. The history amplification factors
beta C_old hat I_old/(kappa V) are **75.86/75.61**. Small spatial representation
errors therefore produce a significant full-profile normalization defect.

The following **signed** terms sum to (V-actual supported integral)/V at the
same worst column; independently selected extrema are not added together:

| Contribution | 32x128 | 32x256 |
|---|---:|---:|
| Full-profile identity defect using stored surface I_h | +1.350854e-4 | +1.658528e-4 |
| Omitted instantaneous + history tail | +5.967970e-5 | +6.055741e-5 |
| Actual bulk quadrature versus independently clipped integral | -6.059375e-6 | +9.441063e-7 |
| Actual supported defect (sum) | **+1.887058e-4** | **+2.273543e-4** |

The bulk quadrature contribution improves with normal refinement, while the
full-profile projection contribution worsens. In the fixed interior
x in [.046875,.203125], maximum actual errors are only 5.622702e-5/6.483053e-5.
The dominant discrepancy is localized near the independent open-fault endpoints.
Widening support would not remove a full-profile identity defect already above
1e-4. This does not establish universal support-policy incompatibility.

![Normalization and projected-I_h discrepancy](../../../benchmarks/reconstructed_fault/uniform_shear/evolving/common0375-identity.png)

## Independent projection checks and the limiting discretization

Source inspection confirms `build_owned_normalization_profiles()` uses three
Gauss points per fault element, followed by
`project_normalization_integrals_to_fault()`'s consistent Q1 mass projection.
The offline audit repeats that projection with independently integrated normal
profiles; stored nodal I_h agrees within **8.79e-9/3.73e-9 m**. Thus the
observed effect is reproduced by the specified discrete rule, not evidence of
an implementation violating that rule or a failed adaptive tail search.

Keeping all saved states fixed, compare the existing projection with a more
accurately integrated L2 projection on the **same 16-element space** and an
offline Q1 approximation aligned with the **32 bulk x cells**. The latter is
only a scalar representation experiment: no live fault, topology, C0, V/C,
history, support or coupled operator is modified or re-solved.

| Frozen-data representation | Max full-profile defect, 128 / 256 | Max independently clipped supported defect, 128 / 256 |
|---|---|---|
| Independent reproduction of three-point, 16-element projection | 1.350853e-4 / 1.658528e-4 | 1.947650e-4 / 2.264102e-4 |
| Bulk-cell-split L2 integration, same 16-element Q1 space | 1.153763e-4 / 1.416551e-4 | 1.750572e-4 / 2.022139e-4 |
| Offline bulk-aligned 32-element Q1 approximation | 7.201e-10 / 8.354e-10 | 5.968693e-5 / 6.056646e-5 |
| Exact independent column I_h diagnostic | <=1.02e-14 / <=1.02e-14 | 5.968758e-5 / 6.056722e-5 |

These clipped values intentionally exclude the separately measured ordinary
bulk-QP integration error; they are not claimed as results of corrected ASPECT
mechanics. Improving only quadrature on the coarse surface space is insufficient.
Resolving the tangential variation of the scalar normal integral is the
discriminating issue. A further normal-only 32x512 run leaves this limitation
unchanged and is not justified before reviewing that component.

The small along-fault phase variation is already present with unchanged H,
as the advected particle/domain representation enters successive phase solves.
At the failed state its full range is 2.7843e-6/2.9318e-6; ranges in I_h are
.000530517/.000656694 m, C=.00136890/.00169258 Pa, V=5.2167e-8/6.3037e-8 m/s.
There are no periodic wraps on this prefix. Maximum particle x displacement is
.000478329/.000478740 m, y displacement only 1.164e-9/1.256e-9 m, and H is
identical by stable ID. These observations do not prove a CPDI construction
bug; the normalization limitation is demonstrated independently of its cause.
No history-transfer or pressure investigation is reopened.

## Solver/lifecycle safeguards and next decision

Each case has **21 passing fresh-linear checks**, with maximum fresh/target
ratios .922663/.969278. Final bulk residuals are 9.8232e-12/1.3894e-11 against
targets 1.2090e-5/2.4180e-5; final surface RMS residuals are
2.7154e-13/3.7033e-13 Pa. All 17 surface nodes remain free. Phase entry/exit
and forced-restoration checks, stable-ID preceding-H handoff, exact Theta
updates (maximum discrepancy 1.43e-14 s), nondecreasing H, geometry invariance,
phi admissibility and the sequence-specific homogeneity checks pass. None of
these passing checks overrides the failed normalization gate.

**Recommended targeted next test/design decision:** verify localization
normalization on a frozen, mildly along-fault-varying Q1 bulk profile using the
existing 16-element versus bulk-aligned 32-element surface spaces, including
the amplified current/previous-I_h difference. Then decide whether a coupled
fault-resolution comparison using existing refinement parameters is the right
next K3 test, or whether a broader normalization-consistency design is needed.
Keep physical endpoints independent/open and preserve all other numerical
choices. A coupled test must initialize its own projections, record their
changes and retain the unmodified guard; the offline scalar substitution is
not a tested production fix. No such production change or new run is made here.

This is review conclusion **(2)**, not (1) convergence or (3) demonstrated
systematic tail/support incompatibility. The original K3 smoke remains a
bounded feasibility pass; K3 spatial/temporal convergence and Gate K2 remain
unestablished.

## Recoverable artifacts and commands

Under `benchmarks/reconstructed_fault/uniform_shear/evolving/`:

- `common0375.prm`, `spatial0375_n128.prm`, `spatial0375_n256.prm`: exact overlays.
- `spatial0375_n{128,256}.log`, `.resources.json` and output directories:
  separate failed runs, hashes, every-state bulk/fault/particle/profile exports.
- Matching `-reference/` and `-comparison.json`: exact-time independent prefixes
  and all spatial/history diagnostics, explicitly incomplete/nonpassing.
- Each case's `identity_audit_*.csv` / `.json` and `projection_audit_*.csv` /
  `.json`: column-by-column decomposition and projection experiments.
- `timestep-support/dt0375/`, `common0375-reference-accuracy/`: independent
  preflight and its normal-resolution check; `common0375-identity.png`: plot.

Executed from the repository root (the reference/diagnostic commands used
`OPENBLAS_NUM_THREADS=1 timeout 120`; each completed well below that cap):

```
python3 .../evolving/diagnose_timestep_support.py --dt .375 --output .../evolving/timestep-support/dt0375
python3 .../evolving/run_smoke.py --case spatial0375_n128 --wall-cap 600
python3 .../evolving/run_smoke.py --case spatial0375_n256 --wall-cap 600
python3 .../evolving/reference.py --parameters CASE/parameters.prm --ramp-peak .00225 --accepted-times CASE --output CASE-reference
python3 .../evolving/compare_smoke.py --case CASE_NAME --expected-steps 8
python3 .../evolving/audit_profile_identity.py CASE
python3 .../evolving/audit_ih_projection.py CASE
python3 .../evolving/plot_common_spatial.py
```

Here `...` means `benchmarks/reconstructed_fault/uniform_shear`, and `CASE`
is the corresponding full output path. Resource JSONs preserve unabridged
ASPECT commands, executable/plugin hashes and HEAD
`dc1a96d3f72dce123393c90ad025e729885e8ee9`. No executable/plugin rebuild occurred.
The identity audits assert QP agreement and signed additive accounting for
every column, and the frozen projection experiment uses independent normal
integration. Existing seven reference tests pass after generalizing the
exported-time reader. Changes in this action are benchmark-local scripts,
fixtures, diagnostics and this verification record; unrelated working-tree
changes and all earlier failures remain untouched. No MPI, normal512 or
additional production temporal case was run.
