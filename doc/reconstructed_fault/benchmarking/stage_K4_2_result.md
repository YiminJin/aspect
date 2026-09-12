# K4.2: completed post-transient finite-width/discretization comparison

## Decision

The approved three-case comparison is complete. A and C reproduce their
independent references within every existing K1 observable tolerance on
**[4,6] s**, against both saved dt=.125 and .0625 trajectories. B fails only
the cohesive-traction tolerance. Normal refinement B→C reduces that error
by 4.02, without changing physical data or solver settings.

The width effects in actual weak traction q, cohesive traction C, Theta,
accumulated slip, velocity profiles and total Ih exceed the conservative
empirical uncertainty by the required factor four throughout the interval.
The much smaller V/full-integral and actual supported-integral contrasts
remain unresolved. They are not zero, and passing the 1e-4 normalization
guard does not make support truncation negligible relative to these signals.

This completes **post-transient K4.2**, not full-history K4.1, a universal
sharp-fault limit, or Gate K2/K3. The early transient and its later physical
influence remain in every trajectory. K4.3 was not started.

## Cases, lifecycle and cost

Each one-rank Release case independently initializes at t=0, then takes 48
real steps of .125 s through 6 s. The 17 samples from 4 through 6 s define
the accuracy window. No production history is used to reset a reference.
The reference's timestep-zero evaluated q/C responses remain distinct from
retained q0/C0/Theta0. Accumulated physical slip starts at zero.

| Case | ell (m) | Bulk mesh | Fault elements | Wall time (s) | Peak RSS (GiB) |
| --- | ---: | --- | ---: | ---: | ---: |
| A | .15625 | 32x256 | 32 | 966.847 | 1.061 |
| B | .078125 | 32x256 | 32 | 798.405 | .829 |
| C | .078125 | 32x512 | 32 | 1941.790 | 1.508 |

Tangential/fault spacing is .0078125 m throughout. Normal cell sizes are
.00390625/.001953125 m. Thus A and C both have ell/h_y=40, while B has 20.
The three completed runs cost 3707.042 s (61.8 minutes) in aggregate.
No additional level, temporal run, MPI run or performance campaign was added.

K1 loading, pressure treatment, material constants, core phase, activation
.1, particle density, history transfer, quadrature, open topology and solver
tolerances are unchanged. The ordinary K1 benchmark freezes the converged
initial phase field using phase-only constraints and keeps H fixed using the
existing material parameter; mechanics and its other histories advance.

## Production errors and normal refinement

Numbers below are maximum absolute errors over [4,6] s against the matching
dt=.125 **independent continuum-initialized** reference. They are not errors
after subtracting an initial offset. q is the actual particle/domain-based
weak surface traction, represented through its assembled mass matrix.

| Observable | A error | B error | C error | B/C contraction |
| --- | ---: | ---: | ---: | ---: |
| q (Pa) | .219720 | 1.006778 | .249036 | 4.04 |
| C (Pa) | .218789 | .991835 | .246654 | 4.02 |
| Theta (s) | .023970 | .076143 | .022867 | 3.33 |
| Slip (m) | 3.16736e-7 | 9.68933e-7 | 2.98608e-7 | 3.24 |
| V (m/s) | 8.83956e-9 | 8.77391e-9 | 3.92896e-9 | 2.23 |
| Ih (m) | .00986768 | .03751434 | .01121198 | 3.35 |
| Raw stress maximum (Pa) | .230685 | 1.111996 | .271053 | 4.10 |
| Raw stress RMS (Pa) | .220140 | 1.006375 | .249872 | 4.03 |
| Native-QP velocity maximum (m/s) | 3.71294e-9 | 1.05657e-8 | 5.04645e-9 | 2.09 |

All these error measures contract B→C. The velocity-profile L2 error also
contracts, by 1.65. A and C pass every existing K1 field/trajectory check
against **both** reference timesteps; B's cohesive traction fails both.
Neither failure is hidden by changing the comparison interval or tolerance.
Raw stress uses the actual **constrained FE old history consumed by mechanics**,
not the separately published FE field or newly committed particle stress.
No stress is smoothed.

## Width signal and conservative empirical uncertainty

For the refined-width comparison C−A, at each common time use

\[
 U_Q=\|Q_C-Q_{\ell/2,.125}^{ref}\|
     +\|Q_A-Q_{\ell,.125}^{ref}\|
     +\|\Delta_\ell Q_{.125}^{ref}-\Delta_\ell Q_{.0625}^{ref}\|
     +E_{ref,grid}.
\]

The last term reuses the summed **maximum** 2048/4096 reference-grid changes
from the existing .5-s resolution check (initial Ih for its constant value).
It is an empirical indicator, not a rigorous temporal/asymptotic bound.
No favorable cancellation between production errors is assumed. The original
factor-four criterion is unchanged. Width-specific temporal uncertainty uses
the explicitly approved K4.1b/post-transient scope; it does not certify 0--4 s.

| Observable | C−A at 6 s | Fine reference width difference at 6 s | Minimum signal/U on [4,6] | Decision |
| --- | ---: | ---: | ---: | --- |
| q (Pa) | -16.693049 | -16.664096 | 35.85 | Distinguishable |
| C (Pa) | -16.474816 | -16.447403 | 35.65 | Distinguishable |
| Theta (s) | -1.114756 | -1.116814 | 27.85 | Distinguishable |
| Slip (m) | 1.764271e-5 | 1.766700e-5 | 28.19 | Distinguishable |
| Ih (m), fixed | -.342933 | -.344277 | 16.16 | Distinguishable |
| Velocity maximum norm (m/s) | 1.614018e-5 | 1.614114e-5 | 894.54 | Distinguishable |
| Velocity L2 norm (m/s) | 5.715123e-6 | 5.715316e-6 | 764.10 | Distinguishable |
| V (m/s) | -9.15622e-9 | -4.27048e-9 | .161 | Unresolved |
| Actual supported integral (m/s) | -4.21916e-9 | +5.95954e-10 | .031 | Unresolved |

The ratios use the .125 reference signal; the .0625 signal is shown alongside,
and its difference is included in U. The conclusions are unchanged by using
the finer signal. The reference full crack-strain integral equals V for this
fixed profile; production actually assembles the **supported** integral.
Do not reinterpret the latter as a full-profile measurement. Its tiny width
contrast even changes sign relative to the full-Ih reference prediction with
support truncation measured independently. No renormalization is applied.

At 4 s, q contrasts are -15.782334 (B−A), -17.038149 (C−A) and -17.008751 Pa
(fine reference). Thus normal refinement removes most of the coarse-width
discrepancy while retaining a roughly 17-Pa model-family width effect.

## Initialization and profile representation remain visible

| Quantity | A | B | C |
| --- | ---: | ---: | ---: |
| Retained C0 (Pa) | 317.513763 | 301.118714 | 299.821936 |
| C0 minus independent initial reference (Pa) | -.229713 | +1.038264 | -.258513 |
| Stored Ih0 (m) | 108.144834 | 107.753174 | 107.801901 |
| Center phi0 | .599764017 | .599746894 | .599601351 |
| Maximum nodal phi0 reference error | 7.83736e-5 | 2.08235e-4 | 8.36369e-5 |
| Sampled maximum H0 (Pa) | 29561.738 | 118017.617 | 118093.217 |
| Localization second moment (m2) | .0047451541 | .0011714682 | .0011717474 |

H0's minimum is .5 Pa in all cases; its sampled maximum is not an analytic
peak-reproduction error. The independently computed continuous peaks are
29568.05 and 118118.45 Pa for the two widths. All initialized fields are
retained/frozen according to the approved K1 semantics, not fitted to output.

The half-width second-moment self-error is **not monotone**: 3.08e-7 m2 in B
versus 5.87e-7 m2 in C. Do not infer asymptotic convergence of every small
profile statistic from the mechanical-error contraction. Both errors are
measured against the independent reference and are small relative to the
3.57e-3 m2 width signal; the C−A width-moment error is 1.72e-6 m2. This bounded
result does not require or authorize another profile-moment/Ih investigation.

The width family includes its required degradation recalibration and changed
initial H/C. The traction difference is **not** attributed solely to changing
the diffuse mechanical kernel with identical initialized cohesive state.

## All-time numerical and physical guards

All 147 accepted-state guards pass (49 per case), including t=0. Geometry is
unchanged; all nodes stay strictly above Vmin (all free); frozen phi/H and
retained initial stress/Theta are verified. Every real-step Theta update and
fixed-profile cohesive update passes. Phase values remain below .6; B has
only a roundoff negative minimum (-5.32e-16), within the unchanged lower rule.

| Check | A | B | C |
| --- | ---: | ---: | ---: |
| Maximum omitted fraction | 5.913959e-5 | 1.576141e-5 | 1.622887e-5 |
| Maximum actual column normalization error | 6.006423e-5 | 1.343677e-5 | 1.659291e-5 |
| Maximum relative total particle-measure discrepancy | 2.34e-9 | 3.70e-9 | 4.72e-9 |
| Maximum Ih/independent FE-profile relative discrepancy | 3.46e-11 | 1.18e-11 | 4.23e-10 |
| Maximum prescribed-boundary velocity error (m/s) | 4.74e-20 | 5.42e-20 | 3.39e-20 |
| Fresh linear checks | 158 | 159 | 159 |
| Maximum fresh residual / requested target | .990924 | .988580 | .996171 |
| Maximum final bulk weak residual | 7.99e-10 | 1.28e-11 | 2.94e-11 |
| Maximum final surface RMS (Pa) | 5.31e-7 | 5.29e-7 | 5.29e-7 |

Both separate 1e-4 support/normalization requirements pass at every measured
column and accepted time. The signed history-localization integrals remain
roundoff zero (at most 5.67e-18 m/s); instantaneous and total supported
integrals are separately retained in every guard record. Every recorded
final bulk/surface residual passes its production target; exit zero alone
was not used as convergence evidence. Boundary traces are reconstructed from
native Q2 Gauss values, not a fitted velocity profile.

## Reproducibility, changes and limitations

Artifacts under `benchmarks/reconstructed_fault/uniform_shear/finite-width/`:

- `k42_A/B/C.prm`, their `.log`/`.resources.json`, and full raw export directories;
- `k42_A/B/C-analysis.json` and `-profiles.npz`: all-time errors, retained
  initialization, final residuals and represented profiles;
- `k42-comparison.json`: all 17 post-transient samples and uncertainty terms;
- `k42-export-verification.json`: all-time physical traces and Ih checks;
- `k42-profiles.png`, `k42-velocity-width-t4.csv`, `...-t6.csv`;
- `k42-tests.log`: **11 tests passed in 1.529 s**;
- `k42-configure.log`/`k42-build.log`: benchmark Release plugin built with -j4.

Reproduction commands: `production.py run k42_A` (then B and C),
`analyze_production.py k42_A` (then B and C), `compare_production.py`,
`verify_exports.py`, and `plot_production.py`. The runner refuses to overwrite
an existing run. Focused tests use
`python3 -m unittest discover -s benchmarks/reconstructed_fault/uniform_shear/finite-width -p 'test_*.py'`.

Production is unchanged at commit `86fff739569da2600d095bc2dca4aa4e6ac04035`.
Executable SHA256 is
`61261bbaa827dd4e368f3879aff8cc078e96fc67684e604e898c15605d84c7f5`;
benchmark plugin SHA256 is
`dc3a8d5405747e1572ea9278e6041a0da44b00c88b576457209678ab81964c31`.
The only C++ edit is an opt-in, read-only benchmark export guard. Other edits
are fixtures, analysis/tests and documentation; no production APIs changed.

The first A attempt stopped after 71.386 s on a mistakenly added **diagnostic**
1e-10 total-volume threshold, not an existing acceptance criterion. Saved
passing K3 evidence also violates it. The guard was corrected to the existing
positive-domain/measured-total semantics, and the attempt is preserved in
`attempt1-volume-guard/`. No physical/solver settings changed. The explicit
rerun then passed; there was no automatic retry or hidden failed evidence.

No ASPECT integration suite, MPI campaign, support change, K3 solver/Ih
investigation, or later K4 substage was performed. The small integrated-rate
width contrasts and the all-time temporal transient remain unresolved.
