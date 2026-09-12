# K3 bounded reference-only ramp adjustment

## Conditions declared before candidate execution

The 9e-3 m/s ramp is rejected for smoke execution, not a failed physical model.
For evolving profiles, omitted h fraction alone does **not** certify support
adequacy: truncating the history correction can dominate. Retain full I_h and
both diagnostics; the primary normalization check is the actual supported
integral of the complete crack strain (instantaneous plus history terms).

Test **4.5e-3 m/s first**, and **2.25e-3 m/s only if necessary**. Stop at the
first passing candidate. No other ramp, support adjustment, normalization,
criterion change or ASPECT run is authorized. Keep the same 2-s ramp duration,
initial loading 1e-4 m/s, histories, .1 activation cutoff, and timestep rules.

Each candidate must have finite admissible profiles with raw phi >=-1e-4,
g>0 and max(phi)<.8, bracketed interior mechanical roots with V>V_min and
absolute scalar residual <1e-7 Pa. Initialization and both real steps must have
omitted h fraction <=1e-4 and supported |integral upsilon-V|/max(|V|,Vref)
<=1e-4. Full-profile identities must also hold; no truncation renormalization.

Feedback is measured by max|H1-H0|, max|phi2-phi1| and |I2-I1|, not just a
core maximum that may remain pinned by the history maximum. Each must exceed
the corresponding reference noise by at least 10x. Estimate noise by one
2048/Gauss-6 versus 4096/Gauss-8 reference comparison with unchanged physics;
use the larger of profile/state disagreement and signal disagreement. Enforce
the existing reference accuracy targets (1e-6 absolute phi; 1e-5 relative
I_h/V/C and maximum H). This is a reference self-check, not production
convergence. If both ramps fail, stop for support-policy review.

The independent continuum initialization remains the primary reference. A
separately labeled conditional production-initialized calculation may read
only timestep-zero saved K1 profiles and retained histories, documenting its
1-D reduction; never reset either calculation from later ASPECT output or
present the conditional run as an independent initialization. Initialization
representation effects must remain visible.

Retain source-consistent RK2 wrap estimates. For future smoke, require measured
along-fault H/phi/I_h/C/V including seam and endpoint regions. Wrapping is not
failure; seam perturbations comparable to the intended feedback invalidate
1-D interpretation. Prefer a noncommitting production-residual comparison
R(phi1;H0), R(phi1;H1), R(phi2;H1), with stable-ID verification of committed H1.
Full CPDI weight/gradient export is a fallback only if that check fails.

Reference commands are capped at 120 s each, 600 s aggregate. Expected cost
is seconds, not minutes. No ASPECT execution follows automatically from a
reference pass: report the selected fixture and smoke resource estimate first.

## Decision and measured reference results

**Select U_max=2.25e-3 m/s. Stop the bounded adjustment; no further peaks.**
The .0045-m/s candidate fails the unchanged primary normalization criterion.
The .00225-m/s candidate passes the independent continuum preflight and a
separately initialized conditional calculation. No ASPECT build/run, production
code change, support change or relaxation of the 1e-4 limits occurred.
The smoke overlay now specifies the selected ramp; it remains unexecuted.

| Ramp peak (m/s) | Second-step time (s) | Omitted h fraction | Supported normalization error | Decision |
|---|---:|---:|---:|---|
| .009, retained earlier evidence | 2.43403 | 9.25172e-5 | 5.15884e-4 | Rejected |
| .0045, first authorized candidate | 2.86806 | 6.66708e-5 | 1.47964e-4 | Rejected |
| .00225, second/final candidate | 3.71235 | 6.16728e-5 | 8.69273e-5 | Reference pass |

All use full I_h and unchanged support. For the selected continuum candidate,
initial omission and first-step normalization are 5.90998e-5. At step 2 the
omitted history integral is +5.70184e-8 m/s; the retained history integral has
the opposite sign. Dividing this omitted contribution by V2 adds 2.52545e-5
to the 6.16728e-5 instantaneous deficit. The full-profile normalization error
is <=2e-16. This decomposition is the K3-specific reason to retain **both**
diagnostics, with supported total crack strain as the primary normalization.

### Selected independent continuum trajectory

| Quantity | Initial/retained | Step 1 | Step 2 |
|---|---:|---:|---:|
| Physical time (s) | 0 | 2 | 3.712347667 |
| Real dt (s) | — | 2 | 1.712347667 |
| V (m/s) | evaluated .000315270817 | .002320272775 | .002257753782 |
| q (Pa) | retained 1500 | 1331.148593 | 1295.384606 |
| C (Pa) | retained 317.742650 | 353.939029 | 382.840151 |
| Theta (s) | retained 200 | 2.357309832 | .4830068452 |
| Maximum phi | .5997028553 | .5997028553 | .5997837943 |
| I_h (m) | 108.1349225 | 108.1349225 | 108.2867776 |

The first timestep remains 2 s; the second is convection-limited, without
changing the maximum-step or RSF rules. Both roots are strictly interior:
F(V_min)=5408.09/4632.76 Pa; achieved absolute scalar residuals are
4.12e-12/2.57e-11 Pa. Profiles stay on the admissible branch with positive g;
the .8 envelope is comfortably satisfied. Histories are initialized once,
never reset from later output. The numerical timestep-zero interval does not
physically advance H0 or Theta0 or replace the retained supplied stress/C0.

The H response is transverse, **not growth of the already large core maximum**:
near y=-.204003899 m, H0=.5 Pa becomes H1=4.43555498 Pa. The core H_max remains
29568.04595 Pa. Max|phi2-phi1|=.000953568; its core increase is only
.0000809390. I2-I1=.151855077 m (.140431%). Therefore checking only H_max
or only core phi would mischaracterize this fixture's intended signal.

The reference phase-entry residual at step 2 is .0116649 and falls to
4.61e-15 after solving with H1. Substituting retained H0 at phi2 gives .0113976.
These norms refer only to this independent discretization; future smoke must
repeat the comparison with production assembly, not compare these raw norms
across discretizations.

One same-physics 4096/Gauss-8 accuracy check follows the 2048/Gauss-6 candidate:

| Feedback quantity | Signal | Conservative measured reference noise | Signal/noise |
|---|---:|---:|---:|
| Maximum absolute H change (Pa) | 3.93555 | .00626183 | 628.5 |
| Maximum absolute phi change | .000953568 | 6.43061e-7 | 1482.9 |
| Absolute I_h change (m) | .151855 | 4.36359e-5 | 3480.0 |

Noise includes the entire sampled-state disagreement and the signal
disagreement, whichever is larger. All predeclared accuracy targets pass.
The finer normalization error is 8.69279e-5, only 6.486e-10 different.
This is a preflight estimate, not a fully converged CPDI or ASPECT reference.

## Separately labeled conditional production-initialized reference

Use only saved timestep-zero files from
`nonuniform/global-accumulator/k1_short/{phase_0,particles_0,surface_0}.csv`.
This is the accepted homogeneous **domain-rule** K1 replay, not an old
point-rule trajectory. File hashes are in the conditional JSON report.
The later continuous-stress transfer correction cannot change the constant
supplied stress at zero; no later history/output is consumed here.

The reduction keeps the saved Q1 phase profile, volume-averaged particle H
at each normal row as P0 on row-midpoint strips, and length-averaged saved
Q1 surface C0/I0/Theta0. It retains particle stress q0=1500 Pa. Along-row
H spread is zero; phi spread is 1.255e-14; surface C/I_h/Theta spreads are
7.05e-12/1.46e-12/1.58e-11. Thus tangential reduction is justified for these
initial data, not assumed for a future wrapping/evolving run.

The saved state is **not** re-equilibrated or adjusted to make the independent
weak residual zero before starting. C0=318.7476558 Pa and I0=108.0980951 m
are distinct from the continuum initialization. The independently integrated
saved Q1 profile differs from stored I0 by 8.79e-9 m; preserve this mismatch,
not a normalization reset. Its predicted first-step full-profile normalization
defect is 5.98e-10, explained by the retained I0/profile-integral mismatch and
negligible relative to the 1e-4 support allowance. Step 2's full identity
returns to roundoff after its own consistent previous-I_h snapshot.

| Conditional quantity | Step 1 | Step 2 |
|---|---:|---:|
| Physical time (s) | 2 | 3.712451378 |
| V (m/s) | .002319831526 | .002257726131 |
| C (Pa) | 354.8616385 | 383.7370962 |
| I_h (m) | 108.1190798 | 108.2752145 |
| max phase change | .000303743 | .000975343 |
| max H increment (Pa) | 4.12452964 | .20858939 |
| Omitted h fraction | 5.88060e-5 | 6.14085e-5 |
| Supported normalization error | 6.25053e-5 | 8.70143e-5 |

Initial omission is 5.83027e-5. The first-step phi change is a **discretization
representation adjustment**: saved CPDI phi0/H0 are not an exact equilibrium
of the independent P1/P0-row problem. This change remains visible and is not
counted as H1 feedback. Step 2 responds to that calculation's newly committed
H1 without resetting anything. Conditional H/phi/I_h signal/noise ratios are
1395/1668/3981 under its own single accuracy check. Both references pass but
remain distinct; conditional initialization is not independent validation of
the production initializer or an exact production-reference trajectory.

The first conditional attempt exposed one artificial independent integration
element of width 3.25e-19 m: a saved symmetric row midpoint rounded just below
zero beside an exact mesh zero. Merging only roundoff-coincident **reference
integration splits** resolves it. No production geometry, data, tolerance or
H value was altered. The regression checks that this sliver does not recur.

## Future smoke diagnostic and seam requirement

Use the existing production phase residual with separate scratch matrix/RHS
and explicit probe bulk vectors, not a duplicated CPDI formula. Existing
`tests/phase_field_test_access.h` reaches the private assembler; the assembler
supports residual-only evaluation but currently reads H directly from particles.
A benchmark-only probe must scope/safely restore H substitutions and never
publish trial histories or touch the live solver matrix/RHS. Hold domains,
constraints, other histories and chemistry fixed for the three evaluations.
No new production constitutive API or CPDI weight export is required for this
first check. Verify stable-ID H1 against the preceding accepted commit before
substitution; failures, MPI ownership changes and exceptions must not leak a
modified history. This instrumentation is planned, **not implemented/tested**.

Record R(phi1;H0), R(phi1;H1) and R(phi2;H1), the phase solver's genuine final
criterion and stable-ID lifecycle data. Escalate to full CPDI weights/gradients
only if these checks fail. Do not refresh live fields merely for visualization.

Both selected references predict 0/358 periodic crossing events over steps
1/2. Maximum displacement is .000237600/.002716932 m, against particle spacing
.002604167 m. The RK2 estimate uses old/extrapolated velocity before mechanics,
not the new accepted solution. Wrapping is allowed; CPDI/seam behavior is not
validated by these estimates. Measure along-fault H, phi, I_h, C and V at
accepted times, including endpoints and seam strips, and compare seam-localized
signals to the **actual transverse feedback** above. A comparable artifact
stops 1-D interpretation. Initialization differences must not be subtracted
away as if they had no subsequent mechanical influence.

## Resources, checks and recoverable artifacts

No ASPECT smoke or build has been launched. The future one-rank 32x128 Release
smoke remains estimated at **60--120 s and 0.7--1.2 GiB**, initialization plus
two real steps, hard 120-s cap with no automatic retry. This is an estimate
from saved runs, not a measured evolving-run cost. A build, when authorized,
uses -j4. Report/review precedes execution; no larger campaign is authorized.

Successful reference calculation times (internal seconds): .0522 for .0045,
.0494/.1167 for selected independent coarse/fine, .2178/.2816 for conditional
coarse/fine. Final reference tests: **6 passed in .355 s**. Including process
startup, the failed conditional construction and bounded diagnosis, execution
was below 15 s aggregate; no command approached the 120-s cap. No failed/timed
out ASPECT harness was repaired or retried. `git diff --check` is also checked.

Under `benchmarks/reconstructed_fault/uniform_shear/evolving/`, retain:

- `ramp-0045/` and `.log`: failed first candidate;
- `ramp-00225/`, `ramp-00225-accuracy/`, corresponding logs and
  `ramp-00225-assessment.json`: selected continuum candidate;
- `ramp-00225-conditional/`, `ramp-00225-conditional-accuracy/`, logs and
  `ramp-00225-conditional-assessment.json`: separate initial-data calculation;
- `adjustment-tests.log`, `reference.py`, `assess_adjustment.py`,
  `test_reference.py`, updated `smoke.prm` and README.

Commands from the repository root (each final command exited zero; JSON gate
flags, not process exit alone, determine preflight acceptance):

```sh
# P denotes the already resolved validated K1 parameter file below.
K3_PARAMETERS=benchmarks/reconstructed_fault/uniform_shear/residual-floor/convergence/space32_dt05/parameters.prm
K3_DIR=benchmarks/reconstructed_fault/uniform_shear/evolving
OPENBLAS_NUM_THREADS=1 timeout 120 python3 "$K3_DIR/reference.py" --parameters "$K3_PARAMETERS" --ramp-peak .0045 --output "$K3_DIR/ramp-0045"
OPENBLAS_NUM_THREADS=1 timeout 120 python3 "$K3_DIR/reference.py" --parameters "$K3_PARAMETERS" --ramp-peak .00225 --output "$K3_DIR/ramp-00225"
OPENBLAS_NUM_THREADS=1 timeout 120 python3 "$K3_DIR/reference.py" --parameters "$K3_PARAMETERS" --ramp-peak .00225 --cells 4096 --quadrature 8 --output "$K3_DIR/ramp-00225-accuracy"
OPENBLAS_NUM_THREADS=1 timeout 120 python3 "$K3_DIR/reference.py" --parameters "$K3_PARAMETERS" --ramp-peak .00225 --initial-data benchmarks/reconstructed_fault/uniform_shear/nonuniform/global-accumulator/k1_short --output "$K3_DIR/ramp-00225-conditional"
OPENBLAS_NUM_THREADS=1 timeout 120 python3 "$K3_DIR/reference.py" --parameters "$K3_PARAMETERS" --ramp-peak .00225 --cells 4096 --quadrature 8 --initial-data benchmarks/reconstructed_fault/uniform_shear/nonuniform/global-accumulator/k1_short --output "$K3_DIR/ramp-00225-conditional-accuracy"
OPENBLAS_NUM_THREADS=1 timeout 120 python3 "$K3_DIR/assess_adjustment.py" "$K3_DIR/ramp-00225" "$K3_DIR/ramp-00225-accuracy" --output "$K3_DIR/ramp-00225-assessment.json"
OPENBLAS_NUM_THREADS=1 timeout 120 python3 "$K3_DIR/assess_adjustment.py" "$K3_DIR/ramp-00225-conditional" "$K3_DIR/ramp-00225-conditional-accuracy" --output "$K3_DIR/ramp-00225-conditional-assessment.json"
OPENBLAS_NUM_THREADS=1 timeout 120 python3 "$K3_DIR/test_reference.py" -v
```
