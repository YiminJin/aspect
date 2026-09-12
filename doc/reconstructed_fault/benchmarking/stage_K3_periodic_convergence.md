# K3 corrected-periodic convergence study

## Authorized question and execution sequence

The periodic-domain/seam, fault32 I_h representation, supported normalization,
and whole-fault homogeneity issues are accepted as resolved in the tested
configuration. This study changes only normal resolution and then timestep.
It does not reopen those implementations or change any equation, loading,
H maximum rule, support, full I_h, pressure, solver tolerance or gate.

Reuse the passing corrected-periodic 32x128/fault32 run through 3 s at dt=.375 s.
First run a fresh 32x256/fault32 case with the same accepted time sequence.
The competing possibilities are contraction of the transverse representation
error versus a limiting feedback-increment error. Estimated cost: 230--300 s,
about 1.0--1.5 GiB peak RSS; allow a 600-s safety cap. Do not mix old wall-domain
trajectories into this comparison. Reuse the independent continuum reference
only after checking every accepted time/dt/loading value; its histories remain
initialized once independently.

Add 32x512/fault32 only if contraction leaves feedback increments unresolved.
Once the normal resolution is adequate, compare nested timesteps through the
same 3 s. The independent reference will determine the useful temporal levels;
suppression of feedback by the specified maximum rule is a result, not a
reason to change that rule or the loading. Retain all per-state admissibility,
normalization, geometry, lifecycle, seam and fresh-linear checks.

Production source and binary remain the tested revision recorded in
`stage_K3_periodic_domain_review.md`. Only benchmark overlays, runners and
analysis are changed in this convergence task. Results follow below.

## Gate decision: unmet, new phase nonlinear exhaustion

The fresh 32x256/fault32 case passed initialization and its first real step
at .375 s. At the next phase solve, t=.75 s, all 50 configured nonlinear
iterations were exhausted at relative residual **2.430e-8**, above the unchanged
**1e-8** tolerance. The configured abort policy correctly stopped the run;
no t=.75-s mechanical/history state was accepted. This is a new reproducible
phase convergence obstacle, not a reopened support, seam, I_h representation,
surface or coupled-mechanical failure. No production correction or tolerance
change is made. No 32x512 or production temporal run follows the failure.

Elapsed time was **78.583 s**, peak RSS **1,112,960 KiB (1.061 GiB)**, exit 1.
There was no timeout, retry or parameter adjustment. The tested executable and
plugin hashes match the accepted corrected-periodic baseline. Fully resolved
parameters differ from that baseline only in Y repetitions (4 to 8) and output
directory. The tangential mesh, fault32, initial physical data and dt are fixed.

### What the phase failure establishes, and what it does not

The phase-entry probe records ||R|| = 5.692112860525105e-7, using the previous
committed H and the same domains/phase input as this solve. The phase constraint
pattern is unchanged; the intervening constraint refresh changes prescribed
mechanical boundary values, not phase constraints. This implies an absolute
target of **5.6921e-15**. Combining the entry norm and rounded final log gives
an absolute terminal residual of approximately **1.3832e-14**; this is an
inference from the recorded normalization, not a separately exported final
high-precision residual.

| Nonlinear iteration (zero based) | Logged relative phase residual |
| --- | ---: |
| 0 | 1.130e-7 |
| 1 | 2.552e-8 |
| 5 | 2.448e-8 |
| 6--49 | 2.430e-8 |

Late iterations report 11 CG iterations and exhausted phase line searches.
The preceding successful phase solve ended at absolute residual 1.53598e-14,
but its entry norm was 3.08647e-6, so its relative residual 4.97651e-9 passed.
The comparable absolute residual levels make finite-precision residual/update
effects a credible hypothesis. They do **not** prove whether assembly
cancellation, representation of the Newton update, or an inaccurate
linearization/direction is responsible. Phase CG reports its iterative stopping
test, not the fresh constrained action audit used by the separate coupled
Stokes solve. No conclusion about a new stopping allowance follows from these
logs alone; an iteration-budget increase is not supported by the plateau.

The smallest next action is a bounded capture of this exact phase linearization:
check a fresh constrained linear residual, compare the Jacobian prediction with
fresh residuals using the actually represented trial increment, and split the
reaction and gradient contributions. Use higher-accuracy diagnostic summation
only if cancellation remains implicated. Preserve the current solve and its
stopping rule until that discrimination supports a correction. Do not start a
new convergence level to work around this failure.

## Accepted-prefix spatial comparison

Both levels are compared to independently initialized continuum histories at
their common accepted times 0 and .375 s. The new reference uses only the
fine run's accepted time/dt/U exports, not its evolving histories. Complete
transverse phi and H profile tables are in each run's `comparison_phase_*.csv`
and `comparison_H_*.csv`; `periodic-convergence/accepted-spatial.csv` contains
scalar, history and gate summaries.

| Error or increment at .375 s | 32x128/fault32 | 32x256/fault32 |
| --- | ---: | ---: |
| Maximum phi error | 2.086434e-4 | 7.927690e-5 |
| Absolute I_h error [m] | 3.596363e-2 | 1.077626e-2 |
| Absolute V error [m/s] | 1.852678e-6 | 5.541213e-7 |
| Absolute C error [Pa] | .993822 | .229101 |
| Absolute Theta error [s] | .0845411 | .0252742 |
| Raw q stress RMS error [Pa] | .908435 | .224775 |
| Raw q stress maximum error [Pa] | 2.591732 | .553445 |
| H-profile maximum error [Pa] | .0230953 | .0258859 |
| H-profile RMS error [Pa] | .00359007 | .00370328 |
| Maximum phi increment error | 1.175413e-6 | 1.177936e-6 |
| I_h increment [m], independent increment = 0 | 8.638083e-4 | 8.649452e-4 |

The total phi, I_h and mechanics errors contract, but this accepted prefix
does not establish contraction of feedback increments. H1-H0 is zero on both
levels and in the reference; its profile error is inherited initial
representation/reference-sampling error, not newly generated H feedback.
The initial phi errors are 2.074680e-4 / 7.839128e-5, and initial signed I_h
errors are -.0368274 / +.00991132 m. These changes remain physically present;
increment subtraction is diagnostic accounting, not removal of initial error.
H maximum-error locations differ between particle grids; no exact analytic
reproduction from cell/domain averaging is presumed.

All accepted fine states pass geometry, admissibility, support/normalization,
whole-fault homogeneity and lifecycle guards. Maximum omission / normalization
are **5.915475e-5 / 6.113454e-5**. All **13** fresh coupled-linear checks pass.
Stable-ID H handoff and normal/forced-failure probe restoration also pass on
entry to the failed step. There is no successful phase-exit or committed state
for that step, and no claim of passing its later gates.

## Independent temporal result (not production timestep convergence)

The permitted nested reference calculations use the identical physical ramp,
support, full I_h, retained initialization and H maximum rule, through 3 s.
They use 2048 reference cells with independent 4096-cell checks. Existing
dt=.375 reference evidence is reused. No support search or normalization
adjustment is performed.

| dt [s] | max(H_final-H0) [Pa] | max|phi_final-phi0| | I_h,final-I_h,0 [m] | Maximum candidate/H_old (4096) |
| --- | ---: | ---: | ---: | ---: |
| .375 | .35201199 | 2.052559e-5 | .0033046951 | 1.268689 |
| .1875 | 0 | 0 | 0 | .851374 |
| .09375 | 0 | 0 | 0 | .425796 |

At the two smaller timesteps every admitted driving candidate remains below
the retained history. The specified maximum rule therefore leaves H unchanged;
phi and I_h remain stationary to reference noise (4096-cell changes below
8.72e-15 and 9.10e-13 m). The .375-s feedback's 2048-to-4096 differences are
1.894e-6 Pa, 1.179e-9 and 8.592e-8 m, respectively, much smaller than the
signal. Suppression at smaller dt is consequently a result of this discretized
history law and loading, not reference-resolution noise. The finite-dt feedback
must not be advertised as a timestep-converged nonzero evolution.

Mechanics continues evolving even on the stationary-profile branches:

| dt [s] | V(3) [m/s] | Theta(3) [s] | C(3) [Pa] | q(3) [Pa] | Slip(3) [m] |
| --- | ---: | ---: | ---: | ---: | ---: |
| .375 | .00224318592 | 1.58347820 | 355.626770 | 1270.04679 | .00517921926 |
| .1875 | .00224485094 | 1.81385944 | 353.964870 | 1265.99486 | .00499483266 |
| .09375 | .00224474627 | 1.95086331 | 353.099921 | 1263.89878 | .00489956984 |

All reference states remain admissible and below both 1e-4 support criteria;
maximum total normalization errors are 5.99049e-5, 5.90998e-5 and 5.90998e-5.
The final mechanical values and accumulated slip still have timestep
dependence. No production temporal-convergence claim is made from reference
results alone. The physical equations and maximum rule are retained without
tuning around this finding.

## Artifacts, checks and resumption point

All paths below are relative to
`benchmarks/reconstructed_fault/uniform_shear/evolving/`:

- `spatial0375_n256_f32_periodic.prm`, `.log`, `.resources.json`, and output
  directory: unchanged-physics failed run, with accepted prefix preserved.
- `spatial0375_n256_f32_periodic-reference/` and `-comparison.json`: exact
  accepted-prefix reference and comparison; `complete_smoke=false`.
- `periodic-convergence/reference{01875,009375}/` and all three
  `reference*-4096/`: independent reference accuracy/temporal evidence.
- `periodic-convergence/summary.json`, `accepted-spatial.csv` and
  `accepted-prefix-and-reference.png`: reproducible summary/profile plot.
- `convergence_reference.py`, `periodic_convergence_summary.py`: benchmark-only
  calculations; existing runner/comparison gain the new case selection.

The unchanged smoke-guard regression passes (1 test, 1.427 s), including the
deliberate history-term perturbation that must fail normalization. The summary
script's initial log-parser whitespace mismatch was fixed locally; it affected
only offline reporting, not a simulation or a gate. Production was neither
rebuilt nor edited in this task. No full suite was run.

Gate K3 remains unmet: spatial feedback convergence and production temporal
convergence cannot be certified beyond the new phase failure. The accepted
periodic-domain and fault32 results remain closed; this evidence does not
contradict them. Resume with the bounded phase-residual consistency audit
above, not architecture development or an extra grid/forcing campaign.
