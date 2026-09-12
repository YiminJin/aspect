# Stage K4.1 — Independent finite-width reference comparison

## Decision

**Initial width dependence is independently resolved, but Stage 4.1 does not
yet provide the temporally qualified basis required by the approved Stage
4.2 plan.** Both widths pass admissibility and support checks. Reference
spatial uncertainty is small. Neither the .5/.25 nor .25/.125-s timestep
comparison meets the required quarter of the resolved K1 observable allowance
over the complete transient. Stop at this stated condition; no further
timestep, production run, or Stage 4.2 case was launched.

This is not a solver failure, an I_h-feedback investigation, or a reason to
change initialization. It is an independently observed finite-step temporal
sensitivity, particularly during the initial mechanical transient.

## Unchanged problem and independent calculation

Use the fully resolved validated K1 parameters, recorded by path/hash in each
initialization JSON. The box is 0.25 by 1 m, fault y=0, prescribed normal
stress 1000 Pa, and K1 loading is
U(t)=1e-4[1+0.2 min(t/4,1)] m/s through 6 s. Retain the original G=1e6 Pa,
eta=1e8 Pa s, Gc=26.666666666666667, cohesion=1000 Pa, friction/state/damping
parameters, initial particle shear stress 1500 Pa and Theta0=200 s. Only ell
changes: .15625 to .078125 m. The phase core remains .6 and activation is .1.

Reuse `evolving/reference.py` for the independent P1 normal phase solve and
scalar mechanics, not production constitutive calls or CPDI. Recompute the
stationary H initializer, calibration and phase equilibrium at each ell.
The AT1 calibration m=Gc/[(8/3) ell Hc] changes from 128 to 256; Hc=.5 Pa.
Natural phase boundaries at y=+/-0.5, the existing admissible branch and
activation/truncation rules are unchanged. The converged H/phi are then
frozen: no K3 H update or subsequent phase solve occurs in these K1 histories.

For each independent initialization compute full I_h and the same continuum
admitted-strip average C0 used by K3. Initialize q/C/Theta/I_h histories once.
The artificial 2-s timestep-zero mechanics is evaluated but does not replace
retained q0/C0/Theta0/H0. For real steps, solve

\[
q_k=\beta q_{k-1}+\kappa(U_k-V_k),\quad
C_k=\beta C_{k-1}+\kappa V_k/I_h,\quad
q_k-C_k-\sigma_*\mu(V_k,\Theta_{k-1})-\eta^d V_k=0.
\]

Theta uses the existing exact frozen-V aging update. Fixed h and I_h imply
the signed history-localization correction is identically zero, not omitted
as an approximation. The full integral of crack strain is V; the supported
integral is (1-f_omitted)V. Velocity follows
u(y)=-U/2+(U-V)(y+.5)+V int_{-.5}^y h/I_h dz.
Both prescribed endpoint velocities are checked explicitly.

## Minimum checks performed

- Four independent initializations: 2048 and 4096 uniform normal intervals
  at each width, with extra splits at support and activation boundaries.
- Six/eight-point quadrature comparison of each *same* P1 phase profile;
  this checks integral evaluation separately from phase discretization.
- Scalar trajectories at .5 and .25 s on each fine reference; .125 s was
  added only after that comparison failed. Coarse-reference .5-s trajectories
  quantify initialization/reference-resolution impact on mechanics.
- No production FE/fault grid exists in this calculation. No ASPECT, MPI,
  production rebuild, refinement campaign or new solver verification was run.

The fixed K1 resolved allowance is .002 |reference| + 1e-5 scale, with scales
V/velocity=1e-4 m/s, traction=1500 Pa, Theta=200 s and slip=6e-4 m.
The 4.1 target is one quarter of this allowance. Velocity maximum and L2
errors are both checked. Timestep comparisons use common physical times;
differences are empirical uncertainty estimates, not assumed Richardson bounds.
The finest tested .125-s trajectory is **diagnostic, not a selected timestep**:
there is no finer test certifying its own error.

## Resolved initial width effects

Values use the fine independent references. Uncertainty is the sum of the
absolute 2048-to-4096 changes at the two widths, not a statistical confidence
interval. Analytically prescribed H0 and policy geometry do not have a phase
grid error; their zero grid change is not a separate proof of model accuracy.

| Quantity | ell0 | ell0/2 | Half minus full | Reference-grid uncertainty |
| --- | ---: | ---: | ---: | ---: |
| Full I_h (m) | 108.134966 | 107.790689 | -.344277 | .0002210 |
| Retained C0 (Pa) | 317.743475 | 300.080450 | -17.663026 | .004952 |
| Center phi | .599702212 | .599538659 | -.000163554 | .000003220 |
| Prescribed maximum H0 (Pa) | 29568.05 | 118118.45 | +88550.40 | Analytic input |
| Policy support half-width (m) | .308821594 | .154410797 | -.154410797 | Analytic input |
| Activation distance (m) | .204000486 | .102000243 | -.102000243 | Analytic input |
| Localization second moment (m2) | .004742849 | .001171160 | -.003571689 | 2.3573e-8 |
| Localization RMS width (m) | .068868342 | .034222215 | -.034646127 | 2.5790e-7 |

The localization measure is h dy/I_h, centered on y=0. Symmetric profiles
have zero first moment. Phase and H profiles are saved in physical y and
y/ell coordinates; the width change is not just resampling one stretched
solution. The required calibration and initial-history differences remain
part of the finite-width model-family comparison.

Maximum coarse/fine full-profile phi differences are 1.42161e-6 and
5.69219e-6. Maximum mechanical reference-grid changes are respectively
1.29416e-9/6.48530e-9 m/s in V, .000802654/.00401752 Pa in q,
.000813332/.00406963 Pa in C, and 8.86574e-5/.000436695 s in Theta.
The worst reference-resolution error/quarter-allowance ratios are
**.00497 and .02625**: both comfortably below one.

The maximum omitted h fractions are **5.91008e-5** and **1.62482e-5**.
Complete supported-normalization errors satisfy the approved 1e-4 bound at
every reference state. Full-integral normalization and boundary velocities
pass 1e-12 checks. These are continuum-reference support measurements, not
claims about production particle-domain admission or bulk quadrature.

## Temporal uncertainty prevents selecting a common timestep

| Width | Compared timesteps (s) | Max change V (m/s) | q (Pa) | Theta (s) | Slip (m) | Worst ratio to quarter allowance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ell0 | .5 / .25 | 3.14200e-4 | 47.0239 | 6.02759 | 4.70847e-5 | 1190.9 |
| ell0/2 | .5 / .25 | 3.27514e-4 | 48.6292 | 6.14378 | 4.87044e-5 | 1208.3 |
| ell0 | .25 / .125 | 3.20658e-4 | 41.7343 | 5.84679 | 4.17846e-5 | 910.0 |
| ell0/2 | .25 / .125 | 3.34579e-4 | 43.2918 | 5.98906 | 4.33473e-5 | 927.6 |

Maxima are over common times for each pair; .25/.125 includes t=.25 s.
The worst normalized discrepancy is at t=.5 s. All roots are interior, with
maximum scalar balance residual <=1.41e-11 Pa, well inside the existing
1e-7 Pa root check. An independent replay through the original K1 scalar
`solve_step` agrees at both widths, including exact initial-history retention.
Thus the measured timestep sensitivity is not explained by an inaccurate root
or accidental reuse of timestep-zero evaluated stress as committed history.
This bounded check does not pursue a new temporal asymptotic regime.

## Width differences that can already be reported

At 6 s, the finest tested timestep (.125 s) gives the following. The final
column conservatively sums the .25-to-.125 changes of the two individual
trajectories. It is an empirical temporal uncertainty indicator, not a bound.

| Observable | Half minus full at 6 s | Temporal change of width difference | Sum of individual temporal changes |
| --- | ---: | ---: | ---: |
| V (m/s) | -4.24307e-9 | 6.57613e-11 | 1.33468e-8 |
| q (Pa) | -16.664274 | .0003490 | .311959 |
| C (Pa) | -16.447386 | .00003408 | .0193463 |
| Theta (s) | -1.116300 | .0010398 | .128326 |
| Slip (m) | +1.76608e-5 | 1.21040e-8 | 1.41366e-6 |
| Velocity-profile max difference (m/s) | 1.61408e-5 (unsigned) | 9.30736e-10 | 4.93505e-9 |

The integrated full crack-strain difference equals the V difference; its
history contribution is zero. The velocity-profile L2 width difference is
5.71519e-6 m/s. Several late-time width effects, notably C, q and the velocity
profile, clearly exceed reference and temporal indicators. The very small
final V width difference does not exceed the conservative temporal indicator.
Cancellation in a matched width difference cannot qualify individually
inaccurate early trajectories under the approved 4.1 criterion.

Consequently, this establishes a useful resolved **initial** width comparison
and informative finite-step mechanical effects, but not a temporally resolved
complete width-dependent trajectory for Stage 4.2. Review the next reference
timestep action before production; do not silently drop the early transient,
change supplied histories, or weaken the quarter-allowance requirement.

## Artifacts, verification and cost

Under `benchmarks/reconstructed_fault/uniform_shear/finite-width/`:

- `reference_check.py`: bounded initialization and scalar replay driver.
- `k41/initialization` data live in `ell0-{2048,4096}` and
  `half-{2048,4096}`; JSON records parameters, provenance, residuals and guards.
- `k41/decision.json`: all-time temporal/reference comparisons and explicit
  `ready=false`, `selected_common_dt_s=null`.
- `k41/width-comparison.json`, per-state velocity CSVs, `profile.csv` and
  `width-profiles.png`: physical/scaled profiles and width differences.
- `test_reference_check.py`: three checks passed in .016 s (K1 scalar/history
  cross-check, calibration/support/normalization, comparison/stopping logic).

The complete reference calculation took .327 s inside Python, about .53 s
wall time including interpreter startup. Summary/figure generation and tests
were also cheap offline operations. Existing reference quadrature and scalar
algorithms are reused unchanged. No existing evidence was overwritten, no
production source was edited, and no commit is made in this task.
