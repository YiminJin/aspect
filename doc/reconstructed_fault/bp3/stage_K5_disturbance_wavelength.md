# One positive 600-m disturbance: comparison with the saved 200-m branch

## Conclusion

Relative-state amplification and absolute velocity-difference growth are **not
exclusive to the shortest, 200-m pattern**. The 600-m branch gives almost the
same final relative-state norm gain (1.07599 versus 1.07732), while absolute
velocity growth is weaker (1.18208 versus 1.26989).

There is an important qualification: the 600-m absolute-state norm decays
0.755%, and its relative-velocity norm decays 5.152%. The corresponding
200-m norms grow 0.520% and 6.649%. Thus changing wavelength changes the
response quantitatively and even the sign of some norm changes; neither
branch supports a blanket statement that every disturbance measure grows.
The longer pattern retains its broader spatial lobes rather than becoming
a dominant neighboring-node alternation during this year.

This is one wavelength comparison on one fixed grid, not spatial or temporal
convergence, an eigenmode calculation, or qualification of continuum stability.

## Single authorized run and unchanged inputs

Reused accepted step 11 from
`first_long_run/mechanical-discrimination-roundoff-clock/restart/01`, and the
completed `reference32` and positive `plus32` trajectories. The only new run
is `first_long_run/state-disturbance/plus60032`:

- Same one-year interval, 1144267468.9434748 to 1175825068.9434748 s.
- Same 32 steps of 986175 s, accepted steps 12--43.
- Same positive epsilon=1e-4 and 15--18 km cosine-squared taper, centered at
  16.5 km. Only the cosine wavelength changes from 200 to 600 m.
- Same bulk/fault grids, physical fields, full I_h, loading, work measure,
  split aging and stress updates, tolerances, and timestep guards.
- Four ranks, unchanged Release executable. The prepared PRM equals the
  saved 200-m PRM after replacing only its output-directory path. Launch
  environments differ only by the diagnostic wavelength variable.

The original checkpoint and executable hashes match the saved branch.
The prior diagnostic library is retained as
`performance/build-gmg/libfault_disturbance.200m-reference.release.so`, with
its original SHA256 `0313e634e2d1815f6ecbbe63d784e5e3116f2ffe65e18edd59d21ba8d68a1545`.
Only the separate diagnostic plugin was rebuilt, with `-j4`; no production
material/solver implementation changed. The new source/library snapshot and
exact command/input hashes are retained alongside the new run.

No reference, negative branch, feedback control, loading prefix, or extension
was run. The branch terminated at accepted step 43.

## Normalization and timing

For each branch use its **own exported Q1 pattern** w and the native consistent
work mass M, not the unperturbed reference file's diagnostic pattern:

\[
 m_w=w^TMw,\quad A=1^TM1,\quad
 \|f\|_{\rm RMS}=\sqrt{f^TMf/A},\quad
 P_w(f)=w^TMf/m_w.
\]

The common work measure is 115466.5035 m. Pattern masses are 377.800817 m
(200 m) and 468.067969 m (600 m); pattern RMS values are 0.057200976 and
0.063668769. Initial relative amplitude is verified from the actual incoming
state: `log(Theta_pert/Theta_ref)=epsilon*w`, with maximum error 1.61e-16
for the new branch.

State gains use each branch's actual initial disturbance norm. Velocity gains
use its first **nonzero accepted mechanical response**, step 12. Also retain
pattern-normalized V RMS divided by epsilon*Vp*RMS(w), Vp=1e-9 m/s, and signed
projections divided by the corresponding amplitude. Relative differences
are `(Theta_P-Theta_R)/Theta_R` and `(V_P-V_R)/V_R` at each common time.

State curves show committed outgoing Theta; mechanics used the preceding
incoming Theta. No outgoing state is substituted into a friction balance.
The production-QP diagnostic independently reproduces the incoming-state
weak residual on every accepted step.

## Whole-fault work-weighted results

| Final gain | 200 m, reused | 600 m, new |
|---|---:|---:|
| Absolute state difference / initial norm | 1.005202 | 0.992452 |
| Relative state difference / initial norm | 1.077315 | 1.075987 |
| Absolute velocity difference / first response | 1.269890 | 1.182081 |
| Relative velocity difference / first response | 1.066489 | 0.948482 |
| Relative-state signed pattern projection / epsilon | 1.061862 | 1.050506 |
| Absolute-state signed projection / initial projection | 0.937121 | 0.920951 |

| Physical/scaled RMS | 200 m | 600 m |
|---|---:|---:|
| Initial absolute state difference (s) | 1344.3121 | 1471.3103 |
| Final absolute state difference (s) | 1351.3048 | 1460.2042 |
| Final relative state difference | 6.162347e-6 | 6.850676e-6 |
| First absolute velocity difference (m/s) | 1.051761e-15 | 1.219481e-15 |
| Final absolute velocity difference (m/s) | 1.335621e-15 | 1.441526e-15 |
| Final relative velocity difference | 5.328190e-6 | 5.211052e-6 |
| First V RMS / (epsilon Vp RMS(w)) | 0.183871 | 0.191535 |
| Final V RMS / (epsilon Vp RMS(w)) | 0.233496 | 0.226410 |

The raw 600-m V norm is larger, but its own-mass-normalized final amplitude is
about 3.03% smaller. Comparing raw norms alone would conceal this distinction.

The changing reference denominator remains important. With the initial
reference Theta held as denominator, relative-state norm gains are 0.836215
(200 m) and 0.816562 (600 m). Evaluating the same final differences with the
evolving reference multiplies them by 1.288323 and 1.317704, respectively,
giving the reported relative gains. This is accounting, not a different
trajectory or removal of the reference's physical influence.

## Spatial distributions

Both responses remain concentrated around the imposed band and evolving creep
front. The final absolute velocity-disturbance peak is 3.4830e-14 m/s at
16.3 km for 200 m, versus 3.4589e-14 m/s at 16.2 km for 600 m. Relative-state
peaks occur at those same locations, with magnitudes 1.22472e-4 and 1.32268e-4.
The largest absolute state differences occur shallower, near 15.7 and 15.6 km,
because the reference Theta is much larger there.

| Location | Absolute-state gain, 200 / 600 m | Relative-state gain, 200 / 600 m |
|---|---:|---:|
| 15.9 km | 1.03085 / 0.99241 | 1.09739 / 1.05648 |
| 16.0 km | 1.00818 / 1.10459 | 1.20614 / 1.32148 |
| 16.2 km | 0.72953 / 0.80906 | 1.31863 / 1.46239 |
| 16.5 km | 0.79712 / 0.75489 | 1.11054 / 1.05171 |
| 17.0 km | 0.81815 / 0.79857 | 0.83897 / 0.81889 |
| 17.5 km | 0.76459 / 0.73056 | 0.74911 / 0.71577 |

The two patterns do not have the same sign/amplitude at every location;
these are gains relative to each one's own initial local difference. Signed
profiles are retained, not silently phase-aligned. At 16.2 km both start with
the same negative pattern amplitude; both differences decay absolutely while
amplifying relative to the falling reference state. Local absolute growth
still occurs for 600 m at 16.0 km despite decay of its whole-fault norm.

At one year the outside-15--18-km/whole-fault norm ratio is below 0.1% for
absolute V and below 0.55% for relative V in the new branch. The relative
norm is not dominated by tiny reference velocities far outside the taper.

## Verification and artifacts

All 32 new states pass the unchanged checks:

- Exact matched timestep history and outgoing-to-next-incoming Theta arrays.
- 1156 free nodes and zero lower-active nodes at every accepted state.
- Maximum V*dt/Dc = 0.116738629, below 0.125.
- Fresh linear checks pass; maximum recorded normalized nonlinear residual
  6.57856e-11 and surface RMS 7.94803e-5 Pa.
- Independent aging relative error at most 2.22045e-16, below 1e-12.
- Incoming-state production weak-residual audit error zero in every record.
- Matching native work matrices, original node ordering and physical coordinates.

Wall time: **1249.78 s (20.83 min)**. Reported maximum child RSS: 1358152 KiB
(about 1.30 GiB, not aggregate four-rank memory). Execution placement differs
from the older run, so these wall times are not a performance comparison.

Task edits are benchmark-only: a default-200-m wavelength option in
`disturbance_diagnostic.h` and `run_disturbance.py`, use of the perturbed
branch's pattern in `analyze_disturbance.py`, and the new
`compare_disturbance_wavelengths.py`. Python syntax checks, completed-run
analysis assertions, and `git diff --check` pass. No broad test campaign.

Reproduction (the existing output name must not be rerun/overwritten):

```sh
cmake --build benchmarks/reconstructed_fault/performance/build-gmg --target fault_disturbance -j4
python3 benchmarks/reconstructed_fault/bp3/run_disturbance.py prepare plus60032 --steps 32 --epsilon 1e-4 --limit .125 --wavelength 600
python3 benchmarks/reconstructed_fault/bp3/run_disturbance.py run plus60032 --cores 0,1,3,6
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-bp3-slip-plot python3 benchmarks/reconstructed_fault/bp3/analyze_disturbance.py reference32 plus60032 --nodes-only
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-bp3-slip-plot python3 benchmarks/reconstructed_fault/bp3/compare_disturbance_wavelengths.py
```

Results under `first_long_run/state-disturbance/wavelength-comparison/`:
`summary.json`, `norms.csv`, `profiles.csv`, `norm_gains.png`,
`spatial_profiles.png`, and `pattern_normalized.png`. Native full-fault nodal
and correctly timed QP diagnostics remain in `plus60032/`; `profiles.csv`
contains the 14--19 km comparison at every common accepted time.
