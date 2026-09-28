# Inclined-fault interpolation comparison

## Decision

All six fresh-start runs passed initialization and four real steps, on one and
two ranks. **Unlimited linear least squares with continuous Q2 (B) is promising
for the interior weak-load error; DGQ2 (C) is not an improvement on this measure.**
This is a short prescribed-slip transfer experiment, not qualification of a
replacement interpolator for the free-RSF BP5 trajectory. No production ASPECT
header or source was changed for this task.

## Controlled configuration

The existing `inclined_production.prm` supplies the unrotated 64x64 Cartesian
square, 60-degree straight fault, 37 surface elements, ell=0.15625 m, ell/h=10,
uniform prescribed V=0.005 m/s, affine boundary loading, zero initial stress,
and frozen Q1 phase. Each run takes four real 0.1-s steps, ending at 0.4 s.
Timestep zero uses the original 0.1-s artificial Maxwell interval and retains
zero particle stress. Physical pressure has volume mean zero. The existing
prescribed/adiabatic friction setting is unchanged; the plotted mechanical
normal traction p-n.tau.n does not include its 1000-Pa reference pressure.

Unlike the original frozen-particle audit, these runs **advect particles with
production RK2**, regenerate domains, and publish histories normally. All 36,864
particles survive; maximum displacement is 0.00271782 m (0.174 cell widths).
All cells retain nine particles, and no parent changes cell during this short
interval. Thus this tests moving-particle interpolation and rank publication,
not a long-time migration/exchange stress test.

| Case | Maxwell interpolation | tau_xx/tau_yy/tau_xy FE | Total DoFs |
|---|---|---|---:|
| A | Native DWA, linear weight | continuous Q2 | 124,937 |
| B | Native linear least squares, no limiter | continuous Q2 | 124,937 |
| C | Native linear least squares, no limiter | DGQ2 | 185,606 |

Velocity Q2, pressure Q1, temperature Q2, theta_initial continuous Q2, and phase
Q1 are unchanged. All four compositional fields still use `particles`; there
is **no DG stress-advection equation**. A benchmark-only routing adapter calls
the two existing interpolators: only the three Maxwell components go through
least squares in B/C; every other property still uses DWA. Consequently disabling
the LS limiter does not change treatment of the other properties. No LS fallback
is needed: every sampled cell contains nine particles, versus three required
for the 2-D fit.

The initial particle hash agrees in all six runs. Native bulk-QP coordinates,
phi, chi, prescribed V, beta and kappa agree across all cases, times and ranks
at rtol=2e-13, atol=1e-14. Current fields at steps 0 and 1 are bitwise identical
between A/B/C on one rank, before nonzero transferred history enters mechanics.
`finite_elements.txt` and resolved `parameters.prm` in each output record the
actual spaces and parameters, not just requested settings.

## Load, stress, and velocity results

After each accepted update, the diagnostic noncommittingly applies the ordinary
particle-to-FE transfer and working constraints. It compares this reconstructed
next history with the just-evaluated current mechanical stress using the same
Q2 velocity tests, native quadrature, and homogeneous mechanical constraints:

    J = || assemble[(tau_next_FE - tau_current_QP):sym grad(w_i)] ||_2.

This measures the complete particle publication/reconstruction chain, not solely
the interpolation formula acting on an analytic tensor. It is measured before
the following step's advection. The next actual solve also includes that advection.
The independent Cartesian Q2 assembly reproduces each reported full load norm
within 2e-10 Pa m. No visualization averaging enters this calculation.

The interior uses velocity tests centred at |s|<=0.2 m and |r|<=0.1 m, away from
open fault tips. Its norm is a subset of the globally assembled test rows, not
a newly truncated volume integral.

| Step | A whole / interior J | B whole / interior J | C whole / interior J |
|---|---:|---:|---:|
| 1 | 38.3343 / 0.082196 | 40.0867 / 0.023399 | 66.7225 / 0.153890 |
| 2 | 48.0693 / 0.154998 | 50.1138 / 0.030780 | 80.1210 / 0.272122 |
| 3 | 56.4829 / 0.226660 | 58.2077 / 0.043347 | 89.1155 / 0.396952 |
| 4 | 63.3274 / 0.293767 | 64.3627 / 0.060060 | 96.9949 / 0.523758 |

Units: Pa m. B reduces the final interior jump by **4.89x**, but its whole-domain
jump is 1.6% higher than A. C's jump is 53.2% higher globally and 78.3% higher
in the interior. Therefore the global and interior conclusions must not be
conflated. No claim is made that DG discontinuities themselves are noise.

At step 4:

| Measure | A | B | C |
|---|---:|---:|---:|
| Current tensor RMS (Pa) | 4335.92 | 4343.60 | 4344.37 |
| Current tensor change from evaluated t=0, RMS (Pa) | 3248.86 | 3256.62 | 3257.37 |
| Publication/reconstruction tensor change, RMS (Pa) | 93.2528 | 95.0654 | 104.589 |
| Mean transfer change in tau_xy (Pa) | 0.053111 | 0.076000 | 0.075958 |
| Velocity RMS difference from A (m/s) | 0 | 7.57123e-7 | 9.08005e-7 |
| Current tensor RMS difference from A (Pa) | 0 | 63.5634 | 94.1462 |

The tensor norm includes the off-diagonal component twice. The large accumulated
stress change is mostly the intended elastic response, **not an error estimate**.
The small transfer-induced mean drift and finite transfer-load jump are separate
measurements; no exact evolving stress solution is available for this fixture.
A's velocity RMS is 0.00317566 m/s, so B/C differ by about 0.0238%/0.0286% of that
scale, despite much larger relative changes in the interior transfer-load norm.

### Cellwise history, without graphical smoothing

The analysis recovers each Q2 polynomial independently from its 3x3 native
Gauss values and evaluates **both** face traces. This is exact polynomial
evaluation of the represented FE history, not extrapolation of plotted CSV
averages. At step 4:

| Next FE-history trace diagnostic | A | B | C |
|---|---:|---:|---:|
| Maximum component jump, full mesh (Pa) | 1.21e-10 | 1.36e-10 | 7566.33 |
| Maximum component jump, interior (Pa) | 1.99e-12 | 1.93e-12 | 36.9679 |
| Face-quadrature tensor jump RMS, interior (Pa) | 8.78e-13 | 8.84e-13 | 13.3623 |

The large global DG extreme is not representative of the interior. C faithfully
retains cell-local independent reconstructions; those discontinuities are
allowed by DG. However, the independently integrated load shows that C does not
reduce the mechanically relevant transfer jump here. B's smaller interior load
does not require a smaller pointwise tensor transfer RMS: weak moments matter.

`comparison.png` also shows the mechanical p, d=-n.tau.n, and p+d using identical
native work-row weights JxW*chi*N_i. These are **normalized weak-row diagnostics,
not a consistent-Q1 projection or a replacement friction input**. On |s|<=0.2 m,
the final mean p+d is 684.974, 685.525, 685.527 Pa for A/B/C. The neighboring-chord
RMS is respectively 0.493023, 0.493074, 0.493392 Pa; it includes broad curvature
and is not an exact-solution error. The B/C minus A changes are primarily a mean
shift (~0.55 Pa), not elimination of a distinct normal-traction pattern.

## Convergence, timing, and MPI publication

All 30 accepted states (six runs including initialization) pass the production
nonlinear check; maximum relative bulk residual is 7.95201e-10 against 1e-8.
All 30 fresh linear residual checks pass. Every solve takes one Newton update,
alpha=1, with 19--22 Krylov iterations. All slip nodes are prescribed in this
diagnostic. Boundary-flux closure is below 1e-15.

Particle tensors read at the beginning of each step equal the preceding accepted
particle tensors exactly. Timestep-zero retained stress is zero. For real steps,
the largest one-/two-rank differences across A/B/C are:

- velocity: 5.73e-14 m/s;
- current tensor component: 3.76e-6 Pa (relative tensor L2 <=5.69e-11);
- next FE-history component: 3.13e-6 Pa;
- committed particle tensor component: 5.45e-6 Pa.

These pass the recorded 1e-8 relative field-scale plus 1e-8 absolute comparison.
The separate **evaluated** initialization stress differs by 1.18238e-4 Pa
(relative L2 1.146e-8), exceeding that strict derived-stress comparison. Its
velocity difference is only 4.34e-12 m/s; the tensor difference is reproduced by
2*kappa*sym grad(delta u) within 1.40e-10 Pa. Both initial solves pass their fresh
residual targets, and neither publishes this evaluated stress: initial particle
and reconstructed retained histories are identically zero. This is reported,
not hidden by widening solver tolerances or claiming bitwise MPI equivalence.

| Case | One rank wall time (s) | Two ranks wall time (s) |
|---|---:|---:|
| A | 24.948 | 22.592 |
| B | 24.842 | 20.853 |
| C | 23.443 | 22.399 |

Total simulation wall time: 139.077 s. Each had a 120-s hard cap; no simulation
failed or was retried. Four inexpensive analysis tests pass: independent weak
loads for constant/linear tensors, and continuous/discontinuous exact Q2 traces.
No full ASPECT suite, longer trajectory, free-RSF run, or mesh convergence study
was performed. The result supports retaining A as the baseline and considering
B, not C, if a later targeted BP5 comparison is warranted.

## Reproduction and files

From the repository root:

```sh
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_moment_cycle -j4
source benchmarks/reconstructed_fault/bp5/interpolation-inclined/environment.sh
timeout 120 mpirun -np 1 build-pf-cpdi/aspect-release benchmarks/reconstructed_fault/bp5/interpolation-inclined/A.prm
# B.prm and C.prm likewise; use A-two.prm/B-two.prm/C-two.prm with -np 2.
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/aspect-interpolation-mpl python3 benchmarks/reconstructed_fault/bp5/interpolation-inclined/analyze.py
python3 -m unittest discover -s benchmarks/reconstructed_fault/bp5 -p test_inclined_moment.py
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s benchmarks/reconstructed_fault/bp5/interpolation-inclined -p test_analysis.py
```

Do not source the old `stress_cycle_env.sh`: it intentionally freezes particles.
Keep prior output directories before rerunning. `comparison.json` contains the
full metrics; `comparison.png` contains matched profiles. Each output has raw
per-rank particle CSVs, unaveraged native-QP tensors, next FE history, coefficient
arrays, solver summaries, and resolved parameters. `analyze.py --plot-only`
regenerates the figure from already analyzed data without rerunning tests.

This task changes only benchmark code/configuration:

- `../clean_stress_cycle.cc`: opt-in advection and stable-ID history/position exports;
- `../moment_cycle.cc`: realized FE inventory and coefficient exports;
- `../stress_only_interpolator.cc`: routes Maxwell stress to the existing unlimited
  LS interpolator, leaving unrelated properties on the existing DWA interpolator;
- `../CMakeLists.txt`: adds that adapter to the existing moment-cycle plugin;
- this directory: six case files, shared overrides, environment, analysis/tests,
  report, and local evidence. Existing unrelated working changes were preserved.

Source HEAD was `33228369da82011f509ee07937f27c17663c7f28`, with the existing dirty
working tree retained. This is not claimed to be a clean committed revision.
Executed ASPECT binary SHA256:
`f0aa968d7a299f31a354209f28b72b32ba609585b42d33ecd6117e813a84a66c`.
Executed plugin SHA256:
`2b301e1ae2b4dd75470dbd074defbc241fff063c3947d04a9aea79df538a8b83`.
Benchmark source SHA256 (as tested):

```
4a4725de50fe0c941241a02e554423640f1869ccc691f7d50bd5b1dd5ef23905 stress_only_interpolator.cc
c1f855c7cd9db7a024eeb6cfe09b8ea8a86276c91340104c5e943ab08ef696d3 clean_stress_cycle.cc
5a78629c586bd1e67c5eb2b37e2267ca9f91f9a43b4b0315c9f89de4191ec10b moment_cycle.cc
```
