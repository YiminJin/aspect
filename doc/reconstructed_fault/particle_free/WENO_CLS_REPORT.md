# Pre-PF7 compact WENO candidate comparison

16 October 2026. **The native pulse runs reproduce an improvement from constrained
candidates, but do not support adoption.** Keep unlimited upwind DGQ2 as the
experimental baseline and both WENO paths experimental. The constrained option
reduces staircase formation, sharp-pulse error and broadening relative to the
existing candidates. Every WENO variant fails the frozen temporal screen, and
the constrained fit introduces transverse variation in these scalar, x-only
problems. PF7 has not begun. No thresholds or physical parameters were tuned.

This is a controlled candidate-construction adaptation motivated by Zhu, Zhong,
Shu and Qiu (2016), not a complete implementation of their method. Their neighbor
fit and selected additional cell-mean information motivate the objective below;
their polynomial space, detector, smoothness and characteristic-system treatment
are not adopted. See the [primary manuscript, sections 3.2 and 4.2](https://archive.ymsc.tsinghua.edu.cn/pacm_download/113/1132-CiCP-749.pdf)
and the [frozen formulation](WENO_CLS_FORMULATION.md).

## Construction and source audit

The original `dg_limiter.cc` extrapolates each complete neighboring polynomial
onto the target K and replaces its **target-cell** average by K's original mean:
`q_J = A_J - <A_J>_K + mu_K`. It retains all nine tensor-product Q2 coefficients.
This is the whole-polynomial/mean-shift construction present in the supplied
`../aspect-weno/source/simulator/limiters.cc`; it is not the constrained fit.
No defects from the older `elasticity.cc` were presumed.

The new private, default-off `make_dg_limiter(..., true)` option uses actual face
neighbors only. For each neighbor J, with original own-cell means `mu_I`, solve

```
min_q <|q-A_J|3²>_J + rho sum_(I in U_J) |<q>_I-mu_I|3²,
subject to <q>_K=mu_K,    rho=|J|/|reference Box|,
|A|3² = xx² + yy² + (xx+yy)² + 2xy².
```

The reference box is the unit square in these runs. U_J contains the other face
neighbors whose own-mean tensor distance from K is smaller than the farthest
other neighbor's distance. Relative ties within 64 machine epsilons are excluded.
Missing boundary neighbors are omitted; no artificial exterior candidate is
created. The center candidate remains the original K polynomial. This common
useful set and the constant positive-definite tensor metric give one spatial KKT
system for all components:

```
H = M + rho sum_I c_I c_I^T,    b = M u_J + rho sum_I c_I mu_I,
[H a; a^T 0] [q; lambda] = [b; mu_K],    a=c_K.
```

Native code precomputes Schur-complement correction maps for every useful mask,
caching translation-equivalent uniform-grid stencils. The independent reference
uses exact diagonal Legendre mass and a full 10×10 constrained solve. No mixed
Q2 modes, including x²y², are removed. A separate 30×30 block solve verifies the
full tensor metric's cancellation from the spatial solve.

The changed core files are `dg_limiter.cc`, `dg_history_internal.h` and the private
declaration in `phase_field_fault.h`. The benchmark observer gains candidate,
case, path and trace selection with its previous defaults. A const stage callback
observes the complete separate output; all candidates still read the immutable
incoming snapshot. It records physical means immediately around every limiter
call, all-cell maximum/integrated drift, all 32 cells in the middle row, selected
complete candidate coefficients, beta/weights/useful masks, interface averages,
and the subsequent mean RHS change from the native transport matrix.

Everything else is held fixed: KXRCF threshold **1**, safeguarded RMS Fu–Shu
threshold **.1**, original patch RMS normalization and indicators computed before
fitting, full tensor differences/common weights, `d0=.996`, `dJ=.001`, omitted
boundary weights returned to the center, and dimensionless epsilon **1e-6**.
The unchanged Q2 smoothness is the target-reference-coordinate integral of
`(|q_r|3²+|q_s|3²+|q_rr|3²+2|q_rs|3²+|q_ss|3²)/A²`, where A is the largest
unmodified own-cell RMS tensor amplitude in the patch. Stable normalized weights
are proportional to `dJ/(betaJ+epsilon)²`; all-zero beta returns normalized linear
weights. Gauss3 integrates reconstruction/mass/squared-Q2 exactly on the admitted
affine Cartesian cells. Gauss4 transport/source/faces, Gauss8 initialization,
upwind material transport and all three SSPRK3 limiter calls remain unchanged.
RMS normalization, polynomial-discrepancy safeguard and this Q2 smoothness retain
their status as experimental adaptations. No component clipping or chi limiting.

## Reproduction and independent checks

The pinned incoming executable/observer reproduce **63 non-cost CSVs byte for
byte** from `pf4-stabilization-final-32-1`, including the unlimited smooth control.
The freshly rebuilt instrumented mean-shift path reproduces unlimited fields
exactly and the four WENO pulse fields within **4.23e-15 relative L2** (maximum
coefficient difference 4.26e-11 Pa). Instrumentation does not explain the old
staircases.

Independent/local results:

| Check | Measured maximum |
|---|---:|
| Legendre KKT versus all native correction maps | 6.69e-14 |
| Candidate coefficient discrepancy, scaled | 7.05e-14 |
| Mean constraint / stationarity residual, scaled | 9.10e-16 / 3.87e-14 |
| Independent beta / normalized weights discrepancy | 1.00e-15 / 2.23e-16 |
| Full tensor block-KKT discrepancy, scaled | 1.43e-17 |
| Global complete-Q2 reproduction, scaled | 9.10e-16 |
| Traversal-order discrepancy | 0 |
| In-plane tensor-basis / amplitude-rescaling discrepancy | 9.10e-16 / 9.10e-16 |
| Local physical mean change | 8.66e-13 Pa |
| All transport limiter calls: physical mean change | 1.31e-12 Pa |
| SSPRK3 physical-mean balance discrepancy | 4.55e-13 Pa |
| Face-flux versus native mean-RHS change discrepancy | 6.06e-12 Pa/s |

The objective is positive definite (minimum tested Legendre eigenvalue .04).
Zero, constants, affine, complete Q2, x²y², piecewise constants and non-proportional
tensor patches pass; coefficients remain nine-dimensional. Weights are finite,
nonnegative and normalized; amplitudes 1e-100 and 1e100 and a .43-radian in-plane
basis rotation pass. The original 69 independent limiter checks also pass,
including mixed-xy detector blind spot/safeguard, smooth zero crossing,
normalization/background sensitivity, boundary inflow and nonpublication.
No threshold reselection was performed.

The existing 23 operator-control CSVs reproduce byte for byte. Independent
operator/boundary/polynomial discrepancies are below 4.4e-15; the uncorrected
constant defect remains .3, confirming the material-divergence control is active.
Native cohesive/Maxwell tests pass **628 assertions in eight cases** and PF1
passes **94 checks**. Every transport run verifies accepted T/Z/I/phase unchanged.
These are scratch prescribed-transport tests, not accepted-history evolution.

One numerical failure is retained: cached versus uncached constrained runs miss
the strict 1e-12 final-field equivalence target (**1.85e-10 relative L2**, maximum
3.16e-6 Pa coefficient difference), despite passing independent local KKT checks.
Their first-call inputs match exactly and outputs differ by at most 3.41e-13 Pa.
This is consistent with amplification of translation-dependent quadrature
roundoff by the nonlinear limiter.
The final matrix uses the cached implementation consistently; no tolerance was
relaxed and bitwise equivalence is not claimed. Initial failed library-name
launch and preliminary outputs are retained too.

## What the staircase trace shows

At the first call, target cell 1280 spans x=[.5,.53125] in the middle row.
Its xx mean is -950.291262135922 Pa. The mean-shift construction produces beta
`[9.72569, ~0, ~0, 9.72569, 9.72569]` and weights
`[5.26e-12, .5, .5, 5.29e-15, 5.29e-15]`. Its within-cell tensor variance falls
by a factor **2.79e-23**: mean restoration turns the two constant-neighbor
candidates into the same target-mean constant. The target mean is conserved,
but its right-face xx average changes -850.873786→-950.291262 Pa. The next xx
mean RHS changes 23691.184466→24963.728155 Pa/s, independently matching the
native matrix. Subsequent stage means satisfy the SSPRK3 balance. This is a
flux-mediated change of later means, not immediate loss of mean conservation.

The effect persists late in the old KX run. At the final dt/4 limiter call
(1235), the cell starting at x=.25 m has xx mean 77.3049445 Pa before and after
limiting. Its constant-neighbor weight is .99995992; its right-face value changes
76.6622013→77.3049187 Pa, changing the next mean RHS
-981.276177→-989.502960 Pa/s. The three final-stage beta/weight/interface traces
are also extracted in `staircase-late-trace.json` beside the archive. The plot
shows the resulting steps in polynomials and altered later cell means.

For the identical incoming snapshot, the constrained candidates have beta
`[9.72569,28.46769,.0184935,5.80241,5.80241]` and weights
`[.00358866,4.21e-7,.99639068,1.01e-5,1.01e-5]`. The variance ratio is **.04004**,
the mean remains conserved, the right-face xx average becomes -989.971926 Pa,
and the next mean RHS becomes 25471.640657 Pa/s. A flat neighbor with a different
target mean now acquires nonconstant modes. Independently, with no useful-cell
penalties its one-dimensional Legendre correction is
`Delta_mu*(1+6P1+30P2)/193` for the right neighbor (P1 sign reverses on the left).
It is not a constant mean shift. Full candidate/interface traces and held-call
fields are archived; a smoother profile alone is not the acceptance criterion.

## Bounded transport measurements

All 30 final jobs use the same freshly built core and observer: 32×64 cells,
18,432 scalar DG DoFs, proportional tensor `(1000,-400,700)` Pa, velocity
(.4,0) m/s and final time .2 s. Base dt=.2/103 s; nested runs use 206 and 412
steps. There is no initial limiter. The sharp pulse is fixed at .25/.5/.75 m;
the finite profile uses ell=.15625 m and delta=ell/ln(9), including transition
overlap and exact inflow. Bounds remain confined to the old scalar benchmark;
this experiment imposes none. Sharp errors split cells at exact jumps; smooth
errors use high-order independent quadrature. All L2 errors use the full tensor
metric over the entire two-dimensional domain.

| Case / path | Relative L2, dt | dt/2 | dt/4 |
|---|---:|---:|---:|
| Sharp / Unlimited | 0.177547779 | 0.177546901 | 0.17754684 |
| Sharp / Mean shift KX | 0.37629267 | 0.366655959 | 0.436260298 |
| Sharp / Mean shift FS | 0.376581638 | 0.367920753 | 0.432499616 |
| Sharp / Constrained KX | 0.214634161 | 0.22327916 | 0.224785605 |
| Sharp / Constrained FS | 0.213913137 | 0.218402894 | 0.22284191 |
| Finite / Unlimited | 0.000427637766 | 0.000427623209 | 0.000427621367 |
| Finite / Mean shift KX | 0.000551776196 | 0.000875908057 | 0.051565199 |
| Finite / Mean shift FS | 0.000441034136 | 0.000455246839 | 0.000475415065 |
| Finite / Constrained KX | 0.000444624314 | 0.000526528965 | 0.000771064803 |
| Finite / Constrained FS | 0.000427985883 | 0.000429381932 | 0.000433812642 |

At dt/4; norm changes are percentages of the initial quadratic norm. Held
changes are relative L2 after 12 extra limiter calls, normalized by the incoming
final field norm. FS denotes the fixed safeguarded RMS Fu–Shu detector.

| Case / path | xx min, max (Pa) | Three middle-row widths (m) | Total norm change % | Direct RK-weighted limiter change % | Held L2 change |
|---|---|---|---:|---:|---:|
| Sharp / Unlimited | -1158.278864, 1165.062231 | 0.023529, 0.023529, 0.023529 | -3.57369 | — | — |
| Sharp / Mean shift KX | -1000.134901, 1000.068581 | 0.108368, 0.108246, 0.108459 | -32.3529 | -3.16058 | 0.067012 |
| Sharp / Mean shift FS | -1002.183535, 1000.891335 | 0.108459, 0.108704, 0.108551 | -32.338 | -3.09612 | 0.0614778 |
| Sharp / Constrained KX | -1000.723646, 1002.093648 | 0.049316, 0.049286, 0.043945 | -13.1545 | -7.38053 | 0.000659977 |
| Sharp / Constrained FS | -1000.619995, 1002.068069 | 0.050171, 0.048523, 0.045776 | -13.3356 | -7.56259 | 0.000852006 |
| Finite / Unlimited | -918.514351, 918.471391 | 0.139240, 0.135314, 0.139216 | -0.000892854 | — | — |
| Finite / Mean shift KX | -918.514962, 918.415702 | 0.160201, 0.135299, 0.138778 | -0.647638 | -0.361027 | 0.00147729 |
| Finite / Mean shift FS | -918.514351, 918.471391 | 0.139244, 0.135314, 0.139216 | -0.00125196 | -0.000357416 | 5.72825e-05 |
| Finite / Constrained KX | -918.514721, 918.471395 | 0.139243, 0.135299, 0.139157 | 0.0100651 | 0.0109705 | 3.27615e-05 |
| Finite / Constrained FS | -918.514351, 918.471393 | 0.139255, 0.135314, 0.139216 | -0.00100416 | -0.000111459 | 2.74162e-06 |

Measured dt/4 costs below are observational: two concurrent serial processes
can share host resources. The advance column includes detectors, reconstruction,
mean/norm diagnostics and trace I/O/native flux probes, so it is **not** a pure
transport-kernel timing or a clean production-speed ratio. Setup is outside it.

| Case / path | Limited visits % | Setup (s) | Detector (s) | Reconstruction (s) | Instrumented advance (s) |
|---|---:|---:|---:|---:|---:|
| Sharp / Unlimited | 0.000 | 0.054 | 0.000 | 0.000 | 3.667 |
| Sharp / Mean shift KX | 54.316 | 0.048 | 8.859 | 6.575 | 24.931 |
| Sharp / Mean shift FS | 80.959 | 0.049 | 12.614 | 10.062 | 32.737 |
| Sharp / Constrained KX | 38.530 | 0.122 | 11.938 | 9.535 | 32.636 |
| Sharp / Constrained FS | 70.638 | 0.111 | 13.129 | 16.011 | 40.197 |
| Finite / Unlimited | 0.000 | 0.025 | 0.000 | 0.000 | 1.635 |
| Finite / Mean shift KX | 31.685 | 0.047 | 10.174 | 4.108 | 23.973 |
| Finite / Mean shift FS | 8.965 | 0.050 | 11.123 | 1.093 | 21.271 |
| Finite / Constrained KX | 16.859 | 0.103 | 14.005 | 4.296 | 30.241 |
| Finite / Constrained FS | 16.981 | 0.108 | 12.335 | 3.557 | 25.675 |

The initial sharp pulse is exactly representable on this aligned mesh; its
projection extrema are [-1000,1000] Pa. Finite-width initialization extrema are
±918.322905 Pa and its relative L2 error is 2.71017e-4. These are initialization
measurements, before limiting or evolution. The actual overlapping finite pulse
has peak 918.505516 Pa; its analytic final front widths are
(.139199795,.135360618,.139199795) m. The retained final sharp DGQ2 representation
floor is .1498621703. No stabilization can beat that space's best approximation.

Reported extrema reuse the established all-cell tensor-product Lobatto9 sampling,
not a proof of global bounds. Sharp front widths use the retained 1024 samples
per x cell (resolution 3.05e-5 m); finite widths solve complete Q2 trace roots and
jumps against the actual overlapping analytic profile. The table uses the
middle row, with min/max over all 64 rows in the assessment. For constrained KX
at dt/4, sharp row ranges are (.049225–.049652, .049286–.049683,
.043518–.045441) m. For constrained Fu–Shu they are (.050171–.050873,
.047455–.048523, .045776–.046295) m. Unmodified candidates/unlimited remain
transversely uniform to roundoff.

In contrast, constrained KX/Fu–Shu produce maximum full-tensor nodal transverse
differences **81.55/58.51 Pa** (sharp) and **.3293/.05798 Pa** (finite) at dt/4.
First-call means remain uniform, while face means already change with y; later
fluxes create y-dependent cell means. Exact projection onto y-independent fields
gives transverse variance L2 .9564/1.0234 Pa for sharp and .01421/.0006877 Pa
for finite, respectively. These small integrated contributions do not explain
away the larger local artifacts. The full Q2 fit/useful-cell penalties and
reduced boundary stencils do not preserve an x-only subspace here. This must be
resolved before any general tensor-method recommendation.

At dt/4 the sharp peaks are not attenuated below the analytic amplitude: all
paths still overshoot. Constrained candidates reduce overshoot from unlimited's
16.5% to about .21%, while old candidates give .0069% (KX)/.089% (Fu–Shu), at
much greater broadening/error. Finite positive peak attenuation is about .0037%
except old KX (.0098%); finite negative overshoot is about .00096% for all paths.
These comparisons use the analytic overlapping profile, not ±1000 Pa as a
finite-pulse envelope.

The direct limiter norm changes and total changes differ materially. Constrained
sharp reconstruction removes **more** quadratic norm at the limiter calls than
old reconstruction, while the final total loss is smaller because changed traces
also change subsequent transport. RK weights (1/6,2/3,1) isolate direct limiter
contributions in the established SSPRK3 quadratic identity; they are not the
whole transport norm budget. Finite KX can **increase** norm. Twelve held-field
calls without transport still modify every WENO field, as quantified above.
The limiter is not idempotent and increasing calls per physical time matters.

## Temporal qualification and recommendation

The screen, frozen before reading constrained fine-step results, requires the
dt/2–dt/4 full-tensor field difference to be ≤1% of the dt/4 analytic error and
to decrease from dt–dt/2. The measured ratios are:

| Path | Sharp | Finite width | Both pass? |
|---|---:|---:|---|
| Unlimited | .00001883 | .00003923 | yes |
| Mean shift, KX | .30159 | 1.00183 | no |
| Mean shift, Fu–Shu | .30013 | .13778 | no |
| Constrained, KX | .02630 | .43050 | no |
| Constrained, Fu–Shu | .03217 | .07697 | no |

Only constrained KX sharp has decreasing WENO field differences; it still misses
the 1% criterion. No continuous-time convergence or adequate timestep control is
claimed. Relative to current WENO candidates, the native improvement is real in
this bounded test; it is not a reproduction of the published method's overall
qualification. Unlimited remains more accurate and cheaper on both controls.
Retain the diagnostic option for investigation; **do not adopt it or enable a
limiter for PF7 on this evidence**. A separately selected stabilization task
would first need to address held-call/time dependence and x-only preservation
without compensating tuning, then qualify any tensor extension. No further
threshold or timestep sweeps were performed here.

Production defaults, history ownership/publication/restart paths, constitutive
law, spin/rate conventions, geometry/phase and PF0–PF5 evidence remain unchanged.
Remaining particle dependencies are the PF5 ones: startup H/CPDI/domains,
surface material projection, association/support, fixed-particle checks, derived
samples and particle/RNG persistence. This diagnostic removes none of them.
AMR/MPI, periodic/open-boundary accepted splitting, moving geometry, birth,
coupled propagation and PF7 remain outside scope.

## Provenance and replay

Start revision **a8707307008401ac9cff702a369e2e2a611fa0f7**, branch
`pf-rsf-particle-free`. All 30 final records agree on 880 production source hashes,
core SHA256 `08aa81e49961fd0dc51f4cb49d7416e2881a38b3ae6ea9daf07e90f5578104e2`
and observer SHA256 `cba0563d1638192b2a0b183765897544da0a38fd50d8f720c86917b000517b2e`.
GCC12.4/deal.II9.6, core without fast-math, observer with
`-fno-finite-math-only -ffp-contract=off`. Every process is capped at 180 s,
one rank/thread, at most two concurrent. Observed per-job wall times are
32.22–82.12 s and peak RSS ≤524,980 KiB. Build limits are core -j4/observer -j2.

Rebuild with the existing configured native/plugin trees, then for example:

```sh
cmake --build build-particle-free/native -j4
cmake --build build-particle-free/plugin --target stabilization_checks limiter_checks transport_checks -j2
python3 benchmarks/reconstructed_fault/particle_free/run_candidate_case.py NEW-LABEL --construction cls --detector fu_shu --case signed_jump --divisor 1
```

Use unique labels; the runner refuses overwrites. The archived campaign lists
all exact commands/expanded inputs, 520 final CSVs, local systems, traces,
regression/reference checks, failed preliminary evidence and source manifests.
Independent replay and the [comparison plot](../../../benchmarks/reconstructed_fault/particle_free/evidence/weno_cls/comparison.png)
are described in the [evidence README](../../../benchmarks/reconstructed_fault/particle_free/evidence/weno_cls/README.md).
Standalone replay verified all **1,307 payload hashes**, reproduced the complete
assessment byte for byte, repeated 69 limiter and 94 PF1 checks, and matched the
23 operator/63 original baseline CSV comparisons. It executes the archived
independent mathematical scripts; it does not rerun native tests.
No commit or push was made for this task.
