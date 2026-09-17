# K5 manufactured gradient-kink and prescribed/free trace experiment

## Decision

The conjecture is supported, but it contains two distinguishable mechanisms.
A continuous slope kink excites alternating discretization errors when the
bulk space does not resolve the surface source; refining the bulk at fixed
fault grid suppresses those errors. However, a negative adjacent-node response
persists in the **bulk-resolved Q1 surface operator**. Bulk refinement alone
therefore does not guarantee removal of a notch at an incompatible imposed
rate boundary.

A separate manufactured trace control isolates that second mechanism. Forcing
a single continuous Q1 value across a junction whose free solution has a
different limiting rate produces a 38–48% overshoot relative to the jump.
Giving the free side its own endpoint trace removes that overshoot and yields
second-order convergence, without smoothing or changing the continuum load.
Exact frozen-state RSF checks reproduce both results.

These are independent diagnostic calculations of the specified work form,
**not new ASPECT/BP3 runs**. They identify and test a possible remedy, but do
not establish the unknown free-side limiting rate of the actual 40-km BP3
problem. No production correction or accepted history change is made.

## 1. What was manufactured, and what was held fixed?

The [addendum](stage_K5_gradient_kink_addendum.md) defines the experiment.
The source/authority comparison used specification.tex's K5 work-measure
equations, `ReconstructedFaultSurfaceSystem::assemble_bulk_work_measure`,
the ordinary Stokes bilinear form, and the mature point response. The
independent code uses their same strain/source work pairing:

\[
 \tau=2\kappa(\epsilon(u)-\chi V S),\qquad
 A u=B V,\qquad
 R=f-(K_V-GA^{-1}B)V.
\]

Old stress is zero, state is homogeneous and frozen, and no history is advanced.
The strip is periodic with lengths 8 ell in both directions. This removes
physical tips and permits an independent Fourier bulk elimination. It does
**not** change BP3's open topology. The analytic localization is

\[
 \chi(n)=\tfrac12(1+\cos\pi n),\quad |n|<1;
 \qquad\chi=0\text{ otherwise},\qquad\int\chi\,dn=1.
\]

Here ell=kappa=1 are the diagnostic units. This compact C1 localization is
manufactured, not the actual sampled BP3 phase/I_h profile. There is no tail
truncation, I_h smoothing, or normalization adjustment. The frozen tangent
d=0.036 is representative of `(sigma*a/V)*ell/kappa` at the saved deep BP3
state. All rate errors below are divided by the **manufactured perturbation
amplitude**, not by Vp and not by a physical BP3 error budget.

For tangential wavenumber k and periodic normal wavenumber l, the independently
eliminated elastic operator has symbol

\[
 E(k)=\frac{\kappa}{L_n}\sum_l |\widehat\chi(l)|^2
              \frac{4k^2l^2}{(k^2+l^2)^2}.
\]

The (0,0) term is kappa/Ln: periodic bulk velocity cannot supply mean shear.
Reference and FE problems share this mean constraint. The normal transform is
`pi^2 sin(l)/(l*(pi^2-l^2))`, with its removable singularities evaluated
analytically. The normal-traction action `p-2*kappa*du_n/dn` is assembled and
checked; its weighted contribution cancels by reflection symmetry here.
This is **not** a general replacement of G by B transpose.

The first target is a continuous triangular rate depression, with corners at
s=2,3,4. The right half s>=4 is prescribed at zero perturbation. All selected
fault grids represent this target exactly. The load is `(E+d)V_target`, using
analytic Fourier coefficients of the *continuous* triangle, independently of
the tested Q2/Q1 bulk FE operator. It is not a discrete manufactured load
computed from that operator. A positive base rate can be added and the
perturbation made arbitrarily small; negative plotted perturbations are not
negative physical slip rates.

## 2. Compatible continuous kink: bulk resolution cures the error

Fault spacing stays fixed at h_Gamma/ell=0.125. Surface first/second moments,
B and shear G use one work measure. The main spatial comparison splits the
x integration partition at surface-element breaks, to isolate approximation
spaces from quadrature across breaks. Native three-point bulk quadrature is
tested separately below.

| Bulk h_s/ell | Bulk h_n/ell | Maximum rate error | Q1 RMS rate error |
|---:|---:|---:|---:|
| 0.5 | 0.25 | 2.30234% | 0.646811% |
| 0.25 | 0.25 | 1.33413% | 0.223546% |
| 0.125 | 0.25 | 0.254482% | 0.0371318% |
| 0.125 | 0.125 | 0.266423% | 0.0367978% |
| 0.0625 | 0.125 | 0.00596140% | 0.000946375% |

At fixed normal spacing 0.25, the first two tangential refinements reduce
RMS error by factors **2.89 and 6.02**. The normal-only check barely changes
RMS and slightly increases the maximum through changed cancellation; it is
not claimed as monotone pointwise convergence. The final tangential refinement
at normal spacing 0.125 reduces RMS another **38.88x**, and maximum error
**44.69x**. These are measured contractions, not claimed asymptotic orders.

On the flat shoulder preceding the kink, h_s=0.25 gives errors -0.006730
and +0.003108 at successive nodes s=1.75 and 1.875: an actual alternating
undershoot/overshoot with no state update. The independently resolved target
has neither. Thus a slope kink is sufficient to excite this discretization
mechanism; aging and a friction-parameter change are not necessary.

At ell=400 m the dimensional spacing equivalents are fault 50 m and bulk
tangential 200,100,50,25 m. This comparison does **not** prescribe those
numbers as a universal BP3 resolution criterion: the benchmark is aligned,
homogeneous, periodic, and has a different analytic transverse localization.

### Quadrature and smooth controls

At h_s=0.25, evaluating the independent manufactured traction with the same
native bulk quadrature gives max/RMS errors **1.25977% / 0.222452%**.
Splitting the quadrature, on both sides of the equation, gives
**1.33414% / 0.223549%**. Splitting alone does not cure this case and is not
promoted as a production correction.

An intentionally one-sided comparison (exact load, native matrix) gives
13.46% maximum error. It is retained and labelled, but **not** used as
evidence that a consistent production quadrature change provides a 10x fix.
This distinction prevents manufacturing a false quadrature diagnosis.

A smooth cosine control on the same h_s=0.25 grid gives 0.14385% maximum
FE-versus-exact-bulk-Q1 error, versus 1.33413% for the kink. The smooth target
is not itself Q1, so its small Q1 reference representation error is recorded
separately rather than assumed zero.

See [kink profiles and errors](../../../benchmarks/reconstructed_fault/bp3/gradient-kink/kink_refinement.png)
and [numerical table](../../../benchmarks/reconstructed_fault/bp3/gradient-kink/bulk_comparison.csv).

## 3. Why bulk refinement is not a complete junction remedy

With exactly the same Q1 grid, the independently resolved bulk operator gives
an adjacent response **-0.764248** to a unit prescribed-node disturbance.
The finest bulk FE result is **-0.764974**. Its nearest-neighbour Schur entry
is +0.01067335, versus +0.01067142 in the reference. Thus a negative adjacent
response is not merely failure to resolve bulk relaxation. It belongs to
the finite-width/Q1 weak response to these imposed data.

This is consistent with, but not a numerical prediction of, the earlier
actual BP3 frozen-history response -0.5917. The materials, localization,
geometry and boundaries differ.

For smooth chi, the high-k part of E(k) decreases as
`4*kappa*integral((chi')^2)/k^2`. Unlike the sharp-interface elastic operator,
it does not penalize indefinitely short tangential wavelengths more strongly.
At high k the local frozen friction tangent remains. There is no automatic
maximum principle for its consistent Q1 inverse. In particular, the usual
reaction stencil alone has decaying alternating ratio -2+sqrt(3).

A jump in the free-side limiting rate is finite-energy for this finite-width
problem. Continuous Q1 can approximate it in an integrated norm, but imposing
a shared boundary value can produce a narrow overshoot that does not vanish
pointwise like a regular smooth-solution error. This is a formulation/trace
question, not proof that the continuum requires an oscillatory physical rate.

See [resolved imposed-node response](../../../benchmarks/reconstructed_fault/bp3/gradient-kink/resolved_impulse.png).

## 4. Tested remedy for an incompatible free/prescribed trace

This **separate control deliberately has a jump**, unlike the original kink:

\[
 w^*(s)=-\tfrac12(1-\cos(\pi s/4)),\quad 0<s<4;
 \qquad w^*=0,\quad 4<s<8.
\]

Its free-side limit at s=4 is -1 and prescribed-side rate perturbation is 0.
The independent load is again manufactured analytically with E+d. Compare:

- continuous Q1 with the shared junction node fixed to the prescribed value;
- Q1 on the free interval with its own endpoint value, while the prescribed
  interval remains exactly unchanged. Basis functions cover each interval
  once; mass, load and elasticity are all integrated consistently. Nothing
  is smoothed or lumped, and no physical forcing is retuned.

| h_Gamma/ell | Continuous-Q1 undershoot below -1 | Independent-trace maximum error | Independent-trace L2 error |
|---:|---:|---:|---:|
| 0.25 | 45.3292% | 0.233771% | 0.113590% |
| 0.125 | 47.9910% | 0.0595926% | 0.0284689% |
| 0.0625 | 43.8269% | 0.0144602% | 0.00711153% |
| 0.03125 | 37.8203% | 0.00346102% | 0.00177647% |

The independent-trace L2 contractions are **3.990, 4.003, 4.003**.
Continuous Q1's integrated error decreases as the bad layer narrows, but its
large extremum survives. The first free trace remains continuous at s=0,
where the exact solution is compatible; only the incompatible s=4 trace is
released. This isolates the mechanism from a general discontinuous method.

The manufactured jump amplitude is **not an estimate of BP3's unknown jump**.
One cannot infer that jump by extrapolating two already oscillating BP3 nodes.

See [trace remedy comparison](../../../benchmarks/reconstructed_fault/bp3/gradient-kink/trace_remedy.png)
and [all trace results](../../../benchmarks/reconstructed_fault/bp3/gradient-kink/trace-comparison/summary.json).

### Exact frozen-state RSF check

Reuse the bulk eliminations and evaluate the regularized asinh law with
a=0.025, b=0.015, mu0=0.6, Theta=Dc/Vp and Vref/Vp=1000. A 1%-Vp
perturbation is used; Theta never changes. Sigma is scaled to the same frozen
tangent d=0.036. This removes the linear-friction approximation without
claiming a reproduction of nonuniform BP3 histories.

The kink maximum errors are **1.33109%** and **0.00592433%** on the coarse
and finest compared bulk grids. For the trace control at h_Gamma=0.125,
continuous Q1 still undershoots **47.8987% of the jump**, while the
independent-trace maximum error is **0.0595925%**. All nonlinear solves take
three Newton updates; fresh residuals are at most 8.18e-15 relative.
The friction tangent central-difference error is at most 2.34e-9 relative.

[Nonlinear kink](../../../benchmarks/reconstructed_fault/bp3/gradient-kink/nonlinear-friction/summary.json),
[nonlinear trace](../../../benchmarks/reconstructed_fault/bp3/gradient-kink/trace-nonlinear/summary.json).

## 5. What this means for BP3, and the next action

The continuous-kink test supports resolving the bulk source response instead
of refining the fault alone. But a finer bulk mesh is not a demonstrated
universal repair of the hard 40-km junction. The trace control supplies a
specific second possibility: the current shared prescribed node may be
forcing a free-side rate toward a value it does not approach naturally.

**Recommended next action:** one frozen-history, noncommitting actual-BP3
comparison giving the free side at 40 km an independent kinematic trace,
while keeping Vp on the deep side and all incoming histories/physics fixed.
This should be implemented as an explicitly bounded diagnostic, not adopted
as a production fix from this report alone. The free-side basis must enter
B, G, K, friction and damping on its own side; source measure cannot be
duplicated, and changing only K or plotted rates would not be the test.
It is a discrete-space revision relative to the authoritative continuous Q1
representation, even though the physical fault geometry need not move.

The test should measure the newly determined free trace, remaining last-element
notch, adjacent weak balances, and raw stress—not only whether a smoother
curve can be drawn. If the actual free trace is compatible with Vp, this
proposed mechanism is insufficient and the unresolved bulk/fault response
must instead be addressed. If it is incompatible and the notch collapses,
history/state representation for a committing two-trace method needs a
separate design decision.

This remedy is **not** suggested at 15 or 18 km: neither is a prescribed/free
interface. The earlier saved-data evidence remains: 15 km is primarily a
broad aging-driven slowdown, while the late revised-work 18-km rate is
monotone. The manufactured kink shows a mechanism for transient discretization
ripples, not that all three historical features share one quantitative cause.

## 6. Verification, scope and recoverability

- All 64 bulk response RHSs per FE case are checked against the original
  full saddle matrix, including restored gauge rows. Worst relative fresh
  bulk residual: **5.78e-13**. Worst free-surface residual: **2.65e-14**.
- Independent pointwise integration checks the sparse Stokes form, source
  work, normal traction, and direct surface tangent. Fourier basis integrals
  are checked against 24-point quadrature; a velocity-projector calculation
  independently checks the scalar eliminated symbol.
- The largest reflected normal action is **1.39e-12**; shear symmetry error
  is **3.25e-15**. This verifies the special homogeneous cancellation; no
  symmetry assumption is transferred to general BP3 G or the condensed solver.
- Reference cutoff 512->1024 changes the kink load by **1.84e-12** maximum.
  Trace cutoff 2048->4096 changes the independent endpoint value by
  **2.61e-11**, versus errors of order 1e-4 or larger being interpreted.
- Localization integral is one to roundoff and its squared integral is .75.
  The constant-rate normal FE approximation error is measured rather than
  asserted zero: its weak error falls from 1.6262e-5 to 1.0282e-6 when normal
  spacing halves. Its normalized rows are spatially constant.
- Exact frozen-state friction, fresh nonlinear residual, and derivative
  checks pass. `py_compile`, the diagnostic algebra tests and
  `git diff --check` pass.

Recorded numerical calculation time is **33.18 s**, excluding plotting and
small setup-debug attempts. The longest FE case takes **17.06 s**; peak RSS
is **1,104,988 KiB (1.05 GiB)**. No MPI execution, ASPECT build/test suite,
production trajectory, history update, cache change, or solver change was
performed. The diagnostic is not claimed as an MPI/production implementation
qualification. No commit was requested or made.

Two initial diagnostic assertions were corrected before interpreting results:
the local reaction is now integrated analytically instead of unnecessarily
truncating its Fourier mass expansion; constant reproduction is checked as
row-normalized traction, not equal raw loads on unequal native quadrature
rows. A zero constant-rate FE approximation error was also an invalid
expectation for Q2 integrating a cosine source. Empty failed setup directories
remain; none is counted as a passing experiment. Native-matrix/exact-load
results are retained separately and not confused with matched quadrature.

New maintained files are `manufactured_gradient_kink.py`,
`manufactured_trace_comparison.py`, `check_gradient_kink_nonlinear.py`,
`test_manufactured_gradient_kink.py`, `analyze_gradient_kink.py`, this report
and its addendum. README points to the result. Small CSV/JSON, matrices and
plots are under `benchmarks/reconstructed_fault/bp3/gradient-kink/`.
Earlier uncommitted BP3 probe/plugin changes are preserved, not modified by
this experiment. `source/` and production headers have no new diff.

Base revision and source/authority hashes are in
[provenance.json](../../../benchmarks/reconstructed_fault/bp3/gradient-kink/provenance.json).
The scripts refuse to overwrite run directories. Example reproduction
commands (use a new label or a clean evidence directory):

```sh
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/bp3/manufactured_gradient_kink.py --nx 32 --label repeat-bulk32
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/bp3/manufactured_gradient_kink.py --nx 64 --ny 64 --label repeat-normal64
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/bp3/manufactured_gradient_kink.py --nx 128 --ny 64 --label repeat-bulk128-normal64
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/bp3/test_manufactured_gradient_kink.py
```

The trace and nonlinear scripts have fixed evidence directories and likewise
refuse overwrite; their completed output is reused by the analysis script.
