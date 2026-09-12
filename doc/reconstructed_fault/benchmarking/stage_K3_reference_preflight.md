# K3 corrected reference preflight — smoke held for review

Historical diagnostic for the rejected .009-m/s ramp. The subsequent authorized
two-candidate adjustment and current smoke-preparation status are recorded in
`stage_K3_bounded_adjustment.md`; this original evidence is retained unchanged.

The cutoff is corrected to the configured K1 value **0.1**. The independent
reference predicts clear H -> phi -> I_h feedback and an admissible mechanical
root, but not an acceptable supported slip normalization on the second step.
No ASPECT smoke, build, production correction, support change, loading scan or
criterion change was made. The opt-in benchmark source is **not build-verified**.
K2 references remain provisional and Gate K2 remains unmet.

## Provenance and initialization

Source baseline: `dc1a96d3f72dce123393c90ad025e729885e8ee9`, with the benchmark
and documentation changes listed below. No production source was edited.

- `source/material_model/phase_field_fault.cc`: the virtual
  `get_phase_field_activation_threshold()` returns the parsed model member;
  its declared default is .1.
- `source/reconstructed_fault/manager.cc`: prescribed initialization requests
  that virtual accessor. It does not supply the generic .01 value itself.
- Resolved, validated K1 parameters:
  `benchmarks/reconstructed_fault/uniform_shear/residual-floor/convergence/space32_dt05/parameters.prm`,
  line 8537: `Phase field activation threshold = 0.1`.
  SHA256: `b92b7827deb3d9ed7f3a3d783e9363f972fb49fff3cd0ba5f843fb8c134098cc`.
- The reference reads that parameter directly. The K3 overlay pins the same
  .1 explicitly. Its other changes are precisely the prepared evolving mode,
  boundary ramp, maximum dt and two-real-step termination/output selection.

Retained histories are initialized once: supplied shear stress 1500 Pa,
Theta0=200 s, prescribed-profile H0 with the .1 cutoff, independently projected
C0=317.7426503 Pa and I0=108.1349225 m. The initial phase has
max(phi0)=.5997028553; H0,max=29568.04595 Pa. The analytic initialization support
is a=.3088215939 m and phi_star=.1 occurs at |y|=.2040004863 m.
There is no core Dirichlet condition or phase irreversibility obstacle.

The artificial 2-s timestep-zero evaluation produces V0=.0003152708172 m/s,
q0,evaluated=1044.033232 Pa and C0,evaluated=317.2240643 Pa. Only kinematics is
committed at zero; these evaluated stresses do not replace the retained
1500-Pa/C0 histories. Theta0 and H0 are not physically advanced.

## Independent calculation and cheap verification

`evolving/reference.py` uses a stationary first-integral inversion, a split
normal P1 weak phase solve with natural Neumann boundaries, Gauss quadrature,
a banded analytic Jacobian and bracketed scalar mechanics. It uses the exact
indexed cycle in `stage_K3_preparation.md`, exponential aging and the factorized
finite-step H maximum, checked against its equivalent work expression.
Localization alone uses max(phi,0). No production constitutive calls are used.
This is a continuum reference, **not the production CPDI discretization**.

The requested first timestep is **2 s**: the reference's initial convective
limit is 16.4402 s, configured maximum-first-step is effectively unlimited,
the K3 maximum is 2 s, and a_f>b_f imposes no RSF splitting limit. The second
step is shortened to .4340277152 s by convection. These are source-consistent
predictions, not accepted ASPECT times; a later reference comparison must use
the actual accepted sequence without resetting histories.

| Quantity | First real step | Second real step |
|---|---:|---:|
| Physical time (s) | 2 | 2.434027715 |
| dt (s) | 2 | .4340277152 |
| Imposed U (m/s) | .009 | .009 |
| Interior root V (m/s) | .008673993508 | .009228337970 |
| Shear stress q (Pa) | 2115.834115 | 2007.780462 |
| C (Pa) | 470.286364 | 402.211270 |
| Theta (s) | .1152929942 | .1084881444 |
| Maximum phi | .5997028553 | .6341084886 |
| I_h (m) | 108.1349225 | 135.8261554 |
| H_max / preceding H_max | 1.248714318 | 1 |
| Omitted h fraction | 5.90998e-5 | 9.25172e-5 |
| Supported slip-normalization error | 5.90998e-5 | **5.15884e-4** |

Both root residuals are below 1e-7 Pa with positive F(V_min). H1 increases by
7353.99639 Pa at its maximum; phi2-phi0 reaches .0344056333 and I_h grows
25.6080%. The phi2 residual with retained H0 is .365444, whereas the solve
with H1 converges. Thus feedback is not an artificial timestep-zero update.
The phi<.8 envelope passes with substantial margin.

One fixed-physics accuracy check doubles nominal normal cells 2048 -> 4096
and Gauss order 6 -> 8. Extra splits at cutoff/support yield 2052/4100 cells.
Maximum common-node phi differences for 0/1/2 are
6.431e-7/6.431e-7/1.203e-7. Relative I_h/V/C/H_max-ratio differences are below
1e-5. This checks the stated inexpensive reference targets, not production
convergence or an exact-error bound. The normalization failure changes by only
1.417e-8 absolute. The initial finer check stalled at a double-precision
phase residual 2.40729e-11. The independent iterate/residual now use long-double
arithmetic, with the same stopping tolerance and double banded corrections;
both reference runs converge. This changes no ASPECT numerical mechanism.

Final reference calculation times: .0656/.1338 s internally; both commands
together .907 s wall. Four reference checks pass in .025 s (command .233 s):
configured cutoff/intact equilibrium, directional phase Jacobian, saved feedback
and failed-budget flags, and saved reference-resolution agreement. Including
the first attempt, exploratory execution is under five seconds, far below
120 s per run/600 s aggregate. Exit zero means calculation succeeded, **not
that the smoke budget passed**; both JSON reports set `smoke_budget_pass=false`.

## Why containment does not certify normalization

Full I_h is retained in mechanics and the cohesive law. Over the full profile,
the history integral is approximately -9.61e-18 m/s and total crack-strain
normalization differs from unity by less than 1e-12. Over the unchanged
admitted strip at step 2, however,

- omitted instantaneous fraction: 9.25172e-5;
- omitted history integral: **+3.90697509e-6 m/s**;
- retained history integral: **-3.90697509e-6 m/s**;
- total relative deficit: 9.25172e-5 + 3.90697509e-6 / V2 = 5.15884e-4.

The omitted-fraction proposal <=1e-4 passes; the separate actual-normalization
proposal <=1e-4 fails. Tail cancellation from changing I_h makes checking h
alone insufficient. This is a **prediction over the reference strip**, not an
ASPECT measurement: actual admitted parents/full domains and bulk QP support
must eventually be measured separately. The full-profile scalar trajectory
is not claimed exactly equivalent to truncated production mechanics. No tail
renormalization or criterion adjustment has been introduced. Stop for review
of this failed preflight rather than launch the smoke or search for loading.

## Periodic particles and unresolved smoke instrumentation

For 96x384 particles, horizontal spacing is .0026041667 m. The estimate retains
particle x positions between steps and applies modulo .25 only after each
advection. It follows `source/particle/integrator/rk_2.cc`, the old/predictor
sampling in `source/particle/manager.cc`, and extrapolation in
`source/simulator/helper_functions.cc`: mechanics has not yet supplied the new
accepted velocity when particles move. Step 1 therefore uses u0; step 2 uses
the mean of u1 and u1+(dt2/dt1)(u1-u0).

Predicted maximum displacements are .0002376002 and .0021624804 m, with
**0 and 350 crossing events**, respectively. The finer independent reference
gives the same counts. No periodic-domain construction claim follows from
these kinematic estimates. Eventual diagnostics must measure H, phi, I_h, C
and V along the fault, including endpoint/seam layers. A seam effect comparable
to the intended feedback stops interpretation as a 1-D result; wrapping alone
does not fail it.

The default-off benchmark mode and overlay are implemented but not compiled
or run. It records old H at timestep entry and preserves old C/I_h for the
post-commit crack-strain diagnostic. It does not refresh live histories or
reevaluate Maxwell updates. This diagnostic formula is scoped to the existing
isothermal, single-background-material K1 fixture. The planned actual frozen
CPDI phase-entry weights/residual check is **not implemented** by this minimal
mode; it remains necessary before claiming full smoke verification. No current
source change is silently described as build-verified or smoke-safe.

## Artifacts and exact commands

All paths below are under `benchmarks/reconstructed_fault/uniform_shear/`:

- `evolving/reference.py`, `evolving/test_reference.py`, `evolving/smoke.prm`;
- `evolving/preflight/` and `evolving/accuracy-check/`: JSON reports plus
  initialization/step-1/step-2 phase and H profiles;
- `evolving/preflight.log`, `evolving/accuracy-check.log`,
  `evolving/reference-tests.log`;
- opt-in source diff in `uniform_shear.cc`; no existing plugin binary changed.

From the repository root (each final command exited zero):

```sh
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/evolving/reference.py --parameters benchmarks/reconstructed_fault/uniform_shear/residual-floor/convergence/space32_dt05/parameters.prm --output benchmarks/reconstructed_fault/uniform_shear/evolving/preflight
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/evolving/reference.py --parameters benchmarks/reconstructed_fault/uniform_shear/residual-floor/convergence/space32_dt05/parameters.prm --cells 4096 --quadrature 8 --output benchmarks/reconstructed_fault/uniform_shear/evolving/accuracy-check
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/evolving/test_reference.py -v
```

The preparation and progress record are updated. Next decision: review the
normalization failure and the still-proposed K3-specific budget. No automatic
change of support, full I_h, loading, initialization or acceptance is authorized.
