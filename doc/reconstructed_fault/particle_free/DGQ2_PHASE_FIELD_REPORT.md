# Native DGQ2 phase: implementation and available coupled results

**6 November 2026.** Native DGQ2 transport and constrained SIPG growth are
implemented in their supported paths. Operator, short growth/lifecycle tests and
both positive-translation coupled timesteps pass. **At the user's request, the
reverse-direction, finer-mesh and remaining tests are deferred.** Both completed
runs and their diagnostic reductions finished before the queue was closed;
neither remaining full run started. No new simulation, build or operator/unit test followed that
instruction. This report closes the requested work using the available evidence.

DGQ2 substantially improves phase initialization, balance and shape retention.
It also changes the coupled slip solution appreciably. Coupled timestep
sensitivity and energy error are not uniformly improved; spatial/directional
qualification remains open. Keep DGQ2 selectable and Q1 available, with no
production-readiness or default-change claim.

## Source, build and preservation

Root `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `cf06525471ce89789f1ac10905dc79f3aac7562a` plus preserved
local changes. [Entry](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/entry.json)
records 938 incoming source hashes and the incoming diff. The separate candidate
is `build-particle-free/dgq2-phase/candidate/aspect-release`, SHA256:

```text
5e1e8620a21a5c6dca3a093c5f99d5e31e1d573e167f20e4807130acbba6f137
```

Its `source-identity.json` records 947 source/include/unit hashes, matching
plugins and benchmark seams; `tracked-source.diff` preserves the tested source
difference from HEAD. It was built in `build-particle-free/dgq2-phase/native`
with GCC 12.4, deal.II 9.6, Release, unity/PCH enabled and four build workers.
MPI runs used OpenMPI 5.0.6, at most four ranks and one thread per rank. Builds,
checks and heavy reductions ran sequentially. The candidate stayed frozen during
the full comparisons.

[Closeout preservation](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/preservation.json)
confirms all **1,020 protected artifacts**, the candidate sources/plugins and
eight audited stress/friction source files are unchanged from their recorded
baselines. No checkout, reset, production run or checkpoint conversion occurred.
Cleanup availability losses remain **39/45 native-pilot frames and earlier
temporary logs**. HEAD alone is not the tested binary identity.

## Implemented framework

The [design and source map](DGQ2_PHASE_FIELD_DESIGN.md) and
[current moving-fault guide](MATURE_MOVING_FAULT.md) are the maintained algorithm
description. The implementation provides:

- One persistent native DGQ2 phase component, selected by `Use discontinuous
  phase discretization = true` and `Phase polynomial degree = 2`. Continuous Q1
  remains default. Initialization, mechanics, normalization, output and restart
  consume all nine modes. Face-coincident observations select one trace before
  evaluating nonlinear localization; there is no trace averaging or Q1 return
  projection.
- Upwind **material** advection, including `phi div(v)`, with frozen accepted
  full velocity, Lobatto5 volume/face integration, SSPRK3 subcycling and limiting
  at each required stage. A positive-mean CFL and mean-preserving Bernstein
  enclosure scaling enforce whole-cell physical bounds to roundoff. There is
  no mean clipping, global renormalization, donor-range clamp or epsilon cap.
- SIPG fracture energy, residual and tangent, including both face consistency
  terms, penalties and weak boundary terms. The existing `Discontinuous penalty`
  is reused. Physical coefficients and mechanical/normalization variations stay
  intact; no physical time-mass term is inserted into equilibrium.
- Bernstein lower/upper-gap constraints and sparse distributed quadratic
  obstacle search inside the existing nonlinear endpoint solve. Final physical
  energy/KKT gates remain. The search problem is not a replacement for that
  nonlinear solve, and no transport limiter is applied after equilibrium.
- The existing joint candidate/accepted transaction, schema9 for DG material
  phase packets, early native phase-space restart validation and explicit
  rejection of unsupported CPDI/DG combinations and adaptive DG phase transfer.

Stress DGQ2/SSPRK3, direct physical stress, corrected bulk spin, accepted-spin
lag, endpoint/RSF conventions, stress/mechanical quadrature and convergence
criteria are unchanged. Scalar SSPRK3 does not raise the complete moving-fault
method to third order: geometry and coupled time discretizations are retained.

## Completed verification

The actual native sources are compiled into the operator driver; these are not
Python copies of the new operators. Exact commands, exit codes and wall times
are linked from the [evidence index](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/README.md).

| Check | Result |
|---|---|
| Frozen candidate full unit suite | 154 cases / 185,541 assertions pass; 6.58 s |
| Final scalar transport matrix | 40 cases pass: both directions, rotation, divergence, inflow, quadratic shear, three meshes and time controls |
| Material balance in that matrix | Maximum step defect 9.42e-15; maximum accumulated absolute defect 2.75e-13 |
| Smooth admissible controls | No limiter activation; limited/unlimited comparisons retained |
| SIPG/obstacle MPI matrix | 18 checks pass in serial and on two/four ranks |
| Static nonmatching faces | 12 checks pass in serial/four ranks; independent positive-model minimum eigenvalue .0723586 |
| Moving manufactured obstacle | Transport then constrained growth error at most 1.90e-13; positive growth and no healing |
| Native fixed-fault cohesive growth | 17 steps to .53125, 45.88 s; integral .15636083 → .15888770; final increment .00176223; KKT 3.06e-14 |
| Native growth derivatives/trace | SIPG energy/tangent, full normalization and mechanical action checks pass; Gauss12/32 ray difference about 1.6e-15; face-coincident trace checked |
| Growth MPI control | Two physical steps on four ranks pass |
| DG four-to-two-rank restart | Bulk difference 1.78e-14; surface 8.88e-15; exterior background identical |
| Forced rejection/retry | Bitwise identical to the matching accepted-schedule control |
| Native Q1 regression | Initial state and two steps bitwise identical to retained full Q1 output |
| Q1 metadata-absence continuation | Private fresh-Q1 checkpoint copy without new metadata: bulk 3.57e-14 / surface 3.56e-15; background identical |
| Cross-family restart | Q1→DG and DG→Q1 reject before vector deserialization |

The last Q1 check tests the historical metadata-absence convention; it does not
claim a new continuation from the original campaign checkpoint. That additional
short test was prepared but deferred with the remaining tests.

Final limited transport L2 errors, at fixed travel/width, are:

| Field | Coarse | Middle | Fine |
|---|---:|---:|---:|
| Positive translation | .00297543 | .000303761 | .0000322364 |
| Negative translation | .00293840 | .000302701 | .0000322340 |
| Rotation | .00263094 | .000285567 | .0000305473 |
| Nonzero divergence | .00238032 | .000295507 | .0000268187 |
| Smooth inflow | .000852549 | .000105672 | .0000131864 |
| Quadratic shear | .00782156 | .000950883 | .000115436 |

Quadratic shear uses one coarser set of levels. Positive translation has measured
orders 3.29 and 3.24. Initialization is recorded separately. Independent retained
Q1 replay gives .0668008, .0233907, .00331697 at dt=.05 on the corresponding
translation meshes. Its Python runtime is not compared with native C++ timing.
Fine rotation costs 93.8 s with whole-cell bounds versus approximately 43 s
unlimited; bound certification in nearly empty cells is measurable.
See [operator figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/operator-comparison.png)
and [compact data](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/operator-summary.json).

Retained failed evidence includes the sandbox MPI launch, initial driver build
issues, an MPI active-set timeout caused by mixed insertion/addition collectives,
and a long temporal balance diagnostic affected by naive summation. The MPI
load now accumulates each row locally and inserts once; diagnostic integrals
accumulate in long double. Equations and numerical tolerances were not relaxed.
An initial Lobatto4 transport rule underintegrated general Q2 transverse
products; Lobatto5 and a quadratic-shear manufacture corrected and checked it.
Two native-growth setup failures involved obsolete observers and the missing
retained particle-freeze environment; both logs remain. Earlier valid affine
operator results are preserved as historical evidence.

## Available coupled comparison

The reused layered fixture has G=10, eta(y)=5.5+4.5 tanh(y/.35), U=+2, loading
3.7, 20 fixed material labels, domain [-6,6]×[-2,2], h=.125 (96×32 cells), T=.8,
and matched prescribed exterior stress/top traction. Growth is disabled. New
runs start fresh; completed Q1 outputs supply controls. This velocity-strengthening
fixture is not the dimensional BP3-like crustal example or a nucleation test.

The new runs completed **192 physical steps with zero retries**. Exterior
recurrence checks pass; geometry replay is within 2.30e-16 and age replay within
5.56e-17. Both DG timesteps have identical initial fields and surface state.

Initialization matters: relative phase L2 error against the prescribed analytic
profile is **4.3625% for Q1 versus .2187% for DGQ2**. The analytic integral is
.1941807407407407. Old/new initial phase fields differ by 4.4711% in relative L2;
initial normalization and slip-rate vectors differ by 4.5828% and .5369%.
Initial physical stress, geometry and age agree. Consequently the coupled
old/new comparison includes initialization as well as subsequent transport.

| Method | dt | Integral change | Final sampled peak | Final normal width | Wall time |
|---|---:|---:|---:|---:|---:|
| Q1 retained | .0125 | +6.24045% | .289853 | .173168 | 39.17 min |
| DGQ2 | .0125 | −.005306% | .350309 | .139031 | 43.64 min |
| Q1 retained | .00625 | +7.03733% | .286245 | .175092 | 71.37 min |
| DGQ2 | .00625 | −.007573% | .350421 | .139109 | 75.63 min |

Peak samples use 81 points in every cell, including each side's face traces;
these samples supplement the native whole-cell bound guarantee. Roundoff-scale
negative reconstructed samples are at most 1.3e-31 for the final DG fields.
The small peak overshoot is allowed by the physical [0,1] bounds; no artificial
initial-peak cap was applied. Initial normal widths are .163304 (Q1) and .155209
(DG). Width is the phase-weighted standard deviation about the mean curve normal.

For DG, integrated divergence sources are −1.03039e-5 and −1.47046e-5; outward
phase flux is below 8.0e-15. They account for the integral changes. Accumulated
absolute **native SSPRK material-balance defects are 5.85e-15 and 5.86e-15**.
Both runs use 512 transport substeps in total. The numerical Stokes velocity is
not pointwise incompressible: DGQ2 phase can weight its divergence, whereas Q1
belongs to the pressure test space. Raw integral change is therefore not the
transport defect. The endpoint/trapezoid phase estimates are separately retained;
they do not replace the native stage-weighted balance.

### Stress, traction, slip and state

All comparisons retain absolute and reference norms. Bulk norms use the union
of both cell partitions; surface quadrature splits at labels, mesh crossings
and region boundaries. Central labels are [.2,.8]; ends are the remaining labels.
The following are **relative L2 differences (%) under timestep halving**, not
errors against an exact coupled solution:

| Quantity | Q1 central | DG central | Q1 ends | DG ends |
|---|---:|---:|---:|---:|
| Surface stress | .700 | .538 | .891 | .893 |
| Shear traction | 2.355 | 2.404 | 1.340 | 1.782 |
| Pressure-inclusive normal traction | .425 | .380 | 1.028 | 1.355 |
| Slip rate | .281 | .427 | .518 | .968 |
| Cumulative slip | .709 | .801 | 1.357 | 1.403 |
| Age | .347 | .443 | .952 | 1.911 |
| Normalization I | .688 | .154 | 2.103 | .280 |
| Geometry | .152 | .176 | .149 | .167 |

Whole-domain stress difference is .2384% (Q1) versus .2447% (DG); phase difference
is 1.4695% versus 1.1093%. Better phase balance does not uniformly improve all
coupled time sensitivities. The unchanged geometry, endpoint and spin lag remain.

At dt=.00625, changing Q1 to DG changes whole-domain stress by 1.678%, central/end
surface stress by 13.96%/9.23%, slip rate by 7.56%/12.13%, cumulative slip by
4.07%/7.45%, and age by 9.81%/22.17%. These are substantial solution changes,
not a physical-error certificate. Raw shear differences are large relative to
the small perturbation shear; including prescribed loading gives driving-shear
differences of 1.44%/1.27%, and work-driving shear .249%/.427%. Effective compression
changes .539%/1.084%. Normalization I changes 16.18%/17.36%.

### Energy, normalization and cost

| Method / dt | Endpoint signed | Endpoint absolute | Trapezoid signed | Trapezoid absolute |
|---|---:|---:|---:|---:|
| Q1 / .0125 | −.00311206 | .0754080 | .000546914 | .00232096 |
| DG / .0125 | −.00340549 | .0751197 | .000243675 | .00261682 |
| Q1 / .00625 | −.00143523 | .0385043 | .000385410 | .000599813 |
| DG / .00625 | −.00162173 | .0383207 | .000191993 | .000697970 |

Units are energy per out-of-plane length. DG endpoint absolute remainder divided
by accumulated active work decreases **.2686% → .1370%**, almost the same as Q1
(.2695% → .1376%). Trapezoidal remainders also decrease, but are somewhat larger
for DG. Small signed sums conceal cancellation; they are not the absolute error.
Accepted reaction-projection losses sum to 2.98e-5 and 1.63e-5 for DG, separately
recorded from transport-stage and total-work diagnostics.

The volume localization-integral/curve-length differences are −.534% and −.623%
for DG, versus −.585% and −.440% for Q1. They include finite-end integration,
curved segment ownership and fixed-label normalization interpolation. Final
q12-minus-q8 localized powers/measure remain sensitive (for the first DG case,
friction power changes .00146 and localization measure .00275); bulk storage,
profile integral and unlocalized terms agree near roundoff. These are sensitivity
checks, not a converged finite-end integral. Tiny signed energy or normalization
improvements must not be interpreted beyond this precision. No frozen audit or
native mechanical-quadrature change was added.

DG uses 27,648 phase DoFs versus 3,201 Q1 DoFs. Total wall time increases 11.4%
and 6.0% in these runs; this includes changed fields and solver effort. Native
phase transport itself takes 28.6 s (1.1%) and 36.9 s (.81%). Linear iterations
are 14,813/24,354 for DG versus 14,625/24,151 for Q1. Maximum concurrently sampled
sum of rank RSS is 1,258/1,268 MiB; shared pages can be counted per rank. Old Q1
RSS was not sampled, so no measured memory-overhead ratio is claimed.

## Figures, reproducibility and decision

- [Profile balance, peak, width and localization](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled/profile-comparison.png)
- [Central/end timestep sensitivity](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled/refinement-comparison.png)
- [Stress traction, slip and state traces](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled/surface-comparison.png)
- [Signed and absolute energy diagnostics](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled/energy-comparison.png)
- [Runtime, phase DoFs and linear effort](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled/cost-comparison.png)
- [Complete compact comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled/comparison-summary.json)
  and [CSV](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled/comparison-summary.csv)

`dgq2_phase_field/compare_coupled.py --available-only` reproduces the coupled
figures and comparison from completed outputs. The [benchmark README](../../../benchmarks/reconstructed_fault/particle_free/dgq2_phase_field/README.md)
records entry points and build instructions. `dgphase-coupled01` has wrapper exit
−15 because the user deferred its remaining queue; **both individual completed
simulation commands have exit 0**. The held queue and closure are recorded in
`coupled/queue-hold.json` and `queue-finish.json`. No native jobs from this task remain running.

The evidence supports retaining the implemented DG option and using it for
further moving-profile research. It does not yet establish spatial convergence
of the changed coupled solution or coupled reverse-direction performance. The
most informative next selected experiment, when resumed, is **U=+2, h=1/12,
dt=.00625**, with the same 20 labels and initialization convention; the already
planned reverse run then completes the direction check. Both remain deferred.

Mature moving growth stays disabled. The manufactured moving obstacle and short
fixed-fault cohesive growth do not supply a missing moving-cohesive law or resolve
older cohesive tip/energy failures. Static nonmatching operators are checked;
adaptive DG phase/obstacle transfer is rejected. Native compact-profile inflow
is zero; general nonzero inflow is an operator capability, not a general native
provider API. Replicated scalar/normalization structures remain in experimental
cohesive coupling despite sparse gradient/search algebra. The inherited adaptive
controller's older-surface-state issue remains; these runs use fixed schedules.
Production BP3 and cohesive histories, energy inventories and readiness claims
remain distinct. No production trajectory is qualified here.
