# PF4 prescribed DG material transport — 15 October 2026

**PF4 prescribed transport scope complete; no limiter enabled.** Selected scope is the PF3 handoff: prescribed smooth DG material
transport, fixed mesh/geometry/constants, no limiter initially, separate
space/time refinement and bulk/surface commutator controls. Existing defaults,
Maxwell/friction/phase laws and all PF0–PF3 evidence remain unchanged. The
transport operator is noncommitting; native outer physical DG stepping and
particle motion remain guarded. No surface birth, mesh transfer or coupled
algorithm is selected.

## Operator and controls declared before implementation

For each in-plane component of a full-3D plane-strain deviator, solve

```
M dA/dt = integral A v_h . grad(test)
        - integral_faces (v_h . n) A_up test
        + integral (div(v_h) A - r A + f_A) test.
```

Use the existing independent DGQ2 space and native continuous Q2 FE velocity.
Internal upwind states use the same physical face points and one normal
velocity. Boundary inflow is explicit prescribed tensor data; outflow uses
the interior trace. Gauss3 gives a full-rank scalar mass; Gauss4 integrates
native affine-Cartesian Q2 velocity/advection products, including transverse
degree6. Source/inflow quadrature refinement is tested separately. The optional
diagnostic reaction rate is either zero (isolate transport) or existing G/eta;
no new material parameter, spin, stress clipping or normalization floor.

SSPRK3 stage times are t, t+dt, t+dt/2. Every call anchors one immutable incoming
DG field and returns a candidate; no stage publishes accepted state. Freeze a
spatial operator only for the explicit fixed FE velocity used to build it.
Transported phase bounds, native surface-history transport and complete coupled
publication are outside this operator experiment.

Verification targets: representable polynomial/constant weak-operator defects
<=1e-11 scaled; tensor symmetry/trace roundoff; decreasing smooth characteristic
errors under independent spatial refinement; third-order time convergence
against a same-space fine-time reference (report its uncertainty); source/rule
refinement distinct from representation error. Constant data with nonzero
divergence must stay constant, with the uncorrected density operator as a
negative control. Signed nonsmooth data have no arbitrary component bounds;
measure extrema, full tensor norm errors and loss of quadratic norm before
deciding on limiting.

The expected Q2 spatial asymptotic rate is tested at >=2.5 on the finest pair,
with decreasing errors on the final three meshes. SSPRK3 time rates must be
>=2.7, the512/1024 reference difference <=5% of the128-step error, and a finest
spatial pulse dt/2 control must contribute <=1% of its spatial error. These
targets precede the finer mesh runs. Characteristic error integrals are evaluated
independently; nonsmooth reference jumps are split before quadrature.

Prescribed V3 controls use the unchanged non-objective rate: frozen localization,
compatible co-translation, material passage through fixed localization, and
differing bulk/surface tangential labels. Compare direct stress, cohesive
redistribution and mature residual representations; omitting a nonzero
commutator is a negative control. A separately forced rotating-tensor identity
is labeled objective, never adopted as a production spin law.

## Implemented boundary and source ownership

Worktree `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `3232056cde12dbae17ef4265ecb804f9098b88f1` plus
uncommitted PF2–PF4 work. Root `AGENTS.md` is effective; no override found.
Incoming manual instructions/derivation/plans and PF0–PF3 evidence are preserved.
No other checkout/build, staging, commit or push is involved.

| File / owner | Responsibility |
|---|---|
| `phase_field_fault.h` | Private fixed-velocity transport/project/SSPRK3 interfaces; no new runtime selector or physical parameter |
| `dg_history_internal.h` | Existing DGQ2 space/tensor definitions moved from `dg_history.cc`, now genuinely shared with transport; fixed operator data only |
| `dg_history.cc` | Existing accepted history/source/stress behavior, using the shared definition; numerical routines unchanged |
| `dg_transport.cc` | Serial native-FE material operator, full-rank mass, upwind neighbor traces at matching physical face points, explicit inflow/source loads, analytic-fixture projection and noncommitting SSPRK3 |
| `transport_checks.cc`, four `transport_*.prm` inputs | Separate observer and bounded fixed-grid controls in the existing native fixture |
| `assess_transport.py` | Independent physical-coordinate assembly, characteristic fields, full-3D norms, quadrature and separate refinement studies |
| Benchmark CMake / runner | Build/load only the selected observer; provenance includes all new source/include files |
| `current_design.md`, `specification.tex` | Matching restricted operator contract; existing outer physical stepping/restart guards retained |

The accepted material mode remains the fresh unloaded cohesive DG option. The
observer projects arbitrary **scratch** analytic tensors into its native space;
this is not loaded-state initialization or particle/checkpoint conversion.
Each run initializes ASPECT once and uses controlled transport time T=.2 s,
without advancing the outer simulator clock or moving particles. All accepted
DG, phase and Z/I values are exactly unchanged at the end. Source/inflow NaN
rejection, unsafe-CFL rejection, immutable incoming candidates and identical
repeated trials are checked separately. No full physical-step rollback/restart
or terminal transport publication is claimed.

The matrix is fixed only for its explicit FE velocity. It must be rebuilt when
velocity or mesh changes. The speed used by the diagnostic CFL guard is sampled
at volume/face points; .125 is not a general stability theorem or production
controller. Spatial tests use a smaller sampled CFL <=.025. Periodic faces,
hanging faces, moving meshes and MPI transport are rejected/unsupported.
Homogeneous reaction r=G/eta=.01 s^-1 is included only in named rate controls;
pure-advection controls use r=0. The native exact-Maxwell mechanical update is
unchanged. No combined native operator split is adopted.

## Independent operator and characteristic checks

The independent checker imports no production routine. It rebuilds DGQ2 basis
values/gradients from exported support points and physical cell coordinates,
forms/inverts its own mass, and independently integrates volume/neighbor/face
blocks at Gauss5. It checks the native matrix, divergence correction and inflow
load. A Q2 tensor-source control uses scalar x²y² with a native Q2 cross-term
velocity `(0.2+0.1*x²*y, -0.15+0.05*x*y²)`; its exact material derivative is
integrated separately. Full tensor norms always include zz=-(xx+yy) and both
xy entries; no second deviator operation is applied to stored stress.

| Check | Measured result |
|---|---|
| Full matrix / divergence-correction errors | 1.489e-15 /1.554e-15 scaled; target1e-11 |
| Inflow-load error | 5.921e-16 scaled; target1e-11 |
| Constant weak derivative / polynomial weak derivative | 1.099e-14 /4.330e-15; target1e-11 |
| Omit divergence correction, constant input | Derivative defect .300000000000001; detected negative control |
| Native candidate repeat | Exact equality; invalid source and unsafe dt rejected with unchanged incoming data |
| Constant tensor under nonzero divergence | Maximum relative full-tensor error8.28e-16 across all meshes |
| Density negative control | Material-constant error .03921056; agrees with the distinct density characteristic to3.43e-9→3.07e-11 relative |
| Frozen localization/reaction | Relative tensor error1.38e-15 |
| Native non-objective uniform tensor in rotating flow | Relative error below1e-14; tensor components remain material constants |
| Externally forced uniform objective rotation | Relative error1.194e-8; no spin is added to the kernel |

The compressible velocity `(0.2+0.3*x, -0.1*(y-0.5))` has divergence .2.
Its independent material characteristics are
`x0=(x+2/3)*exp(-.3t)-2/3`, `y0=.5+(y-.5)*exp(.1t)`.
Material constants remain constants; density constants acquire exp(-.2t).
These are manufactured operator controls, not compressible Stokes solutions.
Continuous Q2 FE interpolation supplies the actual production mapping/face
velocity; particle or mapped-stress values do not enter the operator.

Smooth exact-error integrals are checked with independent Gauss12/16; all
reference-uncertainty gates pass1e-8 scaled. Native Gauss8 norm/error diagnostics
agree with independently evaluated Gauss8 results to1e-11. The native DG tensor
coefficients, not images or summary lines, supply every comparison.

## Separate space, time and quadrature refinement

All spatial cases retain the physical fields, coefficients and T=.2 s on
4x8,8x16,16x32,32x64 affine grids. Independent initial-pulse projection errors
are7.524%,.7732%,.1332%,.01689%; final errors below include transport as well
as representation. Meshes are native ASPECT meshes, not interpolated outputs.

| Case / relative full-tensor L2 error | 4x8 | 8x16 | 16x32 | 32x64 | Finest order |
|---|---:|---:|---:|---:|---:|
| Translating smooth pulse | .075205 | .018173 | .0020976 | .00026275 | 2.997 |
| Pulse in nonzero-divergence flow | .073709 | .014097 | .0017455 | .00022116 | 2.981 |
| Fixed-profile cohesive normal passage | .024715 | .0028122 | .00033720 | .000041656 | 3.017 |
| Mature direct normal transport | .017398 | .0026710 | .00033537 | .000041715 | 3.007 |
| Mature residual reconstruction, normal | .0098222 | .0025163 | .00032786 | .000041792 | 2.972 |
| Cohesive tangential-label mismatch | .016690 | .0017452 | .00022009 | .000027650 | 2.993 |
| Mature direct tangential transport | .016695 | .0017447 | .00022016 | .000027651 | 2.993 |
| Mature residual reconstruction, tangential | .016538 | .0017078 | .00021706 | .000027250 | 2.994 |
| Compatible co-moving profile/labels | .017498 | .0026998 | .00033950 | .000042201 | 3.008 |
| Rotating profile, native non-objective tensor | — | .0024995 | .00031677 | .000039549 | 3.002 |
| Rotating profile/slip tensor, objective forcing | — | .0025015 | .00031712 | .000039576 | 3.002 |

The finest translating-pulse dt/2 change is2.4192e-6 Pa m, only3.222e-5 of
its spatial L2 error (target<=.01). On the fixed8x16 space,16/32/64/128-step
errors against the1024-step solution are .00591882, .000738863, .0000922723,
.0000115101 Pa m. Orders are3.00193,3.00134,3.00300. The512/1024 reference
difference1.57628e-7 Pa m is1.37% of the128-step error (target<=5%). The
continuous characteristic error on this space remains about5.19 Pa m: temporal
convergence is distinct from spatial accuracy, and says nothing about a coupled
phase/mechanical time discretization.

At the same8x16 mesh/history/normal-passage problem, changing source/face/volume
Gauss4→8 changes the final field by .00477000 Pa m (3.707e-7 of its reference
norm, about .0132% of its spatial error). Gauss8→12 changes it by6.397e-12
Pa m. Thus polynomial operator integration is exact at4, while nonpolynomial
load integration has its own measured error. These Gaussian controls do not
qualify the nearly saturated Lorentz profiles whose quadrature failure is
preserved in PF3.

## Actual-discretization commutator controls

Use the named prescribed template
`chi(y)=exp(-((y-.5)/.12)^2)/(.12 sqrt(pi) erf(.5/.12))`, shear S_xy=.5,
Z0=5000 Pa m and matching r=.01. Let
`f(x)=1+.25 sin(2 pi x)`. These are analytic diagnostic fields; native phase,
normalization, surface Z and geometry remain fixed and unpublished.

- **Frozen / compatible translation:** material and surface labels translate
  together and the fixed-amplitude profile template follows the normal motion.
  `D_b chi=0` and matching tangential labels give C=0; unforced direct transport
  converges as shown. The template is extended analytically for inflow. It is
  not renormalized after translation inside the fixed Box, so this is not a
  native moving-normalization or geometry qualification.
- **Normal passage:** bulk normal speed c=.25 through static chi and stationary
  surface labels gives `C=exp(-rt) Z0/G c chi' S`. Cohesive tau_xy is
  `-Z0 exp(-rt) chi(y)`; source Q_geo,xy is `-Z0 exp(-rt)c chi'(y)`.
  Mature direct stress instead advects the old profile. A separately transported
  residual with that source and initial value0 recovers mature stress from the
  current-profile representation. Dropping the cohesive redistribution produces
  a finite relative error .407786 at the finest grid.
- **Tangential mismatch:** bulk speed .35 and surface-label speed .1 give
  `C=chi exp(-rt) Z0/G (.35-.1) f'(x-.1t) S`. The same residual sign is retained
  in the mature representation. Omitting the source leaves .0544635 relative
  error; refinement does not remove this constitutive discrepancy.
- **Rigid rotation:** v=(-(y-.5),x-.5), with rotated normal coordinate
  `y0=.5-sin(t)(x-.5)+cos(t)(y-.5)`. Native unforced non-objective transport
  carries `1000 chi(y0)` in a fixed xy tensor. The distinct objective fixture
  rotates S as well and explicitly supplies W A-A W as a prescribed source;
  its Jaumann material rate vanishes. Both converge at order3.00. Treating the
  unforced native solution as objective leaves .397339 relative discrepancy.
  That is a law distinction, not a justification to add production spin.

Direct mature and residual reconstructions approach the same characteristic
solution. Their mutual full-tensor L2 differences are237.883→28.9688→3.35062→
.409334 Pa m (normal), and29.9773→4.82282→.497160→.064245 Pa m (tangential).
These are convergence results, not roundoff identities. The prescribed continuum
commutator is integrated as a source; DG projection/trace differentiation does
not commute exactly with reconstruction. An algebraically exact discrete remap
would need that discrete commutator. No residual is dropped, and no exact
finite-resolution coupled transfer/work identity is claimed.

## Signed nonsmooth data and limiter decision

Transport a signed pulse with f=+1 on[.25,.5), -1 on[.5,.75), zero elsewhere,
using velocity(.4,0), tensor `(xx,yy,xy)=f*(1000,-400,700)` and zero inflow.
The exact jumps move .08 m. The independent checker splits cells at these
translated jumps before integrating squared error; Gauss24/48 L2 comparisons
pass1e-11. Extrema are native Gauss-Lobatto9 samples per coordinate/cell.

| Grid | Full-tensor relative L2 error | L1 error (Pa m²) | Sampled xx range (Pa) | Quadratic-norm loss |
|---|---:|---:|---:|---:|
| 4x8 | .38165 | 275.95 | [-1293.55,1529.16] |12.826% |
| 8x16 | .32255 | 201.90 | [-1292.11,1244.11] |10.408% |
| 16x32 | .21813 |109.64 | [-1283.31,1141.52] |5.940% |
| 32x64 | .17755 |67.34 | [-1158.22,1165.03] |3.574% |

L1 quadrature differences at24/48 points are .1563,.1260,.0998,.00542 Pa m²;
the table does not imply higher L1 precision. Quadratic loss is the measured
upwind-DG/SSPRK3 norm loss in this no-reaction control, not a full mechanical
energy balance. Small mean changes include numerical outflow tails; this is
an inflow/outflow Box, not a periodic mass-conservation experiment.

**Limiter decision:** keep it disabled for the selected resolved smooth controls.
The signed jump exhibits a significant localized oscillation problem for future
nonsmooth histories:16.50% overshoot and15.82% undershoot remain on32x64. It
warrants a targeted limiter comparison before nonsmooth coupled use. This task
establishes no measured limiter benefit or acceptable mechanical-noise threshold;
it does not qualify that use. No arbitrary component bound, clipping, historical
WENO port or chi filter is introduced. The historical WENO rank/constant-state
issues in the derivation remain unresolved, rather than bypassed.

## Verification, costs and reproduction

**284/284 final transport gates pass.** The preliminary254-gate family, initial
4/8-cell probes and all deliberate negative controls are retained. The final
extension adds the spatially varying rotating-profile controls required by V3;
no tolerance, source sign, physical parameter or expected result was changed.
No unexpected native failure occurred. Raw data and pinned binaries stay beneath
`build-particle-free/`; compact portable evidence is in
[evidence/pf4](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4/README.md).

| Final run | Wall time | Peak child RSS KiB |
|---|---:|---:|
| `pf4-final-4` |4.122s |214360 |
| `pf4-final-8` |5.273s |241216 |
| `pf4-final-16` |11.687s |308028 |
| `pf4-final-32` |40.393s |651508 |
| PF3 phase compatibility |6.176s |234568 |
| Frozen FE / DG compatibility |24.114 /23.665s |222600 /223360 |

Each run has a180s limit, serial/one thread, with at most two native simulations
overlapping. Costs include startup, multiple operator/source controls and compact
field export, not a production-step benchmark. Toolchain is the unchanged
GCC12.4/deal.II9.6.2/Trilinos14.2/OpenMPI5.0.10 Release stack.
Final core SHA256:
`964340e5af68fcff87ef065a151f5247fd343e4b2c1e88acb3376aae3ad0e30d`.
Exact observer/input/source identities are in the evidence provenance.

Compatibility:628 assertions/8 native cohesive-Maxwell cases and94/94 PF1
checks pass. The rebuilt PF3 assessment passes67 gates and is byte-identical to
its retained result. Final frozen FE bulk/surface CSVs remain byte-identical to
the fresh pre-PF2 source baseline; same-core FE/DG field/load discrepancies
remain3.195e-14/7.319e-13 scaled. The existing stress-poisoning/source/work and
publication checks still pass. Both affected DG production translation units
also compile independently without unity/PCH. Default parameters, established
Maxwell/cohesive/friction/H laws and accepted state layouts are unchanged.

From this worktree, use unused labels and output names:

```sh
cmake --build build-particle-free/native -j4
cmake --build build-particle-free/plugin -j2
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf4-repeat-4 --input benchmarks/reconstructed_fault/particle_free/transport_4.prm --plugin build-particle-free/plugin/libtransport_checks.release.so --postprocessor 'PF4 transport checks' --seconds 180
```

Repeat with transport_8/16/32.prm and distinct labels, then:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_transport.py build-particle-free/runs/pf4-repeat-4 build-particle-free/runs/pf4-repeat-8 build-particle-free/runs/pf4-repeat-16 build-particle-free/runs/pf4-repeat-32 --output build-particle-free/pf4/repeated-assessment.json
```

The portable reference archive reproduces the independent assessment without a
native executable. Evidence records archive replay, source correspondence,
preservation, new C++ style, syntax/link checks and whitespace checks.

## Remaining dependencies and next proposal

The advancement kernel reads its fixed operator, DG input, prescribed source/
inflow and mesh identity. It consumes no particle property or CPDI domain.
Operator construction still validates the existing fixed-history restrictions,
including particle count/positions. Startup H/CPDI/domains, surface material
projection, support/association and derived particle sampling still require
managers. No end-to-end particle-free initialization/trajectory follows.

Surface histories in the commutator controls are independent prescribed
characteristic functions. Native Q1 Z/theta transport, materially transported
phase lower bounds, accepted physical-step publication/rollback, coupled
mechanics/phase integration, mesh/geometry transfer, birth, restart and MPI
transport remain subsequent work. The co-moving/rotating templates do not
qualify native moving normalization. No new default, model change, material sweep,
propagation, full energy qualification or PF5 work is selected.

Proposed next task: a **bounded PF4 limiter comparison** on these saved signed
jump and smooth tensor controls, with a reviewed mean-preserving common-tensor
reconstruction. Verify zero/constants, affine/smooth accuracy, cell means,
3D tensor-basis covariance and degenerate weights before judging measured
overshoot/norm-loss benefit. Do not use arbitrary fixed stress bounds or limit
chi. Coupled use and PF5 persistence/transfer remain separate selections.
