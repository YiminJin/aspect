# Objective stress: selected convention and bounded qualification

30 October 2026. Entry source `27178813cebd471b93a7a9308f41215b71b4a20b`, branch
`pf-rsf-particle-free`, plus the source delta recorded with this report.
The default component-rate path and original production inputs remain controls.
This report separates source implementation, local verification and physical
comparison. There is no production-ready objective DG/cohesive trajectory.

The later [bounded-profile coupled comparison](BOUNDED_PROFILE_TRANSPORT_REPORT.md)
improves material-profile transport inside the restricted Stage C transaction.
It does not change the audited total Jaumann convention or promote production BP3.

## Scientific audit and selected scope

The manuscript's `particle_free_history.tex` eq.(jaumann) explicitly specifies
Jaumann with **total** W=skew grad v. It is not an unspecified corotational
symbol. The existing production mature implementation uses the non-objective
component rate, as the governing design/specification already document. The
user has selected a separate opt-in constitutive extension; no other objective
rate is substituted.

deal.II vector FE gradients store Lij=partial_j v_i. Thus W10=(L10−L01)/2 is
positive for counterclockwise rotation. The physical branch tensor is spatial,
symmetric, tension positive and full3D deviatoric in plane strain. Keep
zz=−xx−yy; pressure is an independent scalar multiplier. The branch's BP3 stress
is a **perturbation** about separately prescribed fault traction backgrounds.
Those backgrounds do not specify a complete bulk initial Cauchy tensor. This
feature does not rotate an invented bulk prestress.

The continuous candidate is

    J_W[tau] + r tau = 2G (D3(v) − chi V S),
    J_W[A] = D_t A + A W − W A,  r=G/eta,
    D_t = partial_t + v.grad,  S=sense sym(tangent tensor normal).

The native diffuse velocity contains the regularized slip gradient. Specifying
its symmetric slip-rate subtraction does not select an inelastic spin. For the
illustrative L_slip=chi V s tensor n, W_slip=chi V(s tensor n−n tensor s)/2;
the selected convention uses total W. No W−W_slip replacement, extra damping,
width floor or friction/solver tuning was introduced. Scalar V, age, accumulated
slip, damage and pressure receive no tensor commutators. Segment-frame
projection is applied once as a work/traction functional, independently of
material rotation.

The enabled native configuration is deliberately explicit:

    Material model / Phase field fault:
      Stress rotation = Jaumann lagged Lie
      Bulk history representation = particle stress
      Particle stress strain convention = three dimensional deviator
      Fault constitutive mode = mature frictional
      Evolve phase field = false

It requires fixed mesh, matched constant G and temperature-independent eta,
and the existing `BP3 stationary samples` integrator without replenishment.
Variable-coefficient energy extensions are rejected. The selector rejects DG/FE histories, bulk cohesive mode,
the separate cohesive RSF pilot, transport and AMR. This particle control
retains particle/CPDI dependencies; it is not a new particle-free qualification.

### Why production DG and the cohesive pilot are disabled

In production mature DG, T and Z are independent, while tau=T−B,
B=2chi Z S. At fixed geometry and without advection, the localization
commutator is still

    C[s_mem] = −chi s_mem [W,S],  s_mem=Z/G.

Rotating T but leaving that auxiliary correction unrotated would give
Qgeo=[W,B] in the physical mature stress law, which requires Qgeo=0. This is a
different constitutive equation, not a smaller integration error. The exact
separated extension requires

    tau=T−B−tau_res,
    J_W[tau_res] + r tau_res = [W,B]

along with the existing product/transport terms when geometry/profile move.
Zero initial residual is insufficient. Spatially varying rotations generate
narrow residual history that cannot be recovered by more quadrature without
independent history degrees of freedom. The current DG owner has no such state.

The newer cohesive pilot stores T_DG plus a complete native FE gradient
remainder and analytic lifting memory; it uses a backward-Euler Maxwell law.
Rotating only its DG coefficients would omit two physical tensor contributions.
Generic pointwise rotation does not preserve their finite FE representation,
and its structural-frame/BE energy/pressure adjoint needs a separate derivation.
The shared formulation table in [IMPLEMENTATION_MAPS](IMPLEMENTATION_MAPS.md)
records these distinctions. No unrotated auxiliary tensor hides behind the
new option, and no omitted residual is silently interpreted as cohesion.

## Implemented integration, residual and lifecycle

Freeze the accepted native gradient L− for each attempt, W−=skew L−. Use

    R=exp(dt W−),
    tau_rot=R tau− R^T,
    tau+=beta tau_rot+2 eta_ve D_branch+,
    beta=exp(−r dt),  eta_ve=−eta expm1(−r dt).

This is the composition of the frozen-spin rotation subflow, followed by the
existing exact frozen-source Maxwell subflow. It is a **first-order Lie split**
with endpoint source and accepted-gradient spin lag. The simultaneous exact
solution instead integrates rotated forcing over the whole interval; that is
not claimed here. The zero-spin implementation uses the original tensor
operands exactly; zero forcing gives exact frozen-W rotation/relaxation.
Advection is disabled in this configuration, not confused with tensor rotation.
Consequently the native stationary-sample experiment advances the local spatial
reaction equation `partial_t tau = forcing − r tau + [W,tau]`. It does not solve
the full material equation above when `v.grad(tau)` is nonzero. The local
constitutive/covariance tests qualify the spin/source operator; the fixed
Eulerian specimen is not a globally objective transported continuum model.

The new shared
[`rotate_plane_strain_stress`](../../../include/aspect/reconstructed_fault/objective_stress.h)
uses the2D exponential rotation matrix and symmetric congruence. It does not
relax stress, change its zz convention or limit its localization. Existing
ASPECT [elasticity](../../../source/material_model/rheology/elasticity.cc)
confirms gradient/spin signs but uses a first-order commutator in a different
history architecture; that implementation was not copied into reconstructed
histories. The [official ASPECT documentation](https://aspect-documentation.readthedocs.io/en/latest/doxygen/classaspect_1_1MaterialModel_1_1Viscoelastic.html)
is a convention cross-check, not proof of this branch's implementation.

Changed responsibilities:

| Source | Actual change |
|---|---|
| `phase_field_fault.cc/.h` | One default-legacy selector, admission restrictions and accepted-gradient point inputs. Existing Maxwell coefficients are unchanged. |
| `constitutive.cc` | `rotate_maxwell_history`, spin-aware frozen loads and point response/components. Missing gradient is an error for enabled evaluations. Legacy three-argument frozen-load API remains present. |
| `history.cc` / `history_diagnostics.cc` | Sample accepted gradient at fixed physical particles. Build candidates from the immutable incoming physical tensor. Diagnostic old/new values retain their meaning; spin mode additionally exports angle and rotated incoming tensor. Legacy CSV schemas remain unchanged. |
| `reconstructed_fault_stokes.cc` assembler | Sample W− at the native mechanical QPs and rotate the frozen physical tensor before beta multiplication. Current forcing and slip matrix stay unchanged. |
| `surface_system_{bulk_work,particle,diagnostics}.cc` | Matching accepted-gradient input for native residuals and diagnostic traces, with actual segment tensors and pressure coupling. |
| BP3 runtime `work_audit.cc` | Independent matching-frame observer uses the same spin convention; it still evaluates prepublication FE histories rather than advancing newly accepted particle samples again. |
| Solver `reconstructed_fault_stokes.cc` and `core.cc` | Extend the existing outer-acceptance transaction to spin-enabled particle histories. Rejected trials restore entry vectors/bookkeeping, publish nothing, and retries resample the same accepted anchor with their actual dt. |

W− is independent of the current nonlinear iterate. Consequently the old
current-velocity/slip/pressure Jacobian is the derivative of this declared
discrete residual; no converged current-spin outer iteration or consistent
Newton derivative of W(v+) is claimed. A future current-spin option would need
that additional derivative/iteration and must not reuse this label.

The checkpoint registry marker
`Jaumann total material spin lagged Lie v1` identifies spin, lag and integrator.
Old missing metadata means legacy. Restart checks its presence against the
selected law, so an old trajectory cannot silently become spin enabled. The
default checkpoint schema gets no new property. No production checkpoint was
converted or overwritten.

## Independent local verification

The tests in `unit_tests/phase_field_fault_maxwell.cc` use a componentwise
rotation/eigenbasis solution and a separate RK4 integration of the tensor ODE,
not the implementation's rotation-matrix operation. The RK4 reference uses
10000/20000 steps and agrees within1e−12. Predeclared tolerances follow double
roundoff for homogeneous identities and ratios1.9–2.1 for the first-order
manufactured cases; they were not adjusted to rescue failures.

| Test | Measured result |
|---|---|
| Zero spin, nonzero old stress and forcing | Exact equality with the existing exponential Maxwell helper. |
| Finite rigid rotation1.47 rad | Full tensor error below2e−14; trace, determinant and full3D norm preserved at declared roundoff bounds. |
| Rotation plus relaxation, lambda=23/3 s, t=2.1 s | Full tensor error4.64379e−15; storage plus integrated viscous loss closes to−3.55271e−15. |
| Simultaneous time-dependent spin/forcing, t=1.7 s | N=40/80/160/320: errors .091967/.0459574/.0229723/.0114845 Pa for the dimensioned synthetic G=3 Pa case; measured first order. |
| Superposed time-dependent observer Q(t), angle=.2 t² | Covariance errors .0758517/.0380792/.0190776/.00954823 Pa; first order, rather than a false exact-discrete-covariance claim. |
| Elastic simple shear through gamma=8 | Errors .0642172/.0321085/.0160542 Pa at400/800/1600 steps against tau_xy=G sin gamma, tau_xx=−tau_yy=G(1−cos gamma). Numerical convergence does not remove the hypoelastic oscillation. |
| Fault frame and normal pressure | Rotated tensor/tangent/normal retain shear and compression to4e−15 absolute in the synthetic fixture. This is a frame identity, not a moving-fault trajectory. |
| Fixed physical slip .5 m, widths1/.5/.25 m | Total-spin angles−.25/−.5/−1 rad agree with the analytical rotation. This exposes, rather than removes, diffuse-slip width dependence. |
| Forced Maxwell substep energy | Work .1031096406, viscous loss .02858992885, storage change .07451971175 in consistent synthetic units; closure8.32667e−17. Rotation has zero storage work. |

The first rebuilt cohesive-stack executable passes31 assertions in5 objective
cases and the complete182986-assertion/143-case suite. Exact commands/logs are
under `build-particle-free/objective-stress/`. The first MPI launch was blocked
by sandbox sockets; it did not run a physical step. Subsequent failed startups
preserve the missing old observer declaration and Voro++/CPDI dependency checks.
The existing BP3 build has the required Voro++0.4.6 installation and is used for
the native fixture; disabling particle domains is not a valid workaround.

The final Voro++ BP3 build passes **183073 assertions in148 cases**. Its last
change adds only the matched-constant-material admission guard; a fresh
three-step spin run reproduces the preceding spin run's physical outputs.
Native paired/lifecycle measurements are recorded in the evidence index below.
The native independent reference uses component rotation and direct Maxwell
coefficients with the actual converged strain/slip; its full3D recurrence bound
is2e−12 relative to max(1000 Pa,|tau|3). Rejection checks compare every particle
property and surface history byte for byte, including NaN provenance sentinels.
Same-rank restart and repartition comparisons use persistent physical fields;
transient diagnostic counters are reported separately.

### Native paired, residual, energy and lifecycle evidence

[Native results](../../../benchmarks/reconstructed_fault/particle_free/evidence/objective_stress/native-results.json)
and [exact output hashes](../../../benchmarks/reconstructed_fault/particle_free/evidence/objective_stress/native-output-manifest.json)
are reduced by `objective_stress/analyze_native.py`. The fixture derives from
`section7b-common.prm`: 2×1 km oblique truncated geometry, 59 surface nodes,
25568 stationary particles, uniform strengthening RSF, G=32038120320 Pa,
eta=1e26 Pa s and dt=200000 s. It retains CPDI, actual pressure, Helmholtz20,
all physical parameters/tolerances and independent matching-frame work checks.
This is a small stable BP3-like control, not the remote accelerating event.

| Comparison/test | Measured result |
|---|---|
| Protected legacy binary versus rebuilt legacy option, three steps | Full3D tensor RMS4.48702e−10 Pa, relative L2=1.99173e−14; max component difference2.95404e−9 Pa. Max normal-traction difference3.72529e−8 Pa; V relative L2=4.21936e−15. This comparison is to a distinctly older binary, **not bitwise equality to HEAD**; intervening coefficient/source repairs remain part of the baseline difference. |
| Spin off/on, same current source and inputs, three steps | Full3D tensor RMS=.123827731 Pa, relative L2=5.49656e−6, max component=.978869473 Pa. Max shear/normal differences .0599524/.0639509 Pa. V relative L2=3.89746e−9 (max2.11578e−17 m/s); age relative L2=1.37276e−10 (max .00622380 s); added-slip difference max6.35781e−12 m. |
| Native independent particle recurrence | Across positive cases, max error3.13458e−10 Pa; relative≤1.32898e−14, versus predeclared2e−12 bound. It includes zz and actual localized slip forcing. |
| Residual and tangent audit | Native affine directional tests include pressure rows and a nonzero V probe. Frozen-load changes are exactly zero; max affine/action discrepancies4.90970e−7/5.52814e−8 in native load units, within the existing operand-scaled checks. The new explicitly lagged source has no derivative with respect to current v/V/p. |
| Completed-trial outer rejection | Step2 rejects before publication, retry dt100000 s. All particle properties and surface histories remain byte-identical to the accepted parent. The matched control uses accepted intervals200000/100000/191000 s (the existing growth cap remains active). Final tensor, V, age, slip and weak-traction arrays match exactly; publication count is one. |
| Same-rank native restart | Step3 checkpoint → step4 on two ranks equals uninterrupted step4 exactly in tensor, V, age, slip and weak tractions. No repeated rotation on restore. |
| 2→4 repartition | Tensor RMS2.98281e−10 Pa, relative L2=9.92467e−15; max component1.74623e−9 Pa. V relative L2=5.62861e−15; normal max7.45058e−8 Pa. Existing distributed particle/mesh checkpoint transfer is retained. |
| Negative configuration/restart | DG, cohesive mode, transport and unmatched G reject explicitly. A legacy-control checkpoint cannot become spin enabled: the existing BP3 material-configuration fingerprint rejects it before the core's additional registry-marker check. No physical advancement occurs. |

The branch-energy diagnostic uses full native-QP reconstructed physical stress
and a separate six-point temporal quadrature of the frozen-source Maxwell
solution. Its largest closure over all native controls is1.70530e−13 J/m,
at most .001107 of the predeclared floating-point bound. At step3 spin-on:
old/rotated/new storage=112.835150855/112.835150855/485.993121571 J/m,
source work373.157970750 J/m and viscous loss3.34930e−8 J/m.
Relative to spin off, final reconstructed storage changes5.61658e−5 J/m;
endpoint total work changes5.06017e−4 J/m. The particle square-sum diagnostic
is explicitly unweighted Pa² and is not substituted for bulk storage. The
constitutive substep ledger does **not** establish a complete external/background
loading, friction, radiation and fault energy balance; endpoint and integrated
source work are recorded separately.

Native per-step rotation is≤5.20364e−6 rad in the three-step comparison, with
sum of per-step maxima1.55366e−5 rad (an upper bound on accumulated rotation).
Max |tau|3/G=6.91048e−6. Max |W|scalar lambda≈8.121e4 reflects the extremely
long Maxwell time; the actual short deformation/stress history is also reported,
rather than declaring validity from one dimensionless number. The large-shear
unit test separately exposes the selected hypoelastic limit.
From the saved physical recurrence and rotation's isometry, triangle inequalities
bound the maximum endpoint branch-rate norm between5.85584e−12 and2.86966e−11 s−1,
and its product with lambda between1.82777e4 and8.95700e4. The integrated branch-
rate norm is bounded above by1.02671e−5 over these three steps. These are explicit
bounds, not an invented exact per-point rate/accumulated-strain measurement.

Retained/current legacy three-step controls cost17.15/16.60 s; the spin run with
extra residual auditing costs19.45 s and the four-step spin control19.70 s.
Restart/repartition segments cost7.88/5.38 s; rejection/matched controls18.15/15.10 s.
These include initialization, CPDI and evidence exports and are not isolated
spin-kernel performance measurements. Cases run sequentially, at most four ranks;
core builds use at most four workers. Full commands/inputs/hashes remain in
each `build-particle-free/objective-stress/runs/*/command.json`.

Failed startup evidence remains in `spin-on`, `spin-on-local`, `spin-on-v2` and
`spin-on-v3`. The first matrix assertion expected the later core-marker diagnostic,
but the earlier existing BP3 fingerprint correctly rejected the changed law;
`native-matrix.log` and the continued/final logs preserve that distinction.
Restart summary CSVs are append fragments without a new header; the independent
reader supplies the declared parent schema while retaining the original bytes.

## Interpretation and next experiment

This implements a scoped objective **branch** law. It does not establish
finite-strain hyperelasticity, objectivity of unspecified background bulk
prestress, a moving-fault structural law, or native transport. Those combinations
remain disabled; the prescribed PF4 transport experiment is not relabeled as
qualified by homogeneous rotation tests.

The [Schrank et al. paper](https://doi.org/10.1093/gji/ggx297) compares different
Maxwell stress-rate/strain models at large deformation; the
[MOOSE discussion](https://mooseframework.inl.gov/modules/solid_mechanics/ObjectiveStressRates.html)
likewise distinguishes rate choices and hypoelastic shear limitations. They
motivate measuring branch elastic stress/G, rate times Maxwell time, accumulated
shear and rotation per step. They do not supply a universal admissible total
strain such as .1. Timestep convergence addresses numerical error, not the
constitutive oscillation of the Jaumann hypoelastic model.

The new BP3 export remains a legacy accelerating first-event prefix. Its
successful observer checks and its failed accuracy evidence retain their scope;
the spin feature neither invalidates all old results nor makes them objective.
A new production spin comparison is **not yet justified** on the current DG
owner. First compare direct physical stress to T−B−tau_res in one fixed,
resolved normal-profile fixture with spatially varying W, matched forcing and
pressure/work adjoint. Measure full tensor, shear/normal traction, storage and
projection error under dt and normal-resolution refinement. This will determine
the residual history space needed at BP3 width before native DG implementation
or any short production sensitivity branch.

The controlled-crust example remains a design: finalize material/loading below
40 km, background stress, fixed geometry, normal-pressure convention and run
duration relative to Maxwell times before elastic/layered-viscosity comparisons.
The separate propagation example stays at its protected four-rank step748,
t=374.58003334330067 s, with an incomplete RSF episode and failed fracture/spatial
screens. No new crust, propagation, realistic subduction or server campaign was
launched. The cleanup record's historical field losses remain unchanged.


## Stage C native extension (3 November 2026)

[The Stage C report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md) extends this
historical particle-only qualification with matched mature DGQ2 T/R reaction
history and a distinct cohesive piece/side Q2 reaction option. The original
particle selector and default legacy path keep their original scope. Complete
native/lifting rotation, actual load/tangent/observer consistency, accepted
rejection/restart and restricted T/R AMR have local evidence. Distributed
advection/layered-relaxation operators are verified separately. No native
transported geometry/profile/surface configuration is enabled or qualified.
The uncut cohesive residual space fails and is explicitly disabled; physical
phase-energy and tip-admission screens remain failed. These results do not
upgrade production BP3 or the retained propagation checkpoint.
