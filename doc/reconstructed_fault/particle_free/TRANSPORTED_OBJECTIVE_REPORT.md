# Transported objective histories — staged record

3 November 2026. **Stage C is active and incomplete.** The
[native integration report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md) records
implemented reaction/history paths, distributed transport, cut cohesive state,
AMR/restart/rejection, numerical passes and failed physical screens. Full native
moving geometry/profile/surface transport remains to integrate.

Stage B is complete within its bounded representation scope.
The [fixed-profile report](TRANSPORTED_OBJECTIVE_STAGE_B_FIXED_REPORT.md) and
[moving/layered report](TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md) record
independent direct and complete separated histories, time/space/quadrature
refinements and representation costs. Selected for the first native mature
integration: T DGQ2 + complete R DGQ2 + auxiliary surface memory. Direct physical
stress remains the independent regression reference. No native selector or
checkpoint is changed by Stage B; integration and lifecycle are Stage C.
The completed Stage A derivation and algebra evidence below remain valid.

## Plan and gates

| Gate | Implementation decision / required evidence | Status/evidence |
|---|---|---|
| A1 Physical equation | Material transport, total-spin Jaumann, full3D deviator, pressure/background split | Derived in SPEC §1 |
| A2 Surface motion | Common normal velocity; symmetric interface labels; independent tangential node velocity and ALE scalar transport | Derived in SPEC §2; single-state closure explicit |
| A3 Geometry/maps | Tangent/normal/curvature, profile/normalization, projection mass solve, adjoint and measures | Derived in SPEC §3 |
| A4 Irreversibility | Transport accepted obstacles/material identities before growth; distinguish three freezes | Specified in SPEC §4 |
| A5 Complete transformation | All extension/product/transport/spin and local/projected-rate terms | Derived in SPEC §5; algebra checks below |
| A6 Layered eta | Evaluate along bulk trajectories; convergent frozen-endpoint coefficient rule | Specified in SPEC §6 |
| A7 Prestress/loading | Preserve imposed tractions; no invented bulk prestress; external work separated | Specified in SPEC §7 |
| A8 Mature/cohesive compatibility | Preserve native remainder/lifting; distinguish geometric energy change from numerical remapping | SPEC §8; deformation/growth force gate explicit |
| B1 Fixed-profile representation | Independent direct/separated evolution; oblique Cartesian Q2, nonzero slip/zz, varying spin; time/space/quadrature separated | **Complete for selected q8**; [measured report](TRANSPORTED_OBJECTIVE_STAGE_B_FIXED_REPORT.md), original q6 rejection retained |
| B2 Moving profile / layered eta | Eulerian DGQ2 transport, material shear/translation, profile/orientation, local/projected-rate terms | **Complete**;20 numerical +8 consolidation/profile/selection checks pass; [measured report](TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md) |
| C Native integration | Complete stress/source/tangent, surface transport, moving maps, accepted transaction, versioned restart | **Partial**: native reaction/history and transport operators locally verified; moving complete-state transaction and coupled heterogeneous reaction pending; [report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md) |
| D Integrated verification | Moving/deforming fault, layered eta, unstable RSF, growth, lifecycle and required AMR | Native component/lifecycle and short phase probes run during C; full integrated moving/physical qualification remains pending, with explicit failures |
| E Server packages | Pinned configurations and build/runtime/restart qualification | **Not prepared; depends on B–D** |

Execution order after this design: representation comparison first, then one
fixed-mesh transported mature path, then cohesive/geometry coupling, then
required distributed transfer. Existing PF4 transport and local spin controls
are reusable components, not an integrated qualification. At most four local
build workers/ranks and sequential native runs remain the task limits.

## Entry and preservation

Root `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `27178813cebd471b93a7a9308f41215b71b4a20b` plus
existing dirty source, documentation and manual outputs. No reset/checkout.
[Entry evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_a/entry.json)
records 921 source/header/unit-test hashes, full porcelain status, protected
binary hashes and all files in the three retained RSF restart slots.
The active marker remains2; the matching accepted diagnostics are step748,
t=374.58003334330067 s. No newer local accepted continuation was found in that
run. BP3 accepted4237/latest profile4182 and the accelerating incomplete event
remain unchanged. Retain [cleanup losses](CLEANUP_REPORT.md#legacy-animation-data-loss)
and failed fracture/spatial evidence. No native binary is assumed to match HEAD.

## Decisions and limits

The [specification](TRANSPORTED_OBJECTIVE_SPEC.md) is the derivation for this
extension. The [existing maps](IMPLEMENTATION_MAPS.md) remain the single owner
and formulation inventory and now distinguish current storage from required
transported storage. No existing selector/admission guard changes in Stage A.

The comparison now evolves independent direct physical stress and complete
separated T−B−R under fixed and transported/deforming profiles and layered eta.
Both converge. Measured cost/accuracy selects a complete DGQ2 R beside the
existing DGQ2 T for the first native mature path. Non-affine native remainders
and the separate cohesive lifting remain protected; their native qualification
is not inferred from the affine Stage B coefficient fixture.
A convergent complete direct tensor is the fallback if a separated residual
requires impractical normal resolution. Auxiliary scalar memory is never
relabelled as cohesive energy or discarded at zero resistance.

Cohesive material motion and physical damage/growth require separate operations.
A numerical relabelling must preserve the reconstructed tensor/storage; a
physical localization change uses the existing energy derivative. These cannot
both be implemented as “rebuild geometry and keep all coefficients”. The
smallest counterexamples, including the surface-measure work required during
stretch, are in SPEC §8. Current fixed-geometry source has no such shape-force
coupling. That missing coupling blocks enabling deformed cohesive evolution,
not the independent mature representation comparison. No reference-arclength
fracture model or capillary force is silently introduced.

## Verification

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/check_stage_a.py
```

The checker uses Python3/NumPy, fixed seed20261102 and independent component
Leibniz/coordinate equations, not the production reconstruction or rotation
routine. It performs no native mechanics, profile PDE or Stage B comparison.
[Measured JSON](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_a/algebra.json).

| Stage A audit | Declared criterion | Measured result |
|---|---|---|
| General mature source with transport/profile/orientation and bulk/surface rate mismatch | Exact rational component equality including zz, plus spin power zero | 512 cases; defect0 |
| Positive native lifting plus complete tensor correction and noncommuting surface rate matrix | Exact rational component equality | 128 cases; defect0 |
| Material normal / curvature / association rates | Independent centered coordinate difference, step1e−6; error<2e−9 (roundoff/difference budget) | 1.28e−10 /1.06e−10 /5.49e−11 |
| Surface relabelling and projected mass derivative | Exact rational identity; omission must produce nonzero defect | Pass; omitted ALE source14/15 in synthetic units |
| Variable-rate relaxation, r(t)=.4+.8t, duration1 | N20/40/80; endpoint-coefficient errors decrease with ratios1.9–2.1 | .00889731/.00447090/.00224104; ratios1.99005/1.99501 |
| Counterexamples detect omissions | Nonzero omitted normal/spin, coefficient-gradient, old-cell damage, hidden prestress and stretch work | .85 normal-rate defect; −7/4 lifting-rate defect; unit damage trail;1.69706 stress-rate difference;2.4J/(m s) missing shape power |

These are dimensional synthetic algebra fixtures (G=7/3Pa for component
identities), not inferred Earth parameters. Coefficient verification uses an
exact scalar variable-rate solution; no tensor-transport convergence claim is
borrowed from it. The negative controls are expected failures of the deliberately
incomplete equations, retained in the JSON; no threshold was relaxed.
No full simulator suite/build was rerun because no native source changed.
Previous native spin and PF4 evidence retain their stated scope.

## Acceptance requirements for the next implementation

Before selecting a native representation, Stage B needs the oblique normal
profile with nonzero slip/spin/zz, independent direct tensor, and three levels
**separately** for time, normal spatial resolution and quadrature. Report
absolute and scaled full-tensor, normal/shear traction and storage errors.
Use a nonzero prescribed tensor/traction scale, not a near-zero result as a
denominator. Preserve local identity targets1e−12 and assembled work-adjoint
1e−10 from VERIFICATION.md; first-order convergence applies to the chosen
lagged split only once spatial/reference errors are below the temporal error.
A representation decision requires measured error-versus-cost at the intended
width, not just a formal asymptotic order. Native numerical thresholds and
step/output budgets must be pinned in that fixture before its runs; Stage A
algebra thresholds cannot serve as its PDE tolerances.

For later integrated lifecycle/energy gates: rejected motion must restore all
accepted geometry/history/obstacles; restart/repartition and required AMR must
preserve the complete physical tensor with measured projection error. Maintain
the existing native KKT/contact/work limits and report independent physical
remainders against both sliding and new fracture work. The retained1% fracture
screen remains failed for the old RSF episode (20.8% observed); do not relax it
or count BE numerical terms as physical accuracy. These gates remain pending.

| Configuration | Current enabled state / server readiness | Exact next requirement |
|---|---|---|
| Legacy BP3-derived | Existing legacy control retained; strict_reference_match_passed=false; first event incomplete | Review later data if supplied; no new trajectory selected |
| Stationary particle spin | Prior local qualification retained, particle-only and constant eta | Not the required particle-free production path |
| Transported mature particle-free, constant or layered eta | Disabled / not server-ready | Stage C native residual/tangent and history/lifecycle integration |
| Translated layered crust | Disabled / not server-ready | Native material profile/ALE state, fluxes, compatible boundaries and lifecycle qualification |
| Solution-driven deformed cohesive/growth | Disabled / not server-ready | Complete native/lifting tensor evolution and the shape/pressure/measure-work coupling (SPEC §8), then independent growth/lifecycle/error checks |

Stage A is complete as a design and algebra audit, with the deformed-cohesive
force gate isolated. No formerly unsupported configuration is newly enabled.
Physical additions are the explicitly selected transported total-spin law and
single-state interface-label convention; numerical choices are first-order
lagging, candidate complete tensor representations and transaction/transfer
rules. They are not installed in the solver by this stage.

Next concrete action: Stage C native integration of the selected complete DGQ2
residual with the mature T/s state, native source/tangent/observer and one
accepted geometry/history transaction. Keep the direct tensor as an independent
reference; retain old checkpoint rejection and default controls. No Stage B
fixture ran a native timestep, server job, long physical trajectory or checkpoint
conversion.
