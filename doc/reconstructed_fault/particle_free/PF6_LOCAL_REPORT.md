# Gauss12 and bounded fixed-surface local-transfer results

16 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved
incoming work. **Gauss12 passes the retained-state and short-trajectory controls.
The selected 64→67-cell frozen remap, hanging-node evaluation and native restart
pass at 1/2/4 ranks.** This is a small manual transfer qualification, with a fixed
surface and particle population. Coupled phase evolution on hanging-node meshes
remains guarded. Automatic AMR is not enabled.

[Pre-run rules and documented amendments](PF6_LOCAL_FORMULATION.md),
[source audit](SOURCE_AUDIT.md), [particle dependencies](PF6_PARTICLE_READ_AUDIT.md),
[inputs, measurements and portable replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_local/README.md).
Unlimited DGQ2, constitutive/rate conventions, limiter settings, physical
acceptance rules and production parameter defaults remain unchanged.

The accepted [uniform-transfer evidence](PF6_TRANSFER_REPORT.md) is retained.
Its original native Gauss6 screen remains failed, as does the earlier
**`strict_reference_match_passed=false`**. Every original error, branch comparison,
failed run, checkpoint and reference is preserved. This study does not revisit
the long 512-normal-cell trajectory or its mirror-branch discrepancy.

## Physical quadrature and normalization

The retained worst state is step 2 of the 4x16-cell short fixture, with nonzero
T/Z and the accepted Q1 phase. Restore it and evaluate native Gauss6/12/24/32
normal rules at identical fields, clocks and surface histories, without a trial
or publication. Tangential Gauss3 panels retain the surface-knot partition.
Independent physical integration uses Gauss24/32 in both coordinates. The
Gauss3 DG mass/source rule is unchanged.

The retained equations reconstruct tau=T−2*chi*Z*S, chi=h(phi)/I,
h=m*phi*(1+phi)/(1−phi)^2 and q=Z/I, with S_xy=one-half times the shear sense.
Bulk/cohesive energies are the integrals of |tau|²/(4G) and Z²/(2GI); the bulk
fault functional is the integral of chi*(tau:S)*N_i. Tensor comparisons use
xx²+yy²+(xx+yy)²+2xy². Only physical steps apply the retained exponential Maxwell
update T_e=beta*T_n+2*kappa*D3(symgrad u), Z_e=beta*Z_n+kappa*V; remaps apply none.

Normalization has a different owner: native analytic integration of h(phi)
along each affine-Q1 normal interval, followed by the consistent surface mass
projection. Independently integrate h(phi) and assemble/solve that surface mass
with Gauss24/32. Accumulate reference integrals in extended precision so summation
does not obscure analytic-integration accuracy. Also compare the projected I(x)
with unprojected normal-ray integrals. The retained normal-profile fixture has
negligible surface representation error; this does not qualify arbitrary 2D
phase dependence.

Screens declared before runs remain scaled **1e-9 for energy/weak loads** and
**1e-11 for normalization**, dividing by max(1, reference magnitude), with units
retained in the raw data. No screen is relaxed.

| Retained worst state | Gauss6 | Gauss12 |
|---|---:|---:|
| Bulk energy error, J/m | −1.52022e-8 | 6.66e-16 |
| Maximum weak-shear error, Pa m | 5.90193e-7 | 2.84e-13 |
| Scaled energy / load error | 1.52e-8 / 4.39e-9 | 6.66e-16 / 2.12e-15 |
| Native–reference I discrepancy, m | 5.68e-14 | 5.68e-14 |

Gauss24/32 uncertainty is 2.22e-16 J/m and 2.56e-13 Pa m at that state. I's
scaled discrepancy is 6.19e-16; its reference uncertainty is 4.26e-14 m.
Surface-projection Linf discrepancy is 4.26e-14 m. Replacing native I by the
independent I changes bulk energy by 1.67e-15 J/m and weak loads by at most
1.71e-12 Pa m. Thus the Gauss6 defect is bulk integration error; normalization
differences are at the quantified arithmetic floor. Gauss24/32 native rules
also pass, without changing normalization.

Repeat the same two physical steps, dt=.125 s through t=.25 s, from the existing
physical seed prescription and unloaded T/Z. Mesh, loading, coefficients, phase
irreversibility and solvers match the retained fixture. Native Gauss12/24/32
trajectories give maximum common-rule stress differences of **1.23e-14 / 1.65e-14
scaled L2** for 12-vs-24 / 12-vs-32; maximum phase differences are
**8.40e-16 / 4.65e-16**. Snapshot integration checks also pass throughout.
This isolates quadrature uncertainty at fixed timestep; it is not a new temporal
or spatial convergence study.

The final physical steps remain on 64 cells. The intervening 64→256→64 identity
diagnostic in the original protocol is omitted explicitly because fresh particle
domains expose a genuine support failure there, described below. Its original
outputs and errors remain. The original G12 trajectory is compared separately;
that comparison includes both the omitted identity operation and cache repairs,
not an isolated attribution to either. Its maximum stress/phase differences are
1.18e-14 scaled L2 / 6.46e-16 Linf. The original transfer observer's deep-test
default remains true; this bounded runner explicitly selects false.

## Demonstrated source corrections

Native distributed SolutionTransfer remains the tensor remap. Its three DGQ2
fields retain all nine coefficients each, including x²y². Coarsening still uses
the target-cell L2 projection. Only complete coarsening families are admitted,
and only when the child physical phase/next lower obstacle equals its parent-Q1
interpolant within the retained **256 epsilon** screen. The selected +.01 fine
peak rejects before particle preparation and mesh mutation.

Four implementation issues were exposed and corrected without changing laws:

1. **Local geometry and quadrature.** Accepted caches admit affine Cartesian
   coarse/fine cells and convex homogeneous Q1 hanging constraints. The fixed
   surface knots must remain quadrature panel boundaries. On local meshes use
   a conservative coarse-grid tangential subdivision, identical on every rank,
   with no collective inside cell assembly. Uniform meshes retain their original
   rule. All physical energy, mechanical and weak-load consumers share it.
2. **Canceled refinement.** Default island smoothing cancels the isolated tag.
   Preparing DG/particle transfers before flag finalization left a stale prepared
   owner on the unchanged mesh. Finalize flags first on this selected path, then
   check phase admissibility and prepare data only if the mesh will change.
   Two repeated no-ops preserve histories/RNG/clocks and leave no pending owner.
   Legacy hook order remains unchanged.
3. **Particle-domain freshness and closed boundaries.** The four-rank diagnostic
   changed 81 old slot owners while domain generation stayed 7: 20 sampled slots
   were zero and seven were outside old storage. IDs/positions were unchanged.
   Rebuild selected DG domains after unpacking, bounds handling and final ghost
   exchange, before association consumers. A separate Voro++ 0.4.6 control with
   a closed rectangular candidate patch retains only 9/16 points in the original
   container; its target area is .015625 rather than the independent 1/144 m².
   This is a representative library control, not a replay of every native
   neighbor-list operand. For artificial upper wall b and lower a, use
   delta=gamma_8(|a|+|b|+|b−a|), gamma_k=k*u/(1−k*u), u=epsilon/2, when a candidate
   lies in that bin-roundoff band, followed by an outward nextafter. Four rounded
   reciprocal/bin-map operations give gamma_4; the conservative operand bound
   also covers endpoint construction. The control then retains 16/16 and recovers
   1/144. Every native intended candidate must be resident. Particle coordinates,
   physical-box walls, positive-volume and physical-coverage guards are unchanged.
   The adapter is restricted to nonperiodic 2D.
4. **Completed-checkpoint stopping.** The native do-while attempted one extra
   physical step after loading a completed endpoint. Honor the existing
   termination manager before that loop on restored accepted DG endpoints.
   Restore-only controls now start zero physical steps and publish zero times.
   A same-build uniform half-run/continued restart still advances normally and
   matches uninterrupted step 2 exactly, including all exported histories/RNG.

The old failed attempts, including the test harness's wrapped-message recognition
error and restore attempts stopped by the hanging-phase guard, remain archived.
Earlier fixed-state quadrature observations were made before their unintended
extra uniform step and remain valid; those processes are not counted as passing
restore-only lifecycle checks. The Maxwell source/projection, publication
allowance/validator and matching-CellId trace convention are unchanged.

## Frozen local remap and MPI/restart

The selected parent is CellId **1_2:30**, x=(.5,.75), y=(.375,.4375). Its retained
upper-face particles belong to that surrounding CellId. One isotropic refinement
gives **64→67 cells**, four physical phase hanging nodes and eight coarse/fine
face pairs. After testing default smoothing's no-ops, the diagnostic explicitly
keeps deal.II vertex-level balancing alone to retain this isolated cell. Physical
steps use the original smoothing defaults. Restart reinstates that diagnostic
policy; no automatic AMR or hanging-node phase solve is enabled.

Measure before/refined/coarsened snapshots at common physical points and
independent quadrature, with no intervening physical update. Test non-proportional
global complete-Q2 tensors, actual accepted nonzero-Z history, every native time
level and physical lower obstacle. Verify native Q1/Q2 traces independently
across coarse/fine faces. Preserve Z/I, fixed eight-cell/nine-vertex surface,
committed rates and all properties, accepted clocks and publication counts.

| Frozen transfer, all ranks | Measured maximum |
|---|---:|
| Physical stress scaled L2 | 6.94e-16 |
| Accepted T coefficient roundtrip | 1.46e-11 Pa, 2.67e-16 scaled |
| Target-cell T mean defect | 3.21e-12 Pa, 5.87e-17 scaled |
| Phase, including native time levels | 1.12e-16 |
| Phase-gradient difference | 2.23e-15 |
| Particle-domain volume / bounding-coordinate error | 2.69e-17 m² / 2.23e-16 m |

Maximum common-rule energy/weak-load defects are **1.12e-16 J/m / 5.69e-14 Pa m**;
raw signed quadratic-norm changes,
moments, full-tensor errors and costs are retained. The physical phase is unchanged,
so none is reported as physical evolution or dissipation. The native frozen-load
and high-rule quadrature checks pass after remap. This evaluates identical fields;
it does not measure enriched mechanical approximation error by re-solving them.

Each full 1/2/4-rank case passes **107 native gates**, including actual uniform
outer rejection/retry, terminal source rejection, once-only physical publication,
local no-ops, inadmissible coarsening, fixed particles and guarded unsupported
hanging-node phase evolution. All 144 IDs and physical coordinate strings stay
identical. RK2's effective advection dt is zero; constitutive dt stays .125 s.
Every exported domain is independently checked against the global midpoint-
bisector rectangles of the unchanged 6x24 particle set. Repartition changes three
particle owner ranks in each multi-rank local test; four surrounding CellIds
change under refinement. Local slot changes are not particle motion.

| Comparison | Scaled stress L2 | Complete native fields, scaled max |
|---|---:|---:|
| 1 vs 2 ranks | 1.17e-14 | 2.84e-14 |
| 1 vs 4 ranks | 1.03e-14 | 2.17e-14 |
| Same-count local restart, 1/2/4 | 0 | 0 |
| Serial local checkpoint restored on 4, vs saved serial state | 0 | 0 |

All persistent T coefficients, phase/obstacle/current/old/older/anchor fields,
surface histories, particle properties/coordinates, reconstructed stress and weak
loads compare. Same-count RNG bytes replay exactly. **Changed-rank RNG replay is
unavailable under the retained native policy**: population changes are disabled,
and streams match the fresh four-rank layout, not copies of one serialized serial
stream. Cross-rank deterministic fixed-population restart passes; stochastic
replenishment/birth replay is not qualified.

Full instrumented local runs take **14.54 / 39.19 / 51.32 s** at 1/2/4 ranks;
restore-only runs take approximately two seconds. Each native case has a 180 s
cap, one thread/rank, at most four ranks, sequential jobs. The tiny replicated
endpoint fixture demonstrates no parallel speedup. Local physical quadrature has
19,296 points versus 4,608 on uniform 64 cells, owing to conservative fixed
panel subdivision. At 67 cells: velocity/pressure/phase DoFs are **634/90/90**,
independent tensor DoFs **1809**; native FE total **2399** excludes that separate
tensor space. Surface DoFs remain nine.

The matched serial Gauss12/24/32 trajectories take **12.79 / 16.89 / 20.10 s**
with **4,608 / 9,216 / 12,288** physical bulk points. These include repeated
rejection diagnostics and exports. Gauss12 gives the measured integration
accuracy at lower cost than the two high-rule controls.

The final core also reproduces all **21 frozen-FE control CSVs byte-for-byte**.
The shared nonperiodic 2D closed-box adapter has this focused regression and the
independent/library/native geometry checks above; broader advected or legacy-AMR
inputs are not qualified by this study.

## Remaining limits and next task

Fresh domains on the **256-total-cell (8x32)** diagnostic still fail the original
coverage guard after closed insertion is repaired: particle 3 at
(.25,.020833333333333332), CellId 0_3:001, has patch (0,0)–(.375,.0625) with four
intended/resident candidates. The nearest right particle at x=.4166666666666667
is outside it. This is genuine one-layer support insufficiency, distinct from
stale indices and dropped upper-face candidates. No new neighbors, particles,
volumes or relaxed coverage test are supplied. This failure occurs after mesh
mutation and is fatal; recovery from arbitrary postmutation AMR failures is not
qualified. It concerns the retained sparse 144-particle fixture, not a universal
normal-resolution limit or the earlier 512-normal-cell experiment.

Recommend **Gauss12 for this experimental fixture**, with the production default
unchanged. Retain the measured manual local-transfer/restart subset. Before wider
or automatic refinement, select removal of the remaining particle-domain/material
projection dependency through independently checked FE/cell/surface quadrature,
or a separately qualified neighborhood-completion rule and preflight/recovery
contract. Keep the independent publication check while replacing its consumers.

Before coupled hanging-node phase evolution, specify phi=C*psi on independent
masters, the reduced gradient C^T*g and tangent C^T*H*C, and physical obstacle/KKT
acceptance; verify those operators and the actual outer lifecycle. The current
guard prevents treating redundant constrained nodal coordinates as independent.
No one-pass phase-first update is adopted. Adaptive quadrature, genuinely lossy
obstacle transfer, surface remeshing, transport, moving geometry, birth,
propagation and general PF5 completion remain subsequent work.
