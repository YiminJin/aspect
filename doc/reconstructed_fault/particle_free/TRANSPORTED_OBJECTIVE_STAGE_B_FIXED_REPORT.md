# Stage B first part: resolved fixed-profile stress representation

2 November 2026. The selected first part is complete: a particle-free Cartesian
DGQ2 comparison of independently evolved direct physical stress and T−B−R,
with a fixed oblique profile, nonzero localized slip and spatially varying
total spin. **This is the reaction/history-space experiment, not the remaining
transport/heterogeneous-eta part or native solver integration.** Legacy source,
production inputs/binaries/checkpoints and their guards are unchanged.

[Figure PNG](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_fixed/comparison.png) ·
[vector PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_fixed/comparison.pdf) ·
[results/acceptance JSON](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_fixed/results.json) ·
[evidence index](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_fixed/README.md).

## What was implemented and compared

The [Python fixture](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/stage_b_fixed.py)
implements nine scalar Q2 modes on **each Cartesian cell**, not a normal-only
polynomial surrogate. The oblique analytic profile cuts these cells. Its modal
basis spans the same tensor-product polynomial space as FE_DGQ(2) on affine
Cartesian cells; it does not instantiate ASPECT's native mixed solver. Native
FE-source, pressure/work-adjoint, non-affine remainder and distributed-history
qualification are still required later.

For fixed geometry/profile and no advection, constant r=G/eta:

    tau_dot = [W,tau]−r tau+2G(D−chi V S),
    T_dot = [W,T]−r T+2G D,
    s_dot = V−r s,   B=2G chi s S,
    R_dot = [W,R]−r R+[W,B],   tau_sep=T−B−R.

R is an independently evolved tensor history, initially zero. Its forcing is
projected from [W,analytic B], not inferred from the direct solution and not
replaced by W times a preprojected B. Initial T equals the prescribed physical
tensor. No cohesive energy/resistance is assigned to s or R.

The continuous direct reference solves component equations using
u=(tau_xx−tau_yy)/2+i tau_xy and m=(tau_xx+tau_yy)/2:

    u_dot=(−r+2i omega)u+2G(F_xx−F_yy)/2+2iG F_xy,
    m_dot=−r m,    tau_zz=−2m.

Sinusoidal forcing is integrated analytically. Independent real-component RK4
at2048/4096 steps checks the exact direct reference. A separate integral of
R's **own source** checks the continuous reconstruction; R is never defined by
subtracting direct stress from T−B. This verifies the positive commutator sign
and the factor2G beyond the Stage A local algebra.

For spatial studies the symmetric projected spin multiplication matrix is
diagonalized in the orthonormal Q2 basis. Each eigenmode's linear relaxation/
forcing is integrated exactly in time, including the evolving scalar s(t).
This separates spatial capacity from timestep error. The independent direct
physical DG evolution and T/R evolution share the declared projection space,
but all are checked against the separate unprojected component solution.

## Inputs, bounds and physical limits

The [protocol](../../../benchmarks/reconstructed_fault/particle_free/transported_objective/STAGE_B_FIXED_PROTOCOL.md)
was recorded before runs; its appended follow-up records the quadrature failure
and the unchanged-bound response before that probe. Worktree HEAD is
`27178813cebd471b93a7a9308f41215b71b4a20b`, branch `pf-rsf-particle-free`, with
retained dirty source. It was not reset or assumed to identify retained binaries.

| Quantity | Prescribed value |
|---|---|
| Observation region / fault | [-.36,.36]²m; infinite straight fault at37°, no physical tip |
| Normal profile | chi=exp(−(z/.06m)²)/(.06m sqrt(pi)), normalized on the full normal line |
| Material / duration | G=3MPa, eta=15MPa s, r=.2/s, end1.7s |
| Slip forcing | V=.03[1+.25 sin(1.3t)]m/s, positive and nonzero |
| Gradient data | k(z)=.55+.35 tanh(z/ell), D=kS+.04(t⊗t−n⊗n), W=k(t⊗n−n⊗t)/2 |
| Initial tensor | xx=.07G, yy=.03G, xy=.04G, zz=−.10G; initial s=R=0 |
| Source/operator quadrature / error rule | Gauss8² / independent Gauss12²; energy-rule check16² |
| Spatial cells per direction | N8/16/32/64, h/ell=1.5/.75/.375/.1875 |
| Time-only study | N80/160/320 steps, exact spatial samples at257 normal positions |

The imposed gradient has a compatible divergence-free velocity field, but
material advection is deliberately set to identity here. This is a labelled
spatial reaction control, **not a stationary material solution** of that field.
There is no mechanics/RSF solve, pressure feedback or physical fracture step.
The infinite through-going manufactured fault is viewed through a finite
integration patch; no missing continuation or artificial tip completion is
inferred. Line integrals use the actual patch intersections; no truncated-profile
renormalization or floor is introduced.

Maximum integrated spin magnitude is .765rad; maximum integrated total
symmetric-strain norm bound is1.08614. Final reference stress norm/G reaches
1.63525. These are deliberately nontrivial synthetic hypoelastic scales and
exceed a small-elastic-strain interpretation; numerical agreement does not
validate finite-strain elasticity. Initial zz=−300000Pa decays to
−213531.0968Pa. Full3D norms and storage retain it throughout.

## Measured refinement and acceptance

Every tensor error below is full3D L2/(G sqrt(area)); its dimensional RMS is
that number times G. No near-zero error reference sets the denominator.

| Cells/direction | Direct physical DG error | Complete separated error | Best residual projection error |
|---:|---:|---:|---:|
| 8 | 5.26308e−3 | 4.08328e−3 | 2.93543e−3 |
| 16 | 7.19033e−4 | 5.61254e−4 | 4.10979e−4 |
| 32 | 9.19899e−5 | 7.19189e−5 | 5.29102e−5 |
| 64 | 1.15662e−5 | 9.04633e−6 | 6.66300e−6 |

Last two direct orders are2.9665/2.9916; separated2.9642/2.9910, consistent with
third-order Q2 spatial approximation. No temporal order is inferred from these
analytically integrated runs. The residual's measured N64 error6.66310e−6 is
near its best-projection error6.66300e−6: the remaining error is principally
history-space resolution. Raising quadrature alone cannot remove it.
`space.csv` also records the separately projected analytic strain-rate source
error in1/s; that source is not asserted to be an unprojected native FE gradient.

| N64 observable | Direct DG | Complete separated |
|---|---:|---:|
| Full-tensor RMS error | 34.70Pa | 27.14Pa |
| Pointwise maximum tensor error/G (sampled fine rule) | 1.06180e−4 | 7.90648e−5 |
| Normal-traction RMS error | 14.28Pa | 14.28Pa |
| Shear-traction RMS error | 19.95Pa | 12.82Pa |
| Largest profile-weighted shear error,3 sections | .002115Pa | .000977Pa |
| Largest profile-weighted normal error,3 sections | .002109Pa | .002054Pa |
| Bulk storage relative error | 9.45090e−11 | 1.28828e−11 |

Profile-weighted tractions integrate chi times the actual tensor at xi=−.15,0,
.15m, split at Cartesian-cell crossings. Compression-positive traction uses a
fixed3MPa diagnostic confinement; its offset cancels in errors. Small integrated
errors are **not** point-stress bounds; sampled maximum tensor errors are much
larger. The reference storage is461445.335907J/m. This is a final-state storage
comparison, not a measured boundary-work/friction/Maxwell energy ledger.

The time-only finite-rotation Lie updates rotate old tensors, then relax/apply
endpoint forcing. Direct and separated paths evolve independently, including
an actual discrete s update and commutator source. Their finite-dt closures
are not artificially made identical:

| Steps | Direct error | Separated error | Direct/separated difference |
|---:|---:|---:|---:|
| 80 | 9.75258e−3 | 1.08813e−2 | 2.84445e−3 |
| 160 | 4.87800e−3 | 5.44240e−3 | 1.42221e−3 |
| 320 | 2.43943e−3 | 2.72164e−3 | 7.11099e−4 |

All error ratios are1.9993–2.0001, passing the predeclared first-order1.8–2.2
interval and finest<.01. Time error is substantially larger than the finest
spatial error. The difference is finite-step splitting consistency, not an
extra continuum source or a concealed change of objective rate. Native
integration will need one declared discrete update and its own accepted-state
source/tangent/observer checks; exact equality of unrelated splits is not assumed.

The independent RK4 reference differs from the exact direct solution by
5.67e−15G; refined-reference difference5.90e−15G. Independently integrated
T/B/R reconstruct the exact direct tensor to5.99e−16G. Mass/eigen defects are
below5.6e−15/6.7e−16. The Gauss12 versus16 energy integration difference is
below displayed double precision for this final smooth fixture.

## Retained failure and negative controls

The original six-point quadrature candidate fails its predeclared bound:

| Coarse N8 source rule | Difference from rule12, scaled tensor L2 |
|---:|---:|
| 3 | 7.13975e−4 |
| 4 | 2.56596e−5 |
| 6 | **2.70112e−8 — fails1e−8 bound** |
| 8 (already selected spatial-study rule) | 2.41069e−11 — passes the same bound |

The initial failing code, protocol, JSON and log are retained under `initial/`.
The bound was not relaxed. The focused follow-up qualifies q8 and rejects q6
for this strict coarse-mesh quadrature target. Final JSON explicitly retains
the failed q6 candidate alongside14 passing selected-configuration criteria;
its status is `passed_selected_q8`, not a claim that every tested setting passed.
This is not a change to production quadrature or an assertion that all existing
Gauss6 controls are invalid.

At resolved N64, omitting R leaves scaled tensor error .105787 (approximately
317kPa RMS); reversing its source gives .211574 (approximately635kPa RMS).
Their relative storage errors are3.7415% and9.3689%. These defects persist while
the correct representation converges. Thus rotating only T cannot recover the
mature Qgeo=0 law even when the accepted incoming residual was zero.

## Cost, decision and next experiment

At N64, direct storage is110592 scalar tensor coefficients (884736bytes), versus
221184 (1769472bytes) for T+R, plus the separated scalar surface memory. These
counts deliberately use complete xx/yy/xy arrays, without exploiting this
fixture's constant mean. The joint projection/eigen/history construction is
about.07s; the full numerical study is about5.1s on one local Python process,
BLAS threads1. These are Python algebra costs, not native mechanics timings or
MPI/server scaling estimates. Raw per-case costs are in `cost.csv`.

**Decision for this first part:** a resolved complete DGQ2 residual is viable
in this fixed-profile matched-coefficient experiment; no enrichment is forced
by these measurements. Direct physical DG is also viable, uses half the bulk
tensor coefficients here, and avoids T/B/R cancellation. The analytic B gives
the separated route somewhat smaller same-mesh error. These data do not yet
select the production representation: heterogeneous coefficients, motion,
non-affine native strain and physical work can change that tradeoff.

Next justified experiment is the remaining Stage B manufactured comparison with
moving/deforming profile and heterogeneous eta, explicitly exercising C[s] and
the local-versus-surface rate mismatch. Keep the direct reference independent;
measure separate time/normal-space/quadrature error before native integration.
The cohesive shape-work gate from Stage A remains independent and unresolved.

Reproduce the final bounded comparison:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/stage_b_fixed.py
```

Python3/NumPy/Matplotlib are required. No source build, native timestep, server
job, long trajectory, checkpoint conversion or cleanup ran. Existing BP3/RSF
observations, failed physical screens and cleanup data losses are retained.
[Preservation verification](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_fixed/verification.json)
checks the entry source, binaries/checkpoints, server exports and cleanup-critical
files; no retained executable is assumed to match HEAD.
