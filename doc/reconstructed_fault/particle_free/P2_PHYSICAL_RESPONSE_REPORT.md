# P2 bounded physical-response qualification — 19 October 2026

**PASS for the selected fixed-geometry interval.** The resolved initial state
has no unloaded growth; mechanical loading produces measurable irreversible
phase growth, and all predeclared temporal, spatial and physical-work screens
pass. This supports the next bounded P3 synthetic-extension task. It does not
qualify general propagation or resolve the retained strict mirror-branch failure.

Entry is `612b75d3570d81763407875f3a12c12999b9fa8e` plus the preserved incoming
worktree. [Frozen selection and criteria](../../../benchmarks/reconstructed_fault/particle_free/evidence/p2_response/selection-before-runs.md)
were recorded before trajectories (SHA256 `67028e99272bc4870138a03e27cc6de98e2156b1bfbabc266cfc69e959f91c5e`).
[Verification, commands and raw archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/p2_response/README.md),
[common-field measurements](../../../benchmarks/reconstructed_fault/particle_free/evidence/p2_response/comparison.json)
and [response figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/p2_response/response.png)
retain the actual fields and unaligned differences.

The mesh stays uniform 2×512 cells: 10,250 velocity DoFs, 1,539 physical phase
nodes/independent masters, 27,648 independent DG tensor DoFs, nine fixed surface
nodes and 9,216 fixed diagnostic particles. The thin along-fault discretization
allows an inexpensive matched normal-direction reference. Retain homogeneous
G=1e6 Pa, eta=1e8 Pa s, Gc=26.6666666667 J/m², ell=.15625 m, Lorentz p=1,
m=128.00000000016, cohesive Z=G s_elas, the existing rate-dependent friction law
with equal reference/dynamic mu=.1, and prescribed 10 kPa **adiabatic friction
normal pressure**. Native dynamic pressure is exported separately. No RSF state
is introduced. Domain-free mesh/fault geometry, ordinary-FE fracture, consistent
endpoint/reduced Newton, unlimited DGQ2, normal Gauss12/source3, normalization
and its derivatives, solver tolerances and acceptance remain unchanged.

The original continuum seeded-H BVP defines pre-existing damage. Its original
513-point table equals that BVP exactly at every selected native vertex. Both
native trajectories start from those physical samples with exactly unloaded
T/Z/slip; the 1024-cell reference samples the **same BVP** at its new vertices.
No coarse solution is prolonged as the physical reference, no seed is relaxed
separately, and no physical history is erased. All nine initial phase/history/
normalization/geometry export files are byte-identical across the three native
controls. Initial phi is in [1.54404e-10, .600001937763], I=108.648895577 m.

| Initial representation against the continuum prescription | 512 normal cells | 1024 reference cells |
|---|---:|---:|
| Relative phase L2 | 0.0027691% | 0.00069228% |
| Phase Linf | 3.29436e-5 | 8.23794e-6 |
| Relative phase-gradient L2 | 0.67223% | 0.33611% |
| Fracture-energy error (J/m) | −2.06594e-4 | −5.16505e-5 |
| Independent normalized unloaded obstacle KKT | 0 | 0 |

All initial native phase rows are at the prescribed obstacle, with raw gradient
between .00307997 and .273383 J/m. The correct constrained unloaded equilibrium
therefore holds. A fresh two-step unloaded control through .015625 s has exactly
zero phase drift, healing and KKT defect. Its tiny Z<=1.563e-16 Pa m results from
the retained native rate floor 1e-20 m/s. The earlier 8.01966%-error coarse
initialization and its relaxation remain recorded; resolving the representation
removes that artificial-growth mechanism without changing initialization semantics.

The exact interval mean of the established velocity ramp
u_x=.256*t*(y−.5) is imposed through .5625 s, followed by zero boundary velocity
through .59375 s. Native dt=1/128 and its single halving 1/256 give 76/152
accepted steps. Both stay unchanged through .5234375 s; the halved run records
a subthreshold 6.996e-7 flank increment at .52734375 s. Measurable onset, defined
beforehand as growth>1e-5, is .53125 s in both and in all three references.

At .5625 s the primary peak is .654399205862, cumulative Linf growth .0543972681,
phase L2 growth 7.5467% of its initial norm, and healing zero. Growth is 1,651×
the initial pointwise representation error and exceeds the unloaded-drift screen.
The phase integral increases .156361952→.167314639 m; relative-half-height width
changes .247498→.241006 m as the peak rises, while every physical phase value
is irreversible. I rises to 145.726065560 m and chi_max to 7.96171 1/m (initial
7.06874). Boundary relative displacement is .0405 m; physical cumulative slip
is .03922564196 m, elastic slip Z/G=.03915546920 m, final interval rate
.1443502962 m/s and cohesive traction Z/I=268.692283 Pa. The constant-test shear
weak load is 1268.692283 Pa m over the unit-length fault. Bulk/cohesive/fracture
energies are .804798 /5.260386 /16.153101 J/m, initially 0 / 0 / 14.580127 J/m.

The hold retains the loading history. Both phase and cumulative slip stay
unchanged, V is at its existing floor, and Z/I decays by exp(−.03125/100)=
.999687548823 to268.608329 Pa. Full T and Z agree with Maxwell decay to at most
2.78e-13 scaled T error and2.91e-11 Pa m absolute Z error. Bulk/cohesive viscosity
dissipate .000502842 /.003286714 J/m; phase energy stays fixed. The final raw
sticking fault RMS reaction is .312451 Pa; its projected lower-bound contact
KKT is zero. Zero boundary motion has not removed the stress or cohesive storage.

Measurements below use common physical times, complete native DG polynomials
and the embedded 3D norm xx²+yy²+(xx+yy)²+2xy². Rates are averaged over common
intervals. Percentages use fixed reference loading-peak scales, including
1859.5858 Pa for stress; localized RMS uses chi of the reference. No branch is
reflected or registered. The targeted spatial comparison is 512-phase/projected
Q2 mechanics versus 1024-phase/continuum constant-stress mechanics at dt=1/256.

| Maximum difference throughout loading and hold | Native dt vs dt/2 | Targeted spatial reference |
|---|---:|---:|
| Whole full-tensor stress L2 | .00118635 Pa (0.00006380%) |5.46288 Pa (0.29377%) |
| Localization-weighted stress RMS | .00118635 Pa (0.00006380%) |11.3316 Pa (0.60936%) |
| Phase Linf / relative L2 |6.3091e-8 /0.0000081033% |2.52025e-4 /0.046766% |
| Phase-gradient L2 |2.16058e-7 | .0113384 |
| Cohesive traction |5.45065e-5 Pa | .0178061 Pa |
| Cumulative slip |2.82994e-10 m |1.79999e-8 m |
| Total energy |1.62665e-6 J/m |2.20748e-4 J/m |
| Common elastic phase-driving functional |0.000024914% |0.087583% |

The native along-fault phase spread is<=3.25e-10, supporting the 1D reference
for this uniform fixture. Its equations retain the native projection and
cohesive/Maxwell laws; fresh derivative and independent full-Q2 matrix controls
pass (matrix stress difference<=6.37e-9 Pa). Matching 512-cell native/reference
solutions choose opposite mirror branches: final centroids are .5000085083 /
.4999914917 m at dt=1/128 and .5000085125 /.4999914877 m at dt/2. Maximum
unregistered phase difference is 5.91136e-5 and common-driving difference 0.023842%.
**strict_reference_match_passed=false remains**, with all original errors.
Practical sensitivity screens pass; these branch differences are not counted as
convergence. A single halving establishes bounded temporal uncertainty, not order.

Independent normal integration and Gauss24/32 physical checks give normalization
error<=1.96e-15, energy/load quadrature differences<=6.58e-14 scaled, physical
stress reconstruction<=9.54e-13, weak-load difference<=2.74e-13 and source
recurrence<=2.34e-15 scaled. They are below measured temporal/spatial differences.
The diagnostic reports full reconstructed deviator stress (including implicit zz)
and native dynamic pressure; prescribed friction pressure is not substituted
for that pressure. At selected loaded snapshots native bulk residual norm<=
6.55e-6 and fault RMS<=1.62e-11 Pa; independent projected weak contact defect<=
3.03e-12 Pa m. Raw native phase KKT maxima are 7.37e-12 /5.79e-11 J/m; native
roundoff-aware acceptance and stability criteria are unchanged and pass. Each
trajectory retains one negative translation/CG departure in its stability trace.

External work is assessed independently of algebraic closure:
R_ext=Delta(E_bulk+E_cohesive+E_fracture)−W_boundary+D_bulk+D_cohesive+D_friction.
The retained diagnostic path uses linear physical phase, exact Maxwell histories
and fixed endpoint interval rates. Separate mechanical, pressure, contact and
phase works account for its nonequilibrium interior. The table gives signed
loading sums, except the explicitly absolute sums. All entries are J/m.

| Loading work/energy term | dt | dt/2 |
|---|---:|---:|
| Total energy change |7.63815800 |7.63815942 |
| Boundary work |46.87901001 |46.88342534 |
| Bulk / cohesive viscosity | .006306748 /.014718937 | .006308544 /.014722488 |
| Friction |39.22564196 |39.22564168 |
| Free-mechanical residual work | .143278994 | .036406010 |
| Contact residual work |−.057033942 |−.014304919 |
| Phase work |−.080429406 |−.020694292 |
| Pressure work |−1.63e-16 |−1.62e-17 |
| Signed external remainder | .005815646 | .001406799 |
| Sum of absolute interval external remainders | .006032745 | .001461094 |
| Largest interval external remainder | .003394117 | .000895657 |

The absolute loading remainder is 0.012869% /0.0031164% of boundary work,
below the predeclared 0.5% screen and reduced 4.13×. Absolute interval sums of
free/contact/phase work also decrease .143279/.063822/.082474→
.036406/.016096/.020694. The largest remainder occurs at first sliding,
.09375 /.08984375 s, before phase growth. Hold absolute remainders<=8.72e-15
J/m. Ledger closure<=7.00e-14 J/m, static-time-moment/full-path agreement<=
2.10e-13 J/m and growing-interval time-quadrature uncertainty<=1.73e-14 J/m
are supporting checks; closure alone is not the physical-work qualification.
Finite path-work defects remain, so this is a bounded response result.

The three native controls pass 14/318/622 gates, with exactly 2/76/152 accepted
publications and unchanged independent publication checks (maximum error/
operand-derived allowance .005972). Wall times 20.86/774.35/1439.42 s stay within
180/1200/2400 s budgets. Rejection/retry, pending discard and restart lifecycle
evidence is reused from the [completed same-core hanging-node qualification](PF6_HANGING_REPORT.md),
including its 129-case native suite; no expanded lifecycle/MPI campaign is claimed.
Read-only verification checks 2666 raw CSV identities and all 34 protected BP3
identities. All 858 production source/header identities match entry. Only the
benchmark observer schedule, thin fixture preparation, optional reference hold,
analysis and documentation change. Empty observer schedule retains prior behavior.
Core SHA256 `686288cbbda7cb9793249a5d9dc31cd6f5ea8414737896c376bf71aefa090f80`.

P2 is ready for the **next bounded P3 synthetic straight extension**, with the
strict branch and broader spatial/endpoint temporal findings retained. Before
that extension, specify the joint DG T / cohesive Z=G s_elas / physical-obstacle
mapping on overlap and newly supported surface, measuring stress/storage/load
transfer defects separately from evolution. Rebuild normalization and all
geometry/profile/association caches consistently; define geometry/history atomic
rejection/retry, single accepted publication and same-mode restart. Do not assume
that keeping T unchanged preserves physical tau after chi changes. Distinguish
physical tips from boundary completion. No P3, AMR, transport, moving geometry,
RSF evolution, propagation or server job is begun here; BP3 evidence and its
unresolved accuracy findings are preserved.
