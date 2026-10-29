# P4a: finite-tip integration and phase-driving dependency

19 October 2026. P3 is closed and committed as
`f1f9f29cf40c32d1ddcea000b724375b64853532`. Its native zero-time transfer,
rollback, restart, two figures and limitations remain intact. P4a's **frozen
diagnostic is complete; physical-response qualification is blocked by an
undetermined intact-material activation rule**. No finite-tip solve guard
was removed. BP3 and the existing production equations/defaults are unchanged.

## Actual geometry and support

![Actual native tip quadrature, mesh, phase and normalization rays](../../../benchmarks/reconstructed_fault/particle_free/evidence/p4a_tip/tip_quadrature.png)

**Caption.** The finite surface and continuum cap were prescribed, not grown.
The physical right endpoint is at (.515625,.5) m, inside bulk cell
[.5,.53125]×[.5,.515625]. The Q1 phase samples the original compact radial
profile around this endpoint; its obstacle is identical. White contours are
the represented phase. Orange lines are selected actual normalization-ray
intervals; cyan/pink points are all actual mechanics quadrature points in the
close-up. Endpoint and support membership come from the native owners, not
plot geometry. No append or phase change occurred. Equilibrated mechanical
response and physically driven phase growth remain untested.

There are 393216 native bulk points, of which 202752 are associated. Their
weights sum to 1 m² and .515625 m², respectively (ordinary-double summation
errors 1.83e-12 and 6.1e-14 m²). The close-up contains 1100 points, 572
associated, but their plotted weights sum to .00298707913 and .00157867386
m². Individual JxW range from 2.91e-7 to 5.52e-6 m². Counts are not quadrature
weights; sums over a plotted point window are not exact clipped-window areas.
Normalization integrates each normal interval analytically, followed by
Gauss8 surface integration and consistent surface projection. Rays have
surface weights, not particle masses. The figure records their weight range.

Normal profiles and fault weak coupling end at the actual endpoint plane.
There is no extension into the rounded cap. Bulk stress still exists there,
but its local mechanical phase derivative is zero. The cap contributes ordinary
FE fracture energy everywhere, and remains under physical irreversibility.
It is pre-existing diffuse damage, not a new surface segment. The left
computational-boundary contact has zero completion wedge; boundary completion
does not turn this interior right endpoint into a continued fault.

## Retained equations and the scientific dependency

For fixed Gamma and immutable T/Z, with full 3D plane-strain tensor norm,

\[
 h=g^{-1}-1,\quad I_h=\Pi_\Gamma\int h(\phi_h)\,dn,\quad
 \chi_\Gamma={\bf1}_{\mathrm{normal\ strip}}h/I_h,\quad
 \tau=T-2Z\chi_\Gamma S,
\]
\[
 E_b=\int_\Omega\|\tau\|_{3D}^2/(4G),\quad
 E_c=\int_\Gamma Z^2/(2GI_h),\quad
 E_f=\frac{G_c}{c_0\ell}\int_\Omega(\phi+\ell^2|\nabla\phi|^2).
\]

Here Z=G s_elas, Sxy=Syx=1/2 and tauzz=-(tauxx+tauyy). The native partial
phase variation includes both terms below and the fracture derivative:

\[
 \delta E_b=-\int_{\mathrm{strip}}\frac ZG(\tau:S)
 \left[\frac{h'}I\delta\phi-\frac h{I^2}\delta I\right],\qquad
 \delta E_c=-\int_\Gamma\frac{Z^2}{2GI^2}\delta I.
\]

For the consistent projection I=M_Gamma^{-1}b(phi), the calculation retains
the complete (DI)^T pullback. It does not freeze I or drop evolving-profile
terms. Off the strip, tau=T and a phase perturbation supported wholly ahead
changes neither the bulk nor cohesive mechanical energy. This remains true
even though a basis function centered ahead can extend behind the endpoint.

The historical intact-point expression q²/(2Gg²) requires a defined q. It
does not supply q in unassociated material ahead of Gamma. The retained
particle history implementation also updates its driving history only in an
active association; combining that observer with the endpoint algorithm would
not close this gap. The derivation explicitly says the I=0 activation/limiting
rule must be supplied. At fixed nonzero Z the cohesive energy diverges as
I→0; the path Z=Iq with finite q instead has vanishing storage, but does not
determine q or an activation criterion. No normalization floor was added.
These are constitutive/activation choices, separate from integration accuracy.

The currently verified birth/transfer policy remains P3's restricted one:
positive finite normalization, zero old-tip histories, prescribed zero new
Z/slip/rate records, unchanged physical phase obstacle and the joint
T+=T-+Pi_Q2[2(Z+chi+-Z-chi-)S] map. Old histories/material identities survive;
the ledger separates bulk projection error, changed normalization/cohesive
storage and added-surface weak loads. This does not license free physical
surface creation, an intact I=0 birth, or a general nonzero-tip join. No new
segment was inserted here to generate driving.

**Needed scientific choice:** the mechanical energy/traction and activation
rule for intact material ahead of the finite surface, including its I=0 birth
limit. General nonzero-tip joining remains deferred. The original propagation
PLAN explicitly requires this dependency to be resolved before inserting
geometry. AGENTS likewise requires stopping affected implementation on a
scientific/design conflict while continuing independent useful checks.
An equilibrated finite-tip mechanics/phase-active lifecycle was therefore not
enabled merely by relaxing P3's frozen-observation restrictions.

## One aligned control and one endpoint inside a cell

The [pre-run equations and screens](P4A_TIP_SELECTION.md) fix the two cases.
Retain the 32×64 bulk mesh, unlimited independent DGQ2 with all nine modes,
Gauss8-per-half-cell tangential / Gauss12 normal mechanics diagnostic rule,
Gauss3 source projection, homogeneous P2 materials, existing constant friction
and prescribed normal compression. The current fixture's native special
diagnostic rule is distinct from BP3's untouched Gauss6 mechanics.
Aligned x_tip=.5 m is the control; x_tip=33/64 m is inside a bulk cell and on
an existing surface-knot/integration-panel boundary. This tests a cell-interior
endpoint with an exactly resolved strip cut, not arbitrary off-panel endpoints.
The two phase fields are direct samples around their respective endpoints;
their difference is not a spatial-convergence result.

Native T is the P3 complete non-proportional polynomial. The loaded surface
state is Z=1000 sin²(pi*x/x_tip) Pa m, with zero endpoint records. These are
non-equilibrated snapshots, not mechanical solutions. Independent NumPy
integration reads actual native phase, all DG coefficients and surface histories;
Gauss24/32 explicitly split at bulk cells, every surface knot and the support
endpoint. The represented compact support is resolved by the Q1 cell partition.
No source energy, normalization or differentiation routine is reused.

| Quantity | Aligned | Cell-interior endpoint |
|---|---:|---:|
| Bulk storage [J/m] | .637166455605 | .643919628666 |
| Cohesive storage [J/m] | .000865177768 | .000892328179 |
| Fracture storage [J/m] | 9.43534995312 | 9.66340268703 |
| Integrated shear weak load [Pa m] | -706.478611658 | -728.637161788 |
| Integrated cohesive weak load [Pa m] | 2.31207692743 | 2.38433416712 |
| Integrated friction weak load [Pa m] | 500 | 515.625 |
| Minimum projected I [m] | 108.127890138 | 105.760830749 |
| Max normalized native/reference error | 1.42e-14 | 1.13e-13 |

Errors use max(1, maximum absolute reference value) normalization, with the
stated SI units; they include all mechanical/fracture gradient entries, storage,
I and all three weak-load vectors. Native full mechanical-gradient errors are
7.46e-16/5.97e-16 J/m. Independent Gauss24–32 maximum differences are
1.45e-14/1.57e-14 on this normalization. All pass the declared 1e-10 discrete
integration screens. Directional checks behind/across/ahead of the endpoint
match centered energy differences at epsilon 1e-5 and 1e-6 to at most
7.38e-10/3.77e-10, below the declared 1e-7 finite-difference screen.

Strictly-ahead mechanical derivatives are **exactly zero** in both native and
independent calculations. The corresponding fracture derivatives are
.455822387/.466029270 J/m. A node ahead of the cut tip whose basis straddles
it has mechanical derivative up to 1.05454e-7 J/m; this is not direct intact
driving. The unloaded obstacle KKT has minimum gradient +.00163323917 J/m
and zero negative nodes in both cases. Convex AT1 fracture with the prescribed
seed lower bound therefore has no unloaded energy-decreasing admissible
direction on these meshes. Relative L2 phase representation errors against
the same continuum prescription are .339721%/.321643%; Gauss24–32 differences
in these estimates are 2.40e-10/2.53e-10 absolute. This seed-table estimate is
separate from integration of the represented discrete fields. Phase lies in
[0,.6] and equals its physical obstacle exactly. These checks do not qualify
a loaded evolving event.

![Native physical stress and local mechanical phase derivative](../../../benchmarks/reconstructed_fault/particle_free/evidence/p4a_tip/tip_fields.png)

**Caption.** These are actual native loaded quadrature samples. Auxiliary Txy
is roughly .806–.827 kPa in the exported tip window; reconstructed physical
tauxy reaches -2.550 kPa. The right panel is the local bulk derivative at
fixed I; the separately verified complete nodal derivative also includes
the nonlocal normalization pullback. Negative energy derivatives favor growth;
a colored local value alone is not a phase KKT or an evolving solution.

No time, phase, geometry or persistent history changed. Consequently there is
no dynamic external-work balance or growth result to claim. Frozen storage
and derivatives have been checked, rather than interpreting a zero-time ledger
as physical energy qualification. An eventual evolving comparison must retain
P2's separate external/mechanical/contact/phase residual-work accounting and
unresolved branch findings.

## Evidence and remaining scope

[Replayable evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/p4a_tip/README.md)
contains actual exports, plots, independent checks, commands and hashes. Each
native run has a 180 s timeout. Final native observations take 10.69/10.58 s.
Every accepted owner remains byte-identical during observations. A fresh
default P3 transaction with the augmented test plugin reproduces 16 original
persistent/phase/stress/surface/load/ledger/constraint/DG exports byte for byte.
No production source/header/CMake change occurred after the P3 commit; all 34
protected BP3 identities and manual AGENTS work match entry. P2's
strict_reference_match_passed=false remains false.

Next: select the specific ahead-of-tip activation rule before implementing
finite-tip mechanical eligibility and phase-active evolution. No physical
propagation, RSF, transport, AMR, moving mesh, MPI expansion, parameter search
or general birth/join qualification was attempted.
