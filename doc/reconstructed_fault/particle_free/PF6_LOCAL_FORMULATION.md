# Gauss12 control and conditional fixed-surface local transfers

Selected 16 October 2026 at a8707307008401ac9cff702a369e2e2a611fa0f7 plus
preserved incoming work (`build-particle-free/quad-local/incoming.json/patch`).
The uniform-transfer/MPI/restart results are accepted within their scope.
The original Gauss6 failure and `strict_reference_match_passed=false` remain
immutable evidence. Unlimited DGQ2, equations, rate conventions, limiters and
production defaults remain unchanged.

## Gate before local refinement

Reevaluate the retained worst accepted state (uniform fixture state 2) with
native physical Gauss6/12/24/32 at identical physical phase, T, Z and I, without
advancement/publication. Repeat the same short serial fresh trajectory with
Gauss12. Compare native bulk energy and fault weak loads against independent
Gauss24/32 common physical evaluation; retain the previous scaled 1e-9 screen
and absolute errors. Report Gauss24/32 uncertainty, and trajectory differences
from the preserved Gauss6 case without treating Gauss6 as the accuracy oracle.

Separate normalization from bulk integration. Independently integrate h(phi)
along normal rays using Gauss24/32 and project with the fixed Q1 surface mass;
compare actual native I with a scaled 1e-11 arithmetic screen. Reevaluate the
physical functionals with independent I to quantify normalization's contribution.
Changing physical Gauss points must not change the analytic normalization rule.
The existing Gauss3 mass/source projection remains unchanged. No adaptive rule
or default quadrature change is selected.

## Conditional small local fixture

Only if the Gauss12 gate passes, refine a fixed interior patch isotropically on
the existing Cartesian box, producing real coarse/fine faces and hanging nodes;
keep the surface mesh and all particle positions fixed. Use native distributed
SolutionTransfer for all nine modes of each of three DG components and all native
time levels. Preserve the accepted branch and physical clock/publication counts.
Check global complete Q2, non-proportional modes, and an accepted nonzero-Z state
at common physical points and independent quadrature. Compare phase/lower obstacle
and native fields, normalization, physical energy and weak loads separately.

Permit only complete coarsening families whose physical Q1 phase/obstacle is
parent-representable within the existing 256-epsilon screen; reject a localized
fine peak before mutation. Rebuild fixed-geometry caches and respect native
hanging-node constraints. Check the physical constraint relations themselves.
Use 1/2/4 ranks and same-count plus serial-to-four native restart of the retained
locally refined accepted state. Bound each native run to 180 s, sequential jobs,
one thread/rank, no more than four ranks. Use the previous 1e-11 polynomial and
1e-9 physical-observable screens with raw errors retained.

This stage isolates numerical remapping and hanging-node evaluation/restart:
coupled phase evolution on a hanging-node mesh is not inferred from transfer.
The endpoint minimizer currently treats its phase coordinates as independent;
constrained variations and obstacle KKT would require an explicit reduced-space
design before evolving there. Retain a guard against that unsupported solve.
The short physical trajectory and its native rejection/retry/publication remain
on the previously qualified uniform mesh, followed by a frozen accepted local
remap/checkpoint. No lossy obstacle rule, automatic AMR, surface remeshing,
transport, moving geometry, birth or propagation is selected.

## Bounded support amendment after cache diagnosis

Fresh domain rebuilding exposed the existing one-layer Voronoi support guard in
the retained 64→256 transfer diagnostic. The original four-row local patch is
also outside the analytically sufficient fixed-particle support construction.
Do not add particles, relax a coverage check, or invent domain volumes. Preserve
these failures. The final selected patch is one cell with center in
(.5,.75) x (.375,.4375), yielding 64→67 cells and real hanging nodes. The
retained surrounding CellId owns the particles on this cell's upper faces;
choose the owned side rather than the neighboring cell across those faces.
A single
cell borders coarser cells on all sides. The nearest fixed particles lie strictly
inside that neighborhood; this avoids relying on particles at a Voro container's
upper boundary. The first isolated tag was canceled by default island smoothing,
exposing a stale prepared DG transfer on a no-op; repair that lifecycle and test
two repeated no-ops. During actual local remaps the diagnostic explicitly sets
deal.II's vertex-level balancing alone to permit one isolated cell. Production
mesh-smoothing defaults remain unchanged, as do every physical field and screen.
Verify resulting volumes independently
against the midpoint-bisector Voronoi rectangles of the unchanged 6x24 Cartesian
particle set; do not assume sufficiency from the construction.

The same two physical steps remain at 64 cells, with the same initialization,
loading and timestep. The extra identity-only 64→256 roundtrip between them is
disabled explicitly in this small fixture, and retained as a separately failing
support diagnostic. The original transfer observer's default stays unchanged.
Repeat frozen and fresh G12 controls on the repaired final core, and measure
the trajectory difference from the earlier G12 case including that roundtrip.
Also run the same short trajectory with Gauss24 and Gauss32 to measure evolution
uncertainty directly, alongside the fixed-field quadrature comparison. This is
the prescribed three-rule control, not an adaptive quadrature choice.
This narrows the local qualification; it does not qualify sparse-particle domain
construction on general refined grids. Extended particle neighborhoods or
replacement of this particle/domain dependency requires a later selection.

## Closed-patch insertion diagnosis supersedes the support assumption

The isolated owned-cell attempt still failed for particle 63 at
(.58333333333333337,.39583333333333331), with a neighbor box
(.25,.3125)–(.75,.4375). The geometric sufficiency assumption above was not
established. An independent control against the actual Voro++ 0.4.6 library
shows that the closed-box patch has 16 candidates but the unmodified container
retains only nine: candidates at its upper faces are silently dropped. The
target area is .015625 instead of the independent bisector area 1/144.

Adapt the container, preserving every particle coordinate and physical-box wall.
For an artificial upper wall b and lower wall a, use an outward representable
endpoint when the largest candidate is within the derived bin-roundoff band:
delta=gamma_8 (|a|+|b|+|b-a|), gamma_k=k*u/(1-k*u), u=epsilon/2, followed by
nextafter(b+delta,+infinity). Four rounded reciprocal/bin-map operations give
gamma_4; this larger operation/operand bound covers endpoint construction and
strict bin inclusion. Assert that every intended candidate is actually resident.
The independent control retains all 16 and recovers 1/144. Leave the original
positive-volume and physical-coverage guards intact; no extra neighbors or
invented volumes are supplied. Limit the adapter to nonperiodic 2D; other paths
retain their existing behavior.

Restore the original deep uniform identity roundtrip in the final trajectories;
its earlier failures remain evidence. Independently verify every exported
particle domain, including 256-cell and local snapshots, against the exact
midpoint-bisector rectangles of the unchanged Cartesian particle set. This
qualifies the adapter only for the measured fixture, not general sparse-particle
Voronoi neighborhoods. No further patch search is selected.

With all candidates retained, the deep grid exposes a distinct genuine support
failure: particle 3 at (.25,.020833333333333332), CellId 0_3:001, has a patch
(0,0)–(.375,.0625) with four intended/resident candidates. Its nearest right
neighbor at x=.4166666666666667 is outside the patch. The original coverage
guard still rejects it. Keep that deep control failed; do not extend the
neighborhood algorithm in this task. The final two-step physical trajectories
therefore use the same 64-cell physical steps and explicitly omit the intervening
256-cell identity diagnostic, as in the first bounded-support amendment. Compare
against the original trajectory to quantify this omission. Retain the independent
all-candidate insertion check and the existing coverage guard on the small local
fixture, and validate all its exported domains against the global bisector oracle.

The first local restore loaded and exported the accepted state correctly, then
the native do-while entered an extra physical timestep before checking End time.
The existing hanging-node phase guard stopped that unsupported evolution. Honor
the existing termination manager immediately before this loop only on restored
accepted DG endpoints. A larger-End-time continuation still enters the loop.
Add a restore-only start-timestep negative guard in the observer; final snapshot
and local restart controls must exit without another trial/publication. Earlier
frozen controls' state-2 observations remain valid, but their later extra uniform
step is retained and is not a passing restore-only lifecycle check. Repeat final
controls on one build after this correction.
