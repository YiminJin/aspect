# Bounded polyline domain-quadrature extension

2026-09-09. Written before polyline production edits. Extends the approved
domain-integrated rule, retaining the parent-P0 inputs defined in
`stage_K2_domain_quadrature_addendum.md`. No reconstruction is flattened.

## Coordinate map on an already admitted parent domain

The parent's admitted/excluded status and assigned fault remain immutable.
Integrate its whole existing convex domain D_p. Do not clip by the influence
width, transfer its measure to another fault, or change live CPDI geometry.

For ordered fault segment j, let t_j be its unit tangent, n_j=(-t_y,t_x),
u_j(x)=t_j.(x-a_j), L_j=|b_j-a_j|, and d_j(x)=n_j.(x-a_j).

1. Among segments with 0<=u_j<=L_j, select the smallest |d_j|. Set
   xi=u_j/L_j and use the selected segment's Q1 basis and frame. Exact ties
   use the lower segment index, as in the existing projection loop.
2. If no segment has a finite orthogonal projection, select the nearest
   fault vertex. Surface values are exactly that vertex's values. At a true
   tip use the incident segment frame. At an internal corner select, among
   its two incident segments, the smaller absolute supporting-line distance;
   exact ties use the lower index. Thus frames are one-sided, not averaged or
   radial. In an exactly symmetric tie both frames receive complementary
   regions separated by a bisector; the tie line has zero area.

**Departures from the old point convention:** domain-point mapping does not
reapply the normal-width admission filter. It extends the map to corner/tip
regions where no finite segment-normal coordinate existed. Within finite
projection regions it otherwise retains the closest-normal-distance rule,
not a new global Euclidean closest-point rule mixing segment interiors and
vertices. This is solely the map within an already admitted D_p. The old
width-filtered map still controls parent admission, parent history updates,
and bulk-QP association. Geometry with self-intersections, branching, closed
loops, overlapping faults or 3-D is outside this extension. No mapping to a
different fault or new support policy is introduced.

## Convex partition and coverage

Translate each domain to local coordinates before plane arithmetic. Partition
it by segment endpoint-normal planes u_j=0,L_j and, where two finite segment
projections compete, the two supporting-line distance bisectors
d_i-d_j=0 and d_i+d_j=0. On each resulting convex piece the finite candidate
set and selected segment are constant. Nonintersecting planes leave a piece
untouched. No area is deleted when constructing a complementary split.

For pieces with no finite candidate, use ordinary vertex Voronoi half-planes
|x-v_i|^2<=|x-v_j|^2, and split an internal-corner region by its incident-line
bisectors to determine the one-sided frame. Candidate pruning is permitted
only by geometric bounds proving a segment/vertex cannot win on the domain.
Each positive-area piece has exactly one owner. Shared boundaries have zero
measure. Check summed quadrature area against the original full domain
volume without renormalizing weights to hide uncovered or duplicated area.

At every integration point form S=sym(t tensor n) and N=n tensor n from its
selected segment. Parent bulk gradient/pressure/temperature/phase-field,
chemical fractions and old stress remain P0; only the surface coordinate and
frame vary across D_p. R_Gamma, K_V and G use the same frame and quadrature.
Geometric points/weights are cached and frozen throughout a mechanical solve.

## Moments versus nonlinear integration

For a segment-owned polygon retain the successful transverse-width method:
in that segment's local frame, split at projected polygon vertices; its width
is linear, so three-point Gauss integrates N_i and N_i N_j exactly up to
roundoff. Corner/tip pieces have constant basis and use their full area.
Never replace integral(N_i N_j) by products of first-moment averages.

The nonlinear response is still evaluated at each Gauss coordinate, with
all surface fields interpolated there. Nonlinear response and residual-square
accuracy are checked separately at orders 3/5/7. K_V/G finite differences
differentiate the selected fixed-quadrature residual; derivative agreement
alone is not a quadrature-accuracy test. Residual norms and diagnostic weak
loads use the corresponding integrated mass and nonlinear weights.

## Installed Voro++ comparison and implementation choice

Inspected `/home/ein/local/voro++/0.4.6/include/voro++/cell.hh`. Assignment
copies cells independently. For a temporary prism in local coordinates,
`plane(nx,ny,0,2*c)` keeps n.x<=c; the explicit factor two is essential.
Complementary cuts use (-n,-2*c). The standalone `verify_plane_cuts` probe
checks positive/negative offsets, oblique cuts, empty cuts, copies and slivers.
Complementary unit-box volumes agree within 8.9e-16; an oblique analytic
volume 3.155 is recovered. A width-1e-8 sliver survives, but width 1e-13 is
collapsed (2e-13 volume difference). The original cell remains unchanged.

Choose a small file-local 2-D convex-polygon split routine instead. The
polygons are already retained from production construction; using Voro++ here
would require prism construction, additional polygon extraction and its cut
tolerance semantics. A single computed edge intersection shared by both
halves makes complementary coverage directly testable. No new clipping library
or public polygon framework is introduced. Voro++ remains unchanged as the
particle-domain construction backend.

## Verification gates

Capture the actual formerly failing reconstructed polyline; test it together
with genuine bends, finite-projection overlaps, corner gaps, tips, segment
crossings, complementary/near-degenerate cuts and the straight limit. Verify
full measure, separate first/second moments and nonlinear accuracy. Repeat
production ownership tests on one/two ranks, then the blocked surface/coupled
and affected lifecycle/rollback/restart tests. Only afterward run short K1 and
unchanged 64/128 K2 through 1 s. Report any newly required mapping/support
change instead of silently broadening this convention. The larger temporal
campaign and true-normal-stress branch remain held.
