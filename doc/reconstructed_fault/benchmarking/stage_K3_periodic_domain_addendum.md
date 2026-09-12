# Bounded periodic particle-domain correction

Approved scope: fixed axis-aligned 2-D Box, one declared periodic coordinate.
The global Box extent is the period, never the extent of a local Voro patch.
Images are translated geometric neighbors only: each locally owned real parent
is integrated once and remains the only particle/history owner.

Construction uses deal.II periodic vertex equivalence to complete the existing
vertex-adjacent cell patch across the seam. Opposite-cell particle positions
are translated by the global period into the parent's chart. Voro container
periodic flags remain false. Its local artificial boundary must still be beyond
the domain, using the existing missing-neighbor check. Physical nonperiodic
walls are unchanged. CPDI uses the same full-polygon centroid triangles; sample
positions are periodically mapped to actual FE cells before applying the existing
unique-owner rule. The phase sparsity patch includes the same periodic cells.

`vertices()` retains the full contiguous unwrapped polygon. An additional
`periodic_fragments()` view is empty for an unsplit domain, otherwise contains
all polygon pieces translated into the physical Box. Splits preserve polygon
orientation and full measure, and introduce no normal clipping. The fragment
collection is geometry only; it adds no parent, material sample or history.

Consumer inventory:

- Phase assembly consumes full-parent volume and CPDI weights/gradients; no
  phase equation or constitutive evaluation changes.
- ReconstructedFaultManager consumes physical fragments for its cached domain
  quadrature. Each fragment uses the existing open-polyline map, including
  its tip/corner conventions. Independent endpoint DoFs remain independent.
  All projection, residual/K_V/G, norm and diagnostic consumers already share
  this cache, so no separate rule is introduced downstream.
- Volume-only projection checks and cohesive/history code keep full-parent
  volume. Parent-center admission, P0 bulk/history inputs, and commit timing
  do not change.
- The optional face-based Voronoi linear-reconstruction interpolator requires
  image-relative neighbor positions. This bounded implementation rejects
  periodic domain generation with face data requested rather than silently
  giving that interpolator wrong distances. The approved fixture requests
  CPDI only and retains the cell-average interpolator.
- Domain inspection/tests must distinguish unwrapped vertices from physical
  fragments. Raw first moments at the seam use the parent's coordinate chart,
  not the discontinuous physical-box coordinate function.

No gradient-orientation sign fix, support/normalization change, cyclic surface
topology, history-transfer change, or solver change is included.

Verification order: small nonlinear homogeneous phase/history displacement
tests (pre-wrap and wrapped, one/two ranks), saved frozen seam audit, affected
surface/domain actions, then the existing common-dt K3 fault32 fixture. Keep
the old diagnosis and all failed evidence; changed initial projections are
reported rather than reset from old outputs.
