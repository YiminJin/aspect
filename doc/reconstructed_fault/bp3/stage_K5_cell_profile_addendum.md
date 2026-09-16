# Cell-aware normal profiles: bounded implementation addendum

The approved alternate backend integrates the same normal rays at the same
three Gauss points per fault segment. It retains the surface-uniform material
mixture, full physical-box truncation, consistent surface Q1 projection, raw
phase diagnostics, and existing quadrature/tail tolerances. The remote-point
backend remains the default and the numerical comparison path during validation.

The first backend supports 2-D axis-aligned affine Box cells under exact
MappingCartesian, MappingQ1, or degree-one MappingQ. Unsupported mappings use
the existing remote path; deformed meshes are not admitted. Faults may be
inclined or piecewise straight: each quadrature point keeps its own segment
normal. This does not alter open-tip topology or association support.

For each replicated profile identity, locally owned cells supply clipped signed
ray intervals. A ray on a cell face uses a deterministic half-open convention
(lower face included, upper excluded except the exterior upper boundary).
Thus a positive-measure interval has one bulk-cell owner, including across MPI
partitions. Zero-length corner intersections contribute nothing. No bulk field
is gathered. Each rank samples current ghosted Q1 phase values in its known
cells using their affine reference coordinates.

Intervals are geometric cache data, not phase values. Mesh invalidation clears
them; unchanged profile origin/normal pairs reuse their traversal. New or moved
profiles rebuild only their own cell intersections when the mesh is unchanged.
Material changes update the fixed profile mixture without changing traversal.

Each side advances through ell-wide integral windows, subdivided at every cell
intersection and at the zeros of the Q1 ray polynomial. This explicitly resolves
the kink from max(phi,0): four/eight-point agreement alone cannot detect a narrow
positive sliver when both rules sample only negative values. The stable Boost
quadratic-root implementation supplies these internal subdivisions.
Four/eight-point Gauss comparison controls cell-local bisection;
the absolute scale is the subinterval length rather than ell (a stricter local
budget). Bulk owners reduce window integrals by profile ID. Two successive
small windows use the unchanged tail coefficient and cumulative-integral
scale. Exterior boundaries terminate immediately. Integration never clips to
the mechanical support and never samples beyond an already terminated tail.

Geometric coverage, nonlinear quadrature error, tail termination, and Q1 nodal
projection equivalence are tested separately. The remote path's boundary
lookup tolerance remains unchanged; the alternate path uses exact physical
cell intersections. Any measurable difference beyond the existing accuracy
allowance is a validation failure, not grounds to weaken that allowance.

This is an opt-in numerical backend, not an analytical Ih substitution or
direct bulk-strip projection. Production defaults will not change until the
evolving K3 and long BP3 comparisons establish correctness and performance.

## Validation status and review boundary

The evolving K3 and saved K2 comparisons pass. The long BP3 comparison does
not: the remote reference prematurely finishes a boundary-final panel after
adaptive rejection halves it, omitting an in-domain remainder. Sixteen BP3
profiles reproduce this defect; a separate analytic kernel test confirms it.
There are also smaller unresolved interior discrepancies. The cell backend
therefore remains experimental/opt-in, not a qualified BP3 replacement.
See `stage_K5_bound_and_cell_profiles_report.md`. No legacy algorithm fix is
included in this pass, and no comparison threshold is relaxed.

Mesh changes currently invalidate the complete traversal cache conservatively;
selective AMR invalidation and propagation scaling have not been demonstrated.
Profile origin/normal comparisons support unchanged and appended profile reuse,
but no propagation run is claimed. Particle-property batching is deferred:
measured K3 preparation is already small and the BP3 numerical comparison
must be resolved first.
