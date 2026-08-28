# Stage 5 - Projecting generic particle properties onto the reconstructed fault.

## Projection operator

The particle -> fault operation should NOT be implemented as independent weighted averaging at each vertex.

For each locally owned particle $p$ within the fault influence distance:

1. Find its closest point on the reconstructed fault.
2. Identify the fault segment containing that closest point.
3. Compute the segment reference coordinate $\xi_p$.
4. Evaluate the two Q1 shape functions $N_0(\xi_p)$, $N_1(\xi_p)$.
5. Compute the signed normal distance $r_p$ from the particle to the fault.
6. Compute a geometric weight
\[
    w_p = m_p K(|r_p| / R_p),
\]
   where $K$ is a symmetric compact-support kernel and $m_p$ is the particle sampling measure.

The preferred sampling measure is the particle-domain volume obtained from the existing particle-domain infrastructure when available. Do not substitute CPDI bulk-grid interpolation weights for this projection.

For a scalar particle property $z_p$, assemble
\[
    M_{ab} \mathrel{+}= w_p N_a(\xi_p) N_b(\xi_p), \\
    b_a  \mathrel{+}= w_p N_a(\xi_p) z_p
\]
and solve
\[
    M z_{\text{fault}} = b.
\]
Thus the unknowns are Q1 fault-vertex values.

For multiple scalar/property components that use the same geometric weighting, assemble the geometric matrix M only once and reuse it for all right-hand sides.

## MPI requirements

The reconstructed fault geometry is replicated.

Only LOCALLY OWNED particles should contribute to the local assembly. Do not include ghost particles and thereby double-count contributions.

The intended MPI pattern is:

    local particle contributions
        ->
    MPI reduction of fault-sized matrix/RHS data
        ->
    identical global projection solve on each rank
        ->
    identical replicated fault-property values.

Do not gather the complete particle set onto every rank.

Please examine the existing MPI/data structures and determine the cleanest implementation consistent with ASPECT/deal.II conventions.

## Projection cache

The geometric particle-to-fault calculation will be reused for multiple properties and potentially multiple nonlinear evaluations.

Design a minimal cache entry containing only information justified by the algorithm. It will probably include information equivalent to:

    particle identifier
    fault segment/cell identifier
    reference coordinate xi
    Q1 shape values
    signed normal distance
    geometric weight
    geometry version

but do not copy this list blindly. Inspect the existing classes and determine which entries are actually necessary and which can be cheaply recomputed.

The cache must have explicit invalidation semantics. Consider at least:

- reconstructed-fault geometry changes;
- particle positions change;
- particle ownership/migration changes.

Do not prematurely build a complicated general caching framework.

## Scope

This stage is ONLY particle -> fault projection.

Do not yet implement:

- fault -> Stokes/material quadrature-point interpolation;
- RSF constitutive behavior on the fault;
- modification of the phase-field;
- crack propagation;
- 3-D support.

However, design the interfaces so that the next stage, fault-vertex -> quadrature-point Q1 interpolation, will be straightforward.

## What I want from the plan

Please provide:

1. A concise description of the mathematical algorithm you believe should be implemented.
2. The exact existing classes/functions/files that should participate.
3. The proposed public and private interfaces.
4. The minimal new data structures, especially the projection-cache record.
5. The MPI assembly/reduction strategy.
6. How generic scalar and multi-component properties should be handled.
7. Cache construction and invalidation rules.
8. Failure/degeneracy cases that need explicit treatment, such as insufficient particle support near fault tips.
9. A staged implementation sequence with small independently testable commits.
10. Any place where the current code structure conflicts with this algorithm.

Prefer reusing existing ASPECT/deal.II infrastructure over introducing parallel abstractions.

Most importantly, separate:
(a) mathematical requirements,
(b) implementation necessities,
and
(c) optional optimizations.