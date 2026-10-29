# Finite straight exact-jump insertion and extension

This extends the optional native FE/property/checkpoint component fixture, not
ASPECT's production Stokes/phase evolution driver. The calibration and
[front law](COHESIVE_INTERFACE_FRONT_FORMULATION.md) are unchanged. The bulk mesh,
mixed pressure space and candidate-line quadrature remain fixed.

## Mechanical representation and transfer

On the fixed periodic background lattice, let Q be the Q1-to-Gauss3 interpolation
and let P_I inject the interior nodal jumps of the **finite installed** interval I
into the background trace space. Endpoints and uninstalled nodes have exactly
zero jump. The actual displacement field is

    u = u_affine + u_FE + [H(y−yΓ)−Fχ(y)] δ_h e_x,
    δ_h = Q P_I δ_I.

All components of the regular enrichment gradient, its divergence and pressure
work are retained, using the previously qualified native mixed FE operators. A
background node that is not installed is not subjected to a 3000 Pa surface
reaction bound: its jump variation is absent. An installed intact region is tied
by the bounded weak Gauss multiplier conditions. These are distinct conditions.
All Gauss histories belong to actual installed segments. The native vertex-based
property container stores three histories on each outgoing segment; the terminal
vertex's padding is not an additional physical segment.

The independent upper/lower pressure traces are part of the **fixed bulk FE
approximation**, on the candidate mesh line even outside I. They are not friction
records and do not grant a displacement jump. Insertion changes P_I, its endpoint
constraints and the native finite geometry/history owner; it does not change that
pressure approximation. Thus this is a real enlargement of the mechanical jump
space, not a claim to qualify a simultaneous bulk pressure-space remap. An exact
physical transfer follows because the old δ and its Maxwell memory have zero
extension to the new columns. The complete bulk strain/stress field is identical,
not merely its independent coefficients. Bulk solve responses are reusable
because their mesh, constraints and candidate basis do not change.

For I ⊂ I', the transfer copies existing Gauss damage a, dimensionless state
Θ=V0 θ/Dc, bond reference b, jump δ, relaxed jump memory and absolute slip by the
same material lattice location. New records receive

    a=0, δ=b=0, relaxed jump=0, absolute slip=0,
    Θ=Θvirgin(t),  Θvirgin,new=Θvirgin,old+(V0/Dc)Δt.

Here Θvirgin(0)=1; equivalently the dimensional age starts at Dc/V0=200 s and
ages at rest. The same accepted-step clock is retained for uninstalled material,
so different insertion times agree to floating-point identity. This is not a
steady-state reset. No physical time advances during geometry publication.
No new fracture or bond inventory is created; old fracture inventory remains
Gc∫a ds. Bulk tensor memory is preserved by the unchanged native strain response.
This exact reduced representation applies to the retained homogeneous fixed
geometry/load family; it is not a conversion of arbitrary imported DG histories.

## Weak mechanics, admission and tangent

The mechanical shear and current normal pressure are

    s = (G/H) w − K z,      pn = p0 + N z,
    w_new = β(w_old+ΔU),    z_new=β(z_old+Δδ),
    β = 1/(1+G Δt/η).

K and N come from the same exact-jump mixed FE response and actual normal traces.
Let D contain nodal weights, W surface Gauss weights and R=D^-1 Q^T W. On the
interior installed trace rows,

    P_I^T D(s−R_D r_D−R_T λ_T)=0,
    Q_T δ=b_T,   |λ_T|≤τp.

There is no force residual to impose on excluded endpoint/uninstalled jump rows.
The existing unrestricted weak certificate is evaluated on all represented
virtual directions; the one-sided formation predictor does not replace it.
The full pressure derivative remains in the tangent. The fixed QP·N contraction
is now computed once per trial with its original summation order, rather than
recomputed for every test row. A bitwise control checks this arithmetic change.

A negative eigenvalue in a release matrix that moves tied material is not an
admissible instability certificate. Report that value separately, reconstruct
its mode and contributions, and restrict the actual check to admissible jump
variations. Do not discard a negative admissible curvature or change softening
merely because the unrestricted release value is negative.

## Atomic ownership and restart

Construct the new native manager and new trace constraint map together. Existing
records are copied, new records use the explicit birth values above. Exercise a
real swap/rollback of both geometry and the map; native geometry swaps invalidate
derived associations. The same weak certificate is re-evaluated at the frozen accepted state and rate
before birth commits; an inadmissible newly tied space is rejected, not repaired
by inventing slip or fracture work. Validation failure restores both owners. Deterministic retry
must reproduce the staged archive. One geometry-generation commit follows successful
stress/history/energy validation; it does not increment the physical-step counter.
Every mechanical trial carries both physical and geometry generation IDs, so an
old-geometry trial cannot publish after insertion. Physical acceptance retains
its existing once-only publication check. Restart serializes the finite endpoints,
generation, virgin clock, native manager and accepted bulk memories; changed finite
configuration is rejected. These are component-owner tests, not a qualification
of production outer cutback or distributed topology changes.

## Scope of the future path

The tested path is a prescribed collinear candidate, not a free crack direction.
Bulk phi remains provisional geometry/search metadata, with no second energy or
bulk degradation. Surface a alone creates recoverable compliance and carries
Gc∫a; progressively activated contact resistance is multiplied by a. A future
virgin direction must satisfy maximum-shear intact admission plus a stated tie
preference, closed contact, geometry admissibility and this energy-accounted
transfer. Established material normals and tensor histories must persist. This
work does not select an automatic path-generation algorithm, junction rules or
secondary fracture through old records.

Pure representation insertion is deliberately restricted to an unchanged state
with a feasible tied certificate. It cannot install an already overloaded record
and silently pay its fracture cost with a history reset. Future free-path logic
must either prepare prospective tied candidates before loss of admissibility
(metadata is not fracture formation), or solve insertion and irreversible
admission together with the released energy explicitly accounted. Maximum-shear
selection, tie preferences and the distinction between prospective and established
material normals need that design decision; this test does not settle them.
