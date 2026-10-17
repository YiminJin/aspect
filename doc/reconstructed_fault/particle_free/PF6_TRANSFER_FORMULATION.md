# Fixed-reference transfer qualification: rules declared before runs

16 October 2026; base a8707307008401ac9cff702a369e2e2a611fa0f7 plus preserved
incoming work (build-particle-free/transfer-qualification/incoming.json/patch).
This is numerical remapping and ownership qualification, not physical evolution
or general PF5 completion. Unlimited DGQ2, equations, rates, surface mesh, physical
bounds, limiter/default settings and native acceptance criteria are retained.

## Branch comparisons

The publication study's `strict_reference_match_passed=false`, unregistered
errors, original failures and reference are immutable evidence. A reflection is
only a named diagnostic. It cannot pass the strict trajectory comparison.
Native departure compares each candidate against the current best with its
arithmetic energy allowance; the independent reference compares candidates
strictly after a resolved improvement over entry. Opposite branches below that
allowance are not a demonstrated recurrence failure or a unique trajectory.

A transfer starts from one specified accepted physical branch. Compare that
branch before/after remapping at common physical points, with no phase solve,
reflection, centroid alignment, or physical clock/publication increment. MPI
and restart compare the same initialized branch and prescribed accepted steps;
report primary unaligned differences. A later branch divergence must retain
its original error and be diagnosed, not counted as a passing transfer.

## Matching-cell traces

For independent DG history the particle's surrounding CellId owns BOTH the
native FE gradient and the DG value. Map its physical position to that cell's
reference coordinates. An FE velocity is continuous but its gradient is not:
averaging several matching-cell gradients is a different discrete operator.
Check association ID/order/position explicitly. Keep legacy sampling unchanged.
Include a continuous piecewise-Q2 velocity with unequal one-sided gradients on
a shared face; an averaged-gradient negative control must fail the retained
guard. Arithmetic allowances cannot cover this physical trace difference.

## Transfer contract

Use deal.II distributed SolutionTransfer on all three independent DGQ2 fields;
its FE_DGQ restriction is the cell L2 projection, not cell averaging. Retain all
nine coefficients, including x²y². Refinement is polynomial restriction to
children; coarsening is L2 projection and can lose fine information. Report
mean/moment/full-tensor L2, quadratic norm and physical reconstructed-stress and
weak-load defects separately. No Maxwell/source/relaxation update belongs in
transfer. Clear source-roundoff and evaluation caches.

The accepted Q1 physical phase is also its next physical lower obstacle.
Refinement interpolates the identical physical function. This bounded native
coarsening path accepts only representable Q1 phase/obstacle: child-vertex values
must agree with the parent interpolant, within an explicitly recorded arithmetic
screen. Otherwise reject before changing the mesh or accepted state. This avoids
inventing an obstacle projection, healing, phase caps, or artificial growth.
Lossy obstacle coarsening remains a separate design decision. Test a fine peak
that cannot be represented as a rejected negative control.

The reference surface, its vertices/orientation/properties, Z and committed
rates remain fixed. Rebuild mesh-dependent normalization/association caches;
compare I and q=Z/I and weak loads without applying the cohesive recurrence.
Use only uniform Cartesian mesh changes in the native endpoint fixture; local
nonuniform AMR remains guarded by its existing mechanical/cache restrictions.

First serial tests: zero/constants, nine independent monomials with
non-proportional components, exact refinement/coarsening of common global Q2,
lossy fine-cell DG coarsening against independent quadrature, admissible Q1
phase/obstacle roundtrip, inadmissible coarsening rejection, native accepted
snapshot roundtrip and subsequent physical step. Then 1/2/4 ranks: unique owned
CellIds, ghost values, matching-cell traces, rejection/retry and single
publication; same-state checkpoint/restart and uninterrupted comparison.
Roundoff targets are scaled 1e-11 for polynomial transfer and 1e-9 for native
physical observables; report actual absolute/scaled errors and any misses.

Particles stay fixed: require RK2 and presence of
ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION (presence, even empty/0, means dt=0
in its integrator only). Export IDs, positions, CellIds/reference coordinates
at initialization, before/after each physical step, and transfers. Migration
changes rank/cell ownership, never physical position or ID. Compare globally by
ID. Audit reads separately from authoritative DG mechanics; do not remove the
publication comparison or treat particle persistence as particle-free physics.
