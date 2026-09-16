# Condensed mechanical timing and pivoted surface inverse

This pass implements the user's approved K5 mechanical-performance task, not
the pending I_h boundary fix. The existing I_h defaults/prototype, constitutive
law, constraints, active-set rule, residual criteria and iteration budgets stay
unchanged. Neither K_V nor the condensed operator is assumed SPD.

## Boundaries and measurement

One opt-in rank-local `FaultLinearProfile` begins before surface linearization
and ends after active-set solves and delta-V recovery. Exclusive nested sections
charge factorization, surface RHS solves, A, B, G, Stokes preconditioner, FGMRES
Arnoldi/vector work, and remaining setup/constraint/vector work separately.
The latter also contains surface assembly, G lookup construction, B freezing,
nullspace/fresh-check orchestration and vector conversion. Bulk assembly and
preconditioner construction precede this profile and retain ASPECT's ordinary
timers. The exclusive components sum to the measured profile duration.

The installed deal.II FGMRES exposes no timing hook around Arnoldi. Measure its
solve scope excluding the wrapped operator and preconditioner applications;
the result is **Arnoldi/orthogonalization plus remaining FGMRES vector and
small projected-system work**, not an assertion that every remainder is a dot
product. Do not fork deal.II or change the Krylov algorithm for instrumentation.
Timing adds no MPI collectives; root output describes root-rank elapsed times.

## Direct inverse

The semantic `ReconstructedFaultSurfaceLinearSolve::solve` API remains intact.
A file-local subsystem implementation shared with focused tests owns original
tridiagonal coefficients, the free partition, and reusable direct factors.
The prototype calls the installed Teuchos LAPACK GTTRF/GTTRS wrappers (the
current ASPECT/deal.II Trilinos configuration). It does not implement Thomas,
Cholesky or a custom pivoting algorithm. GTTRF's second superdiagonal and pivot
permutation are retained for each solve. Replicated factors need no MPI solve
communication. Active entries are ignored in the RHS and returned as exact zero.

`ASPECT_FAULT_SURFACE_SOLVER=tridiagonal` selects the prototype;
`umfpack` or an unset variable retains the reference. The new path requires
the installed Trilinos LAPACK wrapper. Full candidate publication and stale
linearization rejection stay unchanged. Original-coefficient backward residuals
are checked for each free block with its own scale, retaining the existing
100*epsilon*n_fault coefficient. Finite input/factor/solution checks and LAPACK
singular-pivot diagnostics identify fault, block or vertex. No constitutive or
linear-solver tolerance is relaxed.

`ASPECT_FAULT_COMPARE_SURFACE_INVERSE=1` additionally factors UMFPACK and
compares every RHS solve at relative 2e-12. This extra work is for numerical
verification, not counted as optimized performance. Tests independently compare
complete condensed actions and recovered delta-V using both semantic inverses
on the same production A/B/G/K_V generation and several active partitions.

## Bounded sequence

1. Time the existing coarse three-real-step BP3 and evolving K3 fixtures.
2. Compare SPD, pivot-requiring indefinite, singular, isolated/endpoint/all-active
   blocks and multiple RHS on one/two ranks; run affected coupled/lifecycle tests.
3. Replay the same production problems with the selected pivoted backend and
   fresh checks, retaining numerical state comparisons and rollback evidence.
4. If inversion is negligible and bulk FGMRES/preconditioning dominates, stop
   with a separate interface-aware preconditioner design. No fine BP3 pilot,
   new I_h algorithm, outer CG substitution, or preconditioner implementation.
