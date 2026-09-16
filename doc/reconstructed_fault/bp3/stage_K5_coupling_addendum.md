# Approved K5 coupling/preconditioner prototype

This stage preserves A, B, G, K, constraints, active sets, equations,
quadrature, histories, pressure handling, and stopping rules. The qualified
adjacent-pivoted tridiagonal inverse is preferred; UMFPACK remains selectable
as an independent reference. Outer FGMRES and fresh residual checks remain.

## Explicit coupling

Opt-in `ASPECT_FAULT_EXPLICIT_B` and `ASPECT_FAULT_EXPLICIT_G` assemble sparse
rectangular operators once per coupled linearization. They retain the original
actions; `ASPECT_FAULT_COMPARE_COUPLING` compares every invocation (including
Krylov, nullspace, RHS and recovery) with those independent actions at 2e-11.
Nothing is promoted based on matvec time alone.

B uses exactly the frozen 2 kappa chi S bulk-QP coefficients and original
Q1 weights. Closed homogeneous constraints expand test rows into masters;
Dirichlet rows vanish. Rank-local sparse rows are additive pieces of the
distributed matrix, completed by vector MPI ADD on application.

G integrates frozen domain test weights/constitutive coefficients by parent.
Its RPE map distributes these to actual bulk-cell owners, divided by the number
of matching cells to retain the original `avg` shared-face convention. Current
FE basis gradients and pressure values build physical full-system columns.
Application reads the same constrained/scaled ghosted physical direction and
MPI-sums the small surface vector. It is not B transpose. Parent-P0 sampling,
open endpoints and all nonlinear domain quadrature are unchanged.

Sparse storage contains only nonempty rows (CSR). It dies with the corresponding
linearization. Both reference geometry/coefficient paths remain available.
Timers distinguish setup, reference actions and sparse applications. Memory
reports include retained sparse arrays; process RSS also includes setup maps.

## Few-mode preconditioner

Opt-in `ASPECT_FAULT_INTERFACE_MODES=1..4` selects constant plus low-order cosine
modes on each contiguous free block, with zero active/prescribed entries.
The prototype caps the total at eight modes, in fault/block order; it does not
probe every vertex. Modes are Euclidean orthonormal in vertex-index coordinates.
This is algebraic preconditioning, not a surface discretization change.

With Q the modes and P_A the unchanged pressure-complement block-AMG Stokes
preconditioner, form Y=P_A BQ and T=Q^T K Q-Q^T GY. Factor the small generally
nonsymmetric T with pivoted LAPACK LU. Apply z=P_A r, then z+Y T^-1 Q^T Gz.
Build once per free-set solve; re-entering after an active-set change rebuilds
the correction. A singular coarse approximation disables the correction,
not the physical surface solve. All accepted directions still pass the
unchanged fresh full/projected residual and compatibility checks.

The existing pressure Schur inverse's zero-RHS fast branch does not write its
destination. Therefore initialize every setup response to zero, not a copy of
the nonlinear RHS used only as a vector-layout template, and clear reused
interface-application destinations before invoking the base. This caller-side
contract is essential because BQ has identically zero pressure RHS. The first
BP3 prototype violated it; its saved failed fresh attempts are diagnostic
evidence, not a valid preconditioner performance comparison. The corrected
prototype optionally verifies exact zero setup-response pressure with
`ASPECT_FAULT_VERIFY_INTERFACE`, without changing the general Stokes solver.

Measure setup probes separately from outer applications. Accept only a lower
total preparation-plus-application cost on the short BP3 and evolving K3
trajectories. No GMG/matrix-free implementation is part of this stage.
