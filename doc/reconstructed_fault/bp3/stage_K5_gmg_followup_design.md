# Separate proposal: reconstructed-fault coupling with the existing GMG path

Design only. This performance stage does **not** implement GMG or matrix-free
fault support. It does not change the accepted block-AMG path, surface inverse,
physics, quadrature, residual criteria, or history lifecycle.

## Decision addressed

Sparse B repays setup on both short trajectories. Sparse G only narrowly
repays setup at the measured application counts. The corrected two-mode
preconditioner does not repay its probes: BP3 retains exactly 502 iterations
and adds 58 setup applications; K3 changes 558 to 559 iterations and adds 74.
The base Stokes preconditioner remains dominant. The separate next proposal
therefore targets its application cost, not a larger dense fault Schur
complement or outer CG. This does not prove all interface spaces ineffective,
only that this bounded candidate fails the cost criterion. See
`stage_K5_coupling_report.md`; the earlier prototype's stale-pressure setup
bug and its preserved failed fresh checks are not performance evidence for
the corrected algorithm.

## Reuse and bounded architectural change

1. Add a solver-side fine-grid Stokes action/preconditioner capability to the
   existing matrix-free handler, referencing its canonical objects. Start with
   the existing global-coarsening implementation, not a second multigrid stack.
   Reuse its `stokes_matrix` vector initialization and
   `internal::ChangeVectorTypes::copy` adapters. Check fine-grid numbering and
   partitioning explicitly; the presence of this copy utility is not proof
   that arbitrary velocity/pressure hierarchy DoFHandlers share system indices.

2. Preserve B's bulk Stokes-QP and G's parent/domain discretizations. Initially
   keep the qualified sparse coupling and its reference actions on the fine
   grid, adapting vectors at the existing solver boundary. Do not move G onto
   bulk QPs, impose B transpose, or assemble fault operators on every MG level.
   A reusable numbering/ownership map, if needed, belongs to this solver
   adapter and is invalidated with mesh/DoF layout changes. Measure adapter
   copies/ghost exchanges separately: they can erase the expected gain.

3. Keep the current nonlinear orchestrator and semantic surface solve. First
   test the GMG base preconditioner with the current assembled A: A's matvec is
   not the measured dominant cost. This isolates preconditioning benefit from
   any fine-operator conversion. Move A to its matrix-free action only after
   the independent fine-operator agreement checks below.
   Outer FGMRES still applies A-B K_FF^-1 G; all active-set masks, homogeneous
   perturbation constraints, solver-to-physical pressure conversion, verified
   pressure quotient, fresh residual checks and rollback stay at their current
   semantic boundaries. Do not enter a separate GMG nonlinear solver that
   omits these safeguards. Bulk residual evaluation remains independently
   assembled, including the separate frozen Maxwell/profile accumulator.

## Qualification before trajectory/performance claims

On one frozen K3 and one BP3 linearization, compare assembled A with native
matrix-free A for basis/random/production directions, using identical viscosity
coefficients, pressure scaling and quadrature. Verify constrained entries,
periodic/hanging-node relations and both pressure branches. Check current GMG
coefficient projection/averaging against the assembled fine-grid definition:
if it changes the fine operator rather than just the preconditioner, stop for
review. This is not permission to change the discrete problem for performance.

Compare complete C, RHS, recovered delta-V and fresh full/projected residuals
on one/two ranks. Only then use the existing short K3/BP3 trajectories and
rollback fixture, with unchanged physics and solver tolerances. Report setup,
base applications, transfers/ghost communication, coupling, memory and total
coupled time. Accept an extension only if setup plus the actual short-trajectory
application count improves materially; a cheaper individual V-cycle alone is
not sufficient. No fine BP3 pilot or propagation is implied by this proposal.
