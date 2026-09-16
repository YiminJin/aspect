# Separate proposal: interface-aware condensed preconditioning

Status: design only, not implemented or authorized by the surface-LU task.
No outer CG/MINRES substitution, tolerance change, new physics or I_h work.

## Question and measured motivation

Is the present Stokes preconditioner expensive because it omits a small number
of important fault-slip coupling modes, or does its bulk application cost
dominate even after those modes are represented? The baseline coarse BP3
trajectory uses 502 preconditioner applications (155.38 s), 81.70 s in B and
32.97 s in G, but only 0.073 s in surface factorization/inversion. Surface LU
acceleration cannot materially change this balance. End-to-end benefit, not
fewer Krylov iterations alone, is the decision metric.

## Candidate and invariant

Let C=A-B K_FF^{-1} G on the existing homogeneous bulk constraints and free
surface space. With exact inverses, Woodbury gives

    C^{-1} = A^{-1} + A^{-1} B (K_FF-G A^{-1} B)^{-1} G A^{-1}.

This sign follows the existing [A,-B; G,-K_V] system. Neither the interface
matrix nor C is assumed SPD. Use the current ASPECT block-Schur application
P_A for the approximate bulk inverse; keep outer FGMRES and every fresh
residual/pressure-compatibility check unchanged.

A bounded first prototype could retain a very small, free-surface coarse
basis Q, precompute Y=P_A B Q, and factor

    T_c = Q^T K_FF Q - Q^T G Y

with pivoted nonsymmetric LU. Apply

    z=P_A r,
    P_C r = z + Y T_c^{-1} Q^T G z.

All B/G/constraint/scaling operations use the existing solver-side semantic
actions. No constitutive evaluation belongs in the preconditioner. Q has zero
active/prescribed entries and no connectivity across disconnected free blocks
or independent open endpoints. Coarse normalization must be documented (for
example the existing surface mass inner product) so units/scaling are explicit;
it does not alter the physical fault discretization.

The current P_A may vary with inner iterative convergence; FGMRES already
permits this. Freeze Y and T_c for one coupled linearization. Rebuild after
A/B/G/K or active/free changes, never silently reuse stale factors. Apply the
same verified pressure-complement projection as the present operator and
preconditioner; do not shift physical pressure in true-normal-stress cases.

If the approximate coarse interface is singular or produces an unusable
correction, diagnose it and fall back to the unchanged P_A for that
linearization. Do not regularize the physical K_V or suppress fresh residual
checks to make a preconditioner usable.

## Minimum discriminating experiment, subject to approval

1. Use a bounded replay of the existing coarse BP3 and K3 fixtures to expose one
   frozen linearization each (the full operators are not checkpointed by this
   timing task). Measure
   velocity/pressure inner iteration cost using existing ASPECT statistics.
   Compare the distribution of slow residual modes with their G trace. If the
   slow modes have little surface trace, stop this interface proposal and
   investigate the bulk preconditioner separately.
2. Start with at most a few smooth free-fault modes (e.g. constant/linear on
   each connected free region), not one inverse probe per vertex. Compare
   unchanged C and RHS with/without the correction, requiring the same fresh
   final residual. Measure setup, applications, iterations, memory and total.
3. Only if the frozen-linearization cost improves, run the existing short K3
   and coarse BP3 trajectories with rollback/active-set checks. No fine pilot
   or new convergence campaign is needed to decide this preconditioner.

## Cost guard and stopping decision

One BP3 inverse probe costs roughly a B, P_A and G action, about 0.5 s in the
baseline; 16 probes per Newton linearization would consume about 8 s before
any Krylov iteration. That is already comparable to the current entire solve,
so a dense per-vertex or casually large coarse-interface construction is not
justified. Four probes would need a substantial iteration reduction to repay
their roughly 2 s setup. Estimate this break-even point before implementation.
Owned Y storage is 8*n_bulk_unknowns*n_modes bytes per rank, in addition to
the ordinary Stokes preconditioner; record actual partition sizes.

Accept a candidate only if setup plus solve improves materially on both chosen
linearizations, then on the short trajectories, without losing robustness.
Stop if added B/G work or rebuilding dominates, if modes are ineffective, or
if success would require a solver/pressure/active-set criterion change. Do not
grow this into a general preconditioning framework merely to obtain a pass.
