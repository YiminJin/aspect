# Bounded within-step Theta A/B diagnostic

The production split equations in `current_design.md` and `specification.tex`
remain authoritative and unchanged. This user-authorized noncommitting
experiment changes only the state entering one mechanical solve. Use the
revised-work accepted step-9 checkpoint, after verifying its clock and retained
state: mechanics 10, ending at 29.24190894 yr, with its original real dt.
Copies of the identical checkpoint supply A and B. No diagnostic result is
published as a timestep or fed into later history.

A uses committed nodal Theta^n. B computes, for every candidate nodal rate,

    x_i = V_i dt / Dc
    T_i(V_i) = Theta_i^n exp(-x_i) - Dc/V_i expm1(-x_i)
    Theta_q = sum_i N_i(q) T_i(V_i).

The same incoming Theta^n is read on every evaluation; no iteration advances
state. Use the existing aging routine and unchanged interpolation, including
prescribed nodes. G uses the resulting mu with V fixed. The exact extra K term
is

    K_ij = sum_q J chi N_i N_j
           [2 kappa chi S:S + sigma mu_V + eta_d + sigma mu_Theta T'_j].

T'_j is the derivative of that same exact aging expression. Stable small-x
evaluation uses the Taylor limit of ((1+x)exp(-x)-1)/x^2. Because T'_j depends
on the trial column, K is generally **nonsymmetric tridiagonal**. Retain separate
upper/lower entries; use the existing adjacent-pivoting LU and backward checks,
which do not require symmetry or definiteness. The default fixed-state assembly
continues to use identical off-diagonals. No B, bulk/history, pressure, support,
boundary or solver-tolerance change is justified by this diagnostic.

Verify the aging derivative, the nonsymmetric inverse and the complete surface
directional derivative, including neighbouring-node directions. Retain the
existing fresh-linear and nonlinear checks and require full exception rollback
of bulk, current/committed V, particle properties/positions and surface data.
Compare A to saved mechanics 10 before interpreting B. Export actual work-QP
terms on the two elements incident to 39.95 km and sum each element's driving,
friction, damping and residual with the exact row test weight.

One A and one B solve are intended (300/600-s mechanical process caps); failure
is preserved without tolerance/budget retuning or a continuation trajectory.
