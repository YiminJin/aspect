# Selected coupling repair (2026-10-07)

The incoming recoverable-energy result and all 981 manifest artifacts were
copied and hash-verified under `build-propagation-integration/coupling-baseline`.
This task retains recoverable H, unloaded initialization and all material inputs.

## Surface history, before work-measure changes

The constitutive internal variable is Z=I q, with elastic slip Z/G. Select the
optional `Cohesive history representation = surface elastic slip`. Store nodal
q and previous I in the existing property pool; interpret Z_h=sum N_i I_i q_i
and q(s)=Z_h(s)/I_h(s). Thus neither the product of two interpolated Q1 fields
nor a particle projection defines the internal variable. This is an explicit
discrete representation decision; the default remains `particle projection`.

For homogeneous G and temperature-independent homogeneous viscosity, beta and
kappa are constant on the fault. Then Z_new=beta Z_old+kappa V is exactly Q1.
Advance the nodes, divide by current nodal I for storage, and use the identical
Z_h/I_h in trial residuals and accepted H. At fixed phase and zero input this
preserves EVERY representable field in the elastic limit and gives exact
exp(-Gt/eta) relaxation for arbitrary partitions. Nonconstant coefficients
would destroy closure in Q1; this option explicitly rejects that case rather
than silently nodally approximating it. No generic projector is changed.

The slip-strain law stays v=chi V+C, chi=h/I, with
C=beta Z_old/kappa (h/I-h_old/I_old). It contains changes in both profile and
normalization. Its V derivative remains chi, and dq/dV=kappa/I. Thus the bulk
residual, surface residual and their linearization retain the same trial law;
terminal publication now preserves it. Stored histories are never modified
by nonlinear trials. The selected representation has a restart identity marker.

## Work conjugacy to audit next

At a frozen phase/history state, define L at bulk Stokes quadrature points by
(L V)_a=chi_a sum_i N_i(s_a)V_i S_a. Its adjoint in the SAME JxW inner product
is (L* tau)_i=sum_a JxW_a chi_a N_i(s_a)(tau_a:S_a).
Then sum_a JxW tau:L V equals sum_i V_i(L* tau)_i identically.
The bulk tangent is -2 kappa_bulk L; the surface shear derivative uses L*
acting on the same stress derivative. A particle-volume surface measure is
not this adjoint. Particle-to-FE stress transfer is a different operation and
must be measured separately from this identity. Surface resistance has its
own surface integration measure; multiplying it by an approximately normalized
bulk chi quadrature additionally weights its energy. The existing mature-only
alternative therefore cannot be adopted by merely removing its guard.

Endpoint-in-time mechanical equilibrium is not time-integrated energy balance.
For constant effective strain and V over a step, integrate both Maxwell stress
paths analytically when reporting work. Changing phase additionally changes
stored energy at fixed Z and introduces C work. The exact discrete phase-energy
projection discrepancy remains separate from mechanical work conjugacy.

## Selected cohesive mechanical pair

`Fault mechanical work measure = bulk conjugate` selects

R_u(w) = sum_a JxW_a tau_a:eps(w)_a (plus unchanged pressure/body terms),

R_i = sum_a JxW_a chi_a N_i (tau_a:S_a + background_shear)
      - sum_b ds_b N_i (q_b + mu_b P_b + damping_b V_b).

Here tau=beta tau_old + 2 kappa_bulk [eps(u) - (chi V+C) S].
The bulk source and B already implement L; retain that expression and feed it
the repaired common material history. Replace the surface shear residual and G
by its Stokes-QP adjoint. K contains sum JxW 2 kappa_bulk chi^2 N_i N_j S:S
plus sum ds N_i N_j (kappa_surface/I + P mu' + damping).
The chosen surface mass is the ordinary ds Q1 mass. Use the same three-point
Gauss surface quadrature in resistance R and K. This is a cohesive-specific
option, not an extension by deleting the frozen-mature guard. It currently
requires one straight 2-D fault, uniform surface coefficients, and prescribed
friction pressure. The old mature diagnostic path is unchanged.

Current and previous phase at Stokes quadrature are sampled separately; unlike
the mature shortcut they are not assumed equal. Both bulk and surface use the
same common C and the same frozen material/history state. A restart marker
prevents switching work measure with already committed histories. General
true-normal friction and curved/multiple faults are outside this control.
