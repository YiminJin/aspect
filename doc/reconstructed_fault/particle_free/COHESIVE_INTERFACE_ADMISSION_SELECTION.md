# Event-resolved prescribed oblique admission: bounded selection

Reuse the qualified 16×8 fitted Q2/P1disc native kernel and prescribed 26.565°
append. No free-path selection, corner opening, pressure-space remap, production
solver change or new spatial campaign. The corner remains closed with zero vector
jump. Phi is provisional geometry; surface a counts fracture energy once.

Fresh histories on the horizontal old segment are intact, tied and at rest.
Prepare a homogeneous axial prestress with gamma=.0015 and physical time .2 s;
this is an explicitly prepared elastic state, not a reinterpretation of a prior
checkpoint. Start dimensionless age at 1.001 everywhere (200.2 s). The full
Maxwell memory represents this prestress. Insert the prospective append at .25 s
while its weak tied certificate is below 3000 Pa. Compare against a candidate
installed at .2 s, retaining the same material clock and physical history.

Apply u_boundary=(gamma*x,−gamma*y), gamma_dot=.001/s through .65 s, then hold
through .66 s. Use dt=.005 and one halving (.0025); the loading/hold switch is
exact. This reaches bounded damage beyond the analytically predicted tied loss
near gamma_memory=.001875. Stop early on lost contact, failure of the existing
weak/residual/history checks or negative admissible mechanical symmetric tangent;
save any rejected trial. Do not change the law to finish the interval.

Locate the first loss of the *discrete* tied certificate within a timestep by
bisection from the immutable incoming state, to a bracket ≤1e−8 s. Accept the
lower tied endpoint and solve the remaining physical interval with unrestricted
signed slip and the unchanged cohesive/RSF return. This bracket is numerical
resolution of a discrete event, not spatially qualified physical activation time.

Q2 trace functions change sign. Do not reuse Q1 nodal positivity or assume that
partial Gauss ties constrain entire cells. Use the general nullspace of tied
Gauss equalities and a bounded-multiplier certificate on every represented test
row. Tied records have exact zero jump by constraint, not a phase floor. Full
current pressure and both compressive side tractions remain in the solve.

Keep the established 1e−8 Pa weak residual, 1e−6 coupled derivative and 1e−10
scaled BE ledger checks. Compare onset brackets, damage relative to fresh a=0,
integrated/peak slip and added fracture at common times. Target <2% timestep
change in integrated damage/slip after established growth; report tiny-onset
relative differences separately. Screen accumulated absolute trapezoidal-work
remainder against 1% of new fracture and sliding loss separately; do not replace
physical energy accuracy by BE closure. No temporal sweep beyond this halving.

Check immutable repeated trials, rejected publication, deterministic retry,
once-only publication and checkpoint continuation across activation. Reuse the
native property/checkpoint adapter where the reference passes. Preserve the old
65.4% early straight-growth spatial failure and the kink's pointwise sensitivity.
Deliver one concise report, PNG/GIF and indexed CSV/JSON evidence.

## Implemented equations and numerical scope

Let c=(w,z) contain the relaxed affine coefficient and native jump coefficients.
Every physical trial uses c_new=beta(c_old+(Delta gamma,Delta delta)), with
beta=1/(1+G dt/eta). Current sided pressures are p0+N_minus c and p0+N_plus c;
the return uses their mean while contact is checked on each side. K is the
retained complete mixed Schur response, not a fixed-pressure stiffness.

At a previously intact Gauss point, split the return as
r=peak sign(Delta delta_q)+h(Delta delta_q,pn). Solve the signed nonsmooth part
using z=Q Delta delta/delta_f, y=lambda/peak and

    z = shrink(z+y,1),
    F_bulk + Q^T W (h+lambda) = 0.

This is a semismooth *solver* representation of the existing law; it adds no
physical length, viscosity or resistance. The initial multiplier estimate may
be least-squares when a Newton predictor is unequilibrated. Only the final full
bounded-multiplier certificate determines acceptance.

For the final tied set T, construct C=null(Q_T). Polish with C^T F and C^T J C,
then recover multipliers satisfying

    F_D + Q_T^T W_T lambda_T = 0, |lambda_T| <= peak.

C acts on an already mechanically condensed operator; it only removes the
surface tie constraints. The complete J includes the phase/age return derivative
and the current pressure partial r_p beta N. Numerical remnants of analytically
zero ties are bounded by floating-point matrix-product error, scaled to the
computed jump. No fixed phase cutoff is used. The actual evolution has complete
old-branch ties; a separate algebraic check exercises a partial Gauss tie.

Report E_bulk=c^T K c/2, bond storage, Gc Delta a, friction/radiation loss,
Maxwell loss 2G dt E_new/eta, the BE bulk remainder and
q Delta delta−Delta bond−Gc Delta a separately. External endpoint and trapezoidal
reaction–strain work are distinct. The 1% sliding-normalized physical-work screen
is retained even when most of the global remainder originates in the bulk.

The prepared axial state is a prescribed-plane control. No other candidate
orientations are enabled; an unrepresented maximum-shear plane can reach its
strength earlier. It must not be interpreted as a free-path nucleation specimen.
