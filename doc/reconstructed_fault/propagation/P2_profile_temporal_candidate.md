# Proposed restricted endpoint-consistent prototype

This is a diagnostic candidate, not an adopted native algorithm. It retains the
current instantaneous recoverable energy, Maxwell/friction laws, spatial
operators, and previous-physical-timestep irreversibility. The implemented native
and original prototype still solve phase with incoming history. The new candidate
checks whether their persistent endpoint phase defect can be removed by a
consistent within-step solve. It does not reinterpret an algorithmic potential
as physical work or add phase dissipation.

## Equations before implementation

For the compatible one-dimensional reduction, let the immutable accepted state
be (phi_a, R_a, Z_a, delta_a), where R_a is the existing Q2 FE coefficient vector.
Let D be its shear derivative matrix, W the shared six-point bulk weights,
I(phi) the analytic transverse integral (the constant consistent surface
projection in this uniform test), and chi(phi)=h(B phi)/I(phi). F(phi) is the
unchanged CPDI fracture energy. The reduced integrals include the fixture’s unit
tangential width (1 m); that factor is implicit in the formulas below. Keep beta=exp(−dt G/eta), kappa=eta(1−beta).

At every trial (phi,u,V), recompute from that same accepted state:

\[
 R_b=\beta R_a+(\kappa/G)u,\quad Z_b=\beta Z_a+\kappa V,\quad
 \tau_b=G D R_b-Z_b\chi(\phi),\quad q_b=Z_b/I(\phi).
\]

The unknowns must satisfy all three endpoint conditions:

\[
 D_I^T W\tau_b=0,\qquad
 0\le V\ \perp\ \mu P+q_b-\chi^T W\tau_b\ge0,
\]
\[
 \phi_a\le\phi\le1,\qquad
 0\in \partial_\phi E(\phi,R_b,Z_b)|_{R_b,Z_b\ \mathrm{fixed}}
       +N_{[\phi_a,1]}(\phi).
\]

At a lower bound the energy derivative is nonnegative; in the interior it
vanishes. The derivative includes the existing exact normalization adjoint.
Boundary u uses the identical interval-mean ramp. The phase lower bound stays
phi_a throughout all inner iterations, never a previous inner iterate.

For this fixture only, mu P is a fixed constant. These equations are stationary
conditions of the algorithmic endpoint potential

\[
 J(\phi,u,V)=E(\phi,R_b(u),Z_b(V))+(\kappa/G)\mu P V,
\]
\[
 E=\tfrac1{2G}\tau_b^T W\tau_b+\tfrac{Z_b^2}{2G I(\phi)}+F(\phi).
\]

Indeed J_u=(kappa/G) D^T W tau_b, J_V=(kappa/G)(mu P+q_b−chi^T W tau_b),
and J_phi equals the fixed-endpoint-state energy gradient. No factor replaces
the physical friction expenditure dt mu P V. J is an algorithmic potential,
not time-integrated external work, and this derivation does not apply unchanged
to evolving RSF or normal-pressure laws. The full E, including its
phase-dependent history terms, must be retained; varying the legacy mechanical
rate potential alone would give the wrong phase equation.

Mechanically minimize J at each phase trial using the existing endpoint solve.
The reduced gradient is J_phi by the envelope theorem. Its exact reduced Hessian
is H_phi_phi−H_phi_m H_mm^{-1} H_m_phi on the current mechanical contact set.
This Schur term is essential; the fixed-history phase Hessian alone is not the
coupled tangent. Use the existing bounded local phase minimization/Newton polish
with this reduced functional and unchanged KKT target. Negative reduced
curvature or a large jump remains a separate result, not permission to add
kinetics, adjust bounds, or assert global stability.

All evaluations use scratch mechanical state. Reset R=R_a, Z=Z_a and
slip=delta_a before every mechanics call, regardless of the previous trial.
After convergence, reevaluate the final trial and validate phase KKT,
mechanical/contact conditions and histories. Publish R_b,Z_b,phi and
slip=delta_a+dt V only once if the complete step succeeds. Rejected trials do
not advance accepted history. The first implementation is a saved-entry replay,
not a silently altered fresh physical trajectory.

## Energy accounting that accompanies the candidate

Retain the existing path: instantaneous profile redistribution at incoming R_a,Z_a,
then fixed-profile, constant-rate Maxwell evolution with exact general beta.
Define the **signed** profile energy change

\[
 \Delta E_{profile}=E(\phi,R_a,Z_a)-E(\phi_a,R_a,Z_a).
\]

Solving an endpoint phase condition does not guarantee that this quantity is
nonpositive at the incoming state. Therefore its negative cannot automatically
be called positive relaxation dissipation. Record both signs. No compensating
loss is introduced.

Integrate the unchanged bulk/cohesive viscous losses and friction dt mu P V
along that exact path. Retain bulk-rate work W_bulk, each lifting integral,
and constrained FE boundary-reaction work W_reaction separately. The diagnostic
ledger is

\[
 \mathcal R_{ext}=W_{reaction}-[E_b^{end}+E_c^{end}+F^{end}
       -E_a-\Delta E_{profile}+D_b+D_c+D_f].
\]

Also report W_bulk−W_reaction and intermediate mechanical imbalance. A zero
endpoint KKT and force residual does not enforce equilibrium along the entire
Maxwell path or establish a nonnegative profile release. A finite positive
profile cost, large jump, indefinite coupled tangent, or persistent external
work remainder blocks native adoption and requires interpretation of that
specific issue. Prototype derivative and replay checks precede any longer
coupled family; no native algorithm changes are authorized by this document.
