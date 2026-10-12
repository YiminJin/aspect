# P2 profile-energy assessment (2026-10-07)

This is an assessment of the selected model, not an adopted phase-law revision.
Recoverable peak H, direct irreversibility, unloaded initialization, surface Z
and the bulk-conjugate mechanical pair remain unchanged. The reduced fault
energy derivation remains valid; it does not supply the variation of the bulk
elastic energy when the diffuse slip profile changes.

## Native step and its mechanical energy identities

Write chi=h/I_h, e=Z/G, S=sym(t tensor n), with S:S=1/2. For the homogeneous
reference material the actual native step is

\[
 Z_1=\beta Z_0+\kappa V,\quad q_1=Z_1/I_1,
 \quad C=\frac{\beta Z_0}{\kappa}(\chi_1-\chi_0),
 \quad\bm\tau_1=\beta\bm\tau_0+2\kappa[\dot{\bm\varepsilon}-(\chi_1 V+C)\bm{\mathcal S}].
\]

Here \(\beta=\exp(-G\,dt/\eta)\), \(\kappa=\eta(1-\beta)\). \(V\) and effective strain rate
are held constant over the native Maxwell integration. With \(\lambda=\eta/G\),
\(\bm A=2\eta[\dot{\bm\varepsilon}-(\chi_1 V+C)\bm{\mathcal S}]\), \(\bm B=\bm\tau_0-\bm A\),

\[
 \bm\tau(t)=\bm A+\bm B\mathrm e^{-t/\lambda},\qquad
 \bm T=\int_0^{dt}\bm \tau(t)dt=\bm A\,dt+\bm B(1-\beta)\lambda.
\]

Using the **same** owned Stokes JxW measure for every bulk term gives

\[
 \Delta E_b+D_b=W_{\rm ext}-W_V-W_C,\quad
 E_b=\sum_a J_aw_a\frac{\bm\tau_a:\bm\tau_a}{4G},\quad
 D_b=\sum_a J_aw_a\int\frac{\bm\tau_a:\bm\tau_a}{2\eta}dt,
\]
\[
 W_V=\sum_a J_aw_a(\bm T_a:\bm{\mathcal S}_a)\chi_{1,a}V_a,\qquad
 W_C=\sum_a J_aw_a(\bm T_a:\bm{\mathcal S}_a)C_a.
\]

The profile work need not vanish. At zero \(\dot{\bm\varepsilon}\) and \(V\), \(W_C=-\Delta E_b-D_b\).
The controlled test confirms this identity; a negative \(W_C\) increases bulk
storage. Reporting \(W_C\) separately does not mean adding it a second time to
bulk storage. A zero unweighted integral of \(C\) would not make stress-weighted
\(W_C\) zero. In addition, projected normalization and Stokes quadrature only
approximately normalize \(\chi\); the continuum zero-integral statement does not
hold identically for that discrete representation.

On the **same native Q3 surface ds measure**, first change \(I\) at fixed \(Z_0\):

\[
 J_\Gamma=\sum_b ds_b\frac{Z_{0,b}^2}{2G}
          (I_{1,b}^{-1}-I_{0,b}^{-1}).
\]

Then integrate \(Z(t)=\eta V+(Z_0-\eta V)\exp(-t/\lambda)\) at fixed \(I_1\):

\[
 \Delta E_\Gamma+D_\Gamma=W_q+J_\Gamma,\quad
 E_\Gamma=\sum_b ds_b Z_b^2/(2G I_b),\quad
 D_\Gamma=\sum_b ds_b\int Z_b^2/(\eta I_b)dt,\quad
 W_q=\sum_b ds_b\int q_b V_b dt.
\]

Friction work is separately \(W_f=\sum ds\mu P V\,dt\). Define the integrated
ordinary exchange defect \(E_x=W_V-W_q-W_f\); native endpoint adjointness does
not force \(E_x=0\) throughout the step. Including fracture expenditure \(\Delta F\),

\[
 W_{\rm ext}-\Delta E_b-D_b-\Delta E_\Gamma-D_\Gamma-W_f-\Delta F
 =W_C-J_\Gamma+E_x-\Delta F.
\]

This is a **mechanical-path accounting residual**, not a corrected total energy
law. Native-QP energies do not telescope across the stress representation
reset. Separately retain T_FE=sum(E_old_FE,k-E_native_QP,k-1). Adding these
representation changes to both the energy ledger and its state transitions
leaves the above residual unchanged. The direct particle-interpolated QP energy
minus native-QP energy is a different comparison and is not T_FE. Authoritative
particle-volume storage is another measure; do not substitute it into a
Stokes-QP identity.

The retained trajectories give W_C-J_Gamma+E_x-Delta F =
-97.8880, -130.0607, -153.7990 J/m for dt=.125/.0625/.03125. W_C dominates;
ordinary exchange defects decrease -4.0722 to -2.1839 J/m. This is not evidence
that endpoint mechanical convergence closes the fracture-energy budget.

## Required phase variation if total recoverable energy is the potential

The native **elastic-limit** profile update at fixed e, displacement and
already committed viscous/internal strains is

\[
 \delta\tau=-2G e\,\delta\chi\,S=-2Z\,\delta\chi\,S.
\]

Consequently the bulk variation implied by that kinematics is

\[
 \boxed{\delta E_b=-\sum_a JxW_a\frac{Z_a}{G}
                       (\tau_a:S_a)\delta\chi_a},\qquad
 \delta\chi=\delta h/I_h-h\,\delta I_h/I_h^2.
\]

The reduced fault variation at fixed Z is independently

\[
 \delta E_\Gamma=-\sum_b ds_b\frac{q_b^2}{2G}\delta I_{h,b}.
\]

For continuous direct normalization delta I=int h'(phi) delta phi dn.
That fault term gives energy variation +Y g'(phi), or positive driving
force -Y g'(phi), with Y=q^2/(2G g^2). For consistent Q1
normalization the exact discrete expression instead uses
**delta I_h=P(delta I)** and its adjoint. This projection issue remains distinct
from delta E_b. The source h/I accuracy error is also a distinct quantity.

The bulk term vanishes under special conditions: uniform transverse shear,
exact normalized transverse measure, and constant Z along each transverse
profile give int delta chi dn=0. The actual nonuniform stress field does not
satisfy that cancellation condition in general. Mechanical equilibrium does
not establish it: at fixed boundary displacement, the envelope theorem removes
the first-order equilibrium displacement variation, not the explicit profile
variation in E_b. Friction/dissipation are not recoverable-energy substitutes.

Independent centered differences of both energies at fixed Q1 Z verify these
formulas. For the accepted fine-path step31-to32 phase direction, with the
controlled representable shear tau_xy=1500+8000|y-.5| Pa, spacing .025 m:

- delta E_b = +0.6589375 J/m;
- delta E_Gamma = -0.7551587 J/m;
- their sum = -0.0962211 J/m before fracture/history terms;
- Q3/Q8 bulk integration difference = 5.96e-6 J/m.

The omitted bulk variation is 87.26% of the reduced fault variation and opposes
it. Uniform-shear control gives only -1.01e-5 J/m at this spacing. These stress
fields are named frozen diagnostic controls, not reconstructions of the actual
coupled stress field. The retained physical trajectories' work ledger provides
separate evidence that profile exchange is substantial. The test does not
claim a new coupled variational solution.

## Time interpretation and governing decision

Instantaneous elastic redistribution tau*=tau_0-2Z_0(chi_1-chi_0)S followed by
Maxwell relaxation has the **same zero-input endpoint** beta tau*, but different
work/dissipation from spreading C over the native step. For the controlled
initial-to-final profile change (holding Z at its accepted step32 value, not
replaying the unloaded physical initial history), the .025 m gradient-stress case has elastic
bulk jump +140.9202 J/m. Native W_C at the three dt is
-140.6866/-140.8034/-140.8618 J/m; the difference in native versus instantaneous
bulk dissipation shrinks -.23357/-.11683/-.05843 J/m. Thus there is an O(dt)
time interpretation difference, but the nonzero profile energy is O(1).
The forced profile requires a conjugate generalized work input; it is not a
zero-load physical phase-equilibrium experiment.

A closed total-recoverable-energy formulation would need to derive the phase
condition from

\[
 \delta F-\sum_a JxW_a(Z_a/G)(\tau_a:S_a)\delta\chi_a
          -\sum_b ds_b(q_b^2/(2G))\delta I_{h,b}=0
\]

on free phase DoFs, with the corresponding lower/upper KKT signs. This equation
is a **candidate for review**, not the implemented condition. The selected
CPDI equation currently uses the recoverable **fault** peak H and fracture F
alone. Replacing instantaneous Y by a peak H already introduces a history-based
dissipation choice; the additional signed, nonlocal bulk term cannot simply be
accumulated into H or added as a compensation factor. A governing revision
would affect the phase residual/Jacobian, normalization-adjoint terms, retained
internal variables during phase trials, and the phase/mechanics energy and
history-publication interpretation. Mechanical blocks need changes only if
the kinematics or mechanical time integration are also revised.

Alternatively, retaining a phenomenological reduced-fault driver requires
explicitly defining the source of evolving-profile generalized work and
abandoning a claim of closed total fracture-energy balance. Choosing between
these interpretations is a scientific-model decision. **No correction is
implemented here. No P3 readiness is established.**
