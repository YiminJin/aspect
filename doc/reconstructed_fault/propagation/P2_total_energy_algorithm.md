# One total-energy candidate step (2026-10-08)

The subsequent [evolving endpoint comparison](P2_total_energy_endpoint_comparison.md)
tests the previously missing family on identical corrected spatial measures.
The energy, accepted lower bound, immutable internal-variable anchor and exact
Maxwell kernels below are common to both mechanical schemes. Endpoint mechanics
is now the preferred implementation direction; the averaged equations below
remain the diagnostic alternative and describe the original proposal. The
evolving comparison, rather than frozen work error, determines native readiness.

For the endpoint variant replace `(r, eta_bar)` in the mechanical rate potential,
both residual blocks and their common Hessian by `(beta, kappa)`. Enforce the
free bulk and slip conditions on tau_new and q_new, with the same new profile.
The solved rates are endpoint-force approximations held fixed over the interval;
prescribed affine velocity uses its exact interval mean to preserve displacement.
Work and viscous losses still use **exact time integrals**, not endpoint stress
times dt. Record the additional mechanical discrepancy independently of D_relax:

\[
 E_{\rm mech}=W_{\rm boundary}-\Delta(E_b+E_c+F_h)-D_b-D_c-D_f-D_r-D_{\rm relax}
              =E_{\rm exchange}-W_{\rm free}.
\]

Free-mode internal work is not boundary input when its applied force is zero.
The report keeps cumulative absolute and signed discrepancies; mean closure
does not establish endpoint equilibrium and endpoint equilibrium does not close
finite-step work. This update adopts no native phase law or accepted-stress
ownership change.

The subsequent [loaded time comparison](P2_total_energy_time_comparison.md) tests
this same energetic/inelastic-strain candidate from a mechanically admissible
unloaded seed, with one free bulk mode and the full surface system. It compares
endpoint/mean mechanics at common times, separates free-mode internal work from
boundary input, and corrects prototype surface-basis undersampling by integrating
on the bulk/surface intersection partition. No physical energy or production law
changes. Refinement decreases release and field/mechanical errors; onset remains
underresolved and native qualification is separate.

**Isolated proposal, not production integration.** This specifies an energy-
dissipative split: constrained phase relaxation with instantaneous elastic
profile redistribution, followed by exact fixed-profile Maxwell evolution and
**time-integrated** mechanical equilibrium. Its finite-step ledger includes the
measured phase-relaxation energy drop. Endpoint agreement with the existing
constitutive update does not imply identical mechanical rates or identical work.
The [prototype report](P2_total_energy_verification.md) states what is verified
and what remains a modeling/numerical approximation.

Notation follows [pf_rsf.tex](../pf_rsf.tex): \(\bm u\) is velocity,
\(\bm\varepsilon=\operatorname{dev}\operatorname{sym}\nabla\bm u\) is strain rate,
\(\lambda=\eta/G\), \(\bm{\mathcal S}:\bm{\mathcal S}=1/2\),
\(h=g^{-1}-1\), \(\chi=h/I^h\). Introduce elastic cohesive slip \(s^e\) and
\(Z=Gs^e=I^h\tau^{\rm coh}\). Fixed straight geometry, homogeneous G and eta,
small strain, prescribed pressure and stateless friction are the selected scope.
No stress rotation, transported histories, geometry changes or RSF-state evolution.

## Accepted state and reconstruction

At time \(t_a\), retain \(\phi^a\), its projected \(I_h^a\), bulk stress
\(\bm\tau^a\) on the **same authoritative bulk quadrature**, surface Q1 \(Z^a\),
accepted displacement \(\bm d^a\)/total strain, accumulated slip, and committed
inelastic strain \(\bm E^{\rm in,a}\). Equivalently retain the combined strain
\(\bm R^a=\bm E^{\rm tot,a}-\bm E^{\rm in,a}\):

\[
 \boxed{\bm R^a=\frac{\bm\tau^a}{2G}
                  +s^{e,a}\chi^a\bm{\mathcal S}.}
\]

This reconstruction is valid for these homogeneous, nonrotating fixed-geometry
kinematics **provided all terms refer to the same accepted spatial state**.
The inelastic strain includes both accumulated bulk viscous strain and deposited
cohesive viscous slip strain. It is not reprofiled when phi changes. Its evolution
at fixed profile is

\[
 \dot{\bm E}^{\rm in}=\frac{\bm\tau}{2\eta}
                         +\chi\frac{Z}{\eta}\bm{\mathcal S},\qquad
 \dot s^e=V-s^e/\lambda.
\]

Production particle→FE stress transfer does not automatically preserve this
reconstruction or its energy. The prototype retains authoritative QP stress and
performs no such transfers. A future implementation must specify ownership or
account explicitly for a representation change; reconstruction is not permission
to silently replace particle histories by an FE field.

Snapshot the whole accepted state, including the lower bound \(\phi^a\), before
the step. Every phase/mechanical trial is scratch data. No predictor or Newton
iterate can become the lower bound. Rejecting either stage publishes nothing.

## Discrete phase energy and its adjoint residual

Let B evaluate Q1 phase at bulk quadrature, with weights \(w_b\); T evaluate
surface Q1 values at those points. Let N evaluate surface Q1 values at surface
quadrature, with arclength weights \(w_s\). Define the fixed consistent projection

\[
 M=N^T W_sN,\quad P=M^{-1}N^TW_s,\quad
 c(\phi)=P\,i(\phi),\quad I_b=Tc,\quad I_s=Nc,
\]

where \(i_r(\phi)=\int_{\mathcal T_r}h(\phi_h)\,d\zeta\) is the direct transverse
integral. \(A=D_\phi i\), so \(\delta c=PA\delta\phi\). Geometry, quadrature,
basis and M stay fixed during trials. The prototype uses the analytic Q1
antiderivative and its analytic derivatives; no frozen-I derivative is used.

At fixed \(\bm R^a,s^{e,a}\), each trial gives

\[
 \chi_b(\phi)=h(B\phi)/I_b,\qquad
 \bm\tau_b^*(\phi)=\bm\tau_b^a
      -2G s_b^{e,a}[\chi_b(\phi)-\chi_b^a]\bm{\mathcal S}_b.
\]

Use the **actual discrete energy**

\[
 \mathscr E_a(\phi)=
 \sum_b w_b\frac{\bm\tau_b^*:\bm\tau_b^*}{4G}
 +\sum_r w_{s,r}\frac{G(s_r^{e,a})^2}{2I_{s,r}(\phi)}+F_h(\phi),
\]
\[
 F_h=\sum_p V_p\frac{\mathcal G_c}{c_0\ell}
       [\alpha(w_p\phi)+\ell^2|\bm D_p\phi|^2],\quad\alpha(\phi)=\phi.
\]

Here \(w_p,\bm D_p,V_p\) are fixed CPDI value/gradient weights and parent volumes.
The small prototype uses their regular rectangular-domain specialization. It
does not substitute a pointwise Y for the derivative of this energy.

For compactness define the bulk dual load
\(v_b=-w_b s_b^{e,a}(\bm\tau_b^*:\bm{\mathcal S}_b)\), and the projected-I dual load

\[
 t=-T^T\operatorname{diag}(h/I_b^2)v
       -N^TW_s\frac{G(s_s^{e,a})^2}{2I_s^2}.
\]

The positive energy-gradient residual is exactly

\[
 \boxed{R_\phi=F_h'
    +B^T\operatorname{diag}(h'/I_b)v
    +A^TP^Tt,\qquad P^T=W_sN M^{-T}.}
\]

All h and h' refer to supplied trial B phi. Both the bulk denominator term and
the fault term pass through the projection **adjoint**. The free equation is
\(R_\phi=0\). At \(\phi=\phi^a\), \(R_\phi\ge0\); at \(\phi=1\),
\(R_\phi\le0\). With production's opposite residual convention these signs
reverse. Complementarity uses these signs, not equality residuals at contacts.

For a direction d, the exact search Hessian can be derived from

\[
 \delta^2\mathscr E_a=F_h''[d,d]
 +\sum_b w_b\{2G(s_b^{e,a})^2(\mathcal S:\mathcal S)(\delta\chi_b)^2
                -s_b^{e,a}(\bm\tau_b^*:\mathcal S)\delta^2\chi_b\}
 +\sum_r w_{s,r}G(s_r^{e,a})^2
       \{(\delta I_s)^2/I_s^3-\delta^2I_s/(2I_s^2)\},
\]
\[
 \delta\chi=h'\delta\phi_b/I_b-h\delta I_b/I_b^2,\quad
 \delta^2\chi=h''(\delta\phi_b)^2/I_b-2h'\delta\phi_b\delta I_b/I_b^2
              -h\delta^2I_b/I_b^2+2h(\delta I_b)^2/I_b^3,
 \quad\delta^2c=P\delta^2i.
\]

Choose a constrained local relaxation from phi^a satisfying KKT and
\(\mathscr E_a(\phi^*)\le\mathscr E_a(\phi^a)\). The energy is generally nonconvex;
this does not claim a global minimizer or unique branch. Trial phase must also
give finite direct integrals and positive projected denominators. Exact saturation
or nonpositive consistent normalization is rejected, not repaired by a cap/floor.
The prototype uses bound-aware L-BFGS then free-set Newton refinement with the
same analytic gradient and projected energy line search; its finite-difference
search Hessian is an explicitly numerical approximation to the expression above.

## Instantaneous profile stage

After phase convergence, form \(\bm\tau^*\), \(I^*\), \(\chi^*\) once from the
immutable anchor. Displacement, inelastic strain and Z remain fixed during this
instantaneous change. Friction and viscosity dissipate nothing over zero time.
Record separate jumps

\[
 J_b=E_b(\bm\tau^*)-E_b(\bm\tau^a),\quad
 J_c=\sum_r w_{s,r}\frac{(Z_r^a)^2}{2G}(1/I_r^*-1/I_r^a),\quad
 \Delta F=F_h(\phi^*)-F_h(\phi^a).
\]

The **discrete phase-relaxation release** is
\(D_{\rm relax}=-(J_b+J_c+\Delta F)\ge0\). This is the finite-drop slack in
the energy inequality, not another imposed driving law or another copy of
fracture expenditure. The algorithm records it; it cannot claim conservative
phase-jump equality when it is nonzero. A physical interpretation of sizeable
release, or a limit in which it vanishes, remains a qualification requirement.
No damage viscosity or compensating H contribution is added.

## General Maxwell stage and mechanical solve

Over \([t_a,t_a+\Delta t]\), hold the new profile and rates fixed. With
\(\beta=e^{-\Delta t/\lambda}\), \(\kappa=\eta(1-\beta)\),
\(\bm B=\bm\varepsilon-\chi^*V\bm{\mathcal S}\),

\[
 \bm\tau(t)=e^{-t/\lambda}\bm\tau^*
                       +2\eta(1-e^{-t/\lambda})\bm B,
 \quad Z(t)=e^{-t/\lambda}Z^a+\eta(1-e^{-t/\lambda})V.
\]

Consequently

\[
 \boxed{\bm\tau^{\rm new}=\beta[\bm\tau^a-2G s^{e,a}\Delta\chi\bm{\mathcal S}]
                            +2\kappa\bm B,\quad
 Z^{\rm new}=\beta Z^a+\kappa V.}
\]

These endpoints equal the old spread-C update for **the same prescribed phase
and rates**, for every dt/lambda. No small-ratio approximation is made.
Define \(r=\lambda(1-\beta)/\Delta t\), \(\bar\eta=\eta(1-r)\); exact mean fields are

\[
 \bar{\bm\tau}=r\bm\tau^*+2\bar\eta\bm B,\qquad
 \bar q=(rZ^a+\bar\eta V)/I^*.
\]

The **chosen candidate** solves averaged bulk virtual work and its slip adjoint:

\[
 \sum_b w_b\bar{\bm\tau}_b:\bm\varepsilon(\delta\bm u)
 -\langle\bar p,\nabla\cdot\delta\bm u\rangle
 =\bar\ell(\delta\bm u),\qquad\nabla\cdot\bm u=0,
\]
\[
 L(\chi^*)^*W_b\bar{\bm\tau}
  -N^TW_s(\bar q+\mu\sigma_n+\eta^{\rm dpr}V)=0
\]

on free positive-slip DoFs. In the prototype's positive-slip Coulomb branch,
V>=0, the driving-minus-resistance residual is <=0 at sticking contacts and
zero when V>0. This supplies a well-defined V=0 control for the fixture's
constant mu=.1 law. Reverse slip and evolving RSF are outside verification.
Prescribed-rate DoFs instead supply reaction work. The source map L and its
adjoint use identical bulk quadrature; cohesive/friction terms use surface ds.

For the stateless monotone branch these equations derive from the rate potential

\[
 \sum_b w_b[\bar\eta|\bm\varepsilon-\chi V\mathcal S|^2
               +r\bm\tau^*:(\bm\varepsilon-\chi V\mathcal S)]
 +\sum_r w_{s,r}[\bar\eta V^2/(2I^*)+rZ^aV/I^*+\mu\sigma_n V
                       +\eta^{\rm dpr}V^2/2]-\bar\ell(\bm u),
\]

with incompressibility and kinematic constraints. Its Hessian supplies matching
coupled blocks. This differs from production **endpoint** equilibrium; it is a
deliberate time-discretization choice in the isolated proposal. It enforces mean
equilibrium, not equilibrium at every time or necessarily at the final endpoint.
The main controls prescribe the bulk shear-rate field and solve the surface
block. Three additional probes free one incompressible bulk velocity mode and
solve both coupled blocks with matching off-diagonal terms. These test reaction
work and the free bulk equation, not a new native Stokes solver.

## Exact finite-step work, losses and publication

For \(x=\Delta t/\lambda\), define

\[
 J_{ee}=\lambda(1-e^{-2x})/2,\quad
 J_{e1}=\lambda(1-e^{-x})^2/2,\quad
 J_{11}=\Delta t-2\lambda(1-e^{-x})+J_{ee}.
\]

For a path \(A(t)=e^{-t/\lambda}A_0+\eta(1-e^{-t/\lambda})b\),
\(\int |A|^2dt=J_{ee}|A_0|^2+2\eta J_{e1}A_0:b+\eta^2J_{11}|b|^2\).
Use this with bulk \(A_0=\bm\tau^*,b=2\bm B\) and cohesive \(A_0=Z^a,b=V\):

\[
 D_b=\sum_b w_b\int\frac{\bm\tau:\bm\tau}{2\eta}dt,\quad
 D_c=\sum_r w_{s,r}\int\frac{Z^2}{\eta I^*}dt,\quad
 D_f=\Delta t\sum_r w_{s,r}\mu\sigma_n V,\quad
 D_r=\Delta t\sum_r w_{s,r}\eta^{\rm dpr}V^2.
\]

Stable `expm1` and near-zero kernel series evaluate the exact expressions; they
do not assume small dt/lambda in the model. The same fixed measures give

\[
 \Delta E_b^{\rm Maxwell}+D_b=W_{\rm ext}-W_{\rm slip},\quad
 \Delta E_c^{\rm Maxwell}+D_c=W_q,\quad
 W_{\rm slip}=W_q+D_f+D_r,
\]
\[
 \boxed{W_{\rm ext}-\Delta(E_b+E_c+F_h)-D_b-D_c-D_f-D_r=D_{\rm relax}\ge0.}
\]

Work uses exact time means, not endpoint traction times dt. No spread-C work is
added during Maxwell evolution: its role is replaced by the recorded instantaneous
bulk and fault storage jumps. If endpoint equilibrium is used instead, retain
the additional integrated exchange defect; it cannot be called phase dissipation.

After **both** stages and numerical/energy checks pass, atomically publish phi*,
I*, Z_new, q_new=Z_new/I*, QP stress, displacement/total strain, cumulative slip,
and updated inelastic strain using the exact time integrals above. Freeze the new
accepted phase as the next step's bound. Rotate simulator histories only by its
native rules in a future integration; never synchronize older phase blocks by
overwriting them. Cache trial data are discarded on failure and accepted caches
are tied to the published phase.

\(\mathcal H^+\) is now solely a diagnostic maximum of accepted
\(Y=q^2/(2Gg^2)\), initialized from Y0 for these isolated controls. It never
enters phase energy/residual or grows on a trial. Old seeded H is historical
initialization provenance, not an extra source in this model. Initialize the
pre-existing band as prescribed phi0 with its lower bound and unloaded mechanics;
test obstacle stationarity rather than assuming its old seeded-H equilibrium is
an unconstrained equilibrium of this energy. This candidate is verified only in
the isolated report; production remains on the previously selected peak-H law.
