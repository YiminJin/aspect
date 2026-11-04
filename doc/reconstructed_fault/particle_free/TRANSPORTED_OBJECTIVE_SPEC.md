# Transported objective histories: equations and executable design

3 November 2026 — Stages A/B derivation with Stage C native controls in §10.
This extends the selected total-spin model. Stages A/B below began as a
proposal; §§10.1–10.3 now describe the implemented, restricted mature material
configuration and bounded local evidence. Broader transported/cohesive cases
and production readiness remain unqualified. Current
owners and proposed additions are in [IMPLEMENTATION_MAPS](IMPLEMENTATION_MAPS.md);
plan, measured checks and readiness are in [REPORT](TRANSPORTED_OBJECTIVE_REPORT.md).
Existing legacy algorithms, inputs and restart schemas stay authoritative for
those trajectories. Constant G, 2D quasi-static mechanics, closed interfaces and
full3D plane strain are the initial scope. No bulk inertia or finite-strain
hyperelasticity is inferred. Equation numbers below are local to this document.

## 1. Physical tensor and material equation

Let L_ij=∂_j v_i (the deal.II vector-gradient convention),
D=dev3(sym L), W=(L−Lᵀ)/2 and [W,A]=WA−AW. Define

    D_v = ∂t + v·∇,       J_v A = D_v A − [W,A],       r_b(x,t)=G/eta(x,t).
    J_v tau + r_b tau = 2G (D − Lcal_t[V]) + Q_phys.                 (1)
    Lcal_t[s](x) = chi(x,t) s(pi_t(x),t) S(x,t),
    S = sym(tangent ⊗ normal),       chi = h(phi)/E[I_h].

For the mature branch Q_phys=0. This is the physical Maxwell **branch** stress,
not an auxiliary trial tensor. Store xx, yy, xy and reconstruct
zz=−xx−yy, xz=yz=0; full norm is xx²+yy²+(xx+yy)²+2xy².
Retain the configured fault sense in the tangent/normal and signed-slip
orientation; do not infer a new sense from instantaneous V. Deviate FE strain
using trace/3. Pressure does not rotate:
sigma=tau−p' I for the evolved branch; prescribed background loads are separate
(§7). Work and all energy norms retain zz.

On the diffuse mature route v is the continuous native FE velocity; V is the
independent surface slip rate, not a measured jump of that continuous field.
Use its **total** W, including whatever shear that FE field resolves. No
subtraction of an assumed localized-slip spin. On the native sharp-jump route
use the regular sided velocity gradient away from the interface, including the
analytic lifting there. Do not put the distributional delta-gradient of a sharp
jump into exp(dt W); that would be an undefined pointwise rotation. The sharp
and diffuse velocities/forcing operators remain different (§8).

## 2. Interface labels, nodes and scalar histories

Write the oriented curve X(a,t), arclength coordinate s, unit tangent t and
normal n. For a material closed sharp interface,

    v+·n = v−·n = u_Gamma·n,       [v]·n=0.

No-opening fixes only the normal speed. The new **single-state** transported
interface uses the symmetric label convention u_Gamma=(v++v−)/2. This assigns
age/damage/slip to the mid-interface markers; it is not a proof that the two
sliding materials have one common tangential material velocity. A two-sided
age/chemistry model would require additional state and is outside this design.
For the diffuse route use u_Gamma=v(X), not V t. Prescribed-motion tests supply
u_Gamma and a compatible bulk v explicitly; solution-driven tests must obtain
this trace from mechanics and verify the common-normal condition.

Choose node velocity w=u_Gamma+alpha t. alpha is only a redistribution speed;
initial fixtures use alpha=0. Relabelling/remeshing later may choose nonzero
alpha without changing physical fields. For any intensive surface scalar f,

    D_Gamma f = D_w f + (u_Gamma−w)·∇_Gamma f = R_f.                (2)

In curve coordinates D_w f means the derivative at the moving node coordinate;
its advection coefficient is (u_Gamma−w)·t/|X_a|. Thus remeshed nodes do not
carry f unchanged unless they follow its selected labels. For any test psi
transported with nodes, with endpoint fluxes included,

    d/dt ∫ f psi ds = ∫ [R_f psi
                      + f (u_Gamma−w)·∇_Gamma psi
                      + f psi div_Gamma u_Gamma] ds
                      − [f psi (u_Gamma−w)·nu]_ends.              (3)

This is an intensive-field equation. Do not transport age as a conserved
surface density or omit the changing mass matrix. Signed slip rate V is a
relative velocity; it is neither u_Gamma nor w. For mature nonnegative V use

    D_Gamma Theta = 1−|V| Theta/Dc,     D_Gamma delta=V,
    D_Gamma delta_abs=|V|,             D_Gamma s_mem+r_Gamma s_mem=V.

The signed cohesive route keeps |V| in aging/friction and signed V in jump.
Here j is signed accumulated relative slip in the evolving oriented interface
frame; j t is not asserted to be a finite-strain difference of two deformation
maps. Complete bulk tensor correction retains the orientation history lost by
representing accumulated slip with a scalar.
Mature Theta is in seconds; cohesive theta=V0 Theta/Dc is dimensionless and has
source V0/Dc−|V|theta/Dc. Reference jump has zero material source; relaxed jump
has source V−r_Gamma b. Scalar histories receive no tensor commutator. Friction
parameters and other material surface properties follow the same labels when
materially attached; prescribed spatial loading/rheology is sampled instead.

A useful negative control: on a straight stationary curve with u_Gamma=0,
w=c t and Theta(s,0)=Theta0+k s, pure relabelling requires
D_w Theta=c k before the aging source. Holding nodal Theta fixed gives an
O(c k dt) state error while curve geometry and relative slip are unchanged.

## 3. Geometry, profile, extension and quadrature

For physical label velocity u=u_Gamma let A=∇u be a smooth local extension;
only its tangential derivative enters these curve formulas. With
kappa defined by ∂s t=kappa n,

    a_s = t·∂s u,          b_s = n·∂s u,
    D_Gamma t = b_s n = (I−ttᵀ) A t,
    D_Gamma n = −b_s t = −(I−nnᵀ) Aᵀ n,
    D_Gamma ds = a_s ds,   D_Gamma kappa = n·∂s²u − 2 a_s kappa.    (4)

Node-coordinate derivatives additionally contain alpha ∂s. Only rigid motion
has D_Gamma n=W n in general. Example L=[[0,k],[0,0]], n=(0,1):
material geometric n is constant, while W n=(k/2,0). Stress rotation and
geometric normal motion cannot share an update just because both use a tensor.
For a level-set interface D_v F=0 gives D_v ∇F=−Lᵀ∇F and the same normalized
normal law. This is not an instruction to replace the existing geometry owner.

In a nonoverlapping smooth tube x=X(a,t)+zeta n(a,t),

    g=|X_a|,     j=1−kappa zeta>0,     dA=ds j dzeta.

Differentiate the association, rather than assuming it commutes with motion:

    D_v x = v = u + X_a (D_v a) + (D_v zeta)n
                    + zeta (D_Gamma n + n_a D_v a),
    D_v zeta=(v−u)·n,
    D_v a=((v−u)·t + zeta b_s)/(g j),
    D_v pi=u+X_a D_v a,       b_pi=D_v pi−u = g t D_v a.            (5)

Here all curve quantities are evaluated at pi. For normal-constant extension
E[f]=f(pi), D_v E[f]=E[D_Gamma f]+b_pi·E[∇_Gamma f]. At tips, association
switches, cell/fault boundaries or overlapping tubes, use the finite map and
weak transfer; (5) is local and does not invent continuation through a tip.

The physical bulk profile obeys D_v phi=R_phi. With growth disabled R_phi=0:
carry the accepted profile, allowing its spatial width to deform. Rebuilding
an unchanged-width signed-distance profile during general strain is a
constitutive/remapping change, not passive transport. Initial ell is the
regularization length; width sensitivity and deformation must be measured.
For h(phi) in its admissible domain, D_v h=h'(phi)R_phi.

Normalization on a selected tube is

    I_h(a)=∫ h(X+zeta n) j dzeta,
    D_Gamma I_h = ∫ [ (∂t+u_tube·∇)h j + h D_Gamma j ] dzeta
                     + [h j zeta_dot_limit]_lower^upper,          (6)
    u_tube=u+zeta D_Gamma n,   D_Gamma j=−zeta D_Gamma kappa

for fixed-zeta coordinates in the interior. Moving limits need the explicit
endpoint terms; fixed-zeta limits give zero endpoint velocity. In the integrand
(∂t+u_tube·∇)h=h'R_phi+(u_tube−v)·∇h. Therefore

    D_v chi=(D_v h)/E[I_h] − h D_v E[I_h]/E[I_h]².                 (7)

Use the **implemented projected** I_h, not an unprojected analytic denominator,
in production derivatives. A projection M(t) z(t)=f(t) obeys
M z_dot=f_dot−M_dot z. Differentiate basis functions, quadrature locations,
geometry Jacobians, ray intervals/limits and material-property samples.
In a discrete normalization M I=f_h, apply this same rule; division (7) comes
after the mass solve. No positive-I floor, phase cap or fabricated tip completion.

Discrete coupling H(t) and bulk/surface mass matrices must satisfy

    q = M_Gamma^−1 Hᵀ M_bulk tau,
    Vᵀ M_Gamma q = (H V)ᵀ M_bulk tau.                            (8)

If a map changes, all three matrices and its physical measures change together.
Cache keys must include mesh/mapping, curve/material IDs, profile/width,
quadrature/association and coefficient generations. Rebuild derived caches
after trial motion or restart, not just when cell topology changes. Normals and
S use the current geometry; tensor stress remains in the global spatial frame.

## 4. Controls, irreversibility and transfer

These controls are independent:

| Restriction | Precisely frozen | What it does not freeze |
|---|---|---|
| Stationary geometry | Curve location and localization configuration | Bulk transport v·∇tau and surface tangential advection may be nonzero |
| Growth disabled | Material damage/phase source and new formed-segment admission | Curve/profile motion, width deformation, RSF/slip/Maxwell evolution |
| Stationary Eulerian history | Transport operator set to identity at fixed x | Not a material solution if v·∇tau≠0; retain only as a labelled reaction control |
| Prescribed translation | X=X0+c t and all applicable data/boundaries translated | Not solution-driven deforming geometry qualification |

For material bulk damage enforce phi+(x)>=phi−(flow_back(x)), not
phi+(x)>=phi−(x). For interface damage enforce a+(label)>=a−(label) and retain
formed identities. Transport a lower bound **before** enforcing growth; vacated
Eulerian cells must return to their transported background. Under pure
translation phi(x,t)=phi0(x−c t); max(phi_old(x),phi_transported(x)) leaves a
spurious trail, even at zero slip and zero physical fracture growth.

For the newer interface a is physical damage; geometric phi is derived for
visualization/localization and is not a second irreversibility inventory.
Transport existing formed material, not temporary computational completion.
New physical material uses the existing documented virgin age/reference and
birth/admission energy rules; no history is copied from the nearest old tip.

Bulk tensors transfer as independent fields with physical inflow data and
full-tensor projection/storage error recorded. Surface scalars transfer by
material IDs/ALE pullback, not nearest-node overwrite. Convex bounded transfer
of damage/positive age and the transported obstacle is required; overshoots
cannot be repaired by counting extra damage as physical growth. If receiver
space cannot represent the obstacle, retain the richer material record or
reject the transfer. Under MPI migration use CellId/material identity; AMR
must transfer complete physical tensor information and all reference fields.
Pure remapping keeps stress/work/storage to measured transfer tolerance.
No numerical transfer error is a new physical source or dissipation.

## 5. Complete separated-stress product rule

Write s=s_mem (m), B=2G Lcal[s]=2chi E[Z]S (Z=G s), and introduce an independent
full deviatoric residual R (Pa). For possibly distinct surface relaxation
r_Gamma and local bulk r_b, define

    C[s] = J_v Lcal[s] − Lcal[D_Gamma s]
         = E[s] { (D_v chi) S + chi (D_v S−[W,S]) }
                     + chi S b_pi·E[∇_Gamma s],
    Delta_r[s] = r_b Lcal[s] − Lcal[r_Gamma s].                   (9)

D_v S includes changes along the association and current t,n; it is not set to
[W,S] by naming the rate corotational. For normal-constant r_Gamma extension,
Delta_r=(r_b−E[r_Gamma]) Lcal[s]. A projected coefficient/multiplication uses
the actual discrete operator; multiplication and projection do not generally
commute. Surface r_Gamma uses the existing surface material-mixture/property
projection, with constant G; it is auxiliary on the mature branch. It must
never substitute for r_b in physical bulk relaxation.

Select the separated trial and reconstruction

    J_v T+r_b T=2G D,
    D_Gamma s+r_Gamma s=V,
    tau=T−B−R.

Leibniz differentiation gives

    J_v B+r_b B=2G Lcal[V]+2G(C[s]+Delta_r[s]),
    J_v R+r_b R=−2G(C[s]+Delta_r[s])−Q_phys.                      (10)

Substitution recovers (1) exactly. For the mature branch Q_phys=0. For fixed
geometry, no advection, constant matched r,
C[s]=−[W,Lcal[s]] and (10) reduces to
J_v R+r R=[W,B], the previous report's sign and factor. Even zero W cannot
remove R when material crosses a fixed profile or r_b differs from E[r_Gamma].
If a genuine physical localization law sets Q_phys=−2G C[s] in the older
cohesive variable transformation, R=0 is possible only for matched rates and
compatible initial states. That choice is **not** mature Q_phys=0.

Three minimal omitted-term counterexamples (G=1):

- W=0, S fixed, s=1, r_b=r_Gamma, D_v chi=1: residual source −2S.
- Constant profile, W=0, s=1, chi=1, r_b=2, r_Gamma=1: source −2S.
- chi=1, W=0, fixed S, ∂s s=1 and b_pi·t=1: source −2S.

A commutator-only implementation fails all three. These sources are tensors,
not cohesive traction or a limiter. T and R follow bulk v; s follows u_Gamma.
They need not share transport trajectories. DG source projection, reconstruction
and coefficient multiplication errors must be measured separately. More
quadrature cannot supply missing R degrees of freedom.

## 6. Coefficient integration and initial numerical ordering

Eta is physical eta(x,t), evaluated at the **bulk transported location** from
explicit material/composition fields or a declared spatial depth law. Distinguish
vertical depth from down-dip friction coordinate. A spatial layer stays at its
specified depth in an actual moving-fault experiment; in an exact Galilean
comparison the eta field is translated too. Advected material layering and
fixed spatial layering are different prescribed inputs.

Along a bulk trajectory x(t), define a(t)=∫r_b(x(z),z)dz. With frozen-spin
congruence separated, the exact relaxation/forcing subflow is

    tau_end = exp(−a_end) tau_rot
       + 2G ∫ exp(−∫_z^end r_b(x(q),q)dq) F(z) dz.                (11)

For frozen endpoint r*=G/eta(x_end,t_end), beta=exp(−r*dt),
eta_ve=G(1−beta)/r* and tau_end=beta tau_rot+2eta_ve F_end.
Use the existing expm1 formula and its elastic limit. This is a declared
**first-order** coefficient treatment, not an exact variable-eta update.
Split known sharp-layer crossings or integrate the coefficient with bounded
substeps; smooth transitions require dt convergence against (11). Spatial and
surface coefficient projections are recorded independently. No derivative of
eta is missing from the material stress equation itself; its trajectory
variation enters (11), while rate mismatches enter the transformation (10).

Reuse the existing upwind DG material-advection operator and its bounded
explicit stages as a starting component; it currently admits only fixed serial
Cartesian DGQ2, no hanging faces, and is noncommitting. It is not already the
parallel/AMR solution. Use the nonconservative material equation, including the
volume div(v) correction if assembled in conservative flux form. Advect spatial
tensor components, then apply finite rotation and relaxation/source. Project
nonpolynomial rotated products explicitly. Start with no limiter for smooth
controls; if needed later limit independent T/R or direct tau, not B as a
reconstructed localization.

For each attempt: freeze accepted v−/W− as a field; trace bulk/surface/curve
motion using that anchor and the actual attempt dt; form current geometry and
transported obstacles; freeze them for mechanics. Use Rrot=exp(dt W−) evaluated
consistently along the approximate trajectory, then endpoint reaction. Geometry
and coefficient lag is explicit, first-order; require time refinement and the
solution-driven check before admission. Slip, pressure and physical damage use
the current trial solve. Damage/geometry-dependent coefficients inside a solve
require existing derivatives or a declared converged outer iteration; they are
not frozen without recording that changed discrete residual.

In a separated discretization, the robust finite-step closure target is

    R+ = T+ − 2G Lcal_+[s+] − tau_direct,+.                       (12)

Use (12) as an independent audit or to derive finite-step corrections. Evolving
a continuum C source with unrelated splitting/projection does not guarantee
(12). Stage B must compare independent component equations/direct integration;
calling the production reconstruction twice is not an independent reference.
Choose the native history space only after measured normal-resolution cost.
The existing particle control and its label are unchanged, and cannot acquire
DG/transport support merely by removing a guard.

## 7. Prestress, prescribed loading and physical work

Retain the current BP3 interpretation: evolved tau is the perturbation branch;
q0 and pn0 are explicitly imposed fault driving/confinement inputs. They do not
supply a complete bulk prestress to rotate or relax. For example, n=e_y,
t=e_x: adding diag(A,0,−A) to any deviator changes neither supplied normal nor
shear traction on that plane, yet its commutator with nonzero in-plane W is
nonzero. Two admissible full prestresses therefore give different rotated
responses while the supplied scalars are identical. No unique total-prestress
Maxwell prediction can be made from those inputs.

In a new transport test choose either an explicitly provided full initial tensor
(subject to equilibrium/boundary compatibility) or a stated zero perturbation
plus imposed surface loads. The existing cohesive pilot instead has isotropic
p0 plus mean-zero solved p'; it supplies no deviatoric prestress through p0.
Document whether imposed tractions follow material coordinates or are prescribed
in spatial coordinates. Observer covariance rotates their physical vectors;
keeping the same scalar components on a rotating fault describes a specified
follower load, not the rotation of an unknown bulk prestress.

Use branch storage psi=tau:tau/(4G), with constant G. For mature Q_phys=0,

    D_v psi = tau:D − tau:Lcal[V] − tau:tau/(2eta).

The spin power tau:[W,tau] vanishes. For a fixed control volume Omega with
outward nu and positive outward energy flux,

    d/dt ∫Omega psi + ∫boundary psi v·nu
      = ∫Omega [tau:D − tau:Lcal[V] − tau:tau/(2eta)
                                      + tau:Q_phys/(2G) + psi div v].   (13)

Incompressible div v=0 is the selected model. Retain pressure work −p div v
in the independent numerical mechanical balance and report its residual.
If a control boundary moves with c_Omega, flux uses (v−c_Omega)·nu.
Surface balances have analogous endpoint flux and ∫psi_Gamma div_Gamma u_Gamma.
For prescribed fault drive q0 counted outside the branch stress, its positive
input power is ∫Gamma q0 V ds with the declared signed orientation. Isotropic
p0/loading work is included through actual displacement/divergence and normal
constraints, not counted again as tangential frictional loss. On a no-opening
interface common normal traction does zero relative-opening work. Moving outer
boundaries and imposed body/traction loads use their actual work-conjugate
velocities. Work of deliberately imposed geometry/profile motion must be
computed from its shape derivative, not defined as whatever closes a ledger.

## 8. Cohesive compatibility, physical geometry and isolated force gate

The older P2/cohesive transformation and newer sharp-jump pilot remain distinct.
For older physical elastic slip e, E_coh=∫ G e²/(2I_h) ds and
Q_phys=−2G(C[e]+Delta_r[e]); it reduces to −2G C[e] for matched rates. Its phase derivative contains −∫tau:delta Lcal[e] dA and
−∫G e² delta I_h/(2I_h²) ds. Setting cohesive resistance to zero in the mature
model does not make this geometric redistribution law equal to Q_phys=0.

For the newer pilot define

    A = T_DG + 2G D3(d_R) − Pi[2G D3(d_R)],
    K_t[b] = sum_j b_j dev3(sym grad lifting_j(t)),
    H = 2G K_t[b],          tau = A + H,
    D_Gamma b + r_Gamma b = V,       D_Gamma j=V.                 (14)

These are current source signs. Native du is the **smooth** displacement
increment; D_total=D_smooth+K_t[V] uses the jump-lifting velocity once.
K includes tangent derivatives and both normal/tangential support variation;
it is not interchangeable with −chi S. Point relaxed_jump equals the Q1 b
trace; jump includes its retained Gauss offset. Gauss reference and damage
are physical history, not bulk tensor residuals.

If J_v A+r_b A=2G D_smooth and the components are extended to new geometry,

    C_K[b] = J_v K_t[b] − K_t[D_Gamma b],
    Delta_K = r_b K_t[b] − K_t[r_Gamma b],
    J_v tau+r_b tau = 2G(D_smooth+K_t[V])
                                     + 2G(C_K[b]+Delta_K).       (15)

Unlike the scalar localization, K may differentiate b. Even matched pointwise
r does not imply Delta_K=0: the product derivative K[r b] includes grad_Gamma r.
For discrete projected rates use the **whole** linear operator K and surface
rate matrix, not a single nodal beta. The native remainder also has missing
terms under rotation/transport: for a gradient field d_R,
D_v(∇d_R)=∇(D_v d_R)−(∇d_R)L; apply sym/dev and differentiate Pi with its mass
matrix. Spatially varying Rrot times a gradient is generally not another
compatible FE gradient. Retain a full independent tensor correction or a
qualified complete tensor space; advecting/rotating d_R alone is insufficient.

For the shared physical Maxwell law during material transport, separate
C_K=C_K^kin+C_K^phys: kin changes from transport, relabelling, material
orientation and associations; phys changes from an explicitly solved damage/
admission/localization step. A complete numerical representation may use

    tau=A+2G K_t[b]−R_K,
    J_v R_K+r_b R_K=2G(C_K^kin[b]+Delta_K),                       (16)

leaving Q_phys=2G C_K^phys[b] only when the **existing physical energy variation**
selects that redistribution. Pure numerical remap compensates exactly:
A_new−R_K,new=tau_transported−2G K_new[b_new]. It creates no physical work.
Finite physical changes, by contrast, evaluate the existing energy at fixed
transported incoming independent state with its full derivative. Do not cancel
a physical phase derivative just to make a direct-stress comparison agree.
If all geometry changes are treated as physical changes of the stored lifting
in (14), (15) instead holds in full; this is a different material law. Stage B
must label these alternatives. Equations (15) and (16) cannot be combined by
deleting their differing source while leaving the storage/shape force unchanged.

At full decohesion q_bond=0 and a=1, the pilot still has nonzero b and A. To
recover the mature **Maxwell material equation**, use complete tensor evolution
as in (16), retain b in reconstruction/correction, use the same regular total
velocity and compare mechanical work adjoints. This is a conditional reduction
of equations, not equivalence of finite-width sharp-jump and diffuse traction
operators. Original pilot geometry is fixed except physical path admission;
it has not qualified the new transport/shape decomposition.

The actual energy inventories are

    Ebulk = ∫ |tau|3²/(4G) dA,
    Ebond = ∫ (1−a)(j−j_ref)²/(2 C0 a) ds,   C0=2Gc/peak²,
    Efracture = ∫ Gc a ds,

with the documented intact and decohered limits. Current-arclength ds is the
existing measure. Its material derivative contains

    ∫ [D_Gamma psi_surface + psi_surface a_s] ds                 (17)

and endpoint flux/birth terms. No reference-length replacement is selected.
A fixed physical damage a=1 on a straight material segment of length L under
uniform tangential strain epsilon_dot changes Efracture at rate Gc L epsilon_dot,
even at V=0 and da=0. A deformation calculation that reports zero work from this
motion fails energy balance. This is the smallest decisive counterexample to
“move the curve but keep the old energy ledger”.

For current-measure energy the required virtual shape terms include

    delta Ebulk = ∫ tau:delta K[b] dA
                    + terms from A/R transfer and bulk-domain measure,
    delta Esurface = ∫ [delta psi_surface
                          + psi_surface t·∂s(delta X)] ds.       (18)

In (18) the first line uses H=2G K[b]; its derivative is evaluated holding the
chosen independent incoming histories, not two incompatible representations,
fixed. Coefficient, native-remainder and pressure/lifting-divergence terms are
included in the same residual/adjoint. For constant psi_surface the last term
reduces to endpoint tensions and curvature force by integration by parts.
The current source solves fixed geometry and does not assemble these general
shape-force terms. Therefore **solution-driven deformed cohesive evolution is
gated** on deriving/assembling the full variation (18), with common-normal
kinematics and its boundary work, and measuring it against finite differences.
This is an exact missing coupling, not authority to add uncalibrated damping or
to switch to reference-area fracture energy. Prescribed shape tests must supply
and independently measure that work. Translation has a_s=0 and unchanged
geometry energy and can proceed independently.

Keep the two existing time integrators: mature exponential Maxwell versus
pilot backward Euler. After coherent transport/rotation to one frame, BE obeys

    tau+ = beta_BE (tau_tr− + 2G Delta_e),
    tau+:Delta_e = Delta psi + |tau+−tau_tr−|3²/(4G)
                                  + dt |tau+|3²/(2eta*).        (19)

Rotation preserves psi; projection/transport error is reported separately.
Surface endpoint work equals bond increment + fracture + friction + radiation
+ the explicitly measured surface endpoint remainder. For changing ds or
physical K add the independently derived geometry work/flux terms, not a
closing remainder. Report BE numerical terms separately from trapezoidal
physical-work error and normalize against new fracture work as well as sliding
loss. The existing failed fracture-relative/spatial screens remain failures.

## 9. Executable owner/lifecycle contract and next gate

The consolidated [state-owner table](IMPLEMENTATION_MAPS.md#transported-extension-state-contract-stage-a)
defines current owners, proposed fields, transport/transfer and restart rules.
Each trial has an immutable parent accepted state. The required operation is:

1. Pull back complete bulk tensors with v−; transfer curve material histories
   with u_Gamma− and node ALE with w−; transport the physical obstacle. Keep the
   parent identities/clocks, input material fields and accepted gradient anchor.
2. Advance curve/profile, coefficients, maps and weights from that same anchor
   with the actual dt. Compute finite rotations, reaction/source and all
   correction histories consistently. A retry repeats from the parent, not a
   previously transported/rotated candidate.
3. Solve current mechanics/pressure/RSF, then physical damage/admission if
   selected; iterate geometry-dependent physical coefficients to the declared
   tolerance. Construct one candidate with matching maps, stress, surface
   histories and native solution. Observe without advancing any history.
4. Check physical bounds, independent tensor/traction/work closure, timestep
   error and final acceptance. Rejection discards geometry/profile/maps and
   all candidate histories as well as solution vectors. Acceptance swaps the
   complete owner once; output/checkpoint uses that endpoint only.
5. Restart restores that accepted endpoint and provenance, then rebuilds
   derived maps. No extra transport/rotation on load. New schema must name
   objective rate, integrator/lag, coefficient model, representation and
   full tensor/obstacle/material-ID layout. Old-to-new conversion is rejected;
   legacy trajectories keep their legacy schema and law.

The [Stage B first-part comparison](TRANSPORTED_OBJECTIVE_STAGE_B_FIXED_REPORT.md)
and [remaining moving/layered comparison](TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md)
now verify the complete source and history-space capacity with independent direct
stress. The selected first native mature representation is existing T DGQ2 plus
an independent complete R DGQ2 and surface scalar memory. This uses the same
nine-mode tensor space and full3D convention as the current owner. A best P1
residual misses the declared coarse-grid target; no enrichment is required by
the Q2 measurements. Direct stress remains a separate reference and possible
same-mesh memory alternative, not a silent replacement of legacy state.

This decision does not enable a native selector or convert checkpoints. Next is
Stage C integration of the selected tensor owner, source/tangent/observer and
complete accepted geometry/history transaction. Native surface transport,
non-affine remainders, AMR/restart and coupled physical energy remain to qualify.
The cohesive branch retains its distinct physical/native/lifting state and
shape-work gate; it is not collapsed into mature auxiliary memory by this choice.

Source contracts used: current mature `constitutive.cc`, `dg_history.cc`,
`dg_transport.cc`, `fe_strain.cc`, `fe_normalization.cc`, `dg_accepted.cc`;
newer `cohesive_bulk_history.cc`, `cohesive_geometry.cc`, `cohesive_system.cc`,
`cohesive_advance.cc`, `cohesive_state.cc`, `cohesive_interface.cc`; native
solver/assembler and manager listed in the maps. These sources were read in
this worktree; their current guards are part of the implementation contract.
The authoritative design/specification link this proposed extension separately
from their already implemented frozen-geometry sections.

The external [official BP3 specification](https://strike.scec.org/cvws/seas/download/SEAS_BP3.pdf)
(1 October 2021, §§1–2) confirms that its linear-elastic prestress setup supplies
fault tractions, not a unique full prestress. It does not establish the validity
of objective Maxwell evolution about an inferred prestress. Equations above
are repository-specific derivations, not external qualification of this code.


## 10. Implemented Stage C controls and their boundaries

The [native Stage C report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md) records
source, measured criteria, failed evidence and exact scope. The following are
stationary reaction controls. Section10.1 below adds the restricted mature
material transaction; general ALE, resampling and cohesive motion in §9 remain
outside that enabled configuration.

For mature matched constant parameters and fixed profile/geometry, with common
DGQ2 projection P and Q=exp(dt Wminus),

    Tbar=P[Q Tminus Q^T],
    Rbar=P[Q Rminus Q^T + Q Bminus Q^T - Bminus],
    tau_anchor=Tbar-Rbar-Bminus,
    Tplus=beta Tbar+2kappa D3(vplus),
    Rplus=beta Rbar,          Zplus=beta Zminus+kappa Vplus.       (20)

D3(vplus) is exactly in the admitted affine native Q2 space. Thus publication
Tplus-Rplus-Bplus equals the point stress used in mechanics. Equation(20) is
also tau_anchor=tau_minus+P[Q tau_minus Q^T-tau_minus], including the analytic
localization in the complete finite-rotation increment. Constant matched
coefficients are essential to the cancellation. It is not a substitute for
the coefficient terms in §5. The immutable accepted-gradient lag makes the
current mechanical tangent the derivative of the residual actually solved.
Native old-solution storage checkpoints that anchor; derived attempt caches
are keyed by parent/step/actual dt and are not additional physical history.
Schema6 stores the independent R; legacy schema4 is unchanged.

For the newer cohesive path retain A=T_ind+2G D3(d_R)-Pi3[2G D3(d_R)] and
H=2G K[b]. A regular lifting gradient has material-piece/side jumps within
bulk cells. The measured uncut DGQ2 rotation space fails its unchanged target.
The new independent T_ind uses Q2 polynomials on formed lifting pieces/sides
plus exterior regions. The old global DG tensor is then a derived projection
cache, not a second independent physical history. Native d_R is retained.

With Pcut the qualified piece/side projection, its fixed-geometry reaction is

    Tbar_ind=Pcut[T_ind + Q(A+H)Q^T-(A+H)],
    Abar=Tbar_ind+2G D3(d_R)-Pi3[2G D3(d_R)],
    tau_bar=Abar+H,
    tau_plus=beta_BE[tau_bar+2G(D3(du)+K[Delta j])].             (21)

The last equality holds pointwise for the BE reaction; the preceding objective
projection has its separately measured error. Wminus includes accepted native
velocity gradient and regular lifting-rate gradient, excluding the sharp
jump's distributional gradient. Neither b nor a normal vector is tensor-rotated.
Restriction of the native projected source into every cut polynomial preserves
the original complete native BE recurrence. Side-aware evaluation uses the
same material side for T_ind and H. A formed-geometry snapshot, local polynomial
bounds/coefficients and native memory checkpoint together. Temporary candidate
tails never define persistent cut partitions. A distinct law/representation
signature rejects legacy and uncut-objective checkpoint conversion.

The numerical projection energy is evaluated directly from pre/post-anchor
physical tensors before mechanics. Exact point rotation has zero power. This
defect is reported separately from viscous/frictional dissipation and from the
independent physical work remainder; it is not a term chosen to close a ledger.
The tested short phase fixture still fails the1% fracture-relative physical
screen, and the localized tip fixture fails a legacy-independent line search.

Owned-row/ghost SSPRK3 DG material advection and frozen spatial relaxation are
implemented and verified independently, including hanging faces and explicit
inflow. Their restricted coupled use is specified below. Moving resampling,
boundary completion, moving AMR and cohesive shape/pressure coupling in §8
remain unsupported; no general transported selector is thereby qualified.

### 10.1 Restricted complete mature material step

`Jaumann DG material Lie` selects a fixed Cartesian mesh, same interior material
labels, growth-disabled physical phase and constant matched reference G/eta.
A positive prescribed Eulerian `Material viscosity multiplier` may vary in
space/time. It is sampled at endpoint bulk and surface positions; it does not
represent transported composition. Let A be DG SSPRK3 upwind material transport
with accepted v_minus, P common Gauss8 DGQ2 projection and B=2 chi Z S. Stage

```
Tbar = P[Q A(Tminus) Q^T]
Rbar = P[Q A(Rminus+P Bminus) Q^T - Bmapped(Zminus)].           (22)
```

The same accepted FE velocity advances material vertices and Q1 characteristic
phase/obstacle. Age and auxiliary Z follow their unchanged label identities.
Support widths use the inverse-transpose geometric map; normals come from the
new curve. Rebuild I, localization and work maps before mechanics. Tensor and
phase inflow were explicitly zero in the initial local experiment; the selected
bounded-profile extension below adds a separately identified tensor boundary
option. Growth is
off, so there is no extra Eulerian irreversibility obstacle left behind.

Matched reaction uses (20). With spatial eta, define beta_b/kappa_b at bulk
quadrature points and beta_Gamma/kappa_Gamma at surface labels, then retain

```
Zplus_j = beta_Gamma,j Zminus_j + kappa_Gamma,j Vplus_j
Tplus = P[beta_b Tbar + 2 kappa_b D3(vplus)]
Rplus = P[beta_b(Rbar+Bmapped(Zminus))
          + 2 kappa_b L[Vplus] - Bmapped(Zplus)].            (23)
```

The difference between tau_h=Tplus-Rplus-Bmapped(Zplus) and the local
unprojected Maxwell response is frozen inside each native mechanical solve.
A maximum24 outer iteration updates that projection correction to relative
change <=128 epsilon at the common quadrature, then invokes the original
residual/KKT publishing solve. Publication recomputes the candidate and checks
pointwise stress consistency under the existing strict bound. The local
Maxwell matrix is an inner tangent/preconditioner, not the full projected
Jacobian. This declared outer iteration retains the coefficient/profile terms
that a direct use of (20) with varying eta would omit.

Bulk/surface/phase/geometry and normalization are one accepted transaction.
Rejection restores the parent and expires derived caches; restart validates
accepted normalization before staging the first moved profile. Schema7 stores
complete history/geometry/clocks plus law/integrator and material/friction/phase
identity, with no old-to-new conversion. Full3D plane strain and total material
Jaumann spin remain unchanged. This is a first-order split discretization with
measured projection/remapping error, not an exact finite-step transformation.
See [measured local qualification](TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md).
No birth, resampling, moving AMR, cohesive shape work or production continuation
is enabled by these results.

### 10.2 Measured boundary and interpolation limits

[Boundary/profile isolation](BOUNDARY_PROFILE_ISOLATION_REPORT.md) adds evidence,
not a different update law. On prescribed constant translation the native DG
energy satisfies dE_h/dt + outward prescribed-boundary energy flux + nonnegative
upwind jump loss = 0 to arithmetic precision. Exact polynomial inflow and zero
inflow are different boundary-value problems; their errors must use their own
references. A compact tensor isolates approximately third-order spatial error.

For the existing repeated Q1 nodal characteristic pullback and0<=C=|c|dt/h<=1,
the exact interpolation variance increment is N C(1-C)h², equivalent to
numerical diffusivity |c|h(1-C)/2. Reducing dt at fixed h therefore does not
remove its spatial diffusion floor. The native default1e-6 unit-cell point-search
tolerance and average neighboring evaluations are additional measured mapping
effects, not constitutive dissipation. The isolation study made no
tolerance/interpolation change. Coupled qualification must use controlled inflow, joint space/time
refinement, appropriate interior windows and independently measured work/flux.

### 10.3 Opt-in bounded Q1 material profile and accepted-trace tensor inflow

The selected improvement keeps Q1 storage and the physical equation
`partial_t phi + v . grad phi = 0`. With the same accepted velocity as tensor
and label transport, evaluate Q1 pullback A-minus at x-dt*v(x), its return
operator A-plus at x+dt*v(x), and

```
phi_forward = A-minus phi_minus
phi_candidate = phi_forward + (phi_minus - A-plus phi_forward)/2
phi_transport = clamp(phi_candidate, donor_min(phi_minus), donor_max(phi_minus)).
```

The donor bounds use accepted Q1 vertices of all departure cells admitted by
the existing remote-point search. Missing departure coverage gives zero phase;
missing return coverage disables the correction there. No global phase cap,
normalization floor or Eulerian maximum with the old phase is added. For the
admitted growth-disabled state this transported phase is also the transported
irreversibility obstacle. This is a bounded MacCormack discretization, not a
conservative remap or a higher-order material trajectory. The time ordering,
total Jaumann spin, source/endpoint convention and surface histories are unchanged.

`Material profile transport = Q1 remap` remains default; `bounded MacCormack`
selects this correction. `Material tensor inflow = zero` remains default;
`accepted trace` freezes the accepted interior boundary values of T and
U=R+P B at boundary quadrature for all DG advection substeps. It supplies a
constant-normal exterior continuation of accepted tensors, followed by the
original rotation/reaction once. It is a stated finite-domain boundary model,
not arbitrary analytic inflow and not qualification of open-boundary accuracy
for another loading. The coupled comparison uses pure-translation mechanical
Dirichlet data and an unloaded natural top, with a compact interior profile.

Both nondefault options require the existing restricted material selector.
Either nondefault selection appends both selectors to schema7 material identity;
the default/default signature is retained exactly. The transient forward field
and limiter have no independent checkpoint history. Profile, tensors, geometry,
surface state and rebuilt maps still stage/publish/rollback as one transaction.
See [comparison protocol](BOUNDED_PROFILE_PROTOCOL.md) and its measured report.


### Bounded-profile coverage condition (current implementation)

The nondefault bounded Q1 profile and accepted-trace inflow selections require
`Association support = full normal cross-section`. The transported mature
selector admits this existing policy for finite interior fault endpoints;
there is no tangential continuation or cohesive-route admission. This corrects
the support/I mismatch exposed by the retained truncated-support comparisons.
The existing inverse-transpose material support-width map is unchanged.
Schema7 material identity additionally records full-normal support, so a
checkpoint from the truncated or early unidentified-support path is rejected.
The default Q1/zero-inflow/stationary-support identity is unchanged. See
[the measured comparison](BOUNDED_PROFILE_TRANSPORT_REPORT.md) for coverage,
normalization and the remaining spatial/time errors.

### Direct physical-history experiment (2026-11-04; locally verified reference)

The opt-in `Material stress representation = direct physical stress` retains
`Jaumann DG material Lie`, fixed Cartesian DGQ2, constant G, endpoint spatial eta,
growth-disabled mature geometry and the existing exponential Maxwell/age laws.
It changes the discrete tensor representation, not the constitutive equation.
`DGHistory::Representation::physical_stress` owns one xx/yy/xy tensor with
zz=-xx-yy and no residual owner. Z remains a surface diagnostic and never enters
physical stress access. Separated history remains the default.

For each attempt, advection uses the accepted velocity and frozen accepted
physical tensor inflow; finite Jaumann rotation uses the accepted FE gradient.
The staged anchor is P[Q A(tau) Q^T]. The native reaction projects the complete
`beta(x) anchor + 2 kappa(x) (D(v)-chi V S)` at the existing common Gauss rule.
A frozen projection defect corrects the existing local Maxwell inner solve;
constant and variable viscosity direct modes both iterate this correction.
The final residual/publication gate must start from the converged bulk and slip
unknowns, not reset V to its accepted-parent value. No tolerance changes apply.
The unloaded accepted initialization and 0.1 s predictor remain comparison
conventions; no initial physical stress is inferred from imposed loads.

Schema7 direct material identity includes an explicit representation suffix;
its residual-coefficient packet is empty and incompatible separated loads are
rejected. The existing material transaction owns transport, geometry, profile,
surface state, rejection and publication. No old checkpoint conversion, AMR,
cohesive-law change or production qualification follows from this option.
See `DIRECT_TRUE_STRESS_REPORT.md` for actual evidence and remaining limits.

### Frozen finite-end qualification follow-up (2026-11-04)

[FROZEN_FINITE_END_REPORT.md](FROZEN_FINITE_END_REPORT.md) compares the same
frozen Q1 profile, labels/I and complete local reaction on16²–128² tensor grids.
A cut-integrated reference separates uncut DGQ2 approximation from Gauss8–24
quadrature error. The former dominates and is concentrated at finite mature
curve ends; it is not the cohesive physical-tip gate. Projection energy
identities apply in their stated integration measure. This is an offline
qualification result, not a change to P_h, history space, source support,
constitutive rate or work assembly. Direct remains an opt-in reference.

## Consolidated implementation locations — 4 November 2026

The equations/state ownership in this document are unchanged. See the current
[module table](STRESS_HISTORY_CONSOLIDATION_REPORT.md#implemented-boundaries-and-caller-audit)
for physical stress access, material projection, transport/rotation, transaction
and benchmark/audit ownership. Earlier file locations are historical. No new
constitutive, checkpoint, quadrature or convergence convention is selected.
