# Bounded closed-contact point option: formulation before implementation

This is a new experimental material-point law, not a change to the selected
native specimen or its checkpoint format. Retain T, A, K, Q, L, the Maxwell
recurrences, h(phi), physical obstacle, full 3D plane strain and in-plane
trace-free flow. No opening, dilation, locked director or phase-growth mask.
The point option admits fully compressive in-plane physical trial and returned
states; tensile/contact-opening states fail explicitly, without clipping pn.
The retained compliance is h=m phi(1+phi)/(1-phi)^2,
h'=m(1+3phi)/(1-phi)^3, h''=6m(1+phi)/(1-phi)^4.

Conventions: tensile-positive physical Cauchy stress sigma=tau-p_mech Id,
tau=T-2G A, Q=2G A/h. p_mech is solved mechanical pressure, not the current
prescribed friction pressure p_N. Plane tangent t=(cos theta,sin theta), normal
n=(-sin theta,cos theta), S=(t*n+n*t)/2. Directions are axes modulo pi; theta
is a plane tangent, not a phase-ridge/crack angle. In-plane compression is
C=p_mech Id-tau. Its largest eigenvalue defines the major compression axis.

Closed-contact yield on every plane:

    f(theta)=|(tau-Q):S(theta)|-c_r-mu pn(theta) <= 0,
    pn(theta)=p_mech-n.tau.n > 0.

Use zero-dilation slip W=sum_k w_k k S(theta_k), w_k>=0, at critical maxima
of f. Both conjugate branches may be active; arbitrarily keeping only one
is not a closed return. This is non-associated: normal-traction sensitivity
enters yield, but no normal/volumetric flow is introduced. Old A/K are tensors
in physical coordinates; they are never rotated to a selected plane.

Let J(x,y)=(-y,x). Two-vectors denote ((xx-yy)/2,xy), with full 3D contraction
of shear-space tensors equal to twice the Euclidean dot product. For the new
time level beta=(1+dt G/eta)^-1, Tplus=beta(Told+2G de), h=h(phi),
q_ref=beta (h_old/h) Qold, with q_ref=0 if h_old=0 including h=0. Put

    t=P(Tplus), pbar=p_mech-(Tplus_xx+Tplus_yy)/2,
    e=t-(1+h)q, d=t-hq, R0=c_r+mu pbar,
    z_k=-k J e-mu d, F_k=|z_k|-R0, v_k=z_k/|z_k|,
    g_k=k J v_k, k=+1,-1.

The closed update satisfies

    q-q_ref=sum gamma_k g_k, gamma_k>=0, F_k<=0, gamma_k F_k=0.

Stick uses q=q_ref. For a single active branch, let z0 be z(q_ref), s=|z0|,
l=1+h, b=mu h. The positive root is computed without subtracting close roots:

    gamma=(s-R0)(s+R0)/[R0 l+sqrt((R0 l)^2+(l^2+b^2)(s-R0)(s+R0))],
    v=[(R0+l gamma) z0+b gamma k J z0]/s^2,
    q=q_ref+gamma k J v.

Check the other branch. For two active branches, mu>0 gives cross(t,q)=0.
For |t|>0 enumerate the two roots q=rho t/|t| of

    [(1+h)^2+mu^2 h^2]rho^2
       -2|t|[(1+h)+mu^2 h]rho+(1+mu^2)|t|^2-R0^2=0.

Solve the 2×2 flow equation for the nonnegative gammas. At t=0 use the radial
intersection radius R0/sqrt((1+h)^2+mu^2 h^2) along q_ref. Admit only globally
feasible branches; reject ambiguous distinct admissible roots. mu=0 reduces
exactly to the retained constant-resistance radial return with R=c_r.

Recover A=hQ/(2G), tau=Tplus-hQ, W=h(Q-Qref)/(2G beta dt), Kplus=Kold+dt W,
Lplus=Lold+dt sqrt(2)||W||. L retains its existing net tensor-strain meaning.
The sum of absolute slip on the two planes generally exceeds this net norm;
it is a diagnostic, not a replacement for existing L history. Frictional power
is Df=sum(c_r+mu pn_k) w_k=(tau-Q):W>=0, not a constant times Ldot.
Bulk/cohesive storage, viscous power and Y=h' Q:Q/(4G) are unchanged.
The intact h=0 constrained limit retains A=W=0 but can have finite Q and Y.

For a consistent action with old histories/dt fixed, dT=2G beta de,
dh=h' dphi, dpbar=dp_mech-(dTxx+dTyy)/2. At fixed q,
dz_input=-(kJ+mu Id)dt+dh(kJ+mu Id)q; dz_q=[k(1+h)J+mu h Id]dq.
dv=(Id-v*v)dz/|z|, dg=kJ dv, dR0=mu dpbar. Differentiate the 2+m active
equations, including dgamma and both changing plane directions:

    dq-sum(g dgamma+gamma dg)=dq_ref,
    v_k.dz_k=dR0 for active k,
    dq_ref=-(dh/h)q_ref (zero if h_old=0).

Then dA=(dh Q+h dQ)/(2G), dstress=dT-2G dA; differentiate the retained
energy/driving/rates directly. Stick uses dA=0. At yield ties select sticking;
other active-set boundaries have directional, not classical two-sided,
derivatives. At h=0 use admissible right phase derivatives only.

The Maxwell/constitutive BE work identity survives because Df=(tau-Q):W.
The old incremental potential does **not**: associated Coulomb normality
contains normal/dilatant terms that this zero-dilation flow omits. Expect a
nonsymmetric consistent tangent. Do not reuse the native symmetric relaxed
Hessian certificate or claim an unchanged stability acceptance rule.

Calibration: use c_r=0 and the retained G/eta/m=4 in the first point option;
mu=.1 and .6 are explicit controlled tests at mechanical p=10000 Pa. Current
constant-R control remains R=1000 Pa. For physical pure shear with pressure p,
the new residual limit is tau_r=(c_r+mu p)/sqrt(1+mu^2). With
qcrit=sqrt(2G Kf/m)=2000 Pa, intact phase activation solves

    (tau_peak-qcrit)^2+mu^2 tau_peak^2=(c_r+mu p)^2.

Thus retaining m does not retain the old 1000/3000 Pa residual/peak pair. To
match a chosen pair at a reference pressure, set R0=tau_r sqrt(1+mu^2),
c_r=R0-mu p_ref, qcrit=tau_peak-sqrt(R0^2-mu^2 tau_peak^2), and
m=2G Kf/qcrit^2, only when c_r>=0 and the square root exists. Such recalibration
is documented, not applied to the native material or seed. No parameter can
preserve the same resistance for every pressure under a pressure-dependent law.

For Q=0, the Mohr-Coulomb candidate plane angle to the major compression axis
is alpha=pi/4-atan(mu)/2. Its angle to the plane normal to that axis is
pi/4+atan(mu)/2. With nonzero tensor backstress and **physical** pn, the complete
z_k maximization governs; do not falsely apply the bare-stress angle formula.
Conjugate simultaneous activation can give a radial net W even though each
plane is tilted. A Coulomb-plane label alone does not force a tilted phase path.
