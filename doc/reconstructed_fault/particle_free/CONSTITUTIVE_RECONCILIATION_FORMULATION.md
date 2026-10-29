# Endpoint continuation and single-direction RSF: bounded formulation

This is an independent reference, not a native/BP3 constitutive change. Read
the [report](CONSTITUTIVE_RECONCILIATION_REPORT.md) and
[predeclared selection](CONSTITUTIVE_RECONCILIATION_SELECTION.md) for scope.
The two questions are separate: whether an endpoint can continue the accepted
history, and whether a selected kinetic branch has regular mixed mechanics.

## 1. Feng finite-step formulas and global admissibility

Source: Feng et al., [JMPS 189, 105687](https://doi.org/10.1016/j.jmps.2024.105687),
[accepted manuscript](https://air.unipr.it/bitstream/11381/2991093/5/PFM_CM-3.pdf),
equations 59, 71–81 and Appendix E. The previous reference implemented the
stationary equations numerically; this task additionally transcribes the
printed closed-contact angle formula itself. Its elastic update is compared
at fixed h. Our Maxwell transformation and phase energy remain distinct from
the complete published model.

Let alpha=(cos(theta),sin(theta)), beta=J alpha,
`m=k sym(alpha tensor beta)`, `N=sqrt(2)m`, and
`F=(tau-Q):m-mu pn`, with pn the smaller positive compression of the orthogonal
pair. Full embedded 3-D contractions retain both xy terms and plane-strain zz.
The angle is a **plane normal**, not a phase-front or stress-principal angle.

For fixed phase, `h=4 phi(1+phi)/(1-phi)^2`,
`beta_M=1/(1+G dt/eta)`, and accepted A_old,

    tau_trial=T-2G beta_M A_old, Qref=2G beta_M A_old/h,
    tau=tau_trial-2G xi m, Q=Qref+2G xi m/h,
    xi=positive[(tau_trial-Qref):m-mu pn_trial]/[G(1+1/h)],
    Delta K=xi m/beta_M, A_new=beta_M A_old+xi m.

Setting beta_M=1 gives the fixed-phase elastic corrector. The printed
equation 73 uses physical trial shear to sign N, while equation 59 and
Appendix E use relative shear. The two agree in the original zero-memory
tests, but not generally during unloading. They are tested separately.

For the literal angle formula, let Theta be measured from the maximum
principal direction of trial stress, theta_s the corresponding relative-stress
principal angle, a_sigma the mean in-plane total trial stress, r_sigma/r_s the
trial/relative Mohr radii, and j the assumed relative-shear sign. Then

    d=h/(1+h), k=j d mu,
    A=j r_s cos(2 theta_s), B=j r_s sin(2 theta_s)+mu r_sigma,
    C1=B-k A, C2=A+k B,
    C=-sign(C1) k a_sigma mu/sqrt(C1²+C2²),
    Theta=[asin(C)-atan(C2/C1)]/2.

The paper tries j=-1 first and j=+1 if Theta is outside [-pi/4,pi/4]. These
are the principal asin/atan functions as printed; silently using atan2 would
be a different branch algorithm. We restrict this transcription to closed
contact, the applicable branch of equation 78 in these controls.

Three distinct checks follow:

1. **Stationary branch:** its own F=0 and F_theta=0.
2. **Globally admissible endpoint:** max_N F<=operand-scaled roundoff and the
   actual increment N is a maximizer. Own-plane consistency is insufficient.
3. **Reachable continuation:** an admissible time derivative exists from the
   incoming physical histories, phase obstacle and loading, and the discrete
   path approximates it. Neither 1 nor 2 proves 3.

An independent complete trigonometric enumeration solves the same Appendix E
stationary equation before division by C1:

    C1 sin(2 Theta)+C2 cos(2 Theta)=-k a_sigma mu.

It retains both sine branches, normal ordering, the assumed shear-sign check,
and a final global test. This removes the C1=0 representation singularity and
recovers the missing reversed-memory branch of the printed shortcut. It does
**not** change the flow law or make a nonglobal stationary endpoint admissible.
All these roots agree with the previous independent angular root calculation.

At tau_trial,xy=1100 Pa, Qref=0 and phi=.2/.8/.99, the literal formula is
stationary but violates global F by .870256/.0144094/2.64938e-5 Pa. At exact
h=infinity, C1=0 makes the printed expression undefined; complete enumeration
and its finite-h limit recover the mature roots. At phi=.99,Qref,xy=-.05 Pa,
the shortcut's assumed sign is inconsistent; its stationarity residual is
-.01436 Pa. Complete enumeration gives two globally admissible endpoints,
which still require a between-state continuation rule. In a separate synthetic
unloading control (phi=.01,tau_trial,xy=500,Qref,xy=1600 Pa), the printed physical
sign produces no increment whereas relative shear gives a positive reverse
increment; the latter still fails the other global maximum. Its initial
uniform phase force is +2.565 Pa, but reachability is not asserted.

## 2. Reachability with the retained physical phase/history equations

The loading refinement starts from T=A=K=0 and a prescribed uniform phase
lower obstacle phi=.2/.8/.99. This is pre-existing damage, not damage created
without fracture energy. Impose engineering shear gamma=.0012 t for 1 s,
G=1e6 Pa, eta=1e8 Pa s and fixed p=10000 Pa. The initial phase force is +8 Pa.
At first yield A=0 and Y=0, so the physical obstacle holds phase fixed.
The exact elastic Maxwell ramp reaches first yield at

    t_y=-(eta/G) log[1-(G/eta) tau_y/(G gamma_dot)],
    tau_y=mu p/sqrt(1+mu²).

The history equations are `Tdot+(G/eta)T=2G dev D`,
`Adot+(G/eta)A=W`, `Kdot=W`. With `f_i=m_i+mu n_i tensor n_i`, a single
active branch j would require

    w_j=[f_j:(2G D-(G/eta)tau)]/[2G(f_j:m_j+(m_j:m_j)/h)],
    tau_dot=2G(D-w_j m_j)-(G/eta)tau,
    Q_dot=2G w_j m_j/h,
    F_i_dot=f_i:tau_dot-m_i:Q_dot.

For either conjugate j, F_j_dot=0 but the other F_i_dot is positive:
9.47315, .163330 and .000300525 Pa/s at phi=.2/.8/.99. Thus the admissible
single-ray tangent fails already at a reachable first-activation state under
this homogeneous loading. Phase evolution cannot cure this instant: Y=0,
the lower obstacle is active, and a change of h multiplies zero A/Q there.
The exact mature limit has zero other-branch derivative.

Runs with 32/128/512/2048 steps record the first failed endpoint; independent
complete roots also fail there. Continuing the printed update beyond that
point is explicitly a **diagnostic inadmissible path**, not accepted history.
Its global force violation approaches a nonzero limit. Uniform Y stays below
8 Pa throughout, so the defect is not a missing permitted phase relaxation.
The reversed-memory endpoint is obstacle-compatible but is not proved reachable
from an accepted single-direction symmetric path. No conjugate average,
rapid-switching/relaxed-flow limit or spontaneous symmetry-breaking perturbation
is adopted to replace the missing continuation rule.

## 3. New kinetic candidate and slip mapping

Source for the kinetic form and pressure update is specifically Fei et al.,
[*A Phase-Field Model for Quasi-Dynamic Rupture Nucleation and Propagation of
In-Plane Faults*](https://doi.org/10.1002/nag.70416), equations 24/26/28/30
and Appendix A14; [preprint](https://arxiv.org/abs/2503.06576).
This reference borrows those ingredients, not Fei's stress decomposition,
phase-driving law or direction initialization. Our changing global maximizing
direction and Maxwell/cohesive memories are additional assumptions.

There is **no chosen reference band width**. For a fixed straight chart use
the retained AT1 crack density

    Gamma_phi=3/(8 ell)[phi+ell² |grad phi|²],
    I_Gamma=integral Gamma_phi dr, chi=Gamma_phi/I_Gamma,
    W=chi V m, |V|=sqrt(2 W:W)/chi,
    signed interface velocity=2 integral W:S dr

for a coherent signed direction S with S:S=1/2. For the constrained AT1
profile `phi(r)=(sqrt(phi0)-|r|/(2ell))_+²`,
`I_Gamma=phi0^(3/2)`, support halfwidth=`2ell sqrt(phi0)`, and
`chi(0)=3/(4ell sqrt(phi0))`. At maturity chi(0)=16/m. The inverse 1/chi is
derived from the phase profile, not a fitted strain-rate-to-slip-rate length.
For coherent V and direction through the profile, integration recovers actual
interface slip and work. With varying directions or V across a transect,
local V is an interface-equivalent speed; it is not generally the signed
integral slip velocity. This limitation matters before a multidimensional
propagation model is defined. Exactly intact I_Gamma=0 is undefined; no floor,
latent state/birth rule or intact friction replacement is introduced here.
The profile ansatz is an additional kinematic assumption, not an identity
implied by the free bulk tensor equations. In particular, the flow plane must
match the chart for an actual coherent tangential displacement jump. The
power check is not a mechanically equilibrated cross-band solution. This
Gamma normalization does not replace the existing cohesive h/I localization
or its derivatives; it defines only the independent kinetic candidate.

For V>=0 and dimensionless age Theta=theta V0/Dc, the candidate is

    mu=a asinh[(V/(2V0)) exp((mu0+b log Theta)/a)],
    Theta_dot=(V0/Dc)[1-(V/V0)Theta],
    max_N [(tau-Q):m-mu(V,Theta) pn(N)] = eta_rad V,
    active N=sqrt(2)m is a global maximizer, W=chi V m,
    eta_rad=G/(2 cs), cs=sqrt(G/rho).

Directions maximize at fixed speed/state; the coupled derivative then includes
speed, state and direction changes. No conjugates are added together. The
asinh law removes the finite sticking interval on the damaged branch; at V=0
the flow direction is undefined. At a genuine tie only the previous normal
preference is used, as before. Histories remain physical fixed-chart tensors;
there is no finite-rotation or objective-rate claim.

Retained G/eta/m/Gc/ell and residual cohesion zero are unchanged. Exploratory
kinetic inputs are mu0=.1, a=.005, b=.01, V0=5e-6 m/s, Dc=.001 m, with
rho=2670 kg/m³ (the in-plane paper's density, combined with **our** G).
Thus cs=19.3528 m/s and eta_rad=25836.0214 Pa s/m. These are not a calibrated
BP3 material set or inherited Fei stability parameters. The 2000 Pa strength
drop is not added as residual cohesion.

The fixed-phase energy remains
`psi=tau:tau/(4G)+G A:A/h`. The point power partition is

    (tau-Q):W=chi[mu pn V+eta_rad V²],
    D_Maxwell=(tau:tau+h Q:Q)/(2 eta).

Its transect integral equals `mu pn V+eta_rad V²` for coherent uniform
traction/speed. There is no extra K energy or accumulated-slip hardening.
The BE quadratic remainder is retained separately from physical losses.

## 4. Complete pressure/state analysis and the lagged comparison

On a smooth selected branch, let q=V/V0, w0=chi V0,
`deltaW=w0(m deltaq+m_theta deltatheta)`. The two algebraic rows are

    deltaR1=(m+mu P):deltatau-m:deltaQ-mu deltap
             -pn mu_Theta deltaTheta-(pn mu_q+eta_rad V0)deltaq,
    deltaR2=(m_theta+mu P_theta):deltatau-m_theta:deltaQ
             -pn_theta(mu_q deltaq+mu_Theta deltaTheta)
             +F_theta_theta deltatheta.

Here P=n tensor n, pn=p-n.tau.n, `pn_theta=-tau:P_theta` and
`F_theta_theta=(tau-Q):m_theta_theta+mu tau:P_theta_theta`.
These nonsymmetric equations accompany Maxwell tau/Q evolution, aging,
momentum `deltatau n-deltap n=0` and incompressibility `n.deltav=0`.
State is not phase. No symmetric potential/Hessian certificate is imported.

The full plane-wave pencil retains velocity, pressure, all independent
plane-strain stress/history components, rate and angle. Its independently
reduced form is `ydot=A y+D z`, `B y+L y+C z=0`, where y contains mechanical
normal-mode stress, planar mean/zz, Q components and state; z contains rate
and angle. L is precisely the normal-pressure/stress contribution. For
`R=pn mu_q+eta_rad V0>0` and strict angular curvature, the 2x2 C is invertible.
This removes the arbitrary-rate singularity on that smooth branch; it does
not prove continuity across conjugate ties. Full generalized QZ and tensor
finite differences check the elimination rather than assuming fixed pressure.

Fei's published A14 uses `(pn)_n=-(sigma_f)_n:(n tensor n)` in the new friction
residual. In our changing-plane comparison, this becomes
`pn_lag(theta)=p_old-n(theta).tau_old.n(theta)`. Keeping current normals while
projecting the old stress is an explicit extension of that temporal update;
it is not the paper's complete direction algorithm. Direction derivatives
still include the old stress. Freezing an old scalar pn while also changing
the normal would be a different rule.

The lagged algebraic row is `B y_new+L y_old+C z_new=0`. Consequently the
full discrete amplification matrix is

    Mlag=[Id-dt(A-D C^-1 B)]^-1 [Id-dt D C^-1 L],
    Mcoupled=[Id-dt(A-D C^-1(B+L))]^-1.

Both approach the same continuous generator as dt->0. Omitting the L y_old
term would falsely call lag stabilization a physical pressure decoupling.
The full discrete block solve independently verifies these matrices.

Six frozen states cover both mature conjugate limits, an archived developed
damage stress/backstress shape, both reversed-memory conjugates and a strict
noncoaxial reversal. Stress magnitudes are adjusted to the **new** kinetic
surface; these are labeled synthetic, not converted accepted checkpoints.
Angular scans include the earlier oblique failure normals. The full mixed
pencil is checked for k ell=.01–1000. All finite rates are independent of k
in this local quasistatic candidate: bounded short-wave response on a branch,
but no selected wavelength or slip-band width. State-off, direct-effect-off
and damping-off controls separate the mechanisms. With radiation retained,
the constant-friction active-branch limit is regular; removing both direct
effect and radiation makes C singular again. Positive bounded growth is not
a validated earthquake instability or nonlinear propagation qualification.

## 5. Nonlinear checks and unresolved global closure

A fresh exact-mature point starts with zero tensors and Theta=1. Engineering
shear rate .02/s reverses at .15 s and ends at .3 s; prescribed point pressure
is `10000+500 sin(2pi t/.3)` Pa. This local control has prescribed pressure;
the wave analysis above solves pressure. The fully coupled return reaches the
endpoint for dt=.0125/.00625/.003125 s with actual normal motion, reversal,
retained T/A/K and positive friction/radiation. The pressure-lag adaptation
fails its global direction test early. Independent 80-digit roots reproduce
the violations, so they are not repaired by loosening roundoff or accepting
only own-plane stationarity. No successful complete lagged path is claimed.

A separate prescribed coherent-plane velocity-step/reversal/stop reference
checks the bulk–surface mapping, aging and pressure-lag temporal error against
exact segmentwise aging. This is explicitly a fixed-plane **control**, not
permanent locking in the candidate. It does not qualify free-direction
continuation. The old finite-h rate-independent first-activation obstruction,
different-state endpoint multiplicity, intact mapping/state birth and global
branch switching remain unresolved. No propagation follows this task.
