# One explicit scalar-state candidate and its mechanical prerequisite

This is a feasibility formulation, not a production replacement. The constant
energetic total-K gradient is **retired as the mature-RSF candidate**; its
previous results remain a qualified gradient-hardening experiment.

## Candidate, variables and spatial operator

Use full 3-D plane strain, small-strain kinematics and the retained histories:

```
sigma=tau-p Id,  p=p0+p',  div v=0,  div sigma=0,
Tdot+gamma_M T=2G D3(v),  Adot+gamma_M A=W,  Kdot=W,
Ldot=sqrt(2 W:W),
tau=T-2G A,  Q=2G A/h(phi),  gamma_M=G/eta,
h(phi)=4 phi(1+phi)/(1-phi)^2.
```

T is auxiliary tensor history; tau is physical deviatoric stress. A is relaxing
shear memory; K is signed accumulated tensor history, with no new gradient
energy. L is passive accumulated equivalent strain. Frictional state Theta is
dimensionless and is not physical phase phi. D3 is the full three-dimensional
trace-free symmetric strain rate. All rates here are material rates on the
fixed, small-strain configuration; no transport or objective-rate claim is added.

For each freely optimized conjugate family i=+,-, put

```
mi=sign_i sym(ti⊗ni),  mi:mi=1/2,
pni=p0+p'-ni.tau.ni >0,
fi=(tau-Q):mi-c_r-mu(q,Theta) pni <=0,
W=sum_i wi mi,  wi>=0,  wi fi=0,
partial fi/partial psi_i=0,  partial² fi/partial psi_i²<0
```

on an active strict plane maximum. This maximization holds the CURRENT common
mu fixed; the simultaneous solve then enforces mu(q,Theta), and its
linearization includes deltamu. Substituting angle-dependent q before varying
a different functional would add a new direction-selection rule. Psi denotes
plane angle, not state. Both
families and their changing directions are retained; there is no dilation.
One common equivalent rate is

```
s=sqrt(2 W:W) [1/s],  q=s/gamma_ref,
mu=a asinh[(q/2)exp((mu_ref+b lnTheta)/a)],
mu_ref=a ln[2 sinh(mu0/a)],  a>0,
Theta_dot=r0[1-Theta M_ex(q)],
M_ex(q)=q+ell_r² Delta q.
```

Mu(1,1)=mu0 and mu(0,Theta)=0. This is regularized creep, not a finite static
friction interval. The spatial operator acts **only in state evolution**, not
on stress, pressure, frictional traction, phase or flow direction. Use an
isotropic physical-space Laplacian in the 2-D audit and zero external normal
derivative of q. The whole-domain weak state equation contains
`-ell_r² integral grad q . grad(Theta test)` after integration by parts.
No internal phase-boundary no-flux condition or mask is silently inserted.

This is an explicitly labeled adaptation of [Pranger et al. (2022)](https://doi.org/10.1029/2021JB023511),
their equations 15/16/26. Their explicit operator has Fourier multiplier
`1-(k ell_r)²`. The implicit averaging alternative in their equation 25 solves
`qbar-ell_r² Delta qbar=q`, with multiplier `1/[1+(k ell_r)²]`.
These agree only to low order at small k ell_r. In particular, M_ex is not a
positive averaging operator. The paper used the explicit approximation in its
numerical examples. We do not substitute the implicit inverse, inherit its
positivity or claim that either operator has the same localized calibration.

Set c1=c2=1 in the aging law for the uniform reference only. These are not
the paper's localized-profile calibration constants. Normalization of a
convolution or zero integral of a Laplacian alone does not prove equivalence
to nonlinear interface RSF. The paper's invariant-based single flow direction
also differs from our two-conjugate flow cone; this difference is decisive.
The asinh creep regularization and its exact mu(1,1) calibration above are
additional assumptions of this candidate, not the paper's logarithmic law.

Feasibility inputs: G=1e6 Pa, eta=1e8 Pa s, p0=10000 Pa, c_r=0,
mu0=.1, a=.005, b=.01, gamma_ref=1e-4 /s, reference width Lref=.05 m,
V0=gamma_ref Lref=5e-6 m/s, Dc=.001 m, r0=V0/Dc=.005 /s,
ell_r=.003 m. The last rate/state/length inputs are uncalibrated feasibility
choices, not edits to BP3. Gc=1 J/m², phase ell=.046875 m, Kf=8 Pa and m=4
remain the bulk reference parameters. No extra Kelvin viscosity, radiation
damping, slip viscosity or artificial dissipation is added.
The retained 2000 Pa strength-drop parameter is distinct from c_r=0; it is
not added a second time as residual frictional cohesion.

For a fixed straight chart S=sym(t⊗n), define
`delta=2 integral K:S dr` [m], `V=2 integral W:S dr` [m/s]. Under coherent
monotonic W parallel to S, V=integral s dr. Under nonproportional tensor flow,
integral s dr is equivalent slip speed and need not equal signed V. Boundary
velocity also includes elastic/Maxwell deformation. A uniform layer of width
Lref gives q=V/V0 and theta_dimensional=Theta/r0, recovering the scalar
reference aging and friction-coefficient equations. The free-plane traction
still differs from a prescribed-interface traction: in mature pure shear it
is mu p0/sqrt(1+mu²), not mu p0. We do not lock planes or recalibrate mu to
hide that distinction. A nonuniform transect requires separate
calibration before claiming interface agreement.

## Exact mixed linearization, including directions and pressure

At a homogeneous active state q=Theta=1, use perturbations proportional to
`exp(lambda t+i k n.x)`. State variation is

```
lambda deltaTheta=-r0 deltaTheta-r0[1-(k ell_r)²] deltaq,
deltaq=2 W:deltaW/(gamma_ref² q),
deltamu=mu_q deltaq+mu_Theta deltaTheta,
deltaW=sum_i(mi deltawi+wi mi' deltapsi_i).
```

With Ni=ni⊗ni and Fi=tau-Q, the complete active algebraic rows are

```
deltafi=(mi+mu Ni):deltatau-mi:deltaQ-mu deltap-pni deltamu=0,
deltafi_psi=(mi'+mu Ni'):deltatau-mi':deltaQ
            -pni' deltamu+fi_psipsi deltapsi_i=0,
fi_psipsi=Fi:mi''+mu Ni'':tau.
```

The mechanical and history rows are

```
(lambda+gamma_M)deltatau=2G[D3 sym(i k deltav⊗n)-deltaW],
(lambda+gamma_M)deltaQ=(2G/h)deltaW,
deltatau n-deltap n=0,  n.deltav=0.
```

For wave angle alpha, put v2=(cos 2alpha,sin 2alpha). Mechanical equilibrium
requires `(deltatau_xx-deltatau_yy)/2, deltatau_xy = G t v2` and
`deltap=G(t+mean)`, where `mean=(deltatau_xx+deltatau_yy)/(2G)`.
The mean/zz mode relaxes at -gamma_M. This eliminates velocity and pressure
exactly; it does not fix pressure. Dynamic variables are t, mean, Qxx/G,
Qxy/G and Theta. Algebraic variables are both rate and plane-angle variations.
The resulting full mixed pencil is checked against direct generalized QZ
eigenvalues and independent nonlinear finite differences.
Here k>0; entries labeled k ell_r=0 in the spectrum are the long-wave limit,
not the unconstrained spatially uniform boundary-value problem. Spatial
wavelength is 2 pi ell_r/(k ell_r): for the exploratory 3 mm length, k ell_r
of 1/10/100 corresponds to 18.85/1.885/.1885 mm. No phase-length or mesh-length
identification is made.

Finite-h dynamics in tau/Q are an invertible rewrite of the retained T/A
histories. Passive K adds neutral memory modes but no mechanical feedback.
In the explicit mature specialization Q=Ecoh=0; auxiliary A/T memory is still
recoverable from their Maxwell equations and must not be erased.

## Structural failure: scalar rate cannot control conjugate redistribution

On the pure-shear mature branch tau_xy=mu p0/sqrt(1+mu²), Q=0, both conjugates
are critical. Their xy flow components agree; xx components have opposite
sign. Therefore equal-and-opposite rate variations produce

```
deltaW=diag(epsilon,-epsilon,0),  deltaq=0,
deltamu=deltaTheta=deltatau=deltap=deltapsi_i=0.
```

For an oblique n=(1,1)/sqrt(2), a transverse divergence-free velocity has
`D3 sym(i k deltav⊗n)=deltaW`. All mechanical, Coulomb, direction and state
rows vanish for **any k and any temporal lambda**. The unchanged planes are
a consequence of unchanged stress/mu in this mode, not a direction lock.
Both base rates are strictly positive, so sufficiently small opposite changes
remain admissible. Maxwell histories satisfy

```
deltaA_aux=deltaW/(lambda+gamma_M),
deltaT=2G deltaW/(lambda+gamma_M),  deltaK=deltaW/lambda.
```

Their physical stress cancels exactly. Neither Maxwell relaxation, positive
RSF direct effect, the state operator nor a fixed phase can remove this mode.
The mature mixed pencil is singular: an arbitrary growth value is evidence of
nonunique linearized evolution, not a predicted earthquake growth rate.

At finite h, a simultaneous-slip state with Q=kappa tau has a q-preserving
rate-redistribution constraint. For symmetric base W parallel to shear tau,
its mechanically differentiated denominator is proportional to

```
D(alpha)=1/h+kappa cos²(2alpha).
```

More explicitly, the algebraic left-null constraint is
deltaQxx-kappa deltatauxx=0. Its mechanically relaxed rate action on a
q-preserving deltaWxx is 2G D(alpha) deltaWxx. This independently supplies
the denominator used to check the mixed-matrix elimination.

No a, b, state length or scalar-operator multiplier enters this denominator.
For the synthetic admissible reversal state phi=.99, Axy=-.002,
Qxy=-.0507588447 Pa, tau_xy=994.9869339 Pa, both flow rates are positive.
D vanishes at oblique normals 30.04149248 and 59.95850752 degrees.
This state has small tensor memory and strictly compressive physical stress;
it is not labeled a native accepted checkpoint. The retained independent
native constant-friction return separately rejects a reversal at step38 with
three roots. That guard is preserved, not used as acceptance for this state.

The positive-history/single-system controls can look stabilized at large k
for a fixed angle. They cannot certify the missing directional coercivity.
Near the reversed-memory pole, frozen growth is unbounded with wave normal,
also when b=0; a joint high-k/angular sequence defeats any uniform short-wave
bound. This is not merely finite-wavelength velocity weakening. Physical RSF
instability would have determined rates and scales; the mature pencil here
does not determine even the linear evolution at an oblique wave.

## State-off and constant-friction limits

With b=0 and a>0, the direct rate dependence remains but state has no feedback
on mechanics. Consequently the spatial interaction also disappears from
mechanical response. The scalar direct effect still cannot penalize the
equivalent-rate-preserving conjugate mode. The original Maxwell eta is not
this direct rate slope.

Letting a→0 is singular: mu_q tends to zero on a finite-positive-rate branch,
the active rate/angle algebraic matrix loses another rank, and the differential
order/constraints change. At a=b=0 use the original constant-friction inclusion;
the asinh formula with division by a is not defined. No positive-length state
interaction survives as a mechanical cure when state dependence is absent.
There is no claim of a regular constant-friction limit or inherited calibration.

## Formal phase coupling and unresolved physical choices

This candidate proposes retaining the bulk free energy
`tau:tau/(4G)+G A:A/h+Kf[phi+ell²|grad phi|²]`, the physical phase obstacle
and phase force `Kf-2Kf ell² Delta phi-h' Q:Q/(4G)`; no K-gradient energy
remains. Frictional dissipation is still `sum mu pni wi>=0` under closed
contact. Theta has no new recoverable energy. There is no associated
incremental potential or permission to reuse the symmetric phase Hessian.

An explicit provisional birth convention is Theta=1 at fresh initialization
throughout the material and retention of that accepted material clock at
first damage; it is not reset at a phase threshold. In intact regions W=A=0,
q=0; away from active interfaces Theta_dot=r0 (dimensional age rate one).
Across an intact boundary the chosen whole-domain M_ex can affect the latent
clock through Delta q. There is no hidden phase mask or internal no-flux wall.
Its weak regularity/positivity and physical meaning would need qualification;
an alternative contact-age birth rule would be different physics.

For c_r=0 and this regularized mu(0)=0, the exact intact reaction is
Q=P_S tau, not the old finite-Coulomb sticking interval. Thus the intact
phase driving/strength calibration is not automatically preserved merely by
retaining m and Gc. At true maturity Q=Ecoh=0 while auxiliary stress memory
persists. These limits and all coupled nonsymmetric phase/state derivatives,
state birth, branch selection and energy must be settled before a propagation
model could be proposed. They are not implemented here, because the earlier
mechanical prerequisite already fails.
