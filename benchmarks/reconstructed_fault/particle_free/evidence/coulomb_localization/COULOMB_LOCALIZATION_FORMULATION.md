# Mixed mechanical localization and possible remedies

This is an audit of the retained zero-dilation Coulomb law, not adoption of a
regularized or dilatant law. The measured result and scope are in
[COULOMB_LOCALIZATION_REPORT.md](COULOMB_LOCALIZATION_REPORT.md). The actual
native consistent derivative includes changing directions and pressure; the
independent continuous-rate construction below is an additional check.

## Actual law and mixed symbol

Use tensile-positive physical stress `sigma=tau-(p0+p')Id`, `p0=10000 Pa`,
`tau=T-2G A`, `Q=2G A/h`, `h=4 phi(1+phi)/(1-phi)^2`. All tensor contractions
are three-dimensional plane strain. Flow lies in the in-plane trace-free
space. With plane tangent t and plane normal nu, put
`m_i=k_i sym(t_i*nu_i)`, `k_i=+1,-1`. The two conjugate systems satisfy

```
f_i=(tau-Q):m_i-c_r-mu[p0+p'-nu_i.tau.nu_i] <= 0,
W=sum_i w_i m_i, w_i>=0, w_i f_i=0,
Tdot+(G/eta)T=2G D3(v),  Adot+(G/eta)A=W,  Kdot=W.
```

Here c_r=0, mu=.1. Established A/K tensors are not rotated. The native
algorithm solves the complete orientation-maximized return, differentiating
its active equations including `delta m_i`. Stress is not the auxiliary T.

For a spatial wave normal n and displacement amplitude u define
`E(u,n)=D3 sym(u*n)`, `C=partial tau/partial epsilon`, and
`P=partial tau/partial(p0+p')`. Phase and incoming histories are fixed for
this derivative. Momentum and incompressibility have principal symbol

```
M(n) = [ Q(n)       b(n) ],   Q(n)u=[C:E(u,n)]n,
       [ n^T         0  ]    b(n)=(P-Id)n.
```

The pressure amplitude scales with the strain amplitude, and is expressed
in units of G in exported dimensionless symbols. The constant background
pressure has no small-strain geometric stiffness term in the implemented
equations. It enters the physical traction/friction law. A mean-pressure
constraint cannot remove a local zero-mean wave packet.

Let a be the unit transverse vector, `n.a=0`. In the basis (a,n), singularity
of the full 3x3 symbol is equivalent to singularity of

```
[ a.Qa   a.b ]
[ n.Qa   n.b ].
```

When `n.b!=0`, the relevant coefficient is
`Lambda=a.Qa-(a.b)(n.Qa)/(n.b)`. Setting pressure fixed omits the second
term and can give the opposite conclusion. We check actual determinant
zeros and full 3x3 null vectors, not just eigenvalues of a symmetrized
tangent. A real rank-loss normal is a local mechanical localization mode;
its angle is neither a Coulomb slip-plane angle nor a phase-front angle.
Finite-domain matrices can remain invertible in its presence.

## Independent continuous-rate check

At a continuous state, set `N_i=m_i+mu nu_i*nu_i`. A jump in the rate has
continuous stress, history and orientation: the instantaneous consistency
equations involve the present active planes. For strict active systems,

```
H_ji = N_j:m_i + (m_j:m_i)/h,
delta tau = 2G[D3(delta v)-sum_i m_i (H^-1)_ij N_j:D3(delta v)]
            +mu sum_i m_i (H^-1)_ij delta p.
```

The final term means summing over both active indices with a vector of ones
in the pressure right-hand side. Stick is elastic. At h=0, A/W remain zero
and the intact mechanical tangent is elastic, including the intact finite-Q
limit. The audit samples phi through .99; phi=1 is the h->infinity limit,
not a finite-compliance evaluation.

For one active system let alpha measure n from its plane tangent, choosing
the conjugate sign consistently. Independent tensor contraction gives

```
Lambda/G = 1 - cos(2alpha)[cos(2alpha)+mu sin(2alpha)]
                /[1+1/h-mu sin(2alpha)].
```

A common positive Maxwell algorithmic factor does not change the zeros of
this frozen-rate expression. The actual finite-step derivative has additional
direction-response terms and is also checked. Loss begins when

```
1/h < max_t 4t(mu-t)/(1+t^2)^2,    t=tan(alpha).
```

The maximizing root satisfies `mu-2t-3mu t^2+2t^3=0`. At mu=.1,
`hcrit=100.49813740694826`, hence `phi_crit=.7676112792918935` for m=4.
The mirror conjugate has the same minimum. This is a single-system threshold,
not a state-independent threshold for two simultaneously active systems.
The sampled double-system states lose ellipticity at larger phase as well.

## Phase and Maxwell terms

The physical AT1 phase force is `Kf-Y-2Kf ell^2 Laplacian(phi)`, with Kf=8 Pa,
and physical lower obstacle the accepted phase. Its gradient term penalizes
phase gradients. It does not penalize gradients in K, A or velocity when
`delta phi=0`. The selected developed-damage state has strict positive
obstacle force. Its critical phase directions therefore have `delta phi=0`;
the mechanical localization mode is admissible without a phase perturbation.

Even formally releasing phase cannot cure the high-frequency mechanical
principal symbol. With normalized strain amplitude E and `delta p=G pi`,
linearized free-phase elimination reads

```
delta phi = [(Y_e:E)/Kf+(G Y_p/Kf) pi]
             /[2(k ell)^2-Y_phi/Kf].
```

Its feedback through `tau_phi` is O(k^-2). This is a diagnostic free-phase
calculation, not an obstacle-admissible sinusoid at a strict active obstacle,
nor a proposed nonsymmetric stability certificate. At fixed physical ell
the short-wave mechanical symbol is the same negative limit. Long-wave
coupling may introduce additional instabilities; this audit need not classify
them to reject the prerequisite.

Maxwell relaxation has local time eta/G=100 s. At a continuous rate jump,
the local relaxation terms have no jump. Backward Euler contributes a
positive beta=(1+dt G/eta)^-1; finite-step direction effects change the
apparent modulus but approach the nonelliptic rate limit as dt decreases.
There is no slip-gradient length or direct slip-rate viscosity in this law.
This is a continuum ellipticity defect; neither a smaller timestep nor a
more permissive nonlinear acceptance condition supplies missing regularity.
We do not infer a dynamic growth rate from this quasistatic symbol.

## Remedy comparison: scientific choices, not implemented corrections

**Spatial slip regularization is the recommended next design to review.**
For example, retain existing shear memories and add recoverable storage in
the *physical accumulated tensor* K:

```
psi_grad=G ell_s^2 grad(K):grad(K),
M_K=2G ell_s^2 grad(K),  X=-div(M_K),
E_eff=tau-Q-X,  Kdot=W.
```

Yield uses `E_eff:m_i` and the original physical normal pressure. The energy
identity includes boundary microtraction work `(M_K n):Kdot`. A microfree
boundary has `M_K n=0`; another condition requires explicit physical input.
Frictional dissipation is `(tau-Q-X):W=sum_i(c_r+mu pn_i)w_i>=0`.
Uniform material points recover the present law. The flow equations and
their orientation derivatives must be rederived with X; simply smoothing
stress or limiting coefficients is not this model.

For the frozen single-system symbol, the denominator becomes
`1+1/h+(k ell_s)^2-mu sin(2alpha)`. Thus high k returns to elastic mechanical
behavior. At h>hcrit the cutoff is
`k_c ell_s=sqrt(1/hcrit-1/h)` (about .05483 at h=144). This controls the
unbounded short-wave defect; it does not certify every finite-wavelength
coupled mode. Choose ell_s from a physical slip/process-zone scale, independently
of mesh size; identifying it with phase ell is an additional assumption.
This needs a native tensor field, microstress weak form, boundary conditions,
complete mixed tangent, energy and checkpoint qualification. Existing K
becomes energetic: activating the term in loaded archives silently would add
storage. No value, history conversion or implementation is selected here.

A physically measured slip viscosity is another possible rate regularization:
`f_i=eta_s w_i`, `eta_s>0`, giving additional `sum eta_s w_i^2` dissipation.
It changes the rate response. In this quasistatic, inertia-free problem it
does not itself prescribe an independent spatial band width; the present
Maxwell viscosity must not be relabeled as this new slip viscosity.

**A consistently dilatant law is a different physical model.** Let d>=0
be the dilation ratio, retain shear flow `W_s=sum w_i m_i`, and introduce
normal plastic flow `W_d=sum d w_i nu_i*nu_i`. One consistent extension is

```
Adot+(G/eta)A=W_s,
Bdot+(G/eta)B=dev(W_d),
Tdot+(G/eta)T=2G D3(v),  tau=T-2G(A+B),  Q=2G A/h,
zetadot=d sum_i w_i,  div(v)=d sum_i w_i.
```

Here B is a new full-plane-strain deviatoric memory and zeta records plastic
volume. Elastic incompressibility no longer means isochoric total motion.
With the same yield resistance, total stress power gives
`D_pl=sum_i[c_r+(mu-d)pn_i]w_i>=0` for `0<=d<=mu`. For associated d=mu,
c_r=0, plastic frictional heating is zero: shear work pays for dilation against
compression. If mu is instead meant to remain a measured dissipative friction
coefficient, strength calibration must change. Dilation cannot be added only
to the tangent or orientation selector.

For a frozen active plane the rate consistency matrix becomes
`H_ji=N_j:dev(m_i+d nu_i*nu_i)+(m_j:m_i)/h`; momentum, pressure derivatives
and `div(v)-d sum w_i=0` must all be differentiated together. The old n.u=0
symbol is inapplicable. Associated flow with positive hardening removes the
specific non-associated mechanism, but does not establish a finite band width
in the vanishing-hardening limit. Partial dilation remains non-associated and
needs its own localization audit. All-displacement, zero-net-volume block
boundaries cannot support net positive dilation: normal release, physical bulk
compressibility or a porosity law would be required. These are substantial
scientific changes, unlike a solver fix. No dilatancy is adopted.

The general need to test non-associated localization and distinguish gradient
from rate regularization is consistent with the primary discussion in
[de Borst et al. (1993)](https://pure.tue.nl/ws/files/1428273/605429.pdf).
The formulas and numerical thresholds above are derived for the actual retained
law, not imported from that paper. Review a physical slip-length/microstress
model first; native phase-active integration remains blocked meanwhile.
