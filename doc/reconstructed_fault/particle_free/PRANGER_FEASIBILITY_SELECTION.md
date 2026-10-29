# One rate/state candidate — selection before the mechanical audit

The energetic total-K gradient is retired as the candidate mature-RSF
regularizer. Its accepted gradient-hardening experiments and archives remain.
This feasibility task selects no production replacement or phase propagation.

Analyze one explicit adaptation of Pranger et al. (2022), equations 15/16/26:
common equivalent bulk rate q=sqrt(2 W:W)/gamma_ref, one positive frictional
state Theta, regularized RSF mu(q,Theta), and

```
Theta_dot = r0 [1 - Theta M_ex(q)],
M_ex(q) = q + ell_r² Laplacian(q).
```

M_ex enters STATE EVOLUTION ONLY. It does not average stress, friction, flow
direction or phase, and adds no stored energy or extra Kelvin viscosity.
Use an isotropic physical-space Laplacian for the 2-D oblique-wave test;
this is an explicit extension of the paper's transect approximation. Outer
microboundary: zero normal derivative of q. Do not substitute the implicit
Helmholtz inverse (paper Eq25), whose multiplier is 1/(1+(k ell_r)^2), for
this explicit multiplier 1-(k ell_r)^2. No localized-profile calibration or
stability claim is inherited. Set c1=c2=1 for the homogeneous reference only.

Retain Tdot+(G/eta)T=2G D3(v), Adot+(G/eta)A=W, Kdot=W,
tau=T-2G A, Q=2G A/h(phi), physical sigma=tau-p Id, incompressibility,
both orientation-maximized Coulomb conjugates and zero dilation. In explicit
mature mode Q=Ecoh=0 but relaxing stress/slip memory remains auxiliary; do
not insert phi=1 into the finite-h API. State is distinct from physical phase.

For positive a define mu=a asinh[(q/2)exp((mu_ref+b lnTheta)/a)], where
mu_ref=a ln[2 sinh(mu0/a)] makes mu(1,1)=mu0 exactly. q=0 has mu=0;
this is creep regularization, not a finite static-friction interval. The
a=b=0 endpoint is the original constant-friction inclusion, not evaluation
of a singular asinh formula. Removing state dependence means b=0 with a>0.

G=1e6 Pa, eta=1e8 Pa s, m=4, Gc=1 J/m², ell=.046875 m,
p0=10000 Pa and c_r=0 are retained. Feasibility inputs mu0=.1, a=.005,
b=.01, gamma_ref=1e-4 /s, reference layer width .05 m, V0=5e-6 m/s,
Dc=.001 m, r0=V0/Dc=.005 /s, ell_r=.003 m are explicit uncalibrated
inputs, not changed BP3 parameters. Interface V is the integral of projected
bulk slip-strain rate, not boundary velocity or an unnormalized invariant.

Before any transect prototype, derive/check the complete frozen-coefficient
mixed displacement–pressure–flow–state pencil with changing plane directions.
Use a strict single-system archived-state control, simultaneous conjugate
slip, nonproportional flow, an admissible reversed tensor memory and exact
mature Q=0. Match mu=.1, q=1, Theta=1 at the tested instantaneous states.
Report which are synthetic admissible states and which come from archived
native data. They are not checkpoint conversions or new accepted trajectories.

Check angular rank loss and high-k growth against the identical rate law
with M(q)=q, then b=0 and the singular constant-friction limit. Verify
mechanical/pressure constraints and full tensor metrics, not just a scalar
strain-rate diffusion coefficient. If a structural mechanical prerequisite
fails, stop: do not add per-system viscosities, states, selection potentials,
direction locks, phase masks or a different operator to make the test pass.
Only a supported candidate proceeds to the user's small cross-fault prototype.
