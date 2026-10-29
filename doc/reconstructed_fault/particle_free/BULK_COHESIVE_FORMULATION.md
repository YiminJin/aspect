# Selected bulk-cohesive stage: discrete formulation before coupled runs

20 October 2026; entry f1f9f29cf plus retained P4a/manual work. The user-selected
`codex_bulk_phase_implementation_brief.md` and accompanying derivation authorize
a distinct default-off bulk law. Earlier P2/P3/P4a evidence is not relabeled.

## Local backward Euler and the intact limit

Use dt>0, r=dt G/eta, beta=1/(1+r), engineering shear a and signed k,
absolute accumulated shear L. All tensors include zz and both xy components.
For a displacement increment dE=dev3 symgrad(du),

    T+ = beta (Tn + 2G dE),  t0=T+:S,  abar=beta an,
    t = t0-G a+, q=G a+/h,
    w=((1+r)a+ - an)/dt,
    t-q in tau_r partial |w|.

At h>0, let c=G(1+1/h). If |t0-c abar|<=tau_r, a+=abar.
Otherwise s=sign(t0-c abar), a+=h(t0-s tau_r)/(G(1+h)).
Compute the stick/slip test as |h t0-G(1+h)abar|<=h tau_r to avoid division
by tiny h. At h=0, require an=0 (irreversibility); a+=0 and
q=sign(t0)(|t0|-tau_r)+. This is the specified relaxed virgin multiplier,
not an empirical source or a positive-damage floor. For positive phase trials
recompute the return map from the same incoming state; do not freeze a=0.
Publish k+=kn+dt w and L+=Ln+dt|w| only after complete acceptance.

On a sliding branch, da/dt0=h/[G(1+h)], da/dphi=h' q/[G(1+h)],
dq/dt0=1/(1+h), dq/dphi=-h' q/(1+h). On sticking, da/dt0=da/dphi=0,
dq/dt0=0 and dq/dphi=-q h'/h. At intact points use the appropriate
one-sided sliding/sticking limit. Switch derivatives are semismooth.

## Native coupled residual, tangent and constraints

Solve incompressible quasistatic mechanics with physical tau=T-2GaS,
and the physical phase obstacle phi+>=phin, 0<=phi+<1:

    Rv = integral tau:dev3 symgrad(test v) - p div(test v) - external,
    Rp = -integral test p div(v),
    gphi = grad Ef - integral N h' q²/(2G).

Ef=Kf integral(phi+ell²|grad phi|²), counted once. No H maximum is used.
Evaluate all residuals/energies from the same local return. The local phase
derivative is partial at fixed physical histories; its consistent Newton
derivative includes the eliminated-state dq, giving
dY/dphi=(h'' q²+2 h' q dq/dphi)/(2G). Mechanical tangents include da/dt0
and da/dphi, and phase/mechanical cross actions include dq/dt0.
Backward Euler makes the cross blocks differ by beta; this is generally a
nonsymmetric coupled residual, not the old surface endpoint potential.
The state-dependent viscous metric is not differentiated as stored energy.

Use native fixed mesh/FE/constraints and the verified independent-master
phase coordinates: complete C^T g and C^T H C on raw physical contributions.
Accepted physical obstacles must obey the same constraints; no artificial
constrained diagonal participates in KKT/stability checks. Old surface terms
and friction are absent from this bulk mechanical residual.

## Exact discrete accounting

Per quadrature point, the Maxwell/contact equations imply

    tau+:dE = dEb + dEc + dt(Db+Dc+Df)
               + |dtau|³D²/(4G) + hn(dq)²/(2G) + q+² dh/(2G).

Here Db=|tau+|³D²/(2eta), Dc=h+q+²/eta, Df=tau_r|w|.
For the AT1 FE functional,

    Ldt = integral[|dtau|³D²/(4G)+hn(dq)²/(2G)
                   +q+²(dh-h'+ dphi)/(2G)]
          +Kf ell² integral|grad dphi|²,
    Wext-d(Eb+Ec+Ef)-dt Dphysical = Ldt - gphi.dphi - Rmechanical.du.

Signs of Rmechanical follow internal minus external. Ldt includes a negative
convex-h quadrature term and is not guaranteed nonnegative for finite steps.
Report its pieces, external remainder, phase residual work and mechanical
residual work separately. An exact algebraic ledger alone is not qualification.
Rate-independent jumps/phi=1 are not silently regularized or accepted.

## Ownership, scope and ordered screens

M4 owns accepted T, a, signed/absolute shear and physical phase obstacle;
every trial starts from the same accepted state. M5 determines acceptance;
rejection leaves it unchanged and native checkpointing preserves it. Fixed
bulk DGQ2 T retains all nine modes; local a histories must be represented
and integrated consistently, without throwing away the normal residual.
For the subsequently qualified fixed-mesh constant-coefficient non-affine
source, T is represented by independent DGQ2 plus a native FE gradient
complement; see [the exact mapping and lifecycle](BULK_SOURCE_HISTORY_REPORT.md).
The point constitutive equations above are unchanged. The raw DG polynomial
alone is not the complete auxiliary stress history on a bilinear distorted cell.
Surface geometry is diagnostic only. Observer reconstruction never changes
physical histories/solution/obstacle; I=0 profiles have no normalized row.
Use actual positive quadrature weights for moment/residual identities.

Run in order: material points; straight-band moment/residual identities;
then one native fixed-mesh finite-tip event and only its prescribed dt/mesh/
quadrature/lifecycle controls. No continuation past a failed prerequisite.
Local identity screen 1e-12 scaled; branch tangent screen 1e-7; profile identity
screen 1e-10. Before the native event, record exact mesh/load/time interval,
PDE KKT/residual and energy gates in its input selection. Preserve all failed
evidence; do not tune the law or loosen criteria to obtain a growth picture.
