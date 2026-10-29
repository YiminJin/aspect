# Scalar spatial rate/state feasibility: stop before native integration

The selected adaptation **does not control the demonstrated mixed mechanical
defect**. Do not proceed to its cross-fault prototype or phase propagation.
This conclusion concerns our two-conjugate Coulomb extension, not a refutation
of Pranger's different flow law. Constant energetic total-K storage is retired
as the mature-RSF candidate; its qualified gradient-hardening results remain.

Worktree: `pf-rsf-particle-free`, revision
`f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved entry work.
[Predeclared selection](PRANGER_FEASIBILITY_SELECTION.md) and
[complete formulation, derivatives and phase implications](PRANGER_FEASIBILITY_FORMULATION.md)
give the reproducible mathematical contract. This is an independent spectral
audit, not a new native constitutive implementation.

## One candidate, with its operator kept explicit

Retain `tau=T-2GA`, `Tdot+(G/eta)T=2G D3(v)`,
`Adot+(G/eta)A=W`, `Kdot=W`, incompressible momentum and physical
`pn_i=p0+p'-n_i.tau.n_i`. Both freely optimized, zero-dilation Coulomb systems
contribute `W=sum w_i m_i`. Their changing angles and pressure derivatives
remain in the mixed tangent. Phase is held fixed for this audit.

Use a common equivalent rate `q=sqrt(2W:W)/gamma_ref`, positive frictional
state Theta, the specified asinh RSF coefficient, and

```
Theta_dot = r0 [1 - Theta (q + ell_r² Delta q)].
```

The explicit operator enters **state evolution only**. This is an adaptation
of [Pranger et al. (2022), equations 15/16/26](https://doi.org/10.1029/2021JB023511),
with explicitly additional asinh creep, isotropic 2-D interaction, Maxwell
memory, phase energetics and two Coulomb families. Its multiplier is
`1-(k ell_r)^2`. The paper's alternative implicit Helmholtz averaging has
multiplier `1/[1+(k ell_r)^2]`; it is shown only for comparison, never substituted
or assigned the same calibration. Uniform `c1=c2=1` does not calibrate a
localized band. No extra damping or stored state/K-gradient energy is added.

Retained inputs: G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², phase ell=.046875 m,
m=4, Kf=8 Pa, strength drop=2000 Pa, p0=10000 Pa, residual cohesion=0.
Exploratory inputs: mu0=.1, a=.005, b=.01, gamma_ref=1e-4 /s,
reference width=.05 m, V0=5e-6 m/s, Dc=.001 m, r0=.005 /s, ell_r=.003 m.
The spatial length is uncalibrated and distinct from phase/mesh lengths.

Interface slip is `delta=2 integral K:S dr` [m] and
`V=2 integral W:S dr` [m/s], not boundary velocity. Only coherent uniform
shear across the reference width gives q=V/V0 and the scalar interface aging
equations. Free Coulomb planes still give mature traction
`mu p0/sqrt(1+mu²)`, rather than aligned-interface `mu p0`.

## Decisive mixed-response evidence

The audit eliminates pressure using actual incompressible equilibrium, not
fixed-pressure stiffness. It retains full 3-D plane-strain stress, both rates,
both angle derivatives, relaxing tensor memory and state. Six base states
include strict single slip, simultaneous slip, unequal system rates, an exact
steady Maxwell/RSF control, reversed memory and the explicit mature limit.

At maturity, Q=0 and both systems are active. Opposite rate increments give
`deltaW=diag(e,-e,0)` while `deltaq=0`. For a 45-degree wave normal, an admissible
divergence-free velocity has strain rate deltaW. Physical stress, pressure,
state and plane-angle perturbations all vanish. Auxiliary T/A histories still
advance and cancel in reconstructed stress. This solves every linearized row
at arbitrary wavelength and temporal exponent. It is a **singular mixed
evolution**, not a predicted physical instability rate. Positive direct rate
dependence cannot see this directional redistribution.

The finite-damage counterexample has phi=.99, Axy=-.002, Qxy=-.050758845 Pa,
tau_xy=994.986934 Pa and two positive rates. Minimum physical compression is
9005.01 Pa. It is a synthetic admissible instantaneous state, **not an accepted
native checkpoint**. Its independently derived constraint denominator is

```
D(alpha) = 1/h + (Qxy/tau_xy) cos²(2 alpha).
```

It vanishes at wave normals 30.04149248 and 59.95850752 degrees. Neither the
state multiplier nor a/b enters D. At k ell_r=10, approaching the first normal
from .01 to .0001 degrees below increases frozen growth from 1.11e3 to
1.11e5 /s. Removing state feedback gives essentially the same pole. These
instantaneous rates are not event predictions. Fixed-angle high-k damping in
other controls cannot establish a uniform oblique short-wave bound. Ordinary
RSF velocity weakening instead has determined finite-wavelength growth rates;
the mature kernel is present even without state weakening.

| Independent numerical check | Measured result | Existing allowance |
|---|---:|---:|
| Nonlinear rate/angle/pressure/state directional finite differences | 1.37e-9 relative | 1e-6 |
| Full mixed generalized QZ versus constrained elimination | 3.35e-9 relative | 1e-8 |
| Matrix denominator versus analytic expression | 6.00e-16 absolute | 1e-12 |
| Mature null at four wavenumbers and three assigned exponents | 0 recorded relative residual | 1e-12 |

The angular root was independently evaluated with 80-digit arithmetic.
The complete 25,992-row bounded symbol comparison takes 6.48 s wall;
plots take 4.93 s. These are kernel costs, not simulator performance.
The old native constant-friction reversal guard still rejects its separate
step-38 three-root example; its error and incoming state are retained.

With b=0 and a>0, state/spatial interaction has no mechanical feedback and the
directional defect persists. As a tends to zero, a single-slip control's
algebraic condition rises 80.8 to 8083 over two decades; the exact constant
friction endpoint loses another rank. The asinh formula is undefined at a=0:
the original Coulomb inclusion is a singular limit, not a regular stabilized
constant-friction law.

## Limits and next decision

No sustained-sliding, velocity-step, mesh/dt or native qualification is claimed:
the prerequisite for that conditional stage failed. Removing K-gradient energy
would remove its accumulated-slip storage, but does not by itself deliver a
usable mature-RSF model.

The formulation records a provisional whole-material aging clock without a
damage-threshold reset, its intact-boundary issue, retained fracture energy and
physical obstacle. Asinh creep also changes the exact intact reaction/strength
calibration. Those scientific choices and a nonsymmetric phase/branch test
would need review; the old potential Hessian cannot be reused. None is adopted.

**Recommendation: stop this common-scalar-state candidate.** The missing closure
must control independent conjugate/directional flow, not just its equivalent
rate. Selecting directional kinetics or a unique flow rule is a new constitutive
decision for scientific review. No extra viscosity, system state, plane lock,
different spatial operator or phase mask is added to repair the failed test.
Production/BP3 behavior and all earlier energy/spatial/branch failures remain.

## Figures and compact evidence

![Mixed mechanical gate](../../../benchmarks/reconstructed_fault/particle_free/evidence/pranger_feasibility/mechanical-gate.png)

The top row distinguishes operators and a benign finite-h steady control. The
bottom row shows the reversed-memory pole and the mature null residual; assigned
exponents verify nonuniqueness and are not simulated growth rates.

![Mature null mode](../../../benchmarks/reconstructed_fault/particle_free/evidence/pranger_feasibility/mature-null-mode.png)

An imposed analytic transverse velocity produces an oblique tensor-flow mode.
Opposite conjugate rate changes leave the scalar rate/state and physical stress
unchanged to first order. This visualizes a failed regularization mechanism,
not crack geometry or a specimen trajectory.

[PNG/CSV/JSON/code/commands archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/pranger_feasibility.zip)
includes definitions, units, checksums and reproducible commands. No large
solver output, MPI expansion or propagation run was generated.
