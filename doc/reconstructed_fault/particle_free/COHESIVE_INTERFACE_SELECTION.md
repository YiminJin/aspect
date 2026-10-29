# Recoverable compliance plus geometric contact: predeclared small references

Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
New experimental constitutive option only; production/BP3 stay untouched.

Use a surface damage a in [0,1], geometric jump delta, immutable bond reference b,
and one frictional state Theta per record. Recoverable jump e=delta-b;
c(a)=c0 a/(1-a), psi=e²/(2c), fracture inventory Gc a.
Closed contact friction is in parallel: s=q+a R(V,pn,Theta), q=e/c,
V=delta_dot. Damage irreversibility gives a=max(a_old,min(|e|/delta_f,1)),
with the exact intact tied constraint when a=0. No phase or compliance floor.
This splits bond storage from contact dissipation, rather than forcing their
rates through the same old bulk history source. It is not an additive series
spring/slider (which would lose frictional resistance when its spring vanishes).

Gc=1 J/m², G=1e6 Pa, eta=1e8 Pa s, ell=.046875 m, nominal strength drop 2000 Pa,
p_ref=p0=10000 Pa, mu_ref=.1. Calibrate peak tau_p=3000 Pa, delta_f=2Gc/tau_p,
c0=2Gc/tau_p². Peak is fixed at that reference calibration; it is not silently
made a pressure-dependent stored-energy parameter. RSF retains a=.005,b=.01,
V0=5e-6 m/s,Dc=.001 m,rho=2670 kg/m³ and the existing radiation coefficient.
Linear contact activation a is a declared new interpolation assumption.

Fracture energy is surface-owned Gc a, never added again as bulk Gc Gamma.
Gamma/I may spread the surface work/loads, with an exact paired adjoint, but
surface admission is defined even without a pre-existing diffuse phase field.
No zero-support normalization is evaluated. Geometric support is created at
zero damage/zero jump before an active formation solve; candidate intact
segments remain tied, not pre-opened. A display/geometry phase is not the new
surface damage a and cannot be counted as a second fracture energy.

Tests in order:
1. Exact intact onset and homogeneous loading, using a .125 m Maxwell block,
   dt=.002/.001/.0005 s; direct one-sided compliance and zero-support checks.
   Require onset-bracket convergence to the analytic intact traction threshold,
   force <=1e-8 Pa, irreversible a, and whole-path energy error <=1% Gc.
   Check return/pressure derivatives, loaded history mapping with explicit
   switch energy, rejection/retry, once-only publication and serialized restart.
2. Mature a=1 interface sliding through several Dc, velocity step, hold and
   reversal. Compare directly with the same-law interface reference. Preserve
   all stress/slip/state histories, exact zero cohesive energy and contact.
   Energy error <=1% of friction+radiation loss, both signed and absolute sums.
3. A finite seed on a straight prospective interface between finite homogeneous
   incompressible Maxwell layers. Derive the full two-component interface
   compliance from the mixed mechanical layer problem. Unequal layer heights
   exercise current normal-pressure feedback; no pressure lag. Use a bounded
   loading horizon selected before execution, then a time-halving and one
   spatial refinement. All intact segments on the candidate path may form;
   do not mask phase growth near the tip. Off-path branching is explicitly
   outside this first path-restricted demonstration.

Spatial/temporal criteria: <=2% / <=1% in integrated damage, slip and reactions,
with physical absolute floors (1e-5 Pa, 1e-10 m, 1e-10 J/m²). Front positions
must agree within two fine surface cells. Do not demand monotone differences
below demonstrated numerical uncertainty. Separate Gamma-profile visualization,
mechanical discretization and actual surface fracture. Retain physical and
algorithmic energy remainders separately; do not invoke the old symmetric
Hessian certificate for the pressure-dependent contact law.

Seeded test fixed before execution: periodic length 2 m, upper/lower clamped
layer heights .075/.05 m, mature seed |x|<.125 m and otherwise exactly intact
candidate plane. Fresh zero stress/slip, Theta=1 everywhere. Apply relative
wall speed .001 m/s to .30 s, then hold to .32 s; N=128/256 and dt=.002/.001 s.
Initial fracture inventory .25 J/m is reported separately. This is a finite
periodic specimen, not distant-boundary evidence. Measure cohesive-front and
fully-decohered fronts separately. Check the symmetric part of the actual
nonsymmetric incremental force Jacobian on free interface variables: positive
curvature is a sufficient local strong-monotonicity check, not the retired
phase-potential stability certificate. No artificial intact-row diagonals enter.
