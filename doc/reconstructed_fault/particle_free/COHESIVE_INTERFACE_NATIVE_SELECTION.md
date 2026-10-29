# Native fixed-Gamma realization: bounded selection

Keep production/BP3 and previous references unchanged. Add a separate experimental
native constitutive component and opt-in benchmark postprocessor. Use ASPECT's
actual Q2/Q1 mesh/FE/AffineConstraints, reconstructed-fault manager, normal-profile
associations, property ownership and native checkpoint serialization. This is an
isolated native kernel/owner realization, not replacement of the production
Stokes/phase driver.

Fixed horizontal through-going geometry, serial, periodic along fault, clamped
outer walls; full plane-strain mixed mechanics with current contact pressure.
A prescribed normalized hat kernel chi=(1-|y-y_fault|/w)/w inside |y-y_fault|<w
represents the geometric slip. Its support knots coincide with mesh lines.
Use E=dev3(symgrad d_R)-chi*z*S for bulk storage/stress, where z is Maxwell
relaxed geometric jump, not an additional cohesive spring. Surface fracture and
bond/contact law are exactly the prior selected model. Surface nonlinear terms
use explicitly declared nodal quadrature, paired with nodal weak tractions; the
bulk coupling uses the continuous Q1 surface basis and its exact work adjoint.
No old bulk fracture energy is included. Compare this finite-width discretization
with the sharp-interface reference; do not assume equality at finite w.

H=.125 m, fault at y=.05 m, periodic L=1 m, w=.025 m initially. Native geometry
ell is not a second physical band energy; this w is a representation parameter.
G/eta/Gc/tau_p/RSF/radiation match the accepted reference. Pressure perturbation
has zero bulk mean, with p0=10000 Pa and pn=p0+p'-tau_nn. Surface points with
nonpositive pn are rejected. No opening or new-direction birth is enabled.

Tests: homogeneous exact intact initiation (dt=.002/.001 s to .42 s), mature
interface sliding (velocity change, hold/reversal through 6 s, dt=.01 s), then
a short nonuniform developed interface with full pressure feedback. Compare
native exports against the independent return at identical operator/time data.
Use bounded 4x40, 8x80, 16x160 mesh/operator checks and one width control as needed
to separate FE error from finite-width representation. Gauss3 main integration
and a Gauss6 polynomial control; no adaptive quadrature or changed production rule.
Require force <=1e-8 Pa, relative tangent FD error <1e-6 after FD-step refinement,
energy ledger <=1e-10 max(1,work), uniform response agreement <=1e-7 Pa,
operator reciprocity/work identities at roundoff, and <2% successive FE operator
changes for the sampled resolved modes. Do not redefine finite-width differences
as FE error. Native manager properties and bulk history publish jointly once;
rejected candidates preserve accepted state, retries reproduce the same result,
and a real simulator checkpoint/restart matches uninterrupted advancement.

## Recorded amendments from the bounded diagnosis

The initial nodal surface quadrature is **retired**: independent relaxed-tangent
analysis found a negative alternating mode during uniform formation. Its result
and source are preserved in the development archive. The final law uses Gauss3
surface integration with Gauss-owned irreversible damage/state histories and a
prepublication relaxed-curvature check. No material parameter/tolerance changes.
The nonuniform interval was extended from .02 to .15 s to exercise appreciable
frictional sliding and pressure feedback, with both near-zero and loaded FD
checks. FD increments are refined relative to the actual asinh transition scale;
the 1e-6 derivative target is unchanged.

One additional 32x160 operator control isolates the remaining surface/FE resolution
effect; the same mesh also receives the half-width comparison. The old nonnegative
manager rate API is preserved; this signed law publishes a distinct registered
signed-rate property. Native manager ownership is exercised through an isolated
ASPECT postprocessor instance because the simulator-owned manager's phase/particle
startup contract describes the legacy model. This task does not enable or claim
replacement of the production Stokes/phase driver.
