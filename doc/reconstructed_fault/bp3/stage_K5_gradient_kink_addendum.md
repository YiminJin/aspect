# Frozen-history gradient-kink experiment

This is a small independent numerical experiment, not a new BP3 trajectory
or a replacement production operator. The current mature, bulk-work-measure
linearization is tested in a homogeneous straight periodic strip. Periodicity
removes physical tips from the experiment; BP3 endpoint topology is untouched.
Old stress and state are fixed. The positive frozen friction-rate tangent is
constant. This tests the infinitesimal, frozen-state mechanism identified by
the previous production impulse experiments, not finite-amplitude RSF aging.

Use units ell=1, kappa=1, and Ls=8, Ln=8. The compact C1 localization is
chi(n)=(1+cos(pi*n))/2 for |n|<1, zero otherwise, with integral one. It is a
manufactured localization, not a change to the BP3 phase/degradation profile.
The frozen friction tangent is d=0.036, representative of
(sigma*a/V)*ell/kappa in the saved deep BP3 state. Rate perturbations have a
unit triangular depression centered at s=3, with corners at 2,3,4. Rates on
the right half s>=4 are prescribed at zero perturbation; the actual positive
base rate can be added, with an arbitrarily small perturbation amplitude.
The continuous target is exactly representable on every selected fault grid.

For a Fourier mode (k,l), elimination of incompressible bulk velocity gives
the elastic surface symbol

    E(k) = kappa/Ln sum_l |chi_hat(l)|^2 4*k^2*l^2/(k^2+l^2)^2.

The (0,0) mode contributes kappa/Ln: the periodic bulk cannot accommodate a
mean shear strain. chi_hat is the unnormalized Fourier transform. This
periodic-strip mean constraint is shared by the reference and FE problem.
Weighted normal traction cancels by reflection symmetry; verify that in the
discrete bulk response rather than substituting G=B^T in general.

Manufacture the applied load as (E+d) applied to the exact triangle using
its analytic Fourier coefficients, independently of the tested bulk FE
operator. The discrete comparison uses Q2 velocity, Q1 pressure, Q1 surface
rate, and the same 2*kappa strain/source work pairing as production. Compare
bulk-relaxed matrices K-B^T A^-1 B, not only frozen-bulk K. No production
source, history, friction, or solver criterion changes are involved.

At fixed fault spacing 1/8 (50 m if ell=400 m), start with tangential bulk
spacings 1/2,1/4,1/8 and fixed normal spacing 1/4. Refine normal spacing only
to distinguish normal error if necessary. First use common-partition
quadrature so the bulk/fault-space test is not confounded by quadrature across
fault-element breaks; separately compare native three-point bulk quadrature.
Double the reference Fourier cutoff to bound reference uncertainty. Check
constant localization reproduction, symmetry, fresh full block residuals,
Fourier manufactured load, and the normal-traction cancellation.

Decisions: does a continuous gradient kink yield alternating rate errors,
and do these errors contract under bulk refinement at fixed fault grid? If
so, test the smallest resolution/quadrature remedy without modifying BP3.
If the independently resolved finite-width reference itself requires a
notch, report that instead of removing it by smoothing. A manufactured pass
cannot by itself qualify a new BP3 discretization or explain the entire
15/18/40-km history.

## Conditional trace control, selected after the first kink results

The resolved bulk operator retained a strongly alternating Q1 response to
an imposed-node impulse. To distinguish that from bulk under-resolution,
add an inexpensive exact-bulk control. Manufacture

    w*(s) = -(1-cos(pi*s/4))/2 on 0<=s<4; zero on 4<s<8.

This has a *deliberate rate jump* at the imposed/free interface; it is not
the original continuous gradient kink or an assertion that BP3 has this
exact solution. Its Fourier coefficients and continuum load are analytic.
Compare continuous Q1 with a separate free-side endpoint trace at s=4,
leaving the prescribed interval and all integration measures unchanged.
Refine only this tiny surface system; no additional bulk/ASPECT runs are
needed. A successful independent trace is an experimental remedy for a
trace-incompatible problem, not authorization to revise production Q1.

Finally reuse the already assembled bulk eliminations with the exact
regularized asinh RSF law, frozen Theta=Dc/Vp, deep BP3 a,b,mu0, and a 1%-Vp
manufactured perturbation. This checks that the mechanism and trace remedy
are not artifacts of linearizing friction. It is still a homogeneous
diagnostic with the manufactured localization, not a finite-box BP3 replay.
