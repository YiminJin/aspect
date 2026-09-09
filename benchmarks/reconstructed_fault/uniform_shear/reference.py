#!/usr/bin/env python3
"""Independent K1 scalar preflight; SI units throughout.

This deliberately does not clip a non-interior root to V_min. The initial
previous-profile integral is an explicit input, not inferred from a retained
I_h snapshot: those quantities are different in the current initialization.
"""

import argparse
import math
from dataclasses import dataclass

from scipy.integrate import quad
from scipy.optimize import brentq


@dataclass(frozen=True)
class Histories:
    stress: float = 1500.0  # Pa, retained particle tau_xy
    cohesive: float = 1000.0 * 0.4 / math.sqrt(1.6)  # Pa, ideal profile only
    theta: float = 200.0  # s


def ideal_profile(relative_tolerance=1e-12):
    """Stationary AT1 first integral, not the ASPECT/CPDI solution."""
    core, ell, m = 0.6, 0.15625, 128.0

    def h(phi):
        return m * phi * (1.0 + phi) / (1.0 - phi)**2

    # phi=core sin²(t) removes the integrable endpoint singularities.
    def jacobian(t):
        phi = core * math.sin(t)**2
        # Cancel the endpoint factors analytically; the resulting integrand
        # is finite even when an inversion explicitly evaluates an endpoint.
        return (2.0 * ell * math.sqrt(1.0+core) * (1.0-phi)
                / math.sqrt(3.0-core-(1.0+core)*phi))

    options = dict(epsabs=1e-11, epsrel=relative_tolerance)
    half_width = quad(jacobian, 0.0, math.pi/2.0, **options)[0]
    integral = 2.0 * quad(lambda t: h(core * math.sin(t)**2) * jacobian(t),
                          0.0, math.pi/2.0, **options)[0]
    return half_width, integral


def residual(velocity, histories, dt, loading, integral,
             previous_profile_integral):
    """R_Gamma density after eliminating the uniform bulk shear stress.

    The stored previous I_h equals integral for K1. The actual integral of
    h(phi_previous) is separate, and need not equal that snapshot at step zero.
    Domain width W=1 m; coefficients are homogeneous bulk/surface quantities.
    """
    beta = math.exp(-dt / 100.0)
    kappa = -1e8 * math.expm1(-dt / 100.0)
    history_slip = (beta * histories.cohesive / kappa
                    * (integral - previous_profile_integral))
    stress = beta * histories.stress + kappa * (loading - velocity - history_slip)
    cohesive = kappa / integral * velocity + beta * histories.cohesive
    friction = 0.025 * math.asinh(
        velocity / (2.0 * 1e-5)
        * math.exp((0.6 + 0.013 * math.log(histories.theta * 1e-5 / 0.001)) / 0.025))
    return stress - cohesive - 1000.0 * friction - 1e5 * velocity


def solve_step(histories, dt, loading, integral, previous_profile_integral,
               initial=False):
    """Return evaluated response and separate retained/advanced histories.

    Callers initialize Histories once. They must pass this returned history
    to subsequent calls, never replace it with later ASPECT output.
    """
    if not (dt > 0 and integral > 0 and previous_profile_integral >= 0
            and histories.theta > 0):
        raise ValueError("Invalid dimensional reference inputs")
    function = lambda v: residual(v, histories, dt, loading, integral,
                                 previous_profile_integral)
    lower = 1e-12
    if function(lower) <= 0:
        raise ValueError(f"No interior root: R(V_min)={function(lower):.12g} Pa")
    upper = 1e-4
    while function(upper) > 0:
        upper *= 2.0
    velocity = brentq(function, lower, upper, xtol=1e-16, rtol=1e-14)
    if abs(function(velocity)) > 1e-7:
        raise ArithmeticError("Independent traction root did not meet 1e-7 Pa accuracy")

    beta = math.exp(-dt / 100.0)
    kappa = -1e8 * math.expm1(-dt / 100.0)
    stress = (beta * histories.stress + kappa * (loading - velocity)
              - beta * histories.cohesive * (integral - previous_profile_integral))
    cohesive = kappa / integral * velocity + beta * histories.cohesive
    response = dict(V=velocity, stress=stress, cohesive=cohesive)
    if initial:
        return response, histories
    increment = -math.expm1(-velocity * dt / 0.001)
    theta = (histories.theta * math.exp(-velocity * dt / 0.001)
             + 0.001 / velocity * increment)
    return response, Histories(stress, cohesive, theta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-profile", choices=("zero", "initial"), default="initial",
                        help="'initial' is the approved rule; 'zero' reproduces the former initialization defect")
    args = parser.parse_args()
    width, integral = ideal_profile()
    tighter_integral = ideal_profile(5e-13)[1]
    assert abs(integral - tighter_integral) < 1e-9
    histories = Histories()
    previous_integral = 0.0 if args.previous_profile == "zero" else integral
    print(f"Ideal half support = {width:.15g} m; I_h = {integral:.15g} m")
    print(f"Retained initial histories: {histories}")
    print(f"Integral of actual previous h profile = {previous_integral:.15g} m")
    print(f"Initial residual at V_min = {residual(1e-12, histories, 2., 1e-4, integral, previous_integral):.15g} Pa")
    try:
        evaluated, retained = solve_step(histories, 2., 1e-4, integral,
                                         previous_integral, initial=True)
    except ValueError as error:
        print(f"K1 ROOT GATE BLOCKED: {error}")
        return 2
    assert retained == histories
    print(f"Evaluated initial response: {evaluated}")
    print("Initial history retention verified; this is not a production pilot result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
