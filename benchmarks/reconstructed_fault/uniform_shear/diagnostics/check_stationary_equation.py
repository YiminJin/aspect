#!/usr/bin/env python3
"""Independent continuum equation audit; never changes initialized H or phi."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


ELL, A, CORE, ACTIVATION, BASELINE_H = .15625, 64., .6, .1, .5


def degradation(p):
    d = (1-p)**2+128*p*(1+p)
    return (1-p)**2/d


def degradation_derivative(p):
    d = (1-p)**2+128*p*(1+p)
    return -128*(1-p)*(1+3*p)/d**2


def h(p):
    return 128*p*(1+p)/(1-p)**2


def h_derivative(p):
    return 128*(1+3*p)/(1-p)**3


def coordinate(p):
    # Endpoint-regularized quadrature of the intended first integral.
    def jacobian(t):
        value = CORE*math.sin(t)**2
        return 2*ELL*math.sqrt(1+CORE)*(1-value)/math.sqrt(3-CORE-(1+CORE)*value)
    return quad(jacobian, math.asin(math.sqrt(p/CORE)), math.pi/2,
                epsabs=1e-13, epsrel=1e-13)[0]


def stationary_H(p):
    return A*CORE/(h(CORE)*degradation(p)**2)


def second_derivative(p):
    # Differentiate ell² phi'² = phi - CORE*h(phi)/h(CORE).
    return (1-CORE*h_derivative(p)/h(CORE))/(2*ELL**2)


def main():
    output = Path(__file__).parent/"results"
    samples = np.array([.6,.5,.3,.12,.100001,.1,.08,.04,.02,.01,.001,0.])
    analytic_H = stationary_H(samples)
    implemented_H = np.where(samples>ACTIVATION,analytic_H,BASELINE_H)
    analytic_residual = analytic_H*degradation_derivative(samples)+A-2*A*ELL**2*second_derivative(samples)
    implemented_residual = implemented_H*degradation_derivative(samples)+A-2*A*ELL**2*second_derivative(samples)
    np.savetxt(output/"stationary_equation_samples.csv",
               np.column_stack(([coordinate(p) for p in samples],samples,analytic_H,implemented_H,
                                second_derivative(samples),analytic_residual,implemented_residual)),
               delimiter=",",header="y,phi,intended_H,implemented_H,phi_second_derivative,intended_residual_Pa,implemented_residual_Pa",comments="")

    # A finite-difference check of the implicit profile is independent of
    # differentiating its first integral. Stay away from the activation jump
    # and compact-support endpoint for this smooth-branch verification.
    fd_checks=[]
    for p in (.5,.3,.12,.08,.02):
        y=coordinate(p)
        def profile(z):
            return brentq(lambda value: coordinate(value)-z,0.,CORE,xtol=1e-14)
        errors=[]
        for spacing in (2e-4,1e-4,5e-5):
            d2=(profile(y+spacing)-2*p+profile(y-spacing))/spacing**2
            errors.append(abs(stationary_H(p)*degradation_derivative(p)+A-2*A*ELL**2*d2))
        fd_checks.append(dict(phi=p,spacings_m=[2e-4,1e-4,5e-5],residual_errors_Pa=errors))

    particle_rows=np.genfromtxt(output/"raw"/"before_phase_particles.csv",delimiter=",",names=True)
    inactive=particle_rows["H"] == BASELINE_H
    report=dict(
        equation="H*g'(phi) + A - 2*A*ell^2*phi'' = 0",
        A_Pa=A, ell_m=ELL, untruncated_identity_max_abs_residual_Pa=float(abs(analytic_residual).max()),
        half_support_m=coordinate(0.), activation_coordinate_m=coordinate(ACTIVATION),
        activation_phi=ACTIVATION, stationary_H_at_activation_Pa=float(stationary_H(ACTIVATION)),
        retained_baseline_H_Pa=BASELINE_H,
        activation_inactive_side_residual_Pa=float(BASELINE_H*degradation_derivative(ACTIVATION)+A-2*A*ELL**2*second_derivative(ACTIVATION)),
        intended_support_inner_limit_residual_with_baseline_Pa=float(implemented_residual[-1]),
        intact_exterior_residual_Pa=BASELINE_H*degradation_derivative(0.)+A,
        analytic_H_intact_limit_Pa=float(stationary_H(0.)),
        baseline_equals_stationary_H_phi=brentq(lambda p: stationary_H(p)-BASELINE_H,0.,ACTIVATION),
        active_particles=int(np.sum(~inactive)), baseline_particles=int(np.sum(inactive)),
        innermost_baseline_particle_abs_y_m=float(np.min(abs(particle_rows["y"][inactive]))),
        outermost_active_particle_abs_y_m=float(np.max(abs(particle_rows["y"][~inactive]))),
        finite_difference_checks=fd_checks,
        globally_stationary_for_implemented_H=False)
    assert report["untruncated_identity_max_abs_residual_Pa"] < 1e-10
    assert report["intact_exterior_residual_Pa"] == 0.
    assert abs(report["activation_inactive_side_residual_Pa"]) > 1.
    for check in fd_checks:
        assert check["residual_errors_Pa"][-1] < 1e-3
    (output/"stationary_equation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
