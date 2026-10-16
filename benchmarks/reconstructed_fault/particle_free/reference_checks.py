#!/usr/bin/env python3
"""PF1 V1--V4 independent local/characteristic references (no ASPECT import).

SI scales are fixed before execution: 1 Pa, 1 m, 1 s, 1 J/m, and 1 Pa/s.
Tensor norms use all nine entries of the 3D plane-strain embedding. Integral
tests use unit tangential length and the stated normal-coordinate measure.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss


S = np.array([[0., .5, 0.], [.5, 0., 0.], [0., 0., 0.]])
G, ETA = 3., 6.  # Pa, Pa s; lambda = 2 s
LAM = ETA / G
RECORDS = []


def check(name, actual, reference, scale=1., tolerance=1.e-12, units="1"):
    absolute = float(np.linalg.norm(np.asarray(actual) - reference))
    relative = absolute / max(scale, float(np.linalg.norm(reference)))
    passed = bool(np.isfinite(relative) and relative <= tolerance)
    RECORDS.append(dict(name=name, absolute=absolute, scaled=relative,
                        scale=scale, tolerance=tolerance, units=units, passed=passed))
    if not passed:
        raise AssertionError(f"{name}: {relative} > {tolerance}")


def negative(name, measured, minimum, units="1"):
    passed = bool(np.isfinite(measured) and measured > minimum)
    RECORDS.append(dict(name=name, measured=float(measured), minimum=minimum,
                        units=units, passed=passed, kind="expected nonzero defect"))
    if not passed:
        raise AssertionError(f"{name}: {measured} <= {minimum}")


def gaussian(y, center=0., width=.2):
    return np.exp(-.5 * ((y-center)/width)**2) / (math.sqrt(2*math.pi)*width)


def rule(a, b, n=160):
    x, w = leggauss(n)
    return a + (b-a)*(x+1)/2, w*(b-a)/2


def v1():
    # Arbitrary compatible nonzero initial deviator, including zz.
    tau0 = np.array([[.3, .2, 0.], [.2, -.1, 0.], [0., 0., -.2]])
    grad = np.array([[.2, .4, 0.], [.4, -.05, 0.], [0., 0., 0.]])
    strain = grad - np.trace(grad)/3*np.eye(3)
    slip0, rate, chi, I = .13, .07, 1.2, .8
    check("V1 S:S", np.sum(S*S), .5)
    check("V1 trace S", np.trace(S), 0.)
    check("V1 full plane-strain norm", np.sum(strain*strain),
          strain[0, 0]**2+strain[1, 1]**2+(strain[0, 0]+strain[1, 1])**2+2*strain[0, 1]**2)
    for ratio in (1.e-16, 1.e-12, 1.e-6, .1, 1., 50.):
        beta = math.exp(-ratio)
        kappa = -ETA*math.expm1(-ratio)
        with localcontext() as ctx:
            ctx.prec = 70
            rb = (-Decimal(str(ratio))).exp()
            rk = Decimal(str(ETA))*(1-rb)
            beta_ref, kappa_ref = float(rb), float(rk)
        check(f"V1 beta {ratio}", beta, beta_ref)
        # Scaling by eta*r for tiny steps exposes cancellation in the increment.
        check(f"V1 kappa {ratio}", kappa, kappa_ref,
              scale=ETA*min(ratio, 1.), units="Pa s")
        if ratio == 1.e-16:
            negative("V1 naive 1-exp cancellation relative defect",
                     abs(ETA*(1-beta)-kappa_ref)/kappa_ref, .05)
        for forcing in (0., 1.):
            trial = beta*tau0+2*kappa*forcing*strain
            slip = beta*slip0+(kappa/G)*forcing*rate
            physical = trial-2*G*chi*slip*S
            # Integrate the physical Maxwell ODE directly, with frozen sources.
            initial = tau0-2*G*chi*slip0*S
            expected = beta_ref*initial+2*kappa_ref*forcing*(strain-chi*rate*S)
            check(f"V1 physical stress {ratio}/{forcing}", physical, expected, units="Pa")
            check(f"V1 trace stress {ratio}/{forcing}", np.trace(physical), 0., units="Pa")
            check(f"V1 traction units {ratio}/{forcing}", G*slip/I,
                  (G*beta_ref*slip0+kappa_ref*forcing*rate)/I, units="Pa")
    # Compatible zero physical stress and nonzero initial elastic slip can
    # require a narrow trial tensor; the representation is not always smooth.
    ys = np.linspace(-.04, .04, 81)
    initial_slip = .02
    correction = 2*G*initial_slip*gaussian(ys, width=.01)[:, None, None]*S
    narrow_trial = correction.copy()
    check("V1 narrow compatible initialization", narrow_trial-correction,
          np.zeros_like(narrow_trial), units="Pa")
    gradient = 2*G*initial_slip*math.exp(-.5)/(math.sqrt(2*math.pi)*.01**2)*np.linalg.norm(S)
    negative("V1 narrow initial trial history gradient Pa/m",
             gradient, 50., "Pa/m")


def v2():
    y, w = rule(-1., 1.)
    t, width0, widening, s, V = .3, .18, .11, .21, .08
    width = width0+widening*t
    # h=A(t) exp(-y^2/(2 width^2)); integrate the full real line analytically.
    def chi(time):
        return gaussian(y, width=width0+widening*time)
    c = chi(t)
    dc = c*widening*(-1/width+y*y/width**3)
    tr = np.array([[.1, .5, 0.], [.5, -.2, 0.], [0., 0., .1]])
    eps = .04*S
    trdot = 2*G*eps-tr/LAM
    sdot = V-s/LAM
    qgeo = -2*G*s*dc[:, None, None]*S
    # Independent derivative of the reconstructed stress, by complex step.
    h = 1.e-30
    derivative = np.imag((tr+1j*h*trdot)[None, :, :]
                         -2*G*chi(t+1j*h)[:, None, None]*(s+1j*h*sdot)*S)/h
    tau = tr[None, :, :]-2*G*c[:, None, None]*s*S
    rhs = 2*G*(eps[None, :, :]-c[:, None, None]*V*S)+qgeo
    check("V2 evolving-width product rule", derivative+tau/LAM, rhs, units="Pa/s")
    negative("V2 omitted redistribution norm", np.linalg.norm(qgeo), .1, "Pa/s")
    # Pure h amplitude changes cancel exactly in chi=h/I, even when h_dot != 0.
    amplitude = lambda time: np.exp(.7*time)
    normalized = lambda time: amplitude(time)*np.exp(-y*y/(2*width*width))/(amplitude(time)*math.sqrt(2*math.pi)*width)
    check("V2 amplitude cancellation", np.imag(normalized(t+1j*h))/h, np.zeros_like(y), tolerance=1.e-12, units="1/(m s)")
    new_c = gaussian(y, width=width*1.2)
    changed = tr-2*G*new_c[:, None, None]*s*S
    jump = -2*G*(new_c-c)[:, None, None]*s*S
    check("V2 cohesive physical phase jump", changed-tau, jump, units="Pa")
    negative("V2 physical phase jump nonzero", np.linalg.norm(jump), .1, "Pa")
    # Mature remap changes the residual, preserving the physical tensor.
    residual0 = .12*S
    residual1 = residual0+2*G*(c-new_c)[:, None, None]*s*S
    remapped = tr-2*G*new_c[:, None, None]*s*S-residual1
    check("V2 mature stress-preserving remap", remapped, tau-residual0, units="Pa")

    # Eq variableG: manufactured solution of the explicitly stated stress law,
    # no extra (D_t G/G) tau term. Separate rate and modulus controls.
    for varying_rate, g_rate in ((True, 0.), (False, .17)):
        rb = (1+.4*y)/LAM if varying_rate else np.full_like(y, 1/LAM)
        rf = 1/LAM
        gt = G*np.exp(g_rate*t)
        st = V/rf*(1-np.exp(-rf*t))+.09*np.exp(-rf*t)
        # Tensor amplitude a solves a_dot+rb*a=2G(t)*e0.
        e0, a0 = .03, .4
        def transformed(time):
            a = a0*np.exp(-rb*time)+2*G*e0*(np.exp(g_rate*time)-np.exp(-rb*time))/(g_rate+rb)
            memory = V/rf*(1-np.exp(-rf*time))+.09*np.exp(-rf*time)
            return a+2*G*np.exp(g_rate*time)*c*memory
        T = transformed(t)
        Tdot = np.imag(transformed(t+1j*h))/h
        # Original forcing uses e=e0+chi V, so a obeys the stated law.
        extra = 2*g_rate*gt*c*st+2*gt*(rb-rf)*c*st
        predicted = 2*gt*(e0+c*V)+extra
        check(f"V2 variableG rate={varying_rate}", (Tdot+rb*T)[:, None, None]*S,
              predicted[:, None, None]*S, units="Pa/s")
        negative(f"V2 omitted coefficient term rate={varying_rate}", np.linalg.norm(extra), .01, "Pa/s")

    # Restricted fixed-profile energy identity, per unit length along the fault.
    I = .8
    tau = 12*S-2*G*c[:, None, None]*s*S
    eps = .1*S
    td = 2*G*(eps-c[:, None, None]*V*S)-tau/LAM
    power = np.sum(w*np.einsum("nij,ij->n", tau, eps))
    loss = np.sum(w*np.einsum("nij,nij->n", tau, tau))/(2*ETA)
    drive = np.sum(w*c*np.einsum("nij,ij->n", tau, S))
    q = G*s/I
    friction = drive-q  # Define balance; verifies energy algebra, not RSF.
    negative("V2 positive fixed-profile resistance", friction, 0., "Pa")
    edot = np.sum(w*np.einsum("nij,nij->n", tau, td))/(2*G)+G*s*sdot/I
    check("V2 fixedenergy power", edot, power-loss-friction*V-G*s*s/(LAM*I), units="W/m")

    # Phase energy: fixed incoming trial stress and s; unprojected I integral.
    # Native CPDI fracture and projected-I derivatives are expressly not tested.
    base = .3*np.exp(-y*y/(2*.25**2))
    by = -y/.25**2*base
    m, ell, fracture = 2., .12, .4
    def energy(a):
        phi = a*base
        hh = m*phi*(1+phi)/(1-phi)**2
        ii = np.sum(w*hh)
        stress = 12*S-2*G*s*(hh/ii)[:, None, None]*S
        return (np.sum(w*np.einsum("nij,nij->n", stress, stress))/(4*G)
                +G*s*s/(2*ii)+fracture*np.sum(w*(phi+ell*ell*(a*by)**2)))
    a = 1.1
    phi = a*base
    hh = m*phi*(1+phi)/(1-phi)**2
    dh = m*(1+3*phi)/(1-phi)**3*base
    ii, di = np.sum(w*hh), np.sum(w*dh)
    delta_chi = dh/ii-hh*di/ii**2
    tau = 12*S-2*G*s*(hh/ii)[:, None, None]*S
    variation = (-np.sum(w*np.einsum("nij,ij->n", tau, S)*s*delta_chi)
                 -G*s*s/(2*ii**2)*di
                 +fracture*np.sum(w*(base+2*ell*ell*a*by*by)))
    check("V2 phasevar complex derivative", np.imag(energy(a+1j*h))/h, variation, units="J/m")
    errors = []
    for step in (1.e-2, 5.e-3, 2.5e-3):
        fd = (energy(a+step)-energy(a-step))/(2*step)
        errors.append(abs(fd-variation))
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    negative("V2 phase derivative central-difference minimum order", min(orders), 1.98)
    check("V2 phase derivative central-difference finest", fd, variation, tolerance=1.e-5, units="J/m")
    return dict(phase_fd_steps=[.01, .005, .0025], phase_fd_errors=errors, phase_fd_orders=orders.tolist())


def v3():
    T, rate = .8, .23
    beta = math.exp(-T/LAM)
    st = rate*LAM*(1-beta)
    # Direct characteristic integral versus transformed residual's independent
    # forcing integral. The latter uses the analytic commutator and scalar ODE.
    for name, velocity, profile_velocity, bulk_tan, surface_tan in (
            ("frozen", 0., 0., 0., 0.),
            ("rigid translation", .31, .31, 0., 0.),
            ("fixed profile crossing", .31, 0., 0., 0.),
            ("tangential mismatch", 0., 0., .27, -.14)):
        direct, rebuilt, omitted, commutators, cohesive = [], [], [], [], []
        for y0 in np.linspace(-.45, .45, 9):
            def localization(t):
                return gaussian(y0+(velocity-profile_velocity)*t, width=.17)
            def surface_factor(t):
                return 1.+.3*np.sin(.2+(bulk_tan-surface_tan)*t)
            def cfield(t):
                z = y0+(velocity-profile_velocity)*t
                dc = -z/.17**2*(velocity-profile_velocity)*localization(t)
                tangential = .3*np.cos(.2+(bulk_tan-surface_tan)*t)*(bulk_tan-surface_tan)
                mem = rate*LAM*(-math.expm1(-t/LAM))
                return mem*(dc*surface_factor(t)+localization(t)*tangential)
            integral = quad(lambda t: math.exp(-(T-t)/LAM)*localization(t)*rate*surface_factor(t),
                            0., T, epsabs=2.e-14, epsrel=2.e-14)[0]
            res = -2*G*quad(lambda t: math.exp(-(T-t)/LAM)*cfield(t),
                            0., T, epsabs=2.e-14, epsrel=2.e-14)[0]
            # Integrate the physical cohesive stress law with Q_geo=-2G C.
            physical_cohesive = -2*G*quad(
                lambda t: math.exp(-(T-t)/LAM)
                *(localization(t)*rate*surface_factor(t)+cfield(t)),
                0., T, epsabs=2.e-14, epsrel=2.e-14)[0]
            d = -2*G*integral
            compact = -2*G*localization(T)*st*surface_factor(T)
            direct.append(d)
            rebuilt.append(compact-res)
            omitted.append(compact)
            commutators.append(cfield(T))
            cohesive.append(physical_cohesive)
        check("V3 characteristic/residual "+name, np.asarray(rebuilt)[:, None, None]*S,
              np.asarray(direct)[:, None, None]*S, units="Pa")
        check("V3 cohesive redistribution "+name, np.asarray(cohesive)[:, None, None]*S,
              np.asarray(omitted)[:, None, None]*S, units="Pa")
        if name in ("frozen", "rigid translation"):
            check("V3 commuting "+name, omitted, np.asarray(direct), units="Pa")
        else:
            negative("V3 omitted commutator "+name,
                     np.linalg.norm(np.asarray(omitted)-direct)*np.linalg.norm(S), .001, "Pa")
            negative("V3 C nonzero "+name, np.linalg.norm(commutators), .001, "1/s")

    # Counterfactual objective law: production has no spin. Rotate full tensors.
    omega = .7
    W = np.array([[0., -omega, 0.], [omega, 0., 0.], [0., 0., 0.]])
    def rotation(t):
        c, s = np.cos(omega*t), np.sin(omega*t)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    def rotated(t):
        R = rotation(t)
        return R@S@R.T
    A = rotated(T)
    Adot = np.imag(rotated(T+1j*1.e-30))/1.e-30
    check("V3 rigid rotation Jaumann", Adot+A@W-W@A, np.zeros((3, 3)), units="1/s")
    negative("V3 nonobjective rotation difference", np.linalg.norm(Adot), .1, "1/s")
    x0 = np.array([.2, .13, 0.])
    x, n = rotation(T)@x0, rotation(T)@np.array([0., 1., 0.])
    check("V3 rotating profile material value", gaussian(x@n), gaussian(x0[1]), units="1/m")
    # Exact rotating characteristic stress with relaxation and localized slip.
    tau0 = np.diag([.2, -.3, .1])
    R = rotation(T)
    physical = R@(beta*tau0-2*G*gaussian(x0[1])*st*S)@R.T
    trial = beta*R@tau0@R.T
    check("V3 objective stress reconstruction", trial-2*G*gaussian(x0[1])*st*A, physical, units="Pa")
    check("V3 rotating trace", np.trace(physical), 0., units="Pa")


def v4():
    T, switch, rate, sigma, sep = 1., .4, .2, .12, .7
    y, w = rule(-2., 2., 320)
    old, current = gaussian(y, -sep/2, sigma), gaussian(y, sep/2, sigma)
    first = LAM*rate*(math.exp(-(T-switch)/LAM)-math.exp(-T/LAM))
    second = LAM*rate*(-math.expm1(-(T-switch)/LAM))
    direct = -2*G*(first*old+second*current)[:, None, None]*S
    compact = -2*G*(first+second)*current[:, None, None]*S
    # Independently propagate the residual jump at profile switch and relaxation.
    s_at_switch = LAM*rate*(-math.expm1(-switch/LAM))
    residual = math.exp(-(T-switch)/LAM)*2*G*s_at_switch*(old-current)[:, None, None]*S
    check("V4 direct versus residual closure", compact-residual, direct, units="Pa")
    norm = lambda tensor: math.sqrt(float(np.sum(w*np.einsum("nij,nij->n", tensor, tensor))))
    defect = norm(compact-direct)
    # Closed infinite-domain Gaussian overlap; tails beyond [-2,2] negligible.
    analytic = 2*G*first*math.sqrt(.5/(math.sqrt(math.pi)*sigma)
                                  *(1-math.exp(-sep*sep/(4*sigma*sigma))))
    check("V4 missing-memory L2 analytic", defect, analytic, units="Pa sqrt(m)")
    negative("V4 current-profile-only mature defect", defect, .1, "Pa sqrt(m)")
    rows = []
    for panels in (16, 32, 64):
        integ = np.zeros_like(y)
        for lo, hi, profile in ((0., switch, old), (switch, T, current)):
            dt = (hi-lo)/panels
            times = lo+(np.arange(panels)+.5)*dt
            integ += rate*dt*np.sum(np.exp(-(T-times)/LAM))*profile
        numerical = -2*G*integ[:, None, None]*S
        rows.append(dict(panels_per_interval=panels, direct_error=norm(numerical-direct),
                         compact_error=norm(compact-numerical)))
    orders = [math.log2(rows[i]["direct_error"]/rows[i+1]["direct_error"]) for i in (0, 1)]
    negative("V4 midpoint characteristic minimum order", min(orders), 1.99)
    check("V4 finest characteristic integral", rows[-1]["direct_error"], 0., tolerance=1.e-6, units="Pa sqrt(m)")
    negative("V4 nonvanishing refined compact error", rows[-1]["compact_error"], .99*analytic, "Pa sqrt(m)")
    return dict(analytic_compact_defect=analytic, convergence=rows, orders=orders,
                auxiliary_has_cohesive_resistance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit("Refusing to overwrite "+str(args.output))
    start = time.monotonic()
    detail, error = {}, None
    try:
        v1()
        detail["V2"] = v2()
        v3()
        detail["V4"] = v4()
    except Exception as exc:
        error = repr(exc)
    summary = dict(passed=error is None, failure=error, seconds=time.monotonic()-start,
                   python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scope="independent mathematical references; no production calls, CPDI or DG solver",
                   checks=RECORDS, detail=detail)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": summary["passed"], "checks": len(RECORDS), "seconds": summary["seconds"], "failure": error}))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
