#!/usr/bin/env python3
"""Fixed-chart, fixed-phase material/section reference; not a mechanical FE solve.

Two explicit curved-section constitutive choices share the native bulk-tip
midpoint equations. Neither is registered as a native propagation option.
All arrays retain physical tensors at quadrature points, not section averages.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Parameters:
    G: float = 1e6
    eta: float = 1e8
    m: float = 4.
    pressure: float = 200.
    mu0: float = .6
    a: float = .015
    b: float = .02
    V0: float = .001
    Dc: float = .0005
    radiation: float = 1000.


def composite_gauss(a, b, cells, degree=6):
    x, w = np.polynomial.legendre.leggauss(degree)
    edges = np.linspace(a, b, cells+1)
    return (edges[:-1, None]+(x+1)*(b-a)/(2*cells)).ravel(), np.tile(w*(b-a)/(2*cells), cells)


def shear_tensor(angle):
    angle = np.asarray(angle)
    S = np.zeros(angle.shape+(3, 3))
    S[..., 0, 0] = -.5*np.sin(2*angle)
    S[..., 1, 1] = -S[..., 0, 0]
    S[..., 0, 1] = S[..., 1, 0] = .5*np.cos(2*angle)
    return S


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@dataclass
class Chart:
    s: np.ndarray
    n: np.ndarray
    ds: np.ndarray
    dn: np.ndarray
    angle: np.ndarray
    J: np.ndarray
    S: np.ndarray
    area: np.ndarray
    g: np.ndarray
    phi: np.ndarray

    @classmethod
    def create(cls, curvature=0., rotation_angle=0., law="material_area"):
        s, ds = composite_gauss(-.4, .4, 4)
        n, dn = composite_gauss(-.125, .125, 8)
        angle = curvature*s+rotation_angle
        J = np.broadcast_to(1-curvature*n, (len(s), len(n))).copy()
        assert np.min(J) > 0., "Intersecting normal chart is outside this reference."
        S = np.broadcast_to(shear_tensor(angle)[:, None], J.shape+(3, 3)).copy()
        if law == "material_area":
            g = np.ones_like(J)
        elif law == "normal_integral":
            g = 1/J
        else:
            raise ValueError(law)
        # Off-center finite band deliberately exposes curvature's first moment.
        phi = np.broadcast_to(.6*np.maximum(0., 1-np.abs(n-.025)/.1)**2, J.shape).copy()
        return cls(s, n, ds, dn, angle, J, S, ds[:, None]*dn*J, g, phi)

    def integrate_rate(self, w):
        return np.sum(self.dn*self.J*self.g*w, axis=1)

    def adjoint_force(self, force):
        return self.g*force[:, None]


@dataclass
class History:
    stress: np.ndarray
    A: np.ndarray
    age: np.ndarray
    slip: np.ndarray

    def copy(self):
        return History(*(x.copy() for x in (self.stress, self.A, self.age, self.slip)))


def initial(chart):
    # Independent full tensors: in-plane shear plus deviatoric normal/zz memory.
    stress = 700*chart.S.copy()
    for i, angle in enumerate(chart.angle):
        R = rotation(angle)
        stress[i] += R@np.diag([100., -150., 50.])@R.T
    e = .0002*(chart.phi/.6)*(1+.3*chart.n/.125)
    A = e[..., None, None]*chart.S
    age = .4+.3*(chart.s+.4)/.8
    slip = .0001*np.sin(3*chart.s)
    return History(stress, A, age, slip)


def aging(speed, age, dt, Dc):
    rate = np.asarray(speed)/Dc
    z = dt*rate
    ratio = np.ones_like(z)
    np.divide(-np.expm1(-z), z, out=ratio, where=z != 0.)
    return age*np.exp(-z)+dt*ratio


def resistance(V, age, half_dt, p):
    stage_age = aging(abs(V), age, half_dt, p.Dc)
    z = V/(2*p.V0)*np.exp((p.mu0+p.b*np.log(stage_age*p.V0/p.Dc))/p.a)
    return p.pressure*p.a*np.arcsinh(z)+p.radiation*V


def point_stage(parent, chart, D, dt, local_force, p):
    h = .5*dt
    beta = 1/(1+h*p.G/p.eta)
    phi, S = chart.phi, chart.S
    c, d = p.m*phi*(1+phi), (1-phi)**2
    fraction = c/(c+d)
    old_e = 2*np.sum(parent.A*S, axis=(-2, -1))
    assert np.max(abs(parent.A-old_e[..., None, None]*S)) < 1e-14
    predictor = beta*(parent.stress+2*p.G*h*D)
    force = np.sum(predictor*S, axis=(-2, -1))+p.G*beta*old_e-local_force
    e = fraction*force/p.G
    stress = predictor-2*p.G*(e-beta*old_e)[..., None, None]*S
    A = e[..., None, None]*S
    w = (e/beta-old_e)/h
    Y = p.m*(1+3*phi)*(1-phi)/(c+d)**2*force**2/(2*p.G)
    return stress, A, w, Y, fraction/(p.G*beta*h)


def energy(stress, A, phi, p):
    inverse = np.zeros_like(phi)
    np.divide((1-phi)**2, p.m*phi*(1+phi), out=inverse, where=phi > 0.)
    assert np.max(abs(A[phi == 0.]), initial=0.) == 0.
    return np.sum(stress*stress, axis=(-2, -1))/(4*p.G), p.G*np.sum(A*A, axis=(-2, -1))*inverse


def advance(parent, chart, D, dt, p=Parameters()):
    """Return a candidate; callers retain/commit it. No map or history mutation."""
    _, _, free_w, _, mobility = point_stage(parent, chart, D, dt, 0., p)
    free = chart.integrate_rate(free_w)
    C = chart.integrate_rate(mobility*chart.g)
    assert p.radiation >= p.pressure*(dt/2)/p.Dc*p.b
    V = np.empty_like(free)
    for k, b in enumerate(free):
        if b == 0.:
            V[k] = 0.
        else:
            # Match the native coordinate, not its safeguarded Newton algorithm:
            # an absolute tolerance on V cannot resolve exponentially cold creep.
            scale = 2*p.V0*np.exp(-(p.mu0+p.b*np.log((parent.age[k]+dt/2)*p.V0/p.Dc))/p.a)
            target = abs(b)
            solve = lambda z: (scale*np.sinh(z)+C[k]*resistance(scale*np.sinh(z), parent.age[k], .5*dt, p)-target)/target
            z = brentq(solve, 0., np.arcsinh(target/scale), xtol=1e-15, rtol=4*np.finfo(float).eps)
            V[k] = np.copysign(scale*np.sinh(z), b)
            assert abs(solve(z)) < 1e-13
    f = resistance(V, parent.age, .5*dt, p)
    force = chart.adjoint_force(f)
    stress, A, w, Y, _ = point_stage(parent, chart, D, dt, force, p)
    measured_V = chart.integrate_rate(w)
    next_state = History(2*stress-parent.stress, 2*A-parent.A,
                         aging(abs(measured_V), parent.age, dt, p.Dc),
                         parent.slip+dt*measured_V)
    eb, ec = energy(stress, A, chart.phi, p)
    old_energy = sum(energy(parent.stress, parent.A, chart.phi, p))
    new_energy = sum(energy(next_state.stress, next_state.A, chart.phi, p))
    work = dt*np.sum(chart.area*np.sum(stress*D, axis=(-2, -1)))
    dissipation = dt*np.sum(chart.area*(2*p.G/p.eta*(eb+ec)+force*w))
    change = np.sum(chart.area*(new_energy-old_energy))
    bulk_power = np.sum(chart.area*force*w)
    fault_power = np.sum(chart.ds*f*measured_V)
    active = abs(w) > 1e-10*np.max(abs(w))
    band_counts = np.sum(active & ~np.pad(active[:, :-1], ((0, 0), (1, 0))), axis=1)
    opposite = np.any((w > 0.) & active, axis=1) & np.any((w < 0.) & active, axis=1)
    return next_state, dict(w=w, V=measured_V, force=force, f=f, Y=Y,
                            stage_stress=stress, stage_A=A, eb=eb, ec=ec,
                            work=float(work), dissipation=float(dissipation),
                            energy_change=float(change), error=float(work-change-dissipation),
                            power_error=float(bulk_power-fault_power),
                            fault_power=float(fault_power),
                            opposite_sections=int(np.count_nonzero(opposite)),
                            max_active_components=int(np.max(band_counts)),
                            rate_error=float(np.max(abs(measured_V-V))),
                            negative_power=float(np.sum(chart.area*np.minimum(0., force*w))))


def history_run(chart, dt=.00125, rotation_angle=0., sign=1., p=Parameters()):
    state = initial(chart)
    records, samples = [], []
    # Globally affine, divergence-free prescribed rate; no equilibrium claim.
    R = rotation(rotation_angle)
    for step in range(round(.1/dt)):
        gamma = .04 if (step+.5)*dt < .05 else -.06
        D = R@(sign*gamma*shear_tensor(0.))@R.T
        parent = state
        state, response = advance(parent, chart, D, dt, p)
        records.append(dict(time=(step+1)*dt, work=response["work"],
                            dissipation=response["dissipation"], energy_change=response["energy_change"],
                            error=response["error"], power_error=response["power_error"],
                            rate_error=response["rate_error"], negative_power=response["negative_power"],
                            fault_power=response["fault_power"],
                            opposite_sections=response["opposite_sections"],
                            max_active_components=response["max_active_components"],
                            mean_slip=float(np.dot(chart.ds, state.slip)/sum(chart.ds)),
                            mean_age=float(np.dot(chart.ds, state.age)/sum(chart.ds)),
                            mean_V=float(np.dot(chart.ds, response["V"])/sum(chart.ds))))
        if step in (0, round(.05/dt)-1, round(.1/dt)-1):
            samples.append((parent.copy(), D.copy(), dt, response))
    return state, records, samples


def select_virgin_plane(stress, preferred_angle):
    """Maximum intact Y with scalar confinement and zero new-section rate.

    Orthogonal tied planes are real geometric alternatives despite opposite S.
    Geometry preference is an explicit additional tie rule, not a Coulomb law.
    """
    normal_difference = (stress[0, 0]-stress[1, 1])/2
    if np.hypot(normal_difference, stress[0, 1]) == 0.:
        return np.array([]), None  # Every in-plane angle ties.
    angle = .5*np.arctan2(-normal_difference, stress[0, 1])
    candidates = angle+np.array([0., np.pi/2])
    distance = abs((candidates-preferred_angle+np.pi/2) % np.pi-np.pi/2)
    if abs(distance[0]-distance[1]) < 1e-12:
        return candidates, None
    return candidates, candidates[np.argmin(distance)]
