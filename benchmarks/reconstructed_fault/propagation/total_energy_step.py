#!/usr/bin/env python3
"""Isolated total-energy split step; no ASPECT state or production law changes.

Q1 phase, regular rectangular CPDI fracture samples, analytic transverse I,
consistent surface projection, and exact Maxwell work on fixed spatial measures.
The bulk shear rate is prescribed; free surface rates use integrated equilibrium.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
G, ETA, GC, ELL, C0, M = 1e6, 1e8, 26.6666666667, .15625, 8/3, 128.00000000016
FRACTURE_SCALE = GC/(C0*ELL)
PRESSURE, MU = 10000., .1
N = 16


def basis(x, y):
    """Sparse Cartesian Q1 values and physical gradients, x-major nodal order."""
    x, y = np.broadcast_arrays(x, y)
    x, y = x.ravel(), y.ravel()
    i, j = np.minimum((N*x).astype(int), N-1), np.minimum((N*y).astype(int), N-1)
    s, t = N*x-i, N*y-j
    row, col, value, dx, dy = [], [], [], [], []
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        fx, fy = (s if a else 1-s), (t if b else 1-t)
        row.extend(np.arange(len(x))); col.extend((i+a)*(N+1)+j+b)
        value.extend(fx*fy); dx.extend((N if a else -N)*fy); dy.extend((N if b else -N)*fx)
    shape = (len(x), (N+1)**2)
    return tuple(sparse.csr_matrix((v, (row, col)), shape=shape) for v in [value, dx, dy])


def surface_basis(x, count):
    x = np.asarray(x).ravel()
    j = np.minimum((x*count).astype(int), count-1)
    t = x*count-j
    return sparse.csr_matrix((np.r_[1-t, t], (np.r_[np.arange(len(x)), np.arange(len(x))],
                              np.r_[j, j+1])), shape=(len(x), count+1))


def h(phi):
    return M*phi*(1+phi)/(1-phi)**2


def dh(phi):
    return M*(4/(1-phi)**3-3/(1-phi)**2)


class Energy:
    def __init__(self):
        z, w = leggauss(3)
        coord = (np.arange(N)[:, None]+(z+1)/2)/N
        x, y = np.meshgrid(coord.ravel(), coord.ravel(), indexing='ij')
        self.x, self.y = x.ravel(), y.ravel()
        self.wb = np.outer(np.tile(w/(2*N), N), np.tile(w/(2*N), N)).ravel()
        self.B = basis(self.x, self.y)[0]
        # Surface spacing .015625 m; every bulk x knot is a surface vertex.
        self.ns = 64
        xs = (np.arange(self.ns)[:, None]+(z+1)/2)/self.ns
        self.xs, self.ws = xs.ravel(), np.tile(w/(2*self.ns), self.ns)
        self.S = surface_basis(self.xs, self.ns)
        self.T = surface_basis(self.x, self.ns)
        self.mass = (self.S.T @ sparse.diags(self.ws) @ self.S).toarray()
        self.factor = cho_factor(self.mass)
        # Direct transverse antiderivative evaluated at surface Q3 locations.
        xx, yy = np.meshgrid(self.xs, np.arange(N+1)/N, indexing='ij')
        self.L = basis(xx[:, :-1], yy[:, :-1])[0]
        self.R = basis(xx[:, 1:], yy[:, 1:])[0]
        # Rectangular Voronoi domains: CPDI value/gradient equals Q1 centroid
        # for this regular, axis-aligned case (not general deformed domains).
        pc = (np.arange(3*N)+.5)/(3*N)
        px, py = np.meshgrid(pc, pc, indexing='ij')
        P, Dx, Dy = basis(px, py)
        vp = 1/(3*N)**2
        self.flinear = FRACTURE_SCALE*np.asarray(P.T @ np.full(P.shape[0], vp))
        self.fhessian = (2*FRACTURE_SCALE*ELL**2*vp*(Dx.T @ Dx+Dy.T @ Dy)).tocsr()

    def direct(self, phi):
        a, b = 1-self.L@phi, 1-self.R@phi
        z = (b-a)/a
        small = np.abs(z) < 1e-3
        ratio = np.empty_like(z); prime = np.empty_like(z)
        q = z[small]
        ratio[small] = sum((-1)**k*q**k/(k+1) for k in range(13))
        prime[small] = sum((-1)**k*k*q**(k-1)/(k+1) for k in range(1, 13))
        q = z[~small]
        ratio[~small] = np.log1p(q)/q
        prime[~small] = (q/(1+q)-np.log1p(q))/q**2
        values = M/N*(2/(a*b)-3*ratio/a+1)
        left = M/N*(2/(a*a*b)-3/a**2*(ratio+(1+z)*prime))
        right = M/N*(2/(a*b*b)+3/a**2*prime)
        # Group transverse cells for each surface quadrature point.
        group = sparse.csr_matrix((np.ones(len(a)), (np.repeat(np.arange(len(self.xs)), N),
                                                    np.arange(len(a)))), shape=(len(self.xs), len(a)))
        jacobian = group @ (sparse.diags(left) @ self.L+sparse.diags(right) @ self.R)
        return values.reshape(-1, N).sum(axis=1), jacobian

    def profile(self, phi):
        if np.min(phi) < 0 or np.max(phi) >= 1:
            raise ValueError('Undefined phase/normalization; no cap or floor.')
        direct, jacobian = self.direct(phi)
        I = cho_solve(self.factor, np.asarray(self.S.T @ (self.ws*direct)))
        ib, isurf = self.T@I, self.S@I
        if np.min(I) <= 0:
            raise ValueError('Nonpositive consistent normalization; reject trial.')
        phib = self.B@phi
        return dict(I=I, ib=ib, isurf=isurf, hb=h(phib), dhb=dh(phib),
                    chi=h(phib)/ib, J=jacobian)

    def fracture(self, phi):
        return float(self.flinear@phi+.5*phi@(self.fhessian@phi))

    def objective(self, phi, accepted):
        p = self.profile(phi)
        zb, zs = self.T@accepted.Z, self.S@accepted.Z
        tau = accepted.tau-zb*(p['chi']-accepted.profile['chi'])
        eb = np.dot(self.wb, tau*tau)/(2*G)
        ec = np.dot(self.ws, zs*zs/p['isurf'])/(2*G)
        f = self.fracture(phi)
        # c is the derivative of bulk energy with respect to chi.
        c = -self.wb*zb*tau/G
        grad = self.flinear+self.fhessian@phi+self.B.T@(c*p['dhb']/p['ib'])
        # Surface nodal force acting on projected I, from both energy terms.
        t = self.T.T@(-c*p['hb']/p['ib']**2) + self.S.T@(-self.ws*zs*zs/(2*G*p['isurf']**2))
        adjoint = cho_solve(self.factor, np.asarray(t))
        grad += p['J'].T @ (self.ws*(self.S@adjoint))
        return float(eb+ec+f), np.asarray(grad), (float(eb), float(ec), f), tau, p


@dataclass(frozen=True)
class State:
    phi: np.ndarray
    tau: np.ndarray
    Z: np.ndarray
    slip: np.ndarray
    history: np.ndarray
    profile: dict
    strain: np.ndarray
    inelastic: np.ndarray
    time: float = 0.


def state(energy, phi, tau, Z, slip=None, history=None, time=0., *, strain=None, inelastic=None):
    """Accepted arrays are immutable; profile is rebuilt solely from phase."""
    phi, tau, Z = [np.asarray(a).copy() for a in [phi, tau, Z]]
    slip = np.zeros_like(Z) if slip is None else slip.copy()
    p = energy.profile(phi)
    history = ((energy.T@Z)/p['ib'])**2*(1+p['hb'])**2/(2*G) if history is None else history.copy()
    if strain is None:
        assert inelastic is None
        # Diagnostic preloads reside in prescribed initial internal strain;
        # total displacement/strain starts at zero, so compatibility is explicit.
        strain = np.zeros_like(tau)
        inelastic = -(tau+(energy.T@Z)*p['chi'])/G
    else:
        strain, inelastic = strain.copy(), inelastic.copy()
    assert max(abs(strain-inelastic-(tau+(energy.T@Z)*p['chi'])/G)) < 1e-12
    for a in [phi, tau, Z, slip, history, strain, inelastic, *p.values()]:
        if isinstance(a, np.ndarray):
            a.flags.writeable = False
    p['J'].data.flags.writeable = False
    return State(phi, tau, Z, slip, history, MappingProxyType(p), strain, inelastic, time)


def fingerprint(a):
    return hashlib.sha256(b''.join(x.tobytes() for x in [a.phi, a.tau, a.Z, a.slip, a.history,
                                                        a.strain, a.inelastic, a.profile['I']])).hexdigest()


def kkt(gradient, phi, lower):
    projected = gradient.copy()
    at_lower = phi-lower <= 2e-12
    projected[at_lower] = np.minimum(projected[at_lower], 0.)
    return float(np.max(abs(projected))/FRACTURE_SCALE)


def phase_solve(energy, accepted):
    def objective(phi):
        try:
            v, g, *_ = energy.objective(phi, accepted)
            return v/FRACTURE_SCALE, g/FRACTURE_SCALE
        except ValueError:
            return math.inf, np.zeros_like(phi)
    phi = accepted.phi.copy()
    iterations = 0
    for repeat in range(4):
        answer = minimize(objective, phi, jac=True, method='L-BFGS-B',
                          bounds=list(zip(accepted.phi, np.ones_like(phi))),
                          options=dict(gtol=2e-13, ftol=2e-16, maxiter=3000, maxls=60, maxcor=30))
        phi = answer.x; iterations += answer.nit
        _, gradient, *_ = energy.objective(phi, accepted)
        if kkt(gradient, phi, accepted.phi) <= 1e-11:
            break
    # L-BFGS energy reduction can stall at floating-point precision before KKT.
    # Refine its free set with Newton on the same analytic residual; finite
    # differences approximate only the search Hessian, never the phase law.
    corrections = []
    for _ in range(16):
        if kkt(gradient, phi, accepted.phi) <= 1e-11:
            break
        free = np.flatnonzero((phi-accepted.phi > 2e-12) | (gradient < 0))
        H = np.empty((len(free), len(free)))
        for col, index in enumerate(free):
            eps = min(1e-5, phi[index]/4, (1-phi[index])/4)
            assert eps > 0
            plus, minus = phi.copy(), phi.copy()
            plus[index] += eps; minus[index] -= eps
            H[:, col] = (energy.objective(plus, accepted)[1][free]
                         -energy.objective(minus, accepted)[1][free])/(2*eps)
        H = (H+H.T)/2
        direction = np.zeros_like(phi)
        direction[free] = np.linalg.solve(H, -gradient[free])
        corrections.append(float(max(abs(direction))))
        current = energy.objective(phi, accepted)[0]
        for ls in range(60):
            trial = np.maximum(accepted.phi, np.minimum(1., phi+2.**(-ls)*direction))
            try:
                v, g, *_ = energy.objective(trial, accepted)
            except ValueError:
                continue
            # A bounded arithmetic allowance for energy sums, not a KKT change.
            allowance = 32*np.finfo(float).eps*max(1., abs(current))
            if v <= current+allowance:
                phi, gradient = trial, g
                break
        else:
            raise RuntimeError('Phase Newton line search failed.')
    assert kkt(gradient, phi, accepted.phi) <= 1e-11, (answer.message, kkt(gradient, phi, accepted.phi), corrections)
    assert np.min(phi-accepted.phi) >= 0 and np.max(phi) < 1
    initial = energy.objective(accepted.phi, accepted)[0]
    assert energy.objective(phi, accepted)[0] <= initial+1e-10
    return phi, dict(kkt=kkt(gradient, phi, accepted.phi), iterations=iterations,
                     max_increment=float(max(phi-accepted.phi)), healing=float(max(accepted.phi-phi)),
                     lower_contacts=int(np.count_nonzero(phi-accepted.phi <= 2e-12)),
                     Newton_refinement_corrections=corrections)


def coefficients(dt, eta):
    lam = eta/G; x = dt/lam
    beta = math.exp(-x); one = -math.expm1(-x)
    r = one/x
    # Series evaluate analytic kernels near zero; no small-step model assumption.
    if x < .01:
        f = sum((-1)**(k+1)*x**k/math.factorial(k+1) for k in range(1, 13))
        j11 = lam*sum((-1)**(k+1)*(2**(k-1)-2)*x**k/math.factorial(k) for k in range(3, 16))
    else:
        f = 1-r
        j11 = dt-2*lam*one+lam*(-math.expm1(-2*x))/2
    return beta, eta*one, r, eta*f, lam*(-math.expm1(-2*x))/2, lam*one**2/2, j11


def step(energy, accepted, dt, eta, gamma_rate, *, fail=False, endpoint_control=False,
         free_bulk_mode=False, evolve_phase=True):
    before = fingerprint(accepted)
    if evolve_phase:
        phi, phase = phase_solve(energy, accepted)
    else:
        phi = accepted.phi
        phase = dict(frozen=True, max_increment=0., healing=0., iterations=0)
    _, _, parts, jump_tau, p = energy.objective(phi, accepted)
    oldparts = energy.objective(accepted.phi, accepted)[2]
    jump = np.asarray(parts)-oldparts
    release = -float(sum(jump))
    # Independent phase-path virtual work checks the complete finite jump,
    # including normalization's adjoint, rather than defining closure by fiat.
    delta = phi-accepted.phi
    phase_work = []
    for order in [32, 64]:
        if not np.any(delta):
            phase_work.append(0.)
            continue
        zphase, wphase = leggauss(order)
        value = sum(weight/2*(energy.objective(accepted.phi+(zeta+1)/2*delta,
                                             accepted)[1]@delta)
                    for zeta, weight in zip(zphase, wphase))
        phase_work.append(float(value))
    phase_work_error = max(abs(value+release) for value in phase_work)
    assert phase_work_error < 1e-9
    beta, kap, r, eta_bar, jee, je1, j11 = coefficients(dt, eta)
    # In pure shear tau:S=tau_xy, 2G e Delta chi S gives G e Delta chi.
    Zb, Zs = energy.T@accepted.Z, energy.S@accepted.Z
    temporal = beta if endpoint_control else r
    visc = kap if endpoint_control else eta_bar
    load = energy.T.T@(energy.wb*p['chi']*(temporal*jump_tau+visc*gamma_rate))
    load -= energy.S.T@(energy.ws*(temporal*Zs/p['isurf']+MU*PRESSURE))
    K = visc*(energy.T.T@sparse.diags(energy.wb*p['chi']**2)@energy.T
              +energy.S.T@sparse.diags(energy.ws/p['isurf'])@energy.S).toarray()
    mode = 2*np.pi*np.cos(2*np.pi*energy.y)
    if free_bulk_mode:
        # One genuine free incompressible velocity mode u_x=U sin(2 pi y).
        # Condense it with both bulk/slip off-diagonal blocks, not a load fix.
        ku = float(visc*np.dot(energy.wb, mode**2))
        cross = -visc*np.asarray(energy.T.T@(energy.wb*p['chi']*mode))
        gu = float(np.dot(energy.wb*mode, temporal*jump_tau+visc*gamma_rate))
        uncondensed = K.copy()
        K = K-np.outer(cross, cross)/ku
        load = load+cross*gu/ku
    # Exact active-set nonnegative quadratic solve, not equality followed by clip.
    chol = np.linalg.cholesky(K)
    V, _ = nnls(chol.T, np.linalg.solve(chol, load), maxiter=10*len(load))
    mechanical_gradient = K@V-load
    mech_kkt = float(max(np.max(abs(mechanical_gradient[V>0])) if np.any(V>0) else 0.,
                         max(0., -np.min(mechanical_gradient[V==0])) if np.any(V==0) else 0.))
    assert mech_kkt < 1e-9
    U = float(-(gu+cross@V)/ku) if free_bulk_mode else 0.
    force_refinements = 0
    if free_bulk_mode:
        # Refinement on the actual evaluated coupled forces, at unchanged
        # active contacts. Avoid condensation/cancellation roundoff becoming
        # a force-balance failure on a more resolved integration measure.
        free = np.flatnonzero(V>0)
        for _ in range(4):
            trial_tau = temporal*jump_tau+visc*(gamma_rate+mode*U-p['chi']*(energy.T@V))
            trial_z = temporal*Zs+visc*(energy.S@V)
            ru = math.fsum(energy.wb*mode*trial_tau)
            fv = np.asarray(energy.T.T@(energy.wb*p['chi']*trial_tau)
                            -energy.S.T@(energy.ws*(trial_z/p['isurf']+MU*PRESSURE)))
            if abs(ru)<1e-10 and np.max(abs(fv[free]),initial=0)<1e-10:
                break
            block=np.empty((len(free)+1,len(free)+1))
            block[0,0]=ku;block[0,1:]=cross[free];block[1:,0]=cross[free]
            block[1:,1:]=uncondensed[np.ix_(free,free)]
            correction=np.linalg.solve(block,np.r_[ru,-fv[free]])
            U-=correction[0];V[free]-=correction[1:]
            force_refinements+=1
            assert np.min(V)>=0
    gamma = gamma_rate+mode*U
    vb, vs = energy.T@V, energy.S@V
    rate = gamma-p['chi']*vb
    tau_end = beta*jump_tau+kap*rate
    Z_end = beta*accepted.Z+kap*V
    tau_avg = r*jump_tau+eta_bar*rate
    Z_avg = r*Zs+eta_bar*vs
    # Reconstruct E_total-E_in at the same bulk points. Includes committed
    # bulk and cohesive viscous strain; only elastic slip is redistributed.
    residual_strain_old = (accepted.tau+Zb*accepted.profile['chi'])/G
    residual_strain_new = (tau_end+(energy.T@Z_end)*p['chi'])/G
    Z_avg_bulk = r*Zb+eta_bar*vb
    inelastic_increment = dt*(tau_avg+p['chi']*Z_avg_bulk)/eta
    reconstruction_error = float(max(abs(residual_strain_new-residual_strain_old
                                         -dt*gamma+inelastic_increment)))
    assert reconstruction_error < 1e-12
    db = float(np.dot(energy.wb, jump_tau**2*jee/eta+2*jump_tau*rate*je1+eta*rate**2*j11))
    dc = float(np.dot(energy.ws/p['isurf'], Zs**2*jee/eta+2*Zs*vs*je1+eta*vs**2*j11))
    friction = float(dt*np.dot(energy.ws, MU*PRESSURE*vs))
    ordinary_bulk = float(dt*np.dot(energy.wb, tau_avg*p['chi']*vb))
    ordinary_fault = float(dt*np.dot(energy.ws, Z_avg/p['isurf']*vs))
    external = float(dt*np.dot(energy.wb, tau_avg*gamma))
    free_mode_residual = math.fsum(energy.wb*mode*tau_avg) if free_bulk_mode else 0.
    free_endpoint_residual = math.fsum(energy.wb*mode*tau_end) if free_bulk_mode else 0.
    assert abs(free_endpoint_residual if endpoint_control else free_mode_residual) < 1e-9, (free_mode_residual, free_endpoint_residual)
    eb = float(np.dot(energy.wb, tau_end**2)/(2*G))
    ec = float(np.dot(energy.ws, (energy.S@Z_end)**2/p['isurf'])/(2*G))
    bulk_defect = eb-parts[0]+db-external+ordinary_bulk
    fault_defect = ec-parts[1]+dc-ordinary_fault
    exchange = ordinary_bulk-ordinary_fault-friction
    total_defect = external-(eb+ec+parts[2]-sum(oldparts))-db-dc-friction-release
    assert min(db, dc, friction) >= 0 and release >= -1e-10
    assert abs(bulk_defect) < 1e-8 and abs(fault_defect) < 1e-8
    if not endpoint_control:
        assert abs(exchange) < 1e-8 and abs(total_defect) < 1e-8
    # Independent temporal quadrature of the actual post-jump paths.
    zz, ww = leggauss(96)
    t = dt*(zz+1)/2; decay = np.exp(-t/(eta/G)); grow = -np.expm1(-t/(eta/G))
    qtau = jump_tau[:, None]*decay+eta*rate[:, None]*grow
    qz = Zs[:, None]*decay+eta*vs[:, None]*grow
    db_q = float(np.dot(energy.wb, (qtau**2)@(dt*ww/2))/eta)
    dc_q = float(np.dot(energy.ws/p['isurf'], (qz**2)@(dt*ww/2))/eta)
    assert abs(db-db_q) < 1e-9 and abs(dc-dc_q) < 1e-9
    # Algebraic endpoint comparator for the old spread-C law, evaluated stably.
    legacy_end = beta*accepted.tau+kap*rate-beta*Zb*(p['chi']-accepted.profile['chi'])
    endpoint_error = float(max(abs(tau_end-legacy_end)))
    assert endpoint_error < 1e-10
    # Peak remains diagnostic; perturbing it must not affect phase/energy.
    qb = (energy.T@Z_end)/p['ib']
    g = 1/(1+p['hb'])
    diagnostic = np.maximum(accepted.history, qb**2/(2*G*g*g))
    if fail:
        assert fingerprint(accepted) == before
        raise RuntimeError('Injected rejection after phase and mechanical trial; publish nothing.')
    new = state(energy, phi, tau_end, Z_end, accepted.slip+dt*V, diagnostic, accepted.time+dt,
                strain=accepted.strain+dt*gamma, inelastic=accepted.inelastic+inelastic_increment)
    assert fingerprint(accepted) == before
    report = dict(dt=dt, ratio=dt*G/eta, beta=beta, phase=phase, phi_max=float(max(phi)),
                  bulk_profile_jump=float(jump[0]), fault_profile_jump=float(jump[1]),
                  fracture_expenditure=float(jump[2]), phase_release=release,
                  phase_path_virtual_work=phase_work[-1], phase_path_error=phase_work_error,
                  bulk_storage_change=eb-oldparts[0], fault_storage_change=ec-oldparts[1],
                  bulk_viscous=db, fault_viscous=dc, friction=friction, external_work=external,
                  ordinary_bulk_work=ordinary_bulk, ordinary_fault_work=ordinary_fault,
                  mechanical_kkt=mech_kkt, exchange_defect=exchange, total_defect=total_defect,
                  residual_strain_reconstruction_error=reconstruction_error,
                  bulk_identity_defect=bulk_defect, fault_identity_defect=fault_defect,
                  temporal_quadrature_error=max(abs(db-db_q), abs(dc-dc_q)),
                  legacy_endpoint_error=endpoint_error, min_I=float(min(p['I'])),
                  max_slip_rate=float(max(V)), min_slip_rate=float(min(V)))
    report['free_bulk_velocity_amplitude'] = U
    report['free_bulk_residual'] = free_mode_residual
    report['free_bulk_work'] = dt*U*free_mode_residual
    # Zero applied force on the free mode: its internal work is not boundary
    # input. Keep endpoint/mean balance and the actual external-work ledger
    # distinct, especially in the endpoint comparator.
    report['free_bulk_endpoint_residual'] = free_endpoint_residual
    report['prescribed_external_work'] = external-report['free_bulk_work']
    report['mechanical_work_error'] = total_defect-report['free_bulk_work']
    for name, tau, z in [('endpoint', tau_end, energy.S@Z_end),
                         ('mean', tau_avg, Z_avg)]:
        force = np.asarray(energy.T.T@(energy.wb*p['chi']*tau)
                           -energy.S.T@(energy.ws*(z/p['isurf']+MU*PRESSURE)))
        report[name+'_slip_force'] = force.tolist()
        report[name+'_slip_condition_error'] = float(max(
            np.max(abs(force[V>0])) if np.any(V>0) else 0.,
            max(0., np.max(force[V==0])) if np.any(V==0) else 0.))
        report[name+'_slip_complementarity'] = float(max(abs(V*force)))
    report['slip_rate'] = V.tolist()
    report['stick_nodes'] = int(np.count_nonzero(V==0))
    report['mechanical_force_refinements'] = force_refinements
    report['mechanical_kkt'] = report[('endpoint' if endpoint_control else 'mean')+'_slip_condition_error']
    assert report['mechanical_kkt'] < 1e-9
    report['diagnostic_history_max'] = float(max(diagnostic))
    report['diagnostic_history_decrease'] = float(max(accepted.history-diagnostic))
    assert report['diagnostic_history_decrease'] <= 0
    return new, report


def verify_derivative(energy, accepted):
    # Off-bound trial: both direct h and nonlocal projected I must vary.
    direction = .04+.03*np.cos(np.arange(len(accepted.phi))*1.234)
    phi = accepted.phi+.01*direction
    value, gradient, *_ = energy.objective(phi, accepted)
    tests = []
    for v in [direction, np.sin(np.arange(len(phi))*.741), np.eye(1, len(phi), len(phi)//2)[0]]:
        exact = float(gradient@v)
        for eps in [1e-4, 1e-5, 1e-6]:
            fd = (energy.objective(phi+eps*v, accepted)[0]-energy.objective(phi-eps*v, accepted)[0])/(2*eps)
            tests.append(dict(eps=eps, analytic=exact, finite_difference=fd,
                              error=abs(fd-exact)/max(1., abs(exact))))
        assert min(t['error'] for t in tests[-3:]) < 1e-7
    p = energy.profile(phi)
    # Adjoint identity for arbitrary dual load, with nontrivial surface mass.
    dual = np.cos(np.arange(energy.ns+1)*.43)
    dI = cho_solve(energy.factor, np.asarray(energy.S.T@(energy.ws*(p['J']@direction))))
    adj = p['J'].T@(energy.ws*(energy.S@cho_solve(energy.factor, dual)))
    adjoint_defect = float(abs(dual@dI-adj@direction))
    assert adjoint_defect < 1e-9
    poisoned = state(energy, accepted.phi, accepted.tau, accepted.Z, history=accepted.history+1e15)
    assert energy.objective(phi, poisoned)[0] == value
    assert np.array_equal(energy.objective(phi, poisoned)[1], gradient)
    return dict(directional_checks=tests, adjoint_defect=adjoint_defect, diagnostic_history_invariance=True)


def profile_variation_control(energy, accepted, name):
    direction = accepted.phi*(1+.2*np.cos(np.repeat(np.arange(N+1)/N, N+1)*2*np.pi))
    p = accepted.profile
    dI = cho_solve(energy.factor, np.asarray(energy.S.T@(energy.ws*(p['J']@direction))))
    change = p['dhb']*(energy.B@direction)/p['ib']-p['hb']*(energy.T@dI)/p['ib']**2
    bulk = float(-np.dot(energy.wb*(energy.T@accepted.Z)/G, accepted.tau*change))
    # Separate bulk quadrature uncertainty; projection and bulk field fixed.
    z, w = leggauss(12)
    coord = (np.arange(N)[:, None]+(z+1)/2)/N
    x, y = np.meshgrid(coord.ravel(), coord.ravel(), indexing='ij')
    wb = np.outer(np.tile(w/(2*N), N), np.tile(w/(2*N), N)).ravel()
    B = basis(x, y)[0]; T = surface_basis(x, energy.ns)
    value = B@accepted.phi; ib = T@p['I']
    dchi = dh(value)*(B@direction)/ib-h(value)*(T@dI)/ib**2
    tau = np.zeros(x.size) if name == 'unloaded' else np.full(x.size, 1500.)
    if name == 'nonuniform':
        tau += 8000*abs(y.ravel()-.5)
    refined = float(-np.dot(wb*(T@accepted.Z)/G, tau*dchi))
    return dict(bulk_Q3_directional_derivative=bulk, bulk_Q12_comparator=refined,
                bulk_quadrature_difference=abs(bulk-refined))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='verification')
    args = parser.parse_args()
    out = HERE/'total_energy_evidence'/args.label
    assert not out.exists(), 'Preserve earlier prototype evidence.'
    out.mkdir(parents=True)
    energy = Energy()
    path = ROOT/'build-propagation-integration/output-irreversibility_dt03125/material-phase-accepted-0.csv'
    with path.open() as stream:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
    lookup = {(r['x'], r['y']): r['phi'] for r in rows}
    seed = np.array([lookup[(i/N, j/N)] for i in range(N+1) for j in range(N+1)])
    nodes = np.linspace(0, 1, energy.ns+1)
    controls = dict(unloaded=(np.zeros_like(energy.x), np.zeros_like(nodes)),
                    uniform=(np.full_like(energy.x, 1500.), np.full_like(nodes, .06*G)),
                    nonuniform=(1500+8000*abs(energy.y-.5), G*.06*(1+.2*np.cos(2*np.pi*nodes))))
    results = {}; curves = []
    for name, (tau, Z) in controls.items():
        initial = state(energy, seed, tau, Z)
        derivatives = verify_derivative(energy, initial)
        variation = profile_variation_control(energy, initial, name)
        cases = []
        for ratio in [.00125, 1., 4.]:
            accepted = initial
            for cycle in range(3):
                rate = 0. if name == 'unloaded' else 2e-5
                accepted, record = step(energy, accepted, ratio*ETA/G, ETA, rate)
                record['cycle'] = cycle+1; cases.append(record)
                curves.append(dict(control=name, ratio=ratio, cycle=cycle+1,
                                   phase=accepted.phi.reshape(N+1, N+1).tolist()))
            before = fingerprint(accepted)
            try:
                step(energy, accepted, ratio*ETA/G, ETA, rate, fail=True)
            except RuntimeError as error:
                assert 'Injected rejection' in str(error)
            assert fingerprint(accepted) == before
        _, endpoint = step(energy, initial, ETA/G, ETA, 0. if name == 'unloaded' else 2e-5,
                           endpoint_control=True)
        # General elastic limit: stable kernels, not replacing exp by 1 in model.
        _, elastic = step(energy, initial, .125, 1e30, 0. if name == 'unloaded' else 2e-5)
        _, coupled_mode = step(energy, initial, ETA/G, ETA,
                               0. if name == 'unloaded' else 2e-5, free_bulk_mode=True)
        results[name] = dict(derivatives=derivatives, profile_variation=variation,
                             steps=cases, endpoint_balance_comparator=endpoint,
                             elastic_limit=elastic, free_bulk_mode=coupled_mode,
                             rejected_trial_preserves_state=True)
        print(name, 'max phase', max(r['phi_max'] for r in cases),
              'max KKT', max(r['phase']['kkt'] for r in cases),
              'max energy defect', max(abs(r['total_defect']) for r in cases), flush=True)
    artifact = dict(prototype_only=True, production_model_changed=False, native_coupled_qualified=False,
                    input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    settings=dict(G=G, eta=ETA, Gc=GC, ell=ELL, c0=C0, m=M, pressure=PRESSURE, mu=MU,
                                  bulk_cells=N, surface_spacing=1/energy.ns, bulk_q=3, surface_q=3,
                                  phase_kkt_tolerance=1e-11, energy_absolute_tolerance=1e-8),
                    results=results)
    (out/'summary.json').write_text(json.dumps(artifact, indent=2)+'\n')
    (out/'phase-fields.json').write_text(json.dumps(curves)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-total-energy-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    y = np.arange(N+1)/N
    axes[0].plot(y, seed.reshape(N+1, N+1)[N//2], 'k--', label='Prescribed seed')
    for name in controls:
        field = next(r['phase'] for r in curves if r['control'] == name and r['ratio'] == .00125 and r['cycle'] == 1)
        axes[0].plot(y, np.asarray(field)[N//2], label=name)
    axes[0].set(xlabel='Normal coordinate y (m)', ylabel='Phase', title='Center section after first phase stage')
    axes[0].legend(); axes[0].grid(alpha=.2)
    keys = ['bulk_profile_jump', 'fault_profile_jump', 'fracture_expenditure', 'phase_release']
    x = np.arange(3)
    for j, key in enumerate(keys):
        axes[1].bar(x+(j-1.5)*.18, [results[name]['steps'][0][key] for name in controls], .18,
                    label=key.replace('_', ' '))
    axes[1].set(xticks=x, xticklabels=list(controls), ylabel='Energy (J/m)', title='Separate instantaneous-stage terms')
    axes[1].legend(fontsize=7); axes[1].grid(axis='y', alpha=.2)
    fig.savefig(out/'phase-and-energy.png', dpi=160)
    fig.savefig(out/'phase-and-energy.pdf')


if __name__ == '__main__':
    main()
