#!/usr/bin/env python3
"""Matched 1D consistent endpoints with independent phase/mechanical spaces.

Reuse the retained analytic Q1 normal integral and Newton polish. Mechanical
elimination below is checked against the retained Q2 matrix solve; the old
split time driver is never called. No native state is modified.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import eigvalsh, eigh
from scipy.optimize import minimize

PROP = Path(__file__).resolve().parents[1]/'propagation'
sys.path.insert(0, str(PROP))
from localization_1d import Reference
from matched_initialization import evaluate_reference
from total_energy_step import G, ETA, ELL, M, MU, PRESSURE, FRACTURE_SCALE as A, h, dh

SEED = Path(__file__).with_name('pf6a-initial-phase.txt')
TARGET = 1e-11  # Retained independent reference KKT target, dimensionless by A.


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class Endpoint:
    direct = Reference.direct
    defect = staticmethod(Reference.defect)
    refine = Reference.refine

    def __init__(self, phase_cells, mechanical_cells=None, order=12):
        self.n = phase_cells
        self.nm = mechanical_cells
        self.dy = 1/self.n
        self.y = np.linspace(0, 1, self.n+1)
        data = np.loadtxt(SEED)
        self.initial = evaluate_reference(self.y)
        # At every retained mesh <=512 this is byte-identical to the supplied
        # native table; finer phase meshes sample the same continuum definition.
        self.mass = np.full(self.n+1, self.dy)
        self.mass[[0, -1]] *= .5
        self.F = sparse.diags([-np.ones(self.n),
                              np.r_[1., np.full(self.n-1, 2.), 1.],
                              -np.ones(self.n)], [-1, 0, 1]).toarray()*2*A*ELL**2/self.dy
        panels = np.union1d(self.y, np.linspace(0, 1, (self.nm or 1)+1))
        z, w = np.polynomial.legendre.leggauss(order)
        self.yq = (panels[:-1, None]+np.diff(panels)[:, None]*(z+1)/2).ravel()
        self.w = (np.diff(panels)[:, None]*w/2).ravel()
        c = np.minimum((self.n*self.yq).astype(int), self.n-1)
        s = self.n*self.yq-c
        row = np.repeat(np.arange(len(s)), 2)
        cols = (c[:, None]+[0, 1]).ravel()
        self.B = sparse.csr_matrix((np.c_[1-s, s].ravel(), (row, cols)),
                                  shape=(len(s), self.n+1))
        # Hessian of the exact Q1 ray integral, using retained 24-point rule.
        z, w = np.polynomial.legendre.leggauss(24)
        self.hs, self.hw = (z+1)/2, w/(2*self.n)
        if self.nm:
            c = np.minimum((self.nm*self.yq).astype(int), self.nm-1)
            xi = 2*(self.nm*self.yq-c)-1
            self.P = sparse.csr_matrix((np.c_[np.ones(len(c)), xi].ravel(),
                                        (row, (2*c[:, None]+[0, 1]).ravel())),
                                       shape=(len(c), 2*self.nm))
            self.pm = np.tile([1/self.nm, 1/(3*self.nm)], self.nm)
        self.lower = self.initial.copy()
        self.mean_n = self.Zn = self.slip_n = 0.
        self.last = None
        self.evaluations = 0

    def incoming(self, phase, mean, Z, slip, time_end, dt):
        self.lower = phase.copy()
        self.lower.flags.writeable = False
        self.mean_n, self.Zn, self.slip_n = mean, Z, slip
        self.time, self.dt = time_end, dt
        self.beta = np.exp(-G*dt/ETA)
        self.kappa = -ETA*np.expm1(-G*dt/ETA)
        self.mean = self.beta*mean+self.kappa*.256*(time_end-dt/2)
        self.last = None

    def integrals(self, phase, second=False):
        I, J = self.direct(phase)
        ph = self.B@phase
        hh, hp = h(ph), dh(ph)
        K = self.w@hh
        Kp = np.asarray(self.B.T@(self.w*hp))
        C, Cp = K/I, Kp/I-K*J/I**2
        residual = np.zeros_like(hh)
        coeff = np.zeros(0)
        if self.nm:
            coeff = np.asarray(self.P.T@(self.w*hh))/self.pm
            residual = hh-self.P@coeff
        D0 = self.w@(residual**2)
        Dp0 = 2*np.asarray(self.B.T@(self.w*residual*hp))
        D, Dp = D0/I**2, Dp0/I**2-2*D0*J/I**3
        H, Hp = C*C+D+1/I, 2*C*Cp+Dp-J/I**2
        result = dict(I=I, J=J, chi=hh/I, hp=hp, hh=hh, C=C, D=D,
                      H=H, Hp=Hp, Cp=Cp, residual=residual/I, coeff=coeff/I)
        if second:
            s = self.hs
            pp = (1-s)*phase[:-1, None]+s*phase[1:, None]
            hpp = M*(12/(1-ph)**4-6/(1-ph)**3)
            hpp_ray = M*(12/(1-pp)**4-6/(1-pp)**3)
            left = np.sum(self.hw*hpp_ray*(1-s)**2, axis=1)
            right = np.sum(self.hw*hpp_ray*s*s, axis=1)
            cross = np.sum(self.hw*hpp_ray*s*(1-s), axis=1)
            HI = sparse.diags([cross, np.r_[left, 0.]+np.r_[0., right], cross],
                              [-1, 0, 1]).toarray()
            KH = (self.B.T@sparse.diags(self.w*hpp)@self.B).toarray()
            CH = KH/I-(np.outer(Kp, J)+np.outer(J, Kp))/I**2-K*HI/I**2+2*K*np.outer(J, J)/I**3
            DH0 = np.zeros_like(HI)
            if self.nm:
                dcoeff = sparse.diags(1/self.pm)@self.P.T@sparse.diags(self.w*hp)@self.B
                DH0 = 2*((self.B.T@sparse.diags(self.w*(hp*hp+residual*hpp))@self.B)
                         -dcoeff.T@sparse.diags(self.pm)@dcoeff).toarray()
            DH = DH0/I**2-2*(np.outer(Dp0, J)+np.outer(J, Dp0))/I**3-2*D0*HI/I**3+6*D0*np.outer(J, J)/I**4
            result['HH'] = 2*C*CH+2*np.outer(Cp, Cp)+DH-HI/I**2+2*np.outer(J, J)/I**3
            result['CH'] = CH
        return result

    def evaluate(self, phase):
        if self.last is not None and np.array_equal(phase, self.last['phase']):
            return self.last['potential'], self.last['gradient']/A
        if min(phase) < 0 or max(phase) >= 1:
            return np.inf, np.zeros_like(phase)
        a = self.integrals(phase)
        f = MU*PRESSURE
        Z = max(self.beta*self.Zn, (a['C']*self.mean-f)/a['H'])
        V = (Z-self.beta*self.Zn)/self.kappa
        lam = self.mean-Z*a['C']
        tau = lam-Z*a['residual']
        bulk = self.w@(tau*tau)/(2*G)
        cohesive = Z*Z/(2*G*a['I'])
        fracture = A*self.mass@phase+.5*phase@(self.F@phase)
        gb = np.asarray(self.B.T@(-self.w*Z*tau/G*a['hp']/a['I']))
        gb += a['J']*(self.w@(Z*tau/G*a['hh']/a['I']**2))
        gc = -Z*Z/(2*G*a['I']**2)*a['J']
        gf = A*self.mass+self.F@phase
        potential = (bulk+cohesive+fracture+f/G*(Z-self.beta*self.Zn))/A
        self.last = dict(phase=phase.copy(), potential=potential, gradient=gb+gc+gf,
                         bulk_driving=gb, cohesive_driving=gc, fracture_driving=gf,
                         Z=Z, V=V, q=Z/a['I'], slip=self.slip_n+self.dt*V,
                         meanT=self.mean, tau_mean=lam, tau=tau, chi=a['chi'],
                         coeff=a['coeff'], I=a['I'], C=a['C'], D=a['D'],
                         bulk=bulk, cohesive=cohesive, fracture=fracture,
                         contact=f+Z/a['I']-self.w@(a['chi']*tau))
        self.evaluations += 1
        return potential, (gb+gc+gf)/A

    def hessian(self, phase):
        self.evaluate(phase)
        a = self.integrals(phase, True)
        Z = self.last['Z']
        H = self.F-self.mean*Z/G*a['CH']+Z*Z/(2*G)*a['HH']
        if self.last['V'] > 0:
            cross = -self.mean*a['Cp']+Z*a['Hp']
            H -= np.outer(cross, cross)/(G*a['H'])
        return H/A

    def positive_search(self, x):
        # Retained P2 positive spectral search metric. The energy, exact
        # Hessian, obstacle and KKT target are unchanged. This handles the
        # nearly translational branch where indefinite root polishing fails.
        for iteration in range(250):
            value, g = self.evaluate(x)
            if self.defect(g, x, self.lower) <= TARGET:
                return x
            ids = np.flatnonzero((x > self.lower+1e-13) | (g < 0.))
            H = self.hessian(x)[np.ix_(ids, ids)]
            ev, Q = eigh(H)
            metric = np.maximum(ev, max(abs(min(ev)), 1e-10))
            d = np.zeros_like(x)
            d[ids] = -Q@((Q.T@g[ids])/metric)
            allowance = 32*np.finfo(float).eps*max(1., abs(value))
            for ls in range(60):
                trial = np.maximum(self.lower, x+2.**(-ls)*d)
                if max(trial) >= 1:
                    continue
                potential = self.evaluate(trial)[0]
                if potential <= value+1e-4*min(float(g@(trial-x)), 0.)+allowance:
                    if np.array_equal(x, trial):
                        return x
                    x = trial
                    break
            else:
                return x
        return x

    def solve(self, guess):
        x = np.maximum(self.lower, guess)
        departures = []
        for search in range(100):
            x = self.refine(x, self.lower, self.evaluate, self.hessian)
            if self.defect(self.evaluate(x)[1], x, self.lower) > TARGET:
                answer = minimize(self.evaluate, x, jac=True, method='L-BFGS-B',
                                  bounds=list(zip(self.lower, np.ones_like(x))),
                                  options=dict(gtol=2e-13, ftol=2e-16, maxiter=3000,
                                               maxls=60, maxcor=30))
                x = self.refine(answer.x, self.lower, self.evaluate, self.hessian)
            if self.defect(self.evaluate(x)[1], x, self.lower) > TARGET:
                x = self.positive_search(x)
                x = self.refine(x, self.lower, self.evaluate, self.hessian)
            value, g = self.evaluate(x)
            defect = float(self.defect(g, x, self.lower))
            assert defect <= TARGET, ('KKT', self.n, self.nm, self.time, defect)
            assert min(x-self.lower) >= 0 and max(x) < 1
            H = self.hessian(x)
            free = x > self.lower+1e-13
            direction = np.zeros_like(x)
            direction[1:-1] = -(x[2:]-x[:-2])/(2*self.dy)
            direction[~free] = 0.
            if max(abs(direction)) > 0:
                direction /= max(abs(direction))
            curvature = float(direction@(H@direction))
            uncertainty = 0.
            if curvature < 0:
                actions = []
                for eps in [1e-5, 5e-6]:
                    actions.append((self.evaluate(x+eps*direction)[1]
                                    -self.evaluate(x-eps*direction)[1])/(2*eps))
                uncertainty = abs(float(direction@(actions[0]-actions[1])))
            if curvature >= -4*uncertainty:
                self.evaluate(x)
                ids = np.flatnonzero(free)
                minimum = float(eigvalsh(H[np.ix_(ids, ids)], subset_by_index=[0, 0])[0]) if len(ids) else None
                ids = np.flatnonzero(free | (abs(g) <= 10*TARGET))
                enlarged = float(eigvalsh(H[np.ix_(ids, ids)], subset_by_index=[0, 0])[0]) if len(ids) else None
                return x, defect, dict(free_minimum=minimum, enlarged_minimum=enlarged,
                                      translation_curvature=curvature,
                                      translation_uncertainty=uncertainty,
                                      departures=departures,
                                      enlarged_negative_is_not_cone_proof=True)
            # Retained native endpoint: both signs, half feasible distance,
            # actual reduced-potential decrease beyond its arithmetic allowance.
            selected = None
            allowance = 32*np.finfo(float).eps*max(1., abs(value))
            for sign in [1., -1.]:
                d = sign*direction
                amplitude = min(np.min((x-self.lower)[d<0]/(-d[d<0])) if np.any(d<0) else np.inf,
                                np.min((1-x)[d>0]/d[d>0]) if np.any(d>0) else np.inf)
                trial = x+.5*amplitude*d
                if amplitude > 0 and min(trial-self.lower) >= 0:
                    potential = self.evaluate(trial)[0]
                    if potential < value-allowance and (selected is None or potential < selected[0]):
                        selected = potential, trial
            assert selected is not None, ('resolved negative curvature without resolved descent', self.time, curvature, uncertainty)
            departures.append(dict(curvature=curvature, uncertainty=uncertainty,
                                   potential_decrease_J_per_m=float(A*(value-selected[0]))))
            x = self.positive_search(selected[1])
        raise RuntimeError(('translation departure budget', self.time))



def run(output, n, nm, divisor=128, order=12):
    output.mkdir(parents=True, exist_ok=False)
    model = Endpoint(n, nm, order)
    phase = model.initial.copy()
    mean = Z = slip = 0.
    rows, phases, zs, means, slips, coefficients = [], [phase.copy()], [Z], [mean], [slip], []
    driving = [np.stack([np.zeros(n+1), np.zeros(n+1), A*model.mass+model.F@phase])]
    if nm:
        coefficients.append(np.zeros((nm, 2)))
    else:
        coefficients.append(np.zeros((0, 2)))
    started = time.perf_counter()
    previous = np.zeros_like(phase)
    for step in range(1, round(.5625*divisor)+1):
        model.incoming(phase, mean, Z, slip, step/divisor, 1/divisor)
        try:
            x, kkt, stability = model.solve(phase+previous)
        except Exception as error:
            np.savez_compressed(output/'failed-entry.npz',phase=phase,meanT=mean,Z=Z,slip=slip,initial=model.initial)
            write(output/'failure.json',dict(time=step/divisor,error=repr(error)))
            write(output/'history.json',rows)
            raise
        model.evaluate(x)
        d = model.last
        assert abs(d['contact']) < 1e-7 or (d['V'] == 0 and d['contact'] >= 0)
        rows.append(dict(time=step/divisor, KKT=kkt, stability=stability,
                         phi_max=float(max(x)), increment=float(max(x-phase)),
                         **{key:float(d[key]) for key in ['q','V','slip','Z','meanT','I','C','D','tau_mean','bulk','cohesive','fracture','contact']}))
        previous = x-phase
        phase, mean, Z, slip = x.copy(), d['meanT'], d['Z'], d['slip']
        phases.append(phase.copy()); zs.append(Z); means.append(mean); slips.append(slip)
        coefficients.append(np.c_[np.full(nm, d['tau_mean'])+Z*d['coeff'][::2], Z*d['coeff'][1::2]] if nm else np.zeros((0,2)))
        driving.append(np.stack([d[k] for k in ['bulk_driving','cohesive_driving','fracture_driving']]))
        if step % 16 == 0 or step/divisor > .52:
            print(json.dumps(dict(n=n, nm=nm, step=step, phi_max=max(phase), KKT=kkt)), flush=True)
    np.savez_compressed(output/'fields.npz', phase=np.array(phases), y=model.y,
                        Z=zs, meanT=means, slip=slips, coefficients=np.array(coefficients),
                        driving=np.array(driving), time=np.arange(len(phases))/divisor)
    write(output/'history.json', rows)
    write(output/'summary.json', dict(phase_cells=n, mechanical_cells=nm, divisor=divisor,
                                     quadrature=order, completed=True, steps=len(rows),
                                     seed_sha256=hashlib.sha256(SEED.read_bytes()).hexdigest(),
                                     seed_definition='retained continuum seeded-H BVP; native nodal identity checked',
                                     seed_BVP_sha256=hashlib.sha256((PROP/'matched_initialization_evidence/reference.npz').read_bytes()).hexdigest(),
                                     seconds=time.perf_counter()-started, evaluations=model.evaluations,
                                     onset=next((r['time'] for k,r in enumerate(rows,1)
                                                 if max(phases[k]-phases[0])>1e-5),None)))


def verify(output):
    checks = []
    for np_, nm in [(32, 32), (64, 32), (32, 64), (64, None)]:
        p = Endpoint(np_, nm, 24)
        p.incoming(p.initial, 40000., 38000., .1, .5625, 1/128)
        x = p.initial+.01*np.exp(-20*(p.y-.5)**2)
        d = np.sin(4.7*p.y)+.3*np.cos(8*p.y)
        d *= .01
        value, g = p.evaluate(x)
        H = p.hessian(x)
        trials = []
        for eps in [2e-4, 5e-5, 1e-5]:
            vp, gp = p.evaluate(x+eps*d)
            vm, gm = p.evaluate(x-eps*d)
            trials.append(dict(energy=abs((vp-vm)/(2*eps)-g@d),
                               tangent=float(max(abs((gp-gm)/(2*eps)-H@d)))))
        assert min(r['energy'] for r in trials)<1e-7 and min(r['tangent'] for r in trials)<1e-7
        assert np.max(abs(H-H.T))<1e-10
        checks.append(dict(phase_cells=np_, mechanical_cells=nm, derivatives=trials))
    # Independent full Q2 matrix equation, not the projection implementation.
    class Supplied(Reference):
        def initialize(self):
            self.startup_defect=0.
            return np.interp(self.y,np.loadtxt(SEED)[:,0],np.loadtxt(SEED)[:,1])
    for n in [16,32]:
        p=Endpoint(n,n,24);p.incoming(p.initial,40000.,38000.,.1,.5625,1/128)
        p.evaluate(p.initial);d=p.last
        r=Supplied(n,1.,24);r.r=40000./G*(np.linspace(0,1,2*n+1)-.5);r.Z=38000.
        old=r.mechanical(.5625,1/128)
        error=max(abs(d['tau']-r.tau));assert error<1e-4
        checks.append(dict(matrix_equivalence_cells=n,stress_max_error_Pa=float(error),Z_error=abs(r.Z-d['Z']),V_error=abs(old['V']-d['V'])))
    write(output, dict(passed=True,checks=checks))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','verify']);p.add_argument('output',type=Path)
    p.add_argument('--phase-cells',type=int,default=256);p.add_argument('--mechanical-cells',type=int)
    p.add_argument('--divisor',type=int,default=128);p.add_argument('--quadrature',type=int,default=12)
    a=p.parse_args()
    if a.action=='verify':verify(a.output)
    else:run(a.output,a.phase_cells,a.mechanical_cells,a.divisor,a.quadrature)
