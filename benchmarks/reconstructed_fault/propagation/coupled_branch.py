#!/usr/bin/env python3
"""Restricted consistent-endpoint trajectory; no production/history conversion.

Equations are derived before implementation in P2_coupled_branch_equations.md.
All mechanics trials start from immutable incoming R,Z,phase. The continuous
diagnostic path uses the same spatial energy and Maxwell internal variables.
"""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from profile_temporal_diagnosis import directory, fields, projected
from localization_1d import Reference
from total_energy_step import G, ETA, FRACTURE_SCALE as A, MU, PRESSURE, h, dh

HERE = Path(__file__).resolve().parent
E = HERE / 'coupled_branch_fresh_evidence'
START = 0.
FINAL = .5625
TARGET = 1e-11


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class Coupled:
    def __init__(self):
        self.r = Reference(256, 1., 6)
        self.D = self.r.D[:, 1:-1]
        self.gram = self.D.T @ (self.r.w[:, None]*self.D)
        self.factor = cho_factor(self.gram)
        self.evaluations = 0
        self.last = None

    def incoming(self, phi, R, Z, slip, time_end, dt):
        self.lower = phi.copy(); self.lower.flags.writeable = False
        self.Ra = R.copy(); self.Ra.flags.writeable = False
        self.Za = float(Z); self.slip_a = float(slip)
        self.time = time_end; self.dt = dt
        self.beta = np.exp(-G*dt/ETA)
        self.kappa = -ETA*np.expm1(-G*dt/ETA)
        self.anchor_shear = self.r.D @ self.Ra
        self.last = None

    def evaluate(self, phi):
        if self.last is not None and np.array_equal(phi, self.last[0]):
            # Energy-path diagnostics also use this scratch Reference. Restore
            # the cached endpoint before any Hessian or subsequent consumer.
            self.r.r = self.last[4].copy(); self.r.Z = self.last[5]
            self.r.phi = phi.copy()
            return self.last[1:3]
        if min(phi) < 0 or max(phi) >= 1:
            return np.inf, np.zeros_like(phi)
        r = self.r
        I, _ = r.direct(phi); chi = h(r.B@phi)/I
        tau0 = G*self.anchor_shear-self.Za*chi
        rate = .256*(self.time-self.dt/2)
        fixed = np.array([-rate/2, rate/2])
        lift = r.D[:, [0, -1]]@fixed
        c = -self.D.T@(r.w*chi)
        forcing = self.beta*tau0+self.kappa*lift
        rhs = -self.D.T@(r.w*forcing)/self.kappa
        u0, b = cho_solve(self.factor, np.c_[rhs, c]).T
        ss = r.w@(chi*chi)+1/I
        rV = (r.w*chi)@forcing-self.beta*self.Za/I-MU*PRESSURE
        V = max(0., (rV-self.kappa*c@u0)/(self.kappa*(ss-c@b)))
        u = np.r_[fixed[0], u0-b*V, fixed[1]]
        Rb = self.beta*self.Ra+(self.kappa/G)*u
        Zb = self.beta*self.Za+self.kappa*V
        tau = G*(r.D@Rb)-Zb*chi
        force = r.D.T@(r.w*tau)
        rf = MU*PRESSURE+Zb/I-r.w@(chi*tau)
        r.r = Rb.copy(); r.Z = Zb; r.slip = self.slip_a+self.dt*V
        r.phi = phi.copy(); r.velocity = u.copy()
        val, g, terms = r.objective(phi)
        potential = val+self.kappa/G*MU*PRESSURE*V/A
        data = dict(q=float(Zb/I), V=float(V), slip=float(r.slip),
                    bulk=float(terms[0]), cohesive=float(terms[1]),
                    fracture=float(terms[2]), force=float(max(abs(force[1:-1]))),
                    contact_signed=float(rf),
                    contact=float(abs(rf) if V > 0 else max(-rf, 0.)),
                    I=float(I), stress_min=float(min(tau)), stress_max=float(max(tau)))
        self.evaluations += 1
        self.last = (phi.copy(), potential, g.copy(), data, Rb.copy(), Zb)
        return potential, g

    def hessian(self, phi):
        self.evaluate(phi)
        r = self.r
        I, J = r.direct(phi); hp = h(r.B@phi)
        C = (sparse.diags(dh(r.B@phi)/I)@r.B).toarray()-np.outer(hp/I**2, J)
        chi = hp/I; tau = G*(r.D@r.r)-r.Z*chi
        cross = -r.Z*self.D.T@(r.w[:, None]*C)
        solved = cho_solve(self.factor, cross)
        correction = cross.T@solved
        if self.last[3]['V'] > 0:
            slip_cross = -r.Z*J/I**2-C.T@(r.w*tau)+r.Z*C.T@(r.w*chi)
            c = -self.D.T@(r.w*chi)
            schur = r.w@(chi*chi)+1/I-c@cho_solve(self.factor, c)
            row = slip_cross-c@solved
            correction += np.outer(row, row)/schur
        return r.phase_hessian(phi)-correction/(G*A)

    def solve(self):
        anchor = self.Ra.copy(); bound = self.lower.copy(); Z = self.Za
        answer = minimize(self.evaluate, bound.copy(), jac=True, method='L-BFGS-B',
                          bounds=list(zip(bound, np.ones_like(bound))),
                          options=dict(gtol=2e-13, ftol=2e-16, maxiter=3000,
                                       maxls=60, maxcor=30))
        x = self.r.refine(answer.x, self.lower, self.evaluate, self.hessian)
        _, g = self.evaluate(x)
        defect = float(self.r.defect(g, x, self.lower))
        assert np.array_equal(anchor, self.Ra) and np.array_equal(bound, self.lower)
        assert Z == self.Za and not self.lower.flags.writeable and not self.Ra.flags.writeable
        return x, defect, str(answer.message)


def stability(p, x):
    """A positive enlarged-subspace eigenvalue bounds the whole critical cone.

    A negative eigenvalue is called admissible only if its weak-contact entries
    have a common sign. Nonstationary spectra are explicitly not equilibria.
    """
    _, g = p.evaluate(x)
    free = x > p.lower+1e-13
    weak = (~free) & (abs(g) <= 10*TARGET)
    ids = np.flatnonzero(free | weak)
    out = dict(phase_KKT=float(p.r.defect(g, x, p.lower)), free=int(sum(free)),
               weak=int(sum(weak)), strict_lower=int(sum(~(free | weak))),
               lower_y=p.r.y[~free].tolist(), equilibrium=bool(p.r.defect(g, x, p.lower)<=TARGET))
    if not len(ids):
        out.update(minimum=None, negative_count=0, admissible_negative=False,
                   verdict='empty critical cone: strictly constrained stable')
        return out
    H = p.hessian(x)
    eig, vec = np.linalg.eigh(H[np.ix_(ids, ids)])
    d = vec[:, 0]; w = d[weak[ids]]
    admissible = not len(w) or min(w) >= -1e-12 or max(w) <= 1e-12
    out.update(minimum=float(eig[0]), negative_count=int(sum(eig < 0)),
               admissible_negative=bool(eig[0] < 0 and admissible),
               verdict=('positive enlarged critical-space curvature' if eig[0] > 0
                        else 'negative admissible curvature' if admissible
                        else 'negative unrestricted curvature; cone inconclusive'))
    return out


def state_check(r, phi, R, Z, V, lower):
    r.r = R.copy(); r.Z = float(Z)
    _, g, terms = r.objective(phi)
    force = r.D.T@(r.w*terms[-1])
    rf = MU*PRESSURE+Z/terms[6]-r.w@(terms[7]*terms[-1])
    return dict(phase_KKT=float(max(abs(projected(g, phi, lower)))),
                mechanical=float(max(abs(force[1:-1]))), contact_signed=float(rf),
                contact=float(abs(rf) if V > 0 else max(-rf, 0.)), V=float(V))


def path_ledger(r, pa, pb, Ra, Za, Rb, Zb, u, V, dt, order=16):
    """Continuous phase plus exact Maxwell path, no instantaneous profile jump."""
    def state(phi, R, Z):
        r.r = R; r.Z = Z
        return r.objective(phi)[2]
    a = state(pa, Ra, Za); b = state(pb, Rb, Zb)
    cost = sum(state(pb, Ra, Za)[:3])-sum(a[:3])
    z, w = leggauss(order); s = (z+1)*dt/2; wt = w*dt/2
    phidot = (pb-pa)/dt
    du = r.D@u
    lift = np.linspace(u[0], u[-1], len(u))
    integrals = {k:0. for k in ['external_work', 'bulk_work', 'lifting_work',
                  'bulk_viscous', 'cohesive_viscous', 'friction',
                  'mechanical_power', 'friction_residual_power', 'phase_power',
                  'bulk_profile_power', 'normalization_power', 'fracture_power']}
    absolute = {k:0. for k in ['mechanical_power', 'friction_residual_power', 'phase_power']}
    maximum = dict(mechanical=0., contact=0., phase_KKT=0., chi_rate=0., identity_power=0.)
    for t, weight in zip(s, wt):
        beta = np.exp(-G*t/ETA); kappa = -ETA*np.expm1(-G*t/ETA)
        R = beta*Ra+kappa/G*u; Z = beta*Za+kappa*V
        phi = pa+t*phidot
        terms = state(phi, R, Z)
        I, J = r.direct(phi); hp = h(r.B@phi); chi = hp/I
        chid = dh(r.B@phi)*(r.B@phidot)/I-hp*(J@phidot)/I**2
        tau = terms[-1]; f = r.D.T@(r.w*tau)
        g = sum(terms[3:6])
        rf = MU*PRESSURE+Z/I-r.w@(chi*tau)
        db = r.w@(tau*tau)/ETA; dc = Z*Z/(ETA*I)
        bp = -Z/G*r.w@(tau*chid)
        npower = -Z*Z/(2*G*I**2)*(J@phidot)
        fp = (r.flinear+r.F@phi)@phidot
        values = dict(external_work=u[[0,-1]]@f[[0,-1]], bulk_work=u@f,
                      lifting_work=lift@f, bulk_viscous=db, cohesive_viscous=dc,
                      friction=MU*PRESSURE*V, mechanical_power=u[1:-1]@f[1:-1],
                      friction_residual_power=rf*V, phase_power=g@phidot,
                      bulk_profile_power=bp, normalization_power=npower,
                      fracture_power=fp)
        # Direct constitutive chain rule, independent of assembled g.
        taudot = G*du-G*chi*V-G/ETA*tau-Z*chid
        Zdot = G*V-G/ETA*Z
        edot = r.w@(tau*taudot)/G+Z/(G*I)*Zdot+npower+fp
        rhs = values['external_work']-db-dc-values['friction']+sum(values[k] for k in absolute)
        maximum['identity_power'] = max(maximum['identity_power'], abs(edot-rhs))
        maximum['mechanical'] = max(maximum['mechanical'], max(abs(f[1:-1])))
        maximum['contact'] = max(maximum['contact'], abs(rf) if V>0 else max(-rf,0.))
        maximum['phase_KKT'] = max(maximum['phase_KKT'], max(abs(projected(g/A, phi, pa))))
        maximum['chi_rate'] = max(maximum['chi_rate'], max(abs(chid)))
        for k in integrals: integrals[k] += weight*values[k]
        for k in absolute: absolute[k] += weight*abs(values[k])
    changes = np.array(b[:3])-np.array(a[:3])
    remainder = integrals['external_work']-sum(changes)-sum(integrals[k] for k in ['bulk_viscous','cohesive_viscous','friction'])
    residual = sum(integrals[k] for k in absolute)
    return dict(**{k:float(v) for k,v in integrals.items()},
                **{k+'_absolute':float(v) for k,v in absolute.items()},
                bulk_change=float(changes[0]), cohesive_change=float(changes[1]),
                fracture_change=float(changes[2]), signed_profile_cost=float(cost),
                external_remainder=float(remainder), identity_error=float(abs(remainder+residual)),
                intermediate_max={k:float(v) for k,v in maximum.items()}, time_quadrature=order)


def prepare():
    E.mkdir(exist_ok=False)
    # Snapshot relevant incoming production and previous evidence; never mutate it.
    root = HERE.parents[2]
    files = list((root/'source').rglob('*.cc'))+list((root/'include').rglob('*.h'))
    for base in [HERE/'profile_temporal_evidence', HERE/'time_resolution_evidence', HERE/'coupled_branch_evidence']:
        files += [p for p in base.rglob('*') if p.is_file()]
    write(E/'preservation.json', {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    p = Coupled(); phi, R, Z = fields(directory(1024))
    hist = json.loads((directory(1024)/'history.json').read_text())
    # User's fallback rule applies to the LAST unchanged seed, even though an
    # earlier checkpoint passed. Preserve those earlier runs only as controls.
    j = max(i for i in range(len(phi)) if np.array_equal(phi[i], p.r.initial))
    check = state_check(p.r, phi[j], R[j], Z[j], hist[j-1]['V'], phi[j-1])
    assert check['phase_KKT']>TARGET
    seed=p.r.initial.copy();zero=np.zeros_like(p.r.r)
    fresh=state_check(p.r,seed,zero,0.,0.,seed)
    assert fresh['phase_KKT']<=TARGET and fresh['mechanical']==0 and fresh['contact']==0
    np.savez_compressed(E/'start.npz', phi=seed, R=zero, Z=0.,
                        slip=0., time=START, initial=p.r.initial,
                        particle_y=p.r.initial_particle_y, H=p.r.initial_particle_H)
    # Stability of the actual incoming state (not a fictitious extra update).
    p.r.r = zero.copy(); p.r.Z = 0.
    _, g, _ = p.r.objective(seed)
    write(E/'start-check.json', dict(time=START, source='fresh unloaded native seed',
          rejected_last_checkpoint_time=j/1024,rejected_checkpoint=check,
          state=fresh, seed_equal=True, min_scaled_phase_gradient=float(min(g)),
          critical_cone='all lower bounds have strictly positive gradient' if min(g)>10*TARGET else 'weak contacts require tangent check'))
    # Verify cached mechanics against existing native-function prototype.
    p.incoming(phi[round(.5234375*1024)], R[round(.5234375*1024)], Z[round(.5234375*1024)], hist[round(.5234375*1024)-1]['slip'], .52734375, 1/256)
    p.evaluate(p.lower); cached = p.last; velocity = p.r.velocity.copy()
    p.r.r = p.Ra.copy(); p.r.Z = p.Za; p.r.slip = p.slip_a; p.r.phi = p.lower.copy()
    reference = p.r.mechanical(p.time, p.dt)
    error = dict(R=float(max(abs(p.r.r-cached[4]))), Z=float(abs(p.r.Z-cached[5])),
                 velocity=float(max(abs(p.r.velocity-velocity))), V=float(abs(reference['V']-cached[3]['V'])))
    assert max(error.values())<1e-8
    write(E/'cached-mechanics-check.json', error)
    print(json.dumps(dict(start=fresh,rejected_checkpoint=check,cached_mechanics=error)), flush=True)


def replay():
    out = E/'replay'; out.mkdir(exist_ok=False)
    p = Coupled(); x, R, Z = fields(directory(1024))
    hist = json.loads((directory(1024)/'history.json').read_text())
    saved = np.load(HERE/'profile_temporal_evidence/endpoint-replay-1024/trial.npz')
    dt = 1/1024; lower = saved['lower']; final = saved['phase']
    incoming = state_check(p.r, lower, R[-2], Z[-2], hist[-2]['V'], x[-3])
    p.incoming(lower, R[-2], Z[-2], hist[-2]['slip'], .625, dt)
    _, g = p.evaluate(final); data = p.last[3].copy(); Rb = p.last[4].copy(); Zb = p.last[5]
    u = p.r.velocity.copy()
    ledgers = [path_ledger(p.r, lower, final, p.Ra, p.Za, Rb, Zb, u, data['V'], dt, order) for order in [16,32,64]]
    before = stability(p, saved['sequential']); after = stability(p, final)
    write(out/'classification.json', dict(incoming=incoming, endpoint=data,
         endpoint_phase_KKT=float(p.r.defect(g,final,lower)),
         phase_increment=float(max(final-lower)), before_stability=before,
         after_stability=after, continuous_path=ledgers,
         legacy_jump_path=json.loads((HERE/'profile_temporal_evidence/endpoint-replay-1024/qualified-numerical-summary.json').read_text())['energy']))
    print(json.dumps(dict(replay_incoming=incoming, phase_increment=float(max(final-lower)),
                         continuous_path=ledgers[-1], reduced_minimum=after['minimum'])), flush=True)


def trajectory(d):
    out = E/f'dt-1-{d}'; out.mkdir(exist_ok=False)
    started = time.perf_counter(); p = Coupled(); init = np.load(E/'start.npz')
    phi = init['phi'].copy(); R = init['R'].copy(); Z = float(init['Z']); slip = float(init['slip'])
    dt = 1/d; rows=[]; phases=[phi.copy()]; strains=[R.copy()]; zs=[Z]; us=[]
    onsets={}; statuses=[]; failures=[]; publishes=0
    for k in range(round((FINAL-START)/dt)):
        t = START+(k+1)*dt
        p.incoming(phi, R, Z, slip, t, dt)
        final, kkt, message = p.solve()
        p.evaluate(final); data=p.last[3].copy(); Rb=p.last[4].copy(); Zb=p.last[5]; u=p.r.velocity.copy()
        assert np.array_equal(phi,p.lower) and np.array_equal(R,p.Ra) and Z==p.Za
        if kkt>TARGET or data['force']>1e-7 or data['contact']>1e-7 or min(final-phi)<-4.922e-12:
            failures.append(dict(time=t,kkt=kkt,mechanical=data,message=message))
            np.savez_compressed(out/'failed-trial.npz',phi=final,lower=phi,R=Rb,Z=Zb,Ra=R,Za=Z)
            break
        ledger=path_ledger(p.r,phi,final,R,Z,Rb,Zb,u,data['V'],dt)
        ledger.update(time=t,phase_KKT=kkt,phase_increment=float(max(final-phi)))
        if k==0 or max(final-phi)>1e-3 or k==round((FINAL-START)/dt)-1:
            status=stability(p,final); status['time']=t;statuses.append(status)
            # Local optimizer found a stationary point; reject proved unstable
            # equilibria rather than publishing a saddle as physical history.
            if status['admissible_negative']:
                failures.append(dict(time=t,reason='negative admissible reduced curvature',stability=status))
                np.savez_compressed(out/'failed-trial.npz',phi=final,lower=phi,R=Rb,Z=Zb,Ra=R,Za=Z)
                break
        phi=final.copy();R=Rb.copy();Z=Zb;slip=data['slip'];publishes+=1
        phases.append(phi.copy());strains.append(R.copy());zs.append(Z);us.append(u)
        row=dict(time=t,phase_max=float(max(phi)),phase_increment=ledger['phase_increment'],
                 phase_KKT=kkt,**data); rows.append(row)
        for key,value in [('field',max(phi-init['initial'])),('core',phi[p.r.n//2]-init['initial'][p.r.n//2])]:
            if key not in onsets and value>1e-5: onsets[key]=[t-dt,t]
        write(out/'history.json', rows)
        # Small transient cost permits compact continuously recoverable evidence.
        existing=out/'energy.json'; energies=json.loads(existing.read_text()) if existing.exists() else []
        energies.append(ledger);write(existing,energies)
        print(json.dumps(dict(d=d,time=t,phi_max=row['phase_max'],increment=row['phase_increment'],
                              kkt=kkt,external_remainder=ledger['external_remainder'])),flush=True)
    np.savez_compressed(out/'fields.npz',phase=np.array(phases),R=np.array(strains),Z=np.array(zs),
                        velocity=np.array(us),y=p.r.y,yq=p.r.yq,initial=init['initial'])
    write(out/'stability.json',statuses)
    summary=dict(denominator=d,dt=dt,start=START,final=FINAL,completed=not failures,
                 accepted=publishes,failures=failures,onsets=onsets,evaluations=p.evaluations,
                 runtime_seconds=time.perf_counter()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','replay','run'])
    parser.add_argument('--denominator',type=int,choices=[256,512,1024],default=256)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='replay':replay()
    else:trajectory(args.denominator)
