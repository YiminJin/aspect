#!/usr/bin/env python3
"""Saved-entry replay of the derived endpoint-consistent reduced potential."""
import argparse
import json
import time

import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import minimize

from profile_temporal_diagnosis import E, directory, fields, Reference, G, ETA, A, h
from total_energy_step import dh, MU, PRESSURE
from time_resolution import load, energy


class Endpoint:
    def __init__(self,d):
        phi,R,Z=fields(directory(d))
        hist=load(directory(d)/'history.json')
        self.dt=1/d;self.time=.625
        self.lower=phi[-2].copy();self.lower.flags.writeable=False
        self.Ra=R[-2].copy();self.Ra.flags.writeable=False
        self.Za=Z[-2];self.slip_a=hist[-2]['slip']
        self.sequential=phi[-1].copy()
        self.r=Reference(256,1.,6)
        self.beta=np.exp(-G*self.dt/ETA)
        self.kappa=-ETA*np.expm1(-G*self.dt/ETA)
        self.evaluations=0;self.last=None
        self.D=self.r.D[:,1:-1]
        self.gram=self.D.T@(self.r.w[:,None]*self.D)

    def evaluate(self,phi):
        if self.last is not None and np.array_equal(phi,self.last[0]):
            return self.last[1:3]
        if np.min(phi)<0 or np.max(phi)>=1:
            return np.inf,np.zeros_like(phi)
        r=self.r
        r.r=self.Ra.copy();r.Z=self.Za;r.slip=self.slip_a;r.phi=phi.copy()
        data=r.mechanical(self.time,self.dt)
        assert np.max(abs(r.r-self.beta*self.Ra-self.kappa/G*r.velocity))<1e-16
        val,g,_=r.objective(phi)
        potential=val+self.kappa/G*MU*PRESSURE*data['V']/A
        self.evaluations+=1
        self.last=(phi.copy(),potential,g.copy(),data.copy(),r.r.copy(),r.Z)
        return potential,g

    def hessian(self,phi):
        self.evaluate(phi)
        r=self.r
        I,J=r.direct(phi);hp=h(r.B@phi)
        C=(sparse.diags(dh(r.B@phi)/I)@r.B).toarray()-np.outer(hp/I**2,J)
        chi=hp/I;tau=G*r.D@r.r-r.Z*chi
        # Mechanical Schur complement of the full endpoint potential.
        cross_bulk=-r.Z*self.D.T@(r.w[:,None]*C)
        if self.last[3]['V']>0:
            cross_slip=-r.Z*J/I**2-C.T@(r.w*tau)+r.Z*C.T@(r.w*chi)
            cross=np.vstack([cross_bulk,cross_slip])
            coupling=-self.D.T@(r.w*chi)
            mech=np.block([[self.gram,coupling[:,None]],
                           [coupling[None,:],np.array([[r.w@(chi*chi)+1/I]])]])
        else:
            cross=cross_bulk;mech=self.gram
        return r.phase_hessian(phi)-cross.T@solve(mech,cross,assume_a='pos')/(G*A)


def main(d):
    out=E/f'endpoint-replay-{d}';out.mkdir(exist_ok=False)
    start=time.perf_counter();p=Endpoint(d)
    x=p.sequential.copy();v,g=p.evaluate(x);H=p.hessian(x)
    free=np.flatnonzero((x>p.lower+1e-13)|(g<0))
    eigen=np.linalg.eigvalsh(H[np.ix_(free,free)])
    direction=np.cos(np.arange(len(x))*.23)*((x>.05)&(x<.95));direction/=np.linalg.norm(direction)
    checks=[]
    for eps in [1e-4,1e-5,1e-6]:
        vp,gp=p.evaluate(x+eps*direction);vm,gm=p.evaluate(x-eps*direction)
        checks.append(dict(epsilon=eps,energy_derivative_error=abs((vp-vm)/(2*eps)-g@direction),
                           Hessian_direction_error=float(max(abs((gp-gm)/(2*eps)-H@direction)))))
    # Same local bounded minimization and exact Newton polish; only the
    # consistent-time-level functional and its derived Schur tangent differ.
    answer=minimize(p.evaluate,x,jac=True,method='L-BFGS-B',
                    bounds=list(zip(p.lower,np.ones_like(x))),
                    options=dict(gtol=2e-13,ftol=2e-16,maxiter=3000,maxls=60,maxcor=30))
    final=p.r.refine(answer.x,p.lower,p.evaluate,p.hessian)
    vf,gf=p.evaluate(final);defect=p.r.defect(gf,final,p.lower)
    data=p.last[3]
    # No retained accepted history is modified by this saved-entry replay.
    np.savez_compressed(out/'trial.npz',phase=final,lower=p.lower,sequential=p.sequential,
                        incoming_R=p.Ra,incoming_Z=p.Za,trial_R=p.last[4],trial_Z=p.last[5],
                        gradient=gf,y=p.r.y)
    term=energy(p.r,p.lower,final,p.Ra,p.Za,p.last[4],p.last[5],p.dt,data['V'],p.time)
    result=dict(denominator=d,diagnostic_saved_entry_only=True,
                formulation_document='P2_profile_temporal_candidate.md',
                derivative_checks=checks,reduced_curvature_before=dict(minimum=float(min(eigen)),
                negative_count=int(sum(eigen<0)),free_nodes=len(free)),
                original_endpoint_KKT=float(p.r.defect(g,x,p.lower)),
                final_endpoint_KKT=float(defect),same_target=1e-11,
                phase_difference=float(max(abs(final-x))),
                phase_increment=float(max(final-p.lower)),phase_max=float(max(final)),
                solver_success=bool(defect<=1e-11),mechanical=data,
                energy=term,profile_energy_change=-term['D_relax'],
                inner_evaluations=p.evaluations,runtime_seconds=time.perf_counter()-start,
                physical_history_published=False,optimizer_message=str(answer.message))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--denominator',type=int,default=1024)
    args=parser.parse_args();main(args.denominator)
