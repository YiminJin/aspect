#!/usr/bin/env python3
"""Separate boundary location and analytic-localization projection in native exports.

For constant matched coefficients, the native response implies its actual
pre-reaction physical anchor. Independent Gauss8 DGQ2 projections of analytic
B isolate delta_B(new)-Q delta_B(old at foot) from the remaining projected
transport/rotation mismatch. No source term is altered or invented by closure.
"""
import json
from pathlib import Path
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DEAL_II_NUM_THREADS"):
    os.environ[_key]="1"
import numpy as np
from assess_material_native import ROOT, RUNS, read, basis, norm3, rotation
from isolate_profile_projection import ClosedState

OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile'


class MemoryProjection:
    def __init__(self,state,memory=None):
        self.state=state;self.coeff=np.zeros((state.n,state.n,9,3))
        g,w=np.polynomial.legendre.leggauss(8)
        u=np.stack(np.meshgrid(g,g,indexing='ij'),axis=-1).reshape(-1,2)
        weights=np.outer(w,w).reshape(-1);b=basis(u)
        factors=np.array([(2*i+1)*(2*j+1)/4 for i in range(3) for j in range(3)])
        ii,jj=np.meshgrid(np.arange(state.n),np.arange(state.n),indexing='ij')
        ij=np.stack([ii.ravel(),jj.ravel()],axis=1)
        x=-2+state.h*(ij[:,None,:]+.5*(u[None,:,:]+1))
        points=x.reshape(-1,2)
        old=state.surface[:,4].copy()
        if memory is not None:state.surface[:,4]=memory
        values=state.memory(points,state.at(points)[:,0]).reshape(-1,64,3)
        state.surface[:,4]=old
        self.coeff[ii,jj]=np.einsum('qk,cqj,q,k->ckj',b,values,weights,factors).reshape(state.n,state.n,9,3)

    def at(self,x):
        s=self.state;ij=np.floor((x+2)/s.h).astype(int)
        valid=np.all((ij>=0)&(ij<s.n),axis=1)
        values=np.zeros((len(x),3));i,j=ij[valid].T
        u=2*((x[valid]+2)/s.h-ij[valid])-1
        values[valid]=np.einsum('ni,nij->nj',basis(u),self.coeff[i,j])
        return values


def inner(a,b):
    return a[:,0]*b[:,0]+a[:,1]*b[:,1]+(a[:,0]+a[:,1])*(b[:,0]+b[:,1])+2*a[:,2]*b[:,2]


def analyze(name):
    p=RUNS/name;command=json.loads((p/'command.json').read_text());assert command['exit_code']==0
    assert command['arguments']['eta']=='1'
    folder=p/'output';meta={k:float(v) for k,v in read(folder/'material-model.csv')[0].items()}
    rows=read(folder/'material-summary.csv');level=command['arguments']['level'];G=meta['G'];eta=meta['reference_eta']
    states={int(r['step']):ClosedState(folder,int(r['step']),level,meta['m']) for r in rows}
    result=[]
    for row in rows:
        step=int(row['step']);dt=float(row['dt'])
        if step==0 or dt<1e-10:continue
        old,new=states[step-1],states[step];x=new.x;v=new.values;oldv=old.at(x)
        foot=x-dt*oldv[:,1:3];angle=.5*dt*(oldv[:,14]-oldv[:,13])
        beta=np.exp(-G*dt/eta);kappa=-eta*np.expm1(-G*dt/eta)
        trace=v[:,12]+v[:,15]
        D=np.stack([v[:,12]-trace/3,v[:,15]-trace/3,.5*(v[:,13]+v[:,14])],axis=1)
        anchor=(v[:,9:12]-2*kappa*D+kappa*new.memory(x,v[:,0],6))/beta
        reference=rotation(old.stress(foot),angle)
        defect=beta*(anchor-reference)
        old_projection=MemoryProjection(old)
        new_projection=MemoryProjection(new,old.surface[:,4])
        original=new.surface[:,4].copy();new.surface[:,4]=old.surface[:,4]
        delta_new=new_projection.at(x)-new.memory(x,v[:,0]);new.surface[:,4]=original
        old_values=old.at(foot)
        delta_old=old_projection.at(foot)-old.memory(foot,old_values[:,0])
        projection=beta*(delta_new-rotation(delta_old,angle))
        remainder=defect-projection
        interior=np.max(abs(x),axis=1)<1.25
        inflow=command['arguments']['speed']*x[:,0]<-abs(command['arguments']['speed'])*1.25
        core=np.max(abs(x),axis=1)<.75
        upstream_inner=interior & (command['arguments']['speed']*x[:,0]<-abs(command['arguments']['speed'])*.75)
        areas={'whole':np.ones(len(x),dtype=bool),'interior':interior,'core':core,
               'upstream_inner':upstream_inner,'inflow_band':inflow,'other_boundary':~interior&~inflow}
        norms={}
        for region,mask in areas.items():
            w=new.weights[mask]
            integral=lambda a,b:float(w@inner(a[mask],b[mask]))
            square=integral(defect,defect)
            p2=integral(projection,projection);r2=integral(remainder,remainder);cross=2*integral(projection,remainder)
            assert abs(square-p2-r2-cross)<1e-14
            norms[region]=dict(defect_L2=float(np.sqrt(square)),localization_projection_L2=float(np.sqrt(p2)),
                remaining_transport_rotation_L2=float(np.sqrt(r2)),signed_cross_term=cross,squared_identity_error=square-p2-r2-cross)
        total=norms['whole']['defect_L2']**2
        result.append(dict(step=step,time=float(row['time']),dt=dt,regions=norms,
            interior_max_error_location=x[np.flatnonzero(interior)[np.argmax(norm3(defect[interior]))]].tolist(),
            inflow_fraction_squared_error=norms['inflow_band']['defect_L2']**2/total,
            interior_fraction_squared_error=norms['interior']['defect_L2']**2/total))
    return dict(case=name,steps=result)


def main():
    names=['matched-positive','matched-negative','matched-half-dt','matched-fine-final',
           'compact-profile-full','compact-profile-half']
    cases=[analyze(name) for name in names if (RUNS/name/'command.json').exists()]
    report=dict(cases=cases,
        decomposition='beta*(native physical anchor-Q tau_old(foot)) = beta*(delta_B_mapped-Q delta_B_old(foot)) + remaining projected transport/rotation mismatch; delta_B=P8 B-B, mapped B uses incoming Z.',
        limitation='The remainder combines DG transport, characteristic-reference error and finite-rotation projection. It is not a new constitutive source or an independent physical energy. Cross terms prevent assigning additive squared-error percentages to the two terms.')
    (OUT/'tensor-defect.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{c['case']:c['steps'][-1]} for c in cases]))


if __name__=='__main__':main()
