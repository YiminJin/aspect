#!/usr/bin/env python3
"""Reconstruct native trial predictors and independently check new-plane Y0.

Uses full physical endpoint tensors and midpoint reconstruction. No source
moment formula and no native angular search are reused.
"""
import json,sys,os
from pathlib import Path
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[n]='1'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'bulk_tip'));sys.path.insert(0,str(HERE.parent/'single_turn'))
from analyze_resolution_matched import ROOT,RUNS,read,gather
from reference import Parameters,resistance,shear_tensor
import numpy as np
from scipy.optimize import brentq,minimize_scalar
p=Parameters();label=sys.argv[1] if len(sys.argv)>1 else 'first-turn-append-admissible'
step=int(sys.argv[2]) if len(sys.argv)>2 else 15
out=RUNS/label/'output';h=.0025;beta=1/(1+h*p.G/p.eta)
a=gather(out,f'tip-points-{step-1:04d}-r',['cell','q']);b=gather(out,f'tip-points-{step:04d}-r',['cell','q'])
planes=read(out/f'tip-planes-{step:04d}.csv');sections=read(out/f'tip-sections-{step:04d}.csv')
records=[]
for k in np.flatnonzero((planes.formed>0)&(planes.parent_formed==0)):
    take=b.section==k;old=a[take];new=b[take]
    assert max(abs(old.Axx))+max(abs(old.Ayy))+max(abs(old.Azz))+max(abs(old.Axy))==0
    P=np.zeros((len(new),3,3))
    for field,i,j in [('xx',0,0),('yy',1,1),('zz',2,2),('xy',0,1)]:
        P[:,i,j]=.5*(old['tau'+field]+new['tau'+field])+p.G*new['A'+field]
        P[:,j,i]=P[:,i,j]
    phi=new.stage_phi;c=p.m*phi*(1+phi);F=c/(c+(1-phi)**2)
    def objective(angle,detail=False):
        S=shear_tensor(angle);tau=P[:,0,0]*S[0,0]+P[:,1,1]*S[1,1]+2*P[:,0,1]*S[0,1]
        ds=planes.reference_dx[k]/np.cos(angle)
        free=sum(new.weight*F*tau)/(p.G*beta*h*ds);C=sum(new.weight*F)/(p.G*beta*h*ds)
        age=planes.parent_age[k]
        scale=2*p.V0*np.exp(-(p.mu0+p.b*np.log((age+h)*p.V0/p.Dc))/p.a)
        if free==0:V=0.
        else:
            solve=lambda z:(scale*np.sinh(z)+C*resistance(scale*np.sinh(z),age,h,p)-abs(free))/abs(free)
            z=brentq(solve,0,np.arcsinh(abs(free)/scale),xtol=1e-15,rtol=4*np.finfo(float).eps)
            V=np.copysign(scale*np.sinh(z),free)
        f=resistance(V,age,h,p)
        Y=sum(new.weight*(1-F)**2*(tau-f)**2)*p.m/(2*p.G)
        return (Y,V,f) if detail else Y
    angle=planes.angle[k];Y,V,f=objective(angle,True)
    grid=np.linspace(-np.pi/4,np.pi/4,257);vals=np.array([objective(v) for v in grid]);best=int(np.argmax(vals))
    assert 0<best<len(grid)-1
    optimum=minimize_scalar(lambda v:-objective(v),bounds=(grid[best-1],grid[best+1]),method='bounded',options={'xatol':1e-12})
    gain=(-optimum.fun-Y)/max(Y,1e-300)
    assert gain<1e-9,(k,gain)
    assert abs(f-sections.force[k])<1e-7
    records.append(dict(section=int(k),x=float(planes.x[k]),native_angle=float(angle),independent_angle=float(optimum.x),
                        relative_objective_gain=float(gain),Y0_integral=float(Y),force_error=float(abs(f-sections.force[k])),rate_error=float(abs(V-sections.V[k]))))
assert records
(ROOT/f'benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn/{label}-plane-selection-{step:04d}.json').write_text(json.dumps(records,indent=2)+'\n')
print('PASS',len(records),'new sections; maximum independent objective gain',max(r['relative_objective_gain'] for r in records))
