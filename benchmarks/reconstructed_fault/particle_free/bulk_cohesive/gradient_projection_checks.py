#!/usr/bin/env python3
"""Convex-lens variational inequality, covariance and complete projection action."""
import json
import numpy as np
from gradient_point import OUT,project
maximum=0.;covariance=0.;vi=0.;active=set();cases=0
tau=np.array([450.,-150.,1000.]);p=10000.;mu=.1;pb=p-.5*sum(tau[:2]);R=mu*pb
d=np.array([.5*(tau[0]-tau[1]),tau[2]]);center=mu*np.array([-d[1],d[0]])
for z in [np.zeros(2),np.array([4000.,1000.]),np.array([-4000.,-1000.]),np.array([0.,3000.]),np.array([0.,-3000.]),2500*center/np.linalg.norm(center),-2500*center/np.linalg.norm(center),3000*d/np.linalg.norm(d),-3000*d/np.linalg.norm(d)]:
    f,J,mask=project(z,tau);active.add(mask);cases+=1
    for c in range(6):
        eps=.001;zp=z.copy();zm=z.copy();tp=tau.copy();tm=tau.copy();pp=pm=p
        if c<2:zp[c]+=eps;zm[c]-=eps
        elif c<5:tp[c-2]+=eps;tm[c-2]-=eps
        else:pp+=eps;pm-=eps
        fp,_,ap=project(zp,tp,pp);fm,_,am=project(zm,tm,pm)
        assert mask==ap==am
        maximum=max(maximum,float(np.max(np.abs((fp-fm)/(2*eps)-J[:,c]))))
    assert max(np.linalg.norm(f-center),np.linalg.norm(f+center))<=R+1e-12*R
    for a in np.linspace(0,2*np.pi,64,endpoint=False):
        v=.8*R*np.array([np.cos(a),np.sin(a)])
        if max(np.linalg.norm(v-center),np.linalg.norm(v+center))<=R:
            vi=max(vi,float(max(0.,-(f-z)@(v-f)/(R*R))))
    angle=.43;rotation=np.array([[np.cos(2*angle),-np.sin(2*angle)],[np.sin(2*angle),np.cos(2*angle)]])
    dr=rotation@d;mean=.5*sum(tau[:2]);tr=[mean+dr[0],mean-dr[0],dr[1]]
    fr,_,_=project(rotation@z,tr);covariance=max(covariance,float(np.linalg.norm(fr-rotation@f)/R))
assert active=={0,1,2,3},active
assert maximum<=1e-6 and covariance<=1e-12 and vi<=1e-12
(OUT/'projection-checks.json').write_text(json.dumps({'cases':cases,'active_masks':sorted(active),'complete_force_pressure_stress_FD':maximum,
    'basis_covariance_defect':covariance,'normalized_variational_inequality_defect':vi,
    'scope':'Unique convex projection, not a uniqueness claim for the nonlinear joint field problem.'},indent=2)+'\n')
print((OUT/'projection-checks.json').read_text())
