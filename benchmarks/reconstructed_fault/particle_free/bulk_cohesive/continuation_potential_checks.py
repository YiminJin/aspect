#!/usr/bin/env python3
"""Independent local derivative check of the native BE incremental potential."""
from pathlib import Path
import os
import ctypes,json
import numpy as np
ROOT=Path(os.environ.get('ASPECT_P4_ROOT',Path(__file__).resolve().parents[4])).resolve()
lib=ctypes.CDLL(str(ROOT/'build-particle-free/bulk-fracture/strip/libbulk_strip_oracle.so'))
fn=lib.bulk_strip_points;array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');fn.argtypes=[ctypes.c_uint,array,array,array,array]
G,eta,m,R=1e6,1e8,4.,1000.;dt=.00125;beta=1/(1+dt*G/eta);p=np.array([G,eta,m,R]);results=[]
for label,old,gamma,phi in [('sliding',[1284.090909090909,.0001,.00012,.00012,.1,184.0909090909091],.0004,.18),('sticking',[2100.,.0009,.001,.001,.2,600.],0.,.25)]:
    old=np.array([old]);base=np.array([gamma,phi])
    hold=m*old[0,4]*(1+old[0,4])/(1-old[0,4])**2
    assert abs(G*old[0,1]/hold-old[0,5])<1e-10
    assert abs(old[0,0]-G*old[0,1]-old[0,5])<=R+1e-10
    def evaluate(z):
        out=np.zeros((1,19));inc=np.array([[z[0],z[1],dt]])
        assert fn(1,p,old,inc,out)==0
        a=out[0]
        return a[9]+a[10]+beta*dt*a[13],np.array([beta*a[6],-a[8]]),a
    energy,gradient,a=evaluate(base)
    assert (a[7]==0.) == (label=='sticking')
    h=m*phi*(1+phi)/(1-phi)**2;hp=m*(1+3*phi)/(1-phi)**3;hpp=6*m*(1+phi)/(1-phi)**4
    J=np.array([[G*beta**2*(1-G*a[15]),-G*beta*a[16]],[-hp*a[5]*a[17]*beta,-(hpp*a[5]**2+2*hp*a[5]*a[18])/(2*G)]])
    energy_errors=[];action_errors=[]
    for d in [np.array([.0001,.01]),np.array([-.0002,.02])]:
        eps=1e-3;ep,gp,_=evaluate(base+eps*d);em,gm,_=evaluate(base-eps*d)
        energy_errors.append(abs((ep-em)/(2*eps)-gradient@d)/max(1.,abs(gradient@d)))
        action_errors.append(float(np.linalg.norm((gp-gm)/(2*eps)-J@d)/max(1.,np.linalg.norm(J@d))))
    assert max(energy_errors)<1e-7 and max(action_errors)<1e-6 and np.max(abs(J-J.T))<1e-8
    results.append(dict(branch=label,w_s_inv=float(a[7]),potential_energy_derivative_error=max(energy_errors),tangent_action_error=max(action_errors),cross_symmetry_error=float(np.max(abs(J-J.T)))))
out=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_continuation/potential-checks.json';out.write_text(json.dumps(results,indent=2)+'\n');print(out.read_text())
