#!/usr/bin/env python3
"""Independent prescribed-translation reference for the new bounded correction."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import json
import numpy as np
from isolate_profile_projection import grid,quadrature,compact,q1,ROOT

def main():
    rows=[]
    for n in [16,32,64]:
        X=grid(n);x,w=quadrature(n);h=4/n
        for speed in [-.4,.4]:
            for dt in [.1,.05,.025]:
                for method in ['Q1 remap','bounded MacCormack']:
                    a=compact(X);foot=X.copy();foot[...,0]-=speed*dt
                    ret=X.copy();ret[...,0]+=speed*dt
                    ij=np.clip(np.floor((foot+2)/h).astype(int),0,n-1);i,j=ij[...,0],ij[...,1]
                    for _ in range(round(.4/dt)):
                        f=q1(a,foot)
                        if method=='bounded MacCormack':
                            reverse=q1(f,ret);candidate=f+.5*(a-reverse)
                            donors=np.stack([a[i,j],a[i+1,j],a[i,j+1],a[i+1,j+1]])
                            a=np.clip(candidate,donors.min(axis=0),donors.max(axis=0))
                        else:a=f
                    exact=x.copy();exact[:,0]-=.4*speed;value=q1(a,x);err=value-compact(exact)
                    rows.append(dict(n=n,speed=speed,dt=dt,method=method,
                        relative_L2=float(np.sqrt(w@(err*err)/(w@(compact(exact)**2)))),
                        mass=float(w@value),peak=float(a.max())))
    out=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile/prescribed-profile.json'
    out.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
