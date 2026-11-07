#!/usr/bin/env python3
"""Retained Q1 MacCormack algorithm replay, using the earlier isolation evaluator.
Native old/new coupled comparisons separately use the actual retained solver data.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'transported_objective'))
from isolate_profile_projection import grid,quadrature,q1
root=Path(__file__).resolve().parents[4]
output=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/q1-controls.json'
if output.exists():raise SystemExit(f'Preserve {output}')
def profile(x):return .35*np.maximum(0.,1-((x[...,0]+.5)/.7)**2)**4*np.maximum(0.,1-(x[...,1]/.7)**2)**4
rows=[]
for level in (4,5,6):
 n=2**level;X=grid(n);points,weights=quadrature(n);h=4/n
 for speed in (1.,-1.):
  for dt in (.05,.025,.0125):
   nodes=profile(X);initial=q1(nodes,points);departure=X.copy();departure[...,0]-=dt*speed
   ret=X.copy();ret[...,0]+=dt*speed
   indices=np.clip(np.floor((departure+2)/h).astype(int),0,n-1);i,j=indices[...,0],indices[...,1]
   inside=np.all((departure>=-2)&(departure<=2),axis=-1)
   reverse_inside=np.all((ret>=-2)&(ret<=2),axis=-1)
   start=time.monotonic()
   for _ in range(round(.75/dt)):
    forward=q1(nodes,departure);reverse=q1(forward,ret)
    donors=np.stack([nodes[i,j],nodes[i+1,j],nodes[i,j+1],nodes[i+1,j+1]])
    low=np.where(inside,donors.min(axis=0),0);high=np.where(inside,donors.max(axis=0),0)
    corrected=np.where(reverse_inside,forward+.5*(nodes-reverse),forward)
    nodes=np.clip(corrected,low,high)
   seconds=time.monotonic()-start
   field=q1(nodes,points);foot=points.copy();foot[:,0]-=.75*speed;exact=profile(foot)
   mass=float(weights@field);centroid=(weights*field)@points/mass
   width=np.sqrt((weights*field)@((points-centroid)**2)/mass)
   rows.append(dict(level=level,speed=speed,dt=dt,initial_l2=float(np.sqrt(weights@((initial-profile(points))**2))),
                    final_l2=float(np.sqrt(weights@((field-exact)**2))),initial_integral=float(weights@initial),final_integral=mass,
                    peak=float(nodes.max()),centroid=centroid.tolist(),width=width.tolist(),seconds=seconds,
                    storage_bytes=int(nodes.nbytes)))
output.write_text(json.dumps(rows,indent=2)+'\n')
