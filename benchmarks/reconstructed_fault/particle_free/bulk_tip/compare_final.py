#!/usr/bin/env python3
"""Compare retained final native samples on the SAME mesh, without an initial frame."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference');p.add_argument('candidate');p.add_argument('--name',required=True)
p.add_argument('--step',type=int,help='Compare this retained accepted step instead of the final frame')
a = p.parse_args()
def read(path):
    return np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf8')).view(np.recarray)
def final(label):
    folder=ROOT/'build-particle-free/bulk-tip/runs'/label/'output'
    rows=read(folder/'bulk-accepted.csv')
    step=rows[-1] if a.step is None else rows[rows.step==a.step][0]
    q=np.concatenate([read(f) for f in sorted(folder.glob(f'tip-points-{int(step.step):04d}-r*.csv'))]).view(np.recarray)
    q=q[np.lexsort((q.y,q.x))]
    sec=read(folder/f'tip-sections-{int(step.step):04d}.csv')
    return step,q,sec
r,x,s=final(a.reference);c,y,t=final(a.candidate)
assert abs(r.time-c.time)<1e-12 and len(x)==len(y)
assert np.max(np.abs(x.x-y.x))+np.max(np.abs(x.y-y.y))<1e-12
assert len(s)==len(t) and np.max(np.abs(s.s-t.s))<1e-12
result=dict(reference=a.reference,candidate=a.candidate,time=float(r.time),point_max={},point_L2={},section_max={})
for field in ('phi','tauxx','tauyy','tauzz','tauxy','Axy','qcoh','w','signed_shear','absolute_shear'):
    diff=y[field]-x[field]
    result['point_max'][field]=float(np.max(np.abs(diff)))
    result['point_L2'][field]=float(np.sqrt(np.sum(x.weight*diff**2)))
for field in ('age','slip','V','force'):
    result['section_max'][field]=float(np.max(np.abs(t[field]-s[field])))
result['accepted_energy_difference']={k:float(c[k]-r[k]) for k in ('bulk','cohesive','fracture')}
out=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/tip_solver'/(a.name+'.json')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
