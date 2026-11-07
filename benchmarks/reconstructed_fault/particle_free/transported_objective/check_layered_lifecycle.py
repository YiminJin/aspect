#!/usr/bin/env python3
"""Compare matched accepted trajectories, including background checkpoint history."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,json
import numpy as np
from assess_layered_campaign import OUT,load_case,state_at
from layered_fields import read

def main():
    p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');p.add_argument('--restart',action='store_true');a=p.parse_args()
    pa,ca,ra=load_case(a.a);pb,cb,rb=load_case(a.b);by={int(r['step']):r for r in ra};checks=[]
    for row in rb:
        step=int(row['step']);reference=by[step]
        clock=max(abs(float(row[k])-float(reference[k])) for k in ['time','dt'])
        assert clock<1e-13,(step,clock)
        x=state_at(pa,ca,step);y=state_at(pb,cb,step)
        bulk=float(np.max(abs(x.coeff-y.coeff)));surface=float(np.max(abs(x.surface-y.surface)))
        bga=read(pa/'output'/f'background-{step}.csv');bgb=read(pb/'output'/f'background-{step}.csv')
        aa=np.array([[float(v) for v in r.values()] for r in bga]);bb=np.array([[float(v) for v in r.values()] for r in bgb])
        background=float(np.max(abs(aa-bb)));checks.append(dict(step=step,time=float(row['time']),clock=clock,bulk=bulk,surface=surface,background=background))
        tolerance=2e-10 if a.restart else 1e-12
        assert max(bulk,surface)<tolerance and background<1e-12,(step,checks[-1])
    result=dict(a=a.a,b=a.b,restart=a.restart,checks=checks,max_bulk=max(r['bulk'] for r in checks),max_surface=max(r['surface'] for r in checks),max_background=max(r['background'] for r in checks))
    (OUT/('lifecycle-'+a.b+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
if __name__=='__main__':main()
