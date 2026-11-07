#!/usr/bin/env python3
"""Match physical exports across ranks, restart, rejection and retained control."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json
import numpy as np
from compare_bounded_profile import load,ROOT,RUNS

def match(a,b):
    _,ra,_,sa=load(a);_,rb,_,sb=load(b)
    ma={float(r['time']):r for r in ra};mb={float(r['time']):r for r in rb};records=[]
    for t in sorted(ma.keys()&mb.keys()):
        r,q=ma[t],mb[t];x,y=sa[int(r['step'])],sb[int(q['step'])]
        assert r['history_publications']==q['history_publications'] and r['geometry_publications']==q['geometry_publications']
        records.append(dict(time=t,bulk_Linf=float(abs(x.values-y.values).max()),surface_Linf=float(abs(x.surface-y.surface).max()),bitwise=bool(np.array_equal(x.values,y.values) and np.array_equal(x.surface,y.surface))))
    assert records
    return dict(left=a,right=b,states=records)
if __name__=='__main__':
    pairs=[('bulk-old-total-control','total-cd4'),('bulk-cd3','bulk-cd-mpi'),('bulk-cd3','bulk-cd-restart'),('bulk-old-total-control','total-cd-restart'),('bulk-cd-retry-control','bulk-cd-reject'),('bulk-cs4','bulk-cs-restart')]
    results=[match(*p) for p in pairs]
    output=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin/lifecycle-comparison.json'
    output.write_text(json.dumps(results,indent=2)+'\n')
    for r in results:
        print(r['left'],r['right'],max(q['bulk_Linf'] for q in r['states']),max(q['surface_Linf'] for q in r['states']),all(q['bitwise'] for q in r['states']))
    # Same retained consistency thresholds as stress-history consolidation.
    for r in results:
        for s in r['states']:assert s['bulk_Linf']<5e-13 and s['surface_Linf']<5e-12,(r['left'],r['right'],s)
    for name in ['bulk-to-total-refused','total-to-bulk-refused']:
        assert 'PF5 checkpoint version/material/quadrature mismatch' in (RUNS/name/'run.log').read_text(errors='replace')
