#!/usr/bin/env python3
"""Compare complete exported material state after rejection and repartition restart."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
from compare_bounded_profile import load,RUNS,ROOT
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/direct_true_stress'

def compare(a,b,continued=False):
    ca,ra,ma,sa=load(a);cb,rb,mb,sb=load(b);by_time={round(float(r['time']),12):int(r['step']) for r in rb};records=[]
    for row in ra:
        time=round(float(row['time']),12)
        if time not in by_time:continue
        ia=int(row['step']);ib=by_time[time];x,y=sa[ia],sb[ib]
        # Polynomial values at a complete Gauss3 set identify every DGQ2
        # tensor/native Q2 velocity and Q1 phase/pressure coefficient.
        bulk=np.max(abs(x.values-y.values),axis=0);surface=np.max(abs(x.surface-y.surface),axis=0)
        records.append(dict(time=time,step_a=ia,step_b=ib,bulk_component_Linf=bulk.tolist(),surface_component_Linf=surface.tolist(),
          bulk_Linf=float(bulk.max()),surface_Linf=float(surface.max()),history_publications=int(row['history_publications']),geometry_publications=int(row['geometry_publications'])))
        assert bulk.max()<1e-11 and surface.max()<1e-10,(a,b,records[-1])
    assert records
    if continued:
        assert len(records)>=2 and min(r['time'] for r in records)>.1
        crossings=[]
        for row in records:
            i=row['step_b'];old,new=sb[i-1],sb[i]
            crossings.append(int(np.any(np.floor((old.surface[:,1:3]+2)/new.h)!=np.floor((new.surface[:,1:3]+2)/new.h),axis=1).sum()))
        assert sum(crossings)>0
    else:crossings=None
    return dict(cases=[a,b],records=records,crossed_labels= crossings,
      tolerance='bulk per-component absolute Linf <1e-11; surface <1e-10; diagnostic comparisons, not relaxed publication or solver tolerances.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--rejection',nargs=2,required=True);p.add_argument('--restart',nargs=2,required=True);a=p.parse_args()
    result={'rejection':compare(*a.rejection),'restart':compare(*a.restart,continued=True)}
    command,rows,_,_=load(a.rejection[0]);assert max(int(r['retries']) for r in rows)==1
    (OUT/'lifecycle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
