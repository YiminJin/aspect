#!/usr/bin/env python3
"""Close homogeneous temporal sequence, including every common physical time."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from compare_bounded_profile import ROOT,load
from assess_sheared_moving_fault import diagnostics,compare
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_campaign'
names=['exterior-compatible050','exterior-compatible025','exterior-compatible0125','exterior-closure00625']
def main():
    result,_,_=diagnostics(names[-1]);(OUT/'homogeneous-closure.json').write_text(json.dumps(result,indent=2)+'\n')
    sets={}
    for name in names:
        cmd,rows,meta,states=load(name);cum=np.zeros(len(states[0].surface));records={}
        for row in rows:
            i=int(row['step']);t=float(row['time']);dt=float(row['dt'])
            if i:cum+=dt*states[i].surface[:,6]
            records[round(t,10)]=(states[i],cum.copy())
        sets[name]=records
    pairs=[]
    for a,b in zip(names[:-1],names[1:]):
        history=[]
        for t in sorted(set(sets[a])&set(sets[b])):
            x,cx=sets[a][t];y,cy=sets[b][t];ref=sets[names[-1]][t][0]
            history.append(dict(time=t,**compare(x,y,cx,cy,bulk_order=6,region_reference=ref)))
        pairs.append(dict(coarse=a,fine=b,history=history))
    (OUT/'homogeneous-history-comparison.json').write_text(json.dumps(pairs,indent=2)+'\n')
    for pair in pairs:
        last=pair['history'][-1]
        print(pair['coarse'],pair['fine'],json.dumps({r:{k:last['surface'][r][k] for k in ['cumulative_slip','slip_rate','total_normal_traction','work_shear_traction']} for r in ['central','ends']}),flush=True)
    print('closure runtime',result.get('wall_s'),'endpoint',result['coupled_work']['relative_active'],flush=True)
if __name__=='__main__':main()
