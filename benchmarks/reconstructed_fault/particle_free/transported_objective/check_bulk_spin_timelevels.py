#!/usr/bin/env python3
"""Compare logged frozen spin with accepted exported velocity/profile/converged V."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,re
import numpy as np
from compare_bounded_profile import load,quadrature,ROOT,RUNS

def fault_spin(state,x,phase,rates):
    X=state.surface[:,1:3];I=state.surface[:,5];width=state.surface[:,7]
    best=np.full(len(x),np.inf);omega=np.zeros(len(x))
    for j in range(len(X)-1):
        d=X[j+1]-X[j];t=d/np.linalg.norm(d);n=np.array([-t[1],t[0]])
        xi=(x-X[j])@d/(d@d);dist=np.abs((x-X[j])@n)
        mask=(xi>=0)&(xi<=1)&(dist<best)&(dist<=(1-xi)*width[j]+xi*width[j+1])
        u=xi[mask];p=phase[mask];h=state.m*p*(1+p)/(1-p)**2
        q=h*((1-u)*rates[j]+u*rates[j+1])/((1-u)*I[j]+u*I[j+1])
        # Explicit full signed dyad, with retained positive shear sense.
        omega[mask]=.5*q*(t[1]*n[0]-t[0]*n[1]);best[mask]=dist[mask]
    return omega
if __name__=='__main__':
    records=[]
    for name in ['bulk-cd3','bulk-cs4','bulk-ld4','bulk-ls4','bulk-cd-half','bulk-ld-half']:
        _,rows,_,states=load(name);log=(RUNS/name/'run.log').read_text()
        logs=[{k:float(v) for k,v in re.findall(r'(\w+)=([0-9.eE+\-]+)',l)} for l in log.splitlines() if 'Accepted bulk spin:' in l]
        for observed in logs:
            step=int(observed['step']);old=states[step-1];x,w=quadrature(old.n,8);a=old.at(x)
            total=.5*(a[:,14]-a[:,13]);fault=fault_spin(old,x,a[:,0],old.surface[:,6]);bulk=total-fault
            expected={}
            for key,v in [('total',total),('fault',fault),('bulk',bulk)]:expected[key+'_max']=float(np.max(abs(v)));expected[key+'_l2']=float(np.sqrt(w@(v*v)))
            error=max(abs(expected[k]-observed[k]) for k in expected)
            assert error<2e-12,(name,step,error)
            row=dict(case=name,step=step,spin_metrics_Linf=error,accepted_V_min=float(old.surface[:,6].min()),accepted_V_max=float(old.surface[:,6].max()))
            if step==1:
                wrong=fault_spin(old,x,a[:,0],np.full(len(old.surface),.001))
                row['initial_predictor_fault_max_gap']=abs(float(np.max(abs(wrong)))-observed['fault_max'])
                assert row['initial_predictor_fault_max_gap']>1e-3
            records.append(row)
    out=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin/timelevels.json';out.write_text(json.dumps(records,indent=2)+'\n')
    print('max metric error',max(r['spin_metrics_Linf'] for r in records))
