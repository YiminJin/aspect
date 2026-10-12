#!/usr/bin/env python3
"""Replay only retained first saddle entries; no new near-release search."""
import numpy as np
import matched_initialization as m
import asymmetric_branch as a

def main():
    rows=[]
    for n in [511,512,1024]:
        path=m.E/f'n{n}';f=np.load(path/'fields.npz');hist=a.read(path/'history.json');t=a.read(path/'summary.json')['first_saddle'];j=round(t*1024)-1
        p=m.Exact(n);p.incoming(f['phase'][j],f['R'][j],f['Z'][j],hist[j-1]['slip'],t,1/1024)
        increment=f['phase'][j]-f['phase'][j-1]
        x=p.r.refine(np.maximum(p.lower,p.lower+increment),p.lower,p.evaluate,p.hessian)
        status,v=a.spectrum(p,x);status.pop('lower_y',None)
        assert status['admissible_negative'] and status['KKT']<=a.TARGET
        H=p.hessian(x);M=a.mass(p.r);checks=[]
        for eps in [1e-5,2e-5]:
            assert min(x-eps*abs(v))>=0 and max(x+eps*abs(v))<1
            gplus=p.evaluate(x+eps*v)[1];gminus=p.evaluate(x-eps*v)[1]
            curvature=float(a.A*v@(gplus-gminus)/(2*eps)/(v@M@v))
            checks.append(dict(epsilon=eps,curvature_Pa=curvature,error=abs(curvature-status['minimum'])))
        assert min(r['error'] for r in checks)<max(1e-5,.01*abs(status['minimum']))
        p.evaluate(x);data=p.last[3].copy();energy=sum(data[k] for k in ['bulk','cohesive','fracture'])
        accepted=hist[j];rows.append(dict(n=n,time=t,symmetric=status,finite_differences=checks,
              accepted_minimum=accepted['stability']['minimum'],accepted_phase_difference=float(max(abs(f['phase'][j+1]-x))),
              endpoint_energy_reduction=float(energy-sum(accepted[k] for k in ['bulk','cohesive','fracture'])),
              purpose='Replay saved physical entry only; no further near-release amplitudes or gains are searched'))
        np.savez_compressed(m.E/f'retained-mode-n{n}.npz',phase=x,mode=v,lower=p.lower,y=p.r.y)
    a.write(m.E/'mode-comparison.json',rows);print(a.json.dumps(rows),flush=True)

if __name__=='__main__':main()
