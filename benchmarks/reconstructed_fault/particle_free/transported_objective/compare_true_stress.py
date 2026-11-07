#!/usr/bin/env python3
"""Compact native direct/separated comparison; no reference treated as truth."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,re
from pathlib import Path
import numpy as np
from compare_bounded_profile import load,reduce,tensor,quadrature,dot3,RUNS,ROOT
from assess_positive_refinement import comparison
from audit_true_stress_equilibrium import audit
from assess_material_native import assess,read
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/direct_true_stress'

def extra(a,b):
    result=comparison(a,b);x,w=quadrature(max(a.n,b.n),8);ta,tb=tensor(a,x),tensor(b,x)
    va,vb=a.at(x),b.at(x)
    def metrics(mask):
        error=ta-tb
        return dict(full3_absolute=float(np.sqrt(w[mask]@dot3(error[mask],error[mask]))),
            full3_relative=float(np.sqrt((w[mask]@dot3(error[mask],error[mask]))/(w[mask]@dot3(tb[mask],tb[mask])))),
            components_absolute=np.sqrt(w[mask]@(error[mask]**2)).tolist())
    delta=x-b.surface[0,1:3];d=b.surface[-1,1:3]-b.surface[0,1:3];d/=np.linalg.norm(d)
    masks=dict(domain=np.ones(len(x),bool),interior=np.max(abs(x),axis=1)<1.25,
               fault_band=(np.max(abs(x),axis=1)<1.25)&(abs(delta@np.array([-d[1],d[0]]))<.4))
    result['tensor_windows']={k:metrics(mask) for k,mask in masks.items()}
    result['pressure_absolute_L2']=float(np.sqrt(w@((va[:,17]-vb[:,17])**2)))
    result['pressure_relative_L2']=float(np.sqrt((w@((va[:,17]-vb[:,17])**2))/(w@(vb[:,17]**2))))
    return result

def costs(name):
    p=RUNS/name;log=(p/'run.log').read_text();meta=json.loads((p/'command.json').read_text());step=max(int(r['step']) for r in read(p/'output/material-summary.csv'))
    records=[read(f)[0] for f in (p/'output').glob(f'material-{step}-storage-rank*.csv')]
    storage={k:sum(int(r[k]) for r in records) for k in records[0]} if records else {}
    outer=[(int(a),float(b)) for a,b in re.findall(r'Material projected reaction: outer=(\d+) correction relative change=([^\s]+)',log)]
    linear=[int(a) for a in re.findall(r'Fault linear solve: iterations=(\d+)',log)]
    timers={}
    for label,calls,seconds in re.findall(r'\| ([^|]+?)\s*\|\s*(\d+)\s*\|\s*([\d.e+-]+)s\s*\|',log):timers[label.strip()]={'calls':int(calls),'seconds':float(seconds)}
    return dict(wall_s=meta['wall_s'],ranks=meta['arguments']['ranks'],linear_solves=len(linear),linear_iterations=sum(linear),
        outer=outer,max_outer_index=max((x[0] for x in outer),default=None),timers=timers,storage=storage,
        checkpoint_bytes=sum(f.stat().st_size for f in (p/'output/restart').rglob('*') if f.is_file()),
        peak_scope='Sum of per-rank VmHWM including ASPECT, libraries, matrices, checker and all allocations; not simultaneous total RSS. Tensor bytes are vector-reported allocations, not all history caches.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',nargs='+',required=True,help='direct:separated labels');p.add_argument('--output',default='comparison.json');a=p.parse_args()
    names=list(dict.fromkeys(n for pair in a.pairs for n in pair.split(':')));cases={};states={};initial={}
    for name in names:
        r,final=reduce(name);cmd,rows,meta,first=load(name,False);states[name]=final;initial[name]=first[min(first)]
        cases[name]={'reduction':r,'cost':costs(name),'equilibrium':audit(final)}
        print(name,'energy endpoint/trap',r['endpoint_relative'],r['trapezoid_relative'],'free weak residual',cases[name]['equilibrium']['mechanical_relative'],flush=True)
    pairs={}
    for pair in a.pairs:
        direct,sep=pair.split(':');r=extra(states[direct],states[sep]);ia=comparison(initial[direct],initial[sep])
        ca=np.array(cases[direct]['reduction']['endpoint_cumulative_slip']);cb=np.array(cases[sep]['reduction']['endpoint_cumulative_slip'])
        r['cumulative_slip_relative_L2']=float(np.linalg.norm(ca-cb)/np.linalg.norm(cb))
        pairs[pair]={'initial':ia,'final':r}
    (OUT/a.output).write_text(json.dumps(dict(cases=cases,pairs=pairs),indent=2)+'\n')
if __name__=='__main__':main()
