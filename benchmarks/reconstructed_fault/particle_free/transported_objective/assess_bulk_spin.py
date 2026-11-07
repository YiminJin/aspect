#!/usr/bin/env python3
"""Bounded constitutive comparison, independent exported-field energy and tractions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault("MPLCONFIGDIR","/tmp/aspect-bulk-spin-mpl")
import argparse,csv,json,re
from pathlib import Path
import numpy as np
from compare_bounded_profile import reduce,compare,quadrature,tensor,dot3
from assess_material_native import RUNS,ROOT,read
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin'

def surface_tractions(s):
    X=s.surface[:,1:3];d=np.diff(X,axis=0);length=np.linalg.norm(d,axis=1);t=d/length[:,None];n=np.stack([-t[:,1],t[:,0]],axis=1)
    g,w=np.polynomial.legendre.leggauss(4);u=.5*(g+1)
    x=(X[:-1,None,:]+u[None,:,None]*d[:,None,:]).reshape(-1,2)
    t=np.repeat(t,4,axis=0);n=np.repeat(n,4,axis=0);a=tensor(s,x);p=s.at(x)[:,-1]
    traction=np.stack([a[:,0]*n[:,0]+a[:,2]*n[:,1],a[:,2]*n[:,0]+a[:,1]*n[:,1]],axis=1)
    return np.sum(t*traction,axis=1),np.sum(n*traction,axis=1)-p,(.5*length[:,None]*w).ravel(),((np.arange(len(d))[:,None]+u)/len(d)).ravel()

def regional(a,b):
    x,w=quadrature(max(a.n,b.n),8);err=tensor(a,x)-tensor(b,x);base=tensor(b,x)
    X=b.surface[:,1:3];t=X[-1]-X[0];u=(x-X[0])@t/(t@t);normal=np.array([-t[1],t[0]])/np.linalg.norm(t);dist=np.abs((x-X[0])@normal)
    ash,an,_,labels=surface_tractions(a);bsh,bn,sw,_=surface_tractions(b)
    out={}
    for name,lo,hi in [('central',.2,.8),('endpoints',0.,1.)]:
        mask=(u>=lo)&(u<=hi)&(dist<.6);sm=(labels>=lo)&(labels<=hi)
        if name=='endpoints':mask&=((u<.2)|(u>.8));sm&=((labels<.2)|(labels>.8))
        out[name]=dict(stress_L2=float(np.sqrt(w[mask]@dot3(err[mask],err[mask]))),
          stress_relative_L2=float(np.sqrt((w[mask]@dot3(err[mask],err[mask]))/max(w[mask]@dot3(base[mask],base[mask]),1e-300))),
          shear_traction_L2=float(np.sqrt(sw[sm]@((ash[sm]-bsh[sm])**2))),
          normal_traction_L2=float(np.sqrt(sw[sm]@((an[sm]-bn[sm])**2))))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',nargs='+');a=p.parse_args();OUT.mkdir(exist_ok=True)
    data={};states={}
    for name in a.cases:
        r,s=reduce(name);data[name]=r;states[name]=s
        x,w=quadrature(s.n,8);tau=tensor(s,x);r['stress_component_L2']=np.sqrt(np.sum(w[:,None]*tau*tau,axis=0)).tolist()
        sh,n,sw,_=surface_tractions(s);r['shear_traction_L2']=float(np.sqrt(sw@(sh*sh)));r['normal_traction_L2']=float(np.sqrt(sw@(n*n)))
        r['slip_rate_range']=[float(s.surface[:,6].min()),float(s.surface[:,6].max())];r['state_range']=[float(s.surface[:,3].min()),float(s.surface[:,3].max())]
        log=(RUNS/name/'run.log').read_text(errors='replace');r['spin']=[{k:float(v) for k,v in re.findall(r'(\w+)=([0-9.eE+\-]+)',line)} for line in log.splitlines() if 'Accepted bulk spin:' in line]
        r['native_checks']=read(RUNS/name/'output/bulk-spin-native.csv') if (RUNS/name/'output/bulk-spin-native.csv').exists() else []
        print(name,'wall',round(r['wall_s'],2),'energy remainder',r['endpoint_relative'],flush=True)
    pairs=[('bulk-cd3','total-cd4'),('bulk-cs4','total-cs4'),('bulk-ld4','total-ld4'),('bulk-ls4','total-ls4'),('bulk-cd3','bulk-cd-half'),('bulk-ld4','bulk-ld-half'),('bulk-cs4','bulk-cd3'),('bulk-ls4','bulk-ld4')]
    comparisons=[]
    for lhs,rhs in pairs:
        if lhs not in states or rhs not in states:continue
        r=dict(left=lhs,right=rhs,**compare(states[lhs],states[rhs]),regions=regional(states[lhs],states[rhs]))
        delta=np.array(data[lhs]['endpoint_cumulative_slip'])-np.array(data[rhs]['endpoint_cumulative_slip'])
        r['cumulative_slip_L2_labels']=float(np.linalg.norm(delta));r['cumulative_slip_relative_L2']=float(np.linalg.norm(delta)/np.linalg.norm(data[rhs]['endpoint_cumulative_slip']))
        comparisons.append(r)
    (OUT/'coupled.json').write_text(json.dumps(dict(cases=data,comparisons=comparisons),indent=2)+'\n')
    fields=['case','wall_s','linear_solves','linear_iterations','storage_change','endpoint_signed','endpoint_absolute','trapezoid_signed','trapezoid_absolute','endpoint_relative','trapezoid_relative','shear_traction_L2','normal_traction_L2']
    with (OUT/'coupled.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in data.values())
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for name in a.cases:
        r=data[name];style='--' if name.startswith('total') else '-'
        ax[0,0].plot([q['time'] for q in r['steps']],[q['storage'] for q in r['steps']],style,label=name)
        ax[0,1].plot(np.linspace(0,1,len(states[name].surface)),r['endpoint_cumulative_slip'],style,label=name)
        ax[1,0].plot([q['time'] for q in r['steps'][1:]],np.cumsum([q['endpoint_remainder'] for q in r['steps'][1:]]),style,label=name)
        spin=r['spin']
        if spin:ax[1,1].plot([q['step']*r['arguments']['dt'] for q in spin],[q['fault_max'] for q in spin],label=name)
    for a1,title,y in zip(ax.ravel(),['Bulk elastic energy','Accumulated slip at t=0.1 s','Signed endpoint work remainder','Accepted fault spin'],['J/m','m','J/m','1/s']):a1.set_title(title);a1.set_ylabel(y);a1.grid(alpha=.2)
    ax[0,0].set_xlabel('time (s)');ax[0,1].set_xlabel('material label fraction');ax[1,0].set_xlabel('time (s)');ax[1,1].set_xlabel('time (s)');ax[0,0].legend(fontsize=7,ncol=2)
    fig.savefig(OUT/'coupled.png',dpi=160)
if __name__=='__main__':main()
