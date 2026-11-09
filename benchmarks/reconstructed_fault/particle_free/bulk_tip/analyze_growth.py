#!/usr/bin/env python3
"""Native-sample growth diagnostics; no remapping, smoothing or altered contours."""
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-tip-mpl')
from pathlib import Path
import numpy as np
from analyze_efficiency import ROOT,OUT,read,frame,accumulate
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--seed-balance-run',help='Zero-clock seed export for exact SIPG work partition')
p.add_argument('label');p.add_argument('--seed-run',required=True,help='Same-mesh equilibrated zero-clock reference with fracture-density export')
a=p.parse_args();d=ROOT/'build-particle-free/bulk-tip/runs'/a.label/'output'
seed_d=ROOT/'build-particle-free/bulk-tip/runs'/a.seed_run/'output'
seed=frame(seed_d,0);rows=read(d/'bulk-accepted.csv')
command=json.loads((d.parent/'command.json').read_text());parent_d=Path(command['arguments']['resume_from'])
oldrows=read(parent_d/'bulk-accepted.csv');parent_time=rows[0].time-rows[0].dt
old=oldrows[np.abs(oldrows.time-parent_time)<1e-12];assert len(old)==1
parent=frame(parent_d,int(old[0].step));oldseed=frame(parent_d,0)
assert np.max(np.abs(seed.phi-oldseed.phi))<1e-13 and np.max(np.abs(seed.x-oldseed.x))<1e-13
H=lambda q:80*q.phi*(1+q.phi)/(80*q.phi*(1+q.phi)+(1-q.phi)**2)
def extent(q,values,t):
 mask=values>t
 return [float(q.x[mask].min()),float(q.x[mask].max())] if np.any(mask) else None
result=dict(label=a.label,seed_run=a.seed_run,seed_phase_difference=float(np.max(np.abs(seed.phi-oldseed.phi))),
            parent_time=float(parent_time),phase_thresholds=[.001,.01,.1,.5,.99],H_thresholds=[.5,.9,.99],steps=[])
result['initial_extents']={str(t):extent(seed,seed.phi,t) for t in result['phase_thresholds']}
result['parent_extents']={str(t):extent(parent,parent.phi,t) for t in result['phase_thresholds']}
result['parent_H_extents']={str(t):extent(parent,H(parent),t) for t in result['H_thresholds']}
seed_balance=None
if a.seed_balance_run:
    seed_b=ROOT/'build-particle-free/bulk-tip/runs'/a.seed_balance_run/'output'
    seed_balance=np.concatenate([read(f) for f in sorted(seed_b.glob('tip-seed-fracture-r*.csv'))]).view(np.recarray)
    seed_balance=seed_balance[np.lexsort((seed_balance.i,seed_balance.cell))]
    seed_energy=float(read(seed_b/'tip-initial.csv').fracture[0])
for row in rows:
 q=frame(d,int(row.step));sec=read(d/f'tip-sections-{int(row.step):04d}.csv')
 assert len(q)==len(seed) and np.max(np.abs(q.x-seed.x))<1e-13
 local=row.dt*q.weight*q.w*sec.force[q.section.astype(int)]
 bands=0;maximum_bands=0
 for k in np.unique(q.section):
  samples=q[q.section==k];samples=samples[np.argsort(samples.y)]
  active=np.abs(samples.w)>max(1e-10,1e-6*np.max(np.abs(samples.w)))
  n=int(np.sum(active & ~np.r_[False,active[:-1]]));bands+=n>1;maximum_bands=max(maximum_bands,n)
 remote=np.abs(q.x-2)>1.3
 regions={}
 for name,mask in [('left',np.abs(q.x-1.2)<.25),('right',np.abs(q.x-2.8)<.25),('remote',remote)]:
  regions[name]=dict(damage_from_seed=float(q.weight[mask]@(q.phi[mask]-seed.phi[mask])),damage_from_parent=float(q.weight[mask]@(q.phi[mask]-parent.phi[mask])),
    fracture_volume_from_seed=float(q.weight[mask]@(q.fracture_density[mask]-seed.fracture_density[mask])),
    max_Y=float(q.Y[mask].max()),max_H=float(q.H[mask].max()),max_w=float(np.abs(q.w[mask]).max()))
 balance=np.concatenate([read(f) for f in d.glob(f'tip-balance-{int(row.step):04d}-r*.csv')]).view(np.recarray)
 front=(np.abs(balance.x-1.2)<.25)|(np.abs(balance.x-2.8)<.25)
 free=(balance.B>balance.lower+1e-12)&(balance.B<1-1e-12)
 if seed_balance is not None:
  balance=balance[np.lexsort((balance.i,balance.cell))]
  assert np.all(balance.cell==seed_balance.cell) and np.all(balance.i==seed_balance.i)
  fracture_work=.5*(balance.fracture_gradient+seed_balance.fracture_gradient)*(balance.B-seed_balance.B)
  for name,mask in [('left',np.abs(balance.x-1.2)<.25),('right',np.abs(balance.x-2.8)<.25),('remote',np.abs(balance.x-2)>1.3)]:
   regions[name]['SIPG_weak_work_from_seed']=float(fracture_work[mask].sum())
  regions['global_SIPG']=dict(weak_work_from_seed=float(fracture_work.sum()),energy_difference=float(row.fracture-seed_energy),closure_error=float(fracture_work.sum()-(row.fracture-seed_energy)))
  front=(np.abs(balance.x-1.2)<.25)|(np.abs(balance.x-2.8)<.25)
  free=(balance.B>balance.lower+1e-12)&(balance.B<1-1e-12)
 scaled=balance.residual/(40*balance.mass)
 result['steps'].append(dict(step=int(row.step),time=float(row.time),regions=regions,
   extents={str(t):extent(q,q.phi,t) for t in result['phase_thresholds']},H_extents={str(t):extent(q,q.H,t) for t in result['H_thresholds']},
   positive_work=float(np.maximum(local,0).sum()),negative_work=float(np.minimum(local,0).sum()),multiple_band_sections=bands,maximum_bands=maximum_bands,
   rate_cancellation=float(np.sum(sec.ds*(sec.integral_abs_w-np.abs(sec.V)))),
   front_balance=dict(free_rows=int(np.sum(front&free)),lower_rows=int(np.sum(front&~free)),
     max_free_scaled_residual=float(np.max(np.abs(scaled[front&free]))) if np.any(front&free) else None,
     min_lower_scaled_residual=float(np.min(scaled[front&~free])) if np.any(front&~free) else None),
   remote_driving_ratio=float(q.Y[remote].max()/40),remote_max_phase=float(q.phi[remote].max())))
result['sampling']=dict(mesh_spacing=.25,Gauss_points_per_axis=6,section_x_spacing_min=float(np.diff(np.unique(q.x)).min()),section_x_spacing_max=float(np.diff(np.unique(q.x)).max()))
OUT.mkdir(exist_ok=True,parents=True);(OUT/'growth.json').write_text(json.dumps(result,indent=2)+'\n')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
final=frame(d,int(rows[-1].step));last_sec=read(d/f'tip-sections-{int(rows[-1].step):04d}.csv')
old_sec=read(parent_d/f'tip-sections-{int(old[0].step):04d}.csv')
yline=np.unique(final.y)[np.argmin(np.abs(np.unique(final.y)-1))]
fig,axes=plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
for col,limits in enumerate(((.65,1.65),(2.35,3.35))):
 ax=axes[0,col];twin=ax.twinx()
 for q,label,style in [(parent,f'parent {parent_time:.4g} s','--'),(final,f'endpoint {rows[-1].time:.4g} s','-')]:
  mask=np.isclose(q.y,yline,rtol=0,atol=1e-12)
  ax.plot(q.x[mask],q.phi[mask],style,color='C0',label=label+' phase');twin.plot(q.x[mask],H(q)[mask],style,color='C1',label=label+' H')
  axes[1,col].plot(q.x[mask],q.tauxy[mask],style,label=label)
 ax.set_ylabel('phase',color='C0');twin.set_ylabel('H (elimination fraction)',color='C1');ax.legend(fontsize=7,loc='upper left');twin.set_ylim(0,1.05)
 ax=axes[2,col];twin=ax.twinx()
 for sec,label,style in [(old_sec,'parent','--'),(last_sec,'endpoint','-')]:
  ax.plot(sec.s,sec.V,style,color='C0',label=label+' V');twin.plot(sec.s,sec.slip,style,color='C1',label=label+' slip')
 ax.set_ylabel('V (m/s)',color='C0');twin.set_ylabel('slip (m)',color='C1');ax.legend(fontsize=7)
 for row in range(3):axes[row,col].set_xlim(*limits);axes[row,col].set_xlabel('native section x (m)')
 axes[1,col].set_ylabel('shear stress (Pa)');axes[1,col].legend(fontsize=7)
 axes[0,col].set_title(('Left' if col==0 else 'Right')+f' seed end; native y={yline:.6f} m')
fig.savefig(OUT/'growth-profiles.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for ax,values,title in zip(axes[0],[final.phi-parent.phi,H(final)-H(parent)],['Phase increment after .08 s','Elimination fraction H increment after .08 s']):
 xx,yy=np.round(final.x,13),np.round(final.y,13);ux,uy=np.unique(xx),np.unique(yy)
 v=np.empty((len(uy),len(ux)));v[np.searchsorted(uy,yy),np.searchsorted(ux,xx)]=values
 im=ax.pcolormesh(ux,uy,v,shading='nearest',cmap='magma');fig.colorbar(im,ax=ax)
 ax.set(title=title,xlabel='x (m)',ylabel='y (m)',aspect='equal',xlim=(0,4),ylim=(0,2))
for ax,key,title,unit in zip(axes[1],('w','qcoh'),('Fault-deformation rate','Cohesive conjugate stress'),('1/s','Pa')):
 for q,label,style in [(parent,f'parent {parent_time:.4g} s','--'),(final,f'endpoint {rows[-1].time:.4g} s','-')]:
  mask=np.isclose(q.y,yline,rtol=0,atol=1e-12)
  ax.plot(q.x[mask],q[key][mask],style,label=label)
 ax.set(title=title,xlabel='native x (m)',ylabel=unit,xlim=(.6,3.4));ax.legend(fontsize=8)
 for lo,hi in ((.95,1.45),(2.55,3.05)):ax.axvspan(lo,hi,color='grey',alpha=.1)
fig.savefig(OUT/'growth-increments.png',dpi=150);plt.close(fig)
print(json.dumps(result,indent=2))
