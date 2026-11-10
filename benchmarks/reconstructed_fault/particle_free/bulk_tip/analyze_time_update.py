#!/usr/bin/env python3
"""Compact endpoint/work and predeclared-region diagnostics for tip time-update runs."""
import argparse,json,os,re
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-tip-time-mpl')
ROOT=Path(__file__).resolve().parents[4]
RUNS=ROOT/'build-particle-free/bulk-tip/runs'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update'
def read(p):
 return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding=None)).view(np.recarray)
def frame(d,step):
 a=np.concatenate([read(p) for p in sorted(d.glob(f'tip-points-{step:04d}-r*.csv'))]).view(np.recarray)
 return a[np.lexsort((a.q,a.cell))]
def balance(d,pattern):
 a=np.concatenate([read(p) for p in sorted(d.glob(pattern))]).view(np.recarray)
 return a[np.lexsort((a.i,a.cell))]
def regions(x,y):
 return dict(tip=((np.abs(x-1.2)<=.375)|(np.abs(x-2.8)<=.375))&(np.abs(y-1)<=.375),
             interior=(x>1.575)&(x<2.425)&(np.abs(y-1)<=.375),remote=np.abs(x-2)>1.3,
             boundary=np.minimum.reduce([x,4-x,y,2-y])<=.25)
def summary(label,parent_fracture=None):
 d=RUNS/label/'output';rows=read(d/'bulk-accepted.csv');cmd=json.loads((d.parent/'command.json').read_text())
 r=dict(label=label,seconds=cmd['seconds'],exit_code=cmd['exit_code'],steps=len(rows),parent_time=float(rows.time[0]-rows.dt[0]),end_time=float(rows.time[-1]),energy={})
 for name in ['Wext','energy_change','Db','Dc','Df','stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work','ledger_error']:
  v=rows[name];r['energy'][name]=dict(signed=float(sum(v)),absolute=float(sum(abs(v))))
 # Split the work with the actual section measure and integrated rate. The
 # radiation traction uses the root-resolved stage rate, as does the solver.
 radiation=float(re.search(r'set Radiation damping coefficients\s*=\s*([^\n]+)',(d.parent/'input.prm').read_text())[1])
 losses={'friction':[],'radiation':[]};defects=[]
 for row in rows:
  s=read(d/f'tip-sections-{int(row.step):04d}.csv')
  rad=radiation*s.resolved_V
  losses['radiation'].append(float(row.dt*sum(s.ds*rad*s.V)))
  losses['friction'].append(float(row.dt*sum(s.ds*(s.force-rad)*s.V)))
  defects.append(losses['radiation'][-1]+losses['friction'][-1]-row.Df)
 for name,values in losses.items():
  r['energy'][name]=dict(signed=float(sum(values)),absolute=float(sum(abs(np.asarray(values)))))
 r['section_work_split_max_defect']=max(abs(np.asarray(defects)))
 v=rows.Wext-rows.energy_change-rows.Db-rows.Dc-rows.Df
 r['physical_remainder']=dict(signed=float(sum(v)),absolute=float(sum(abs(v))))
 if parent_fracture is None and (d/'tip-initial.csv').exists():parent_fracture=float(read(d/'tip-initial.csv').fracture[0])
 if parent_fracture is not None:
  r['added_fracture']=float(rows.fracture[-1]-parent_fracture)
  r['remainder_over_added_fracture']=float(abs(sum(v))/r['added_fracture']) if r['added_fracture']>1e-12 else None
 r['gates']={k:float(max(abs(rows[k]))) for k in ['mechanical_residual','phase_KKT','ledger_error']}
 for file in ['tip-solver.csv','tip-profile.csv']:
  if (d/file).exists():
   data=read(d/file);r[file]={k:float(sum(data[k])) for k in data.dtype.names if k!='step'}
 return r,rows,frame(d,int(rows.step[-1]))
def compare(a,b):
 assert np.array_equal(a.cell,b.cell) and np.array_equal(a.q,b.q)
 result={}
 for name in ['phi','tauxx','tauyy','tauzz','tauxy','Axy','signed_shear']:
  diff=a[name]-b[name];norm=np.sqrt(sum(b.weight*b[name]**2));err=np.sqrt(sum(b.weight*diff**2))
  result[name]=dict(max_abs=float(max(abs(diff))),L2=float(err),relative_L2=float(err/norm) if norm>1e-25 else None)
 if 'Y_endpoint' in a.dtype.names and 'Y_endpoint' in b.dtype.names and np.any(np.isfinite(a.Y_endpoint)&np.isfinite(b.Y_endpoint)):
  mask=np.isfinite(a.Y_endpoint)&np.isfinite(b.Y_endpoint);diff=a.Y_endpoint[mask]-b.Y_endpoint[mask]
  result['Y_endpoint_positive_phase']=dict(L2=float(np.sqrt(sum(b.weight[mask]*diff**2))),max_abs=float(max(abs(diff))))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['time','screen','growth']);p.add_argument('labels',nargs='+');p.add_argument('--parent-fracture',type=float);p.add_argument('--name',default='analysis');a=p.parse_args()
 OUT.mkdir(exist_ok=True,parents=True);result={'mode':a.mode,'cases':{},'comparisons':{}}
 frames={}
 for label in a.labels:
  r,rows,last=summary(label,a.parent_fracture);result['cases'][label]=r;frames[label]=last;d=RUNS/label/'output'
  if a.mode!='time':
   r['regions_overlap']=True;r['samples']=[]
   initial=frame(d,0);seed_energy=float(read(d/'tip-initial.csv').fracture[0]);seed_file=list(d.glob('tip-seed-fracture-r*.csv'))
   sb=balance(d,'tip-seed-fracture-r*.csv') if seed_file else None
   r['initial_extents']={str(t):[float(min(initial.x[initial.phi>=t])),float(max(initial.x[initial.phi>=t]))] if np.any(initial.phi>=t) else None for t in [.001,.01,.1,.5]}
   for row in rows:
    f=frame(d,int(row.step));sec=read(d/f'tip-sections-{int(row.step):04d}.csv');rr={}
    for region,mask in regions(f.x,f.y).items():
     sub=f[mask];q=int(np.argmax(sub.Y))
     rr[region]=dict(max_stage_Y_over_Kf=float(max(sub.Y)/40),Y_position=[float(sub.x[q]),float(sub.y[q])],max_phase=float(max(sub.phi)),max_delta_phi=float(max(sub.phi-initial.phi[mask])),max_abs_tau=float(max(abs(sub.tauxy))),max_abs_normal=float(max(np.maximum(abs(sub.tauxx),abs(sub.tauyy)))))
     if 'stage_normal_x' in sub.dtype.names:
      rr[region]['max_abs_dynamic_total_normal']=float(max(np.maximum(abs(sub.stage_normal_x),abs(sub.stage_normal_y))))
      rr[region]['stage_pressure_range']=[float(min(sub.stage_pressure)),float(max(sub.stage_pressure))]
    ext={str(t):[float(min(f.x[f.phi>=t])),float(max(f.x[f.phi>=t]))] if np.any(f.phi>=t) else None for t in [.001,.01,.1,.5]}
    sample=dict(step=int(row.step),time=float(row.time),regions=rr,extents=ext,max_section_rate=float(max(abs(sec.V))),max_resistance=float(max(abs(sec.force))),max_age=float(max(sec.age)),max_slip=float(max(abs(sec.slip))),rate_cancellation=float(sum(sec.ds*(sec.integral_abs_w-abs(sec.V)))))
    local=row.dt*f.weight*f.w*sec.force[f.section.astype(int)]
    sample['local_work_positive']=float(sum(np.maximum(local,0)));sample['local_work_negative']=float(sum(np.minimum(local,0)))
    bands=[]
    for k in np.unique(f.section):
     q=f[f.section==k];q=q[np.argsort(q.y)];active=abs(q.w)>max(1e-10,1e-6*max(abs(q.w)));bands.append(int(sum(active&~np.r_[False,active[:-1]])))
    sample['multiple_band_sections']=sum(v>1 for v in bands);sample['max_bands']=max(bands)
    if sb is not None:
     b=balance(d,f'tip-balance-{int(row.step):04d}-r*.csv');assert np.array_equal(sb.cell,b.cell)
     work=.5*(b.fracture_gradient+sb.fracture_gradient)*(b.B-sb.B)
     sample['fracture_weak_work']={k:float(sum(work[m])) for k,m in regions(b.x,b.y).items()}
     tip=regions(b.x,b.y)['tip'];sample['fracture_weak_work']['outside_tip']=float(sum(work[~tip]));sample['fracture_weak_work']['global']=float(sum(work));sample['fracture_partition_error']=float(sum(work)-(row.fracture-seed_energy))
    reaction_files=list(d.glob(f'tip-reactions-{int(row.step):04d}-r*.csv'))
    if reaction_files:
     reaction=np.concatenate([read(p) for p in reaction_files]).view(np.recarray)
     sample['reaction_work_error']=float(sum(reaction.physical_work)-row.Wext)
     sample['net_reaction']={str(c):float(sum(reaction.force[reaction.component==c])) for c in [0,1]}
     sample['top_nodal_reaction_including_corners']={str(c):float(sum(reaction.force[(reaction.component==c)&(abs(reaction.y-2)<1e-12)])) for c in [0,1]}
    r['samples'].append(sample)
 if a.mode=='time':
  for first,second in zip(a.labels[:-1],a.labels[1:]):
   result['comparisons'][first+' vs '+second]=compare(frames[first],frames[second])
   da=RUNS/first/'output';db=RUNS/second/'output';ra=read(da/'bulk-accepted.csv');rb=read(db/'bulk-accepted.csv');sa=read(da/f'tip-sections-{int(ra.step[-1]):04d}.csv');s=read(db/f'tip-sections-{int(rb.step[-1]):04d}.csv')
   for key in ['slip','age']:
    diff=sa[key]-s[key];result['comparisons'][first+' vs '+second][key]=dict(max_abs=float(max(abs(diff))),L2=float(np.sqrt(sum(s.ds*diff**2))))
 result['stage_comparison_note']='Last-stage V, force and Y differ in observation time across time grids. Endpoint slip and fault deformation integrals are compared; last-stage rate/force are not claimed as matched endpoint error.'
 (OUT/(a.name+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:{i:v for i,v in r.items() if i in ['seconds','steps','physical_remainder','added_fracture','remainder_over_added_fracture','gates']} for k,r in result['cases'].items()},indent=2))
 if a.mode=='growth':
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  label=a.labels[-1];d=RUNS/label/'output';rows=read(d/'bulk-accepted.csv');steps=[0,int(rows.step[len(rows)//2]),int(rows.step[-1])];fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained');seed=frame(d,0)
  for j,step in enumerate(steps):
   f=frame(d,step);x=np.unique(f.x);y=np.unique(f.y);order=np.lexsort((f.x,f.y));z=f.phi[order].reshape(len(y),len(x));X,Y=np.meshgrid(x,y)
   axes[0,j].pcolormesh(X,Y,z,shading='nearest',vmin=0,vmax=.8,cmap='viridis');cs=axes[0,j].contour(X,Y,z,levels=[.001,.01,.1,.5],colors='white',linewidths=.6);axes[0,j].clabel(cs,fontsize=6);axes[0,j].set_title(f'{label}, step {step}')
   vals=(f.phi-seed.phi) if j<2 else f.Y/40
   c=axes[1,j].scatter(f.x,f.y,c=vals,s=3,cmap='magma');fig.colorbar(c,ax=axes[1,j]);axes[1,j].set_title('Incremental physical phase' if j<2 else 'Stage Y / Kf')
   for ax in axes[:,j]:ax.set(xlim=(0,4),ylim=(0,2),xlabel='x (m)',ylabel='y (m)',aspect='equal')
  fig.suptitle('Gauss-6 samples on h=.25 mesh; sampled contours, not resolved front tracking')
  fig.savefig(OUT/(a.name+'.png'),dpi=180);fig.savefig(OUT/(a.name+'.pdf'));plt.close(fig)
