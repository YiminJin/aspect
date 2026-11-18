#!/usr/bin/env python3
"""Single-pilot native space/time rate, connected contour and fresh-material evidence."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-propagation-pilot-mpl')
from pathlib import Path
import argparse,sys,json,csv,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase import coefficients,fronts,read,value,connected_front
O=R/'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot';B=R/'build-particle-free/stage4-propagation-pilot'
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--output',type=Path);a=p.parse_args();out=R/'build-particle-free/bulk-tip/runs'/('stage4-propagation-'+a.label)/'output'
if a.output:out=a.output
summary=read(out/'pilot-summary.csv');base=coefficients(out,0);front=fronts(base,16);intact=np.max(abs(base),axis=(2,3))==0
yy,xx=np.mgrid[:128,:256];xx=(xx+.5)/32;yy=(yy+.5)/32
remote=(abs(yy-2.)>.5)|(xx<1.5)|(xx>6.5)
energies=read(out/'bulk-accepted.csv');planes0=np.loadtxt(B/'input-candidate/sections.txt',skiprows=1);rows=[];rates=[];x=None
for r in summary:
 step=int(r['step']);b=coefficients(out,step);f=fronts(b,16);f4=fronts(b,8);sections=read(out/f'pilot-sections-{step:04d}.csv');x=sections['x'];rates.append(sections['rate_stage'])
 row=dict(step=step,time=float(r['time']),stage_time=float(r['stage_time']),right_x=f['right_x'],right_advance=f['right_x']-front['right_x'],left_advance=front['left_x']-f['left_x'],contour_sampling_error=max(abs(f[k]-f4[k]) for k in ['right_x','left_x']),components=f['sampled_components'],detached_area=f['detached_sample_area'],max_phase_increment=float(np.max(b-base)),intact_max_phase_increment=float(np.max((b-base)[intact])),intact_integrated_increment=float(np.sum((b-base)[intact])/9/32**2),peak_rate=float(r['peak_rate']),integrated_slip=float(r['integrated_slip']),work_error=0.,new_formed_sections=0,maximum_old_angle_change=0.,positive_deformation_integral=0.,negative_deformation_integral=0.,added_fracture=0.)
 for side in ['right','left']:
  for half in ['upper','lower']:
   name=side+'_'+half+'_x';row[name]=f[name]
  row[side+'_tip_y_min']=f[side+'_y_range'][0];row[side+'_tip_y_max']=f[side+'_y_range'][1]
 row['minimum_phase_coefficient']=float(b.min())
 row['maximum_phase_coefficient']=float(b.max())
 row['minimum_phase_increment']=float((b-base).min())
 row['remote_max_phase_increment']=float(np.max((b-base)[remote]))
 row['peak_rate_x']=float(x[np.argmax(abs(sections['rate_stage']))])
 row['rate_at_right_tip']=float(np.interp(f['right_x'],x,sections['rate_stage']))
 row['rate_at_left_tip']=float(np.interp(f['left_x'],x,sections['rate_stage']))
 for side in ['right','left']:
  hi=min(len(x)-1,max(1,int(np.searchsorted(x,f[side+'_x']))));lo=hi-1
  row[side+'_tip_left_station_x']=float(x[lo]);row[side+'_tip_right_station_x']=float(x[hi])
  row[side+'_tip_left_station_rate']=float(sections['rate_stage'][lo]);row[side+'_tip_right_station_rate']=float(sections['rate_stage'][hi])
 row['right_taper_peak_rate']=float(np.max(abs(sections['rate_stage'][(x>=5.5)&(x<=6.)])))
 row['left_taper_peak_rate']=float(np.max(abs(sections['rate_stage'][(x>=2.)&(x<=2.5)])))
 row['maximum_stage_slip_over_Dc']=float(np.max(abs(sections['rate_stage']))*float(r['dt'])/3.2211126126299625e-7)
 if step:
  planes=read(out/f'tip-planes-{step:04d}.csv');old=planes0[:,2]>0
  row['new_formed_sections']=int(np.sum((planes['formed']>0)&(~old)));row['maximum_old_angle_change']=float(np.max(abs(planes['angle'][old]-planes0[old,1])))
  local=read(out/f'tip-sections-{step:04d}.csv');row['positive_deformation_integral']=float(np.sum(local['ds']*local['positive_w']));row['negative_deformation_integral']=float(np.sum(local['ds']*local['negative_w']))
  for label,lo,hi in [('left_taper',1.75,2.5),('right_taper',5.5,6.25)]:
   mask=(local['s']>=lo)&(local['s']<=hi)
   positive=float(np.sum(local['ds'][mask]*local['positive_w'][mask]));negative=float(np.sum(local['ds'][mask]*local['negative_w'][mask]))
   row[label+'_positive_deformation']=positive;row[label+'_negative_deformation']=negative
   row[label+'_cancellation_fraction']=1-abs(positive+negative)/(positive-negative) if positive>negative else 0.
  e=energies[energies['step']==step][0];row['added_fracture']=float(e['fracture']-energies[0]['fracture']+read(out/'release-regions.csv')[read(out/'release-regions.csv')['step']==int(energies[0]['step'])]['added_fracture_row_partition'].sum())
  row.update(work_error=float(e['Wext']-e['energy_change']-e['Db']-e['Dc']-e['Df']),phase_KKT=float(e['phase_KKT']) if 'phase_KKT' in energies.dtype.names else None)
 rows.append(row)
with (O/(a.label+'-history.csv')).open('w') as f:
 w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
levels=[1e-6,1e-4,.001,.01,.05]
support={str(level):{'initial':connected_front(base,8,level),'final':connected_front(b,8,level)} for level in levels}
record=dict(initial_front=front,final=rows[-1],support_levels=support,signed_work_error=sum(r['work_error'] for r in rows),absolute_work_error=sum(abs(r['work_error']) for r in rows),note='Sampled component checks and both DG traces; section admission and angle data retained separately. No episodicity or law attribution follows from a single coarse pilot.')
(O/(a.label+'-summary.json')).write_text(json.dumps(record,indent=2)+'\n')
fig,axs=plt.subplots(2,1,figsize=(11,7),constrained_layout=True,sharex=True)
t=np.array([r['stage_time'] for r in rows]);rate=np.array(rates)
vmax=max(.2,float(np.max(abs(rate)))*1e3)
im=axs[0].pcolormesh(t*1e3,x,rate.T*1e3,shading='nearest',cmap='coolwarm',norm=SymLogNorm(linthresh=.2,vmin=-vmax,vmax=vmax));fig.colorbar(im,ax=axs[0],label='Signed V (mm/s); symmetric log scale');axs[0].set(ylabel='Material station x (m)',title='Weakening pilot: native slip rate and connected contour');axs[0].axhline(front['right_x'],color='k',ls='--');axs[0].axhline(front['left_x'],color='k',ls='--')
levels=[v for v in [1.,5.,10.] if np.min(rate)*1e3<v<np.max(rate)*1e3]
if levels:
 cs=axs[0].contour(t*1e3,x,rate.T*1e3,levels=levels,colors='white',linewidths=.6)
 axs[0].clabel(cs,fmt='%g',fontsize=7)
axs[0].plot([r['time']*1e3 for r in rows],[r['right_x'] for r in rows],color='k',lw=.8)
axs[0].plot([r['time']*1e3 for r in rows],[front['left_x']-r['left_advance'] for r in rows],color='k',lw=.8)
axs[1].plot([r['time']*1e3 for r in rows],[r['right_advance']/.03125 for r in rows],label='Right');axs[1].plot([r['time']*1e3 for r in rows],[r['left_advance']/.03125 for r in rows],label='Left');axs[1].set(xlabel='Time (ms); rates at stages, contours at endpoints',ylabel='Connected advance / hx');axs[1].legend()
for ext in ['png','pdf']:fig.savefig(O/(a.label+'-space-time.'+ext),dpi=180)
print(json.dumps(record,indent=2))
