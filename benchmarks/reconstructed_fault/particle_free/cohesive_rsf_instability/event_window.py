#!/usr/bin/env python3
"""Accepted native event diagnostics. Never evolves or remaps physical histories.
Spatial interpolation below is for visualization only; CSV preserves Gauss data.
"""
import argparse,csv,gzip,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/cohesive-rsf-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from analyze import read

COLUMNS=['x_m','y_m','weight_m','damage','jump_m','velocity_m_s','theta_normalized','q_Pa','pn_work_Pa','resistance_Pa','absolute_slip_m','formed','admitted_s','piece']
def initial(x):return np.minimum(1,np.minimum(x-4,np.maximum(0,9-x))).clip(0,1)
def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=float,default=374.5);p.add_argument('--threshold-reference',action='store_true',help='Use this run’s first accepted Vmax>=1e-3 m/s as its explicit reference; report the absolute bracket.');p.add_argument('--prefix',type=Path);p.add_argument('--gif',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 h,s,stems=read(args.run)
 if args.prefix:
  groups={}
  with gzip.open(args.prefix,'rt') as f:
   for row in csv.DictReader(f):groups.setdefault(int(row['step']),[]).append(row)
  meta=list(csv.DictReader(args.prefix.with_name('event-history.csv').open()))
  own={r['step'] for r in h};prior=[r for r in meta if int(r['step']) not in own and int(r['step'])<h[0]['step']]
  hh=[];ss=[]
  for r in prior:
   q=groups[int(r['step'])];hh.append({k:float(v) for k,v in r.items() if v!=''});hh[-1]['step']=int(r['step'])
   ss.append({k:np.array([float(a[k]) for a in q]) for k in COLUMNS});ss[-1]['_ids']=np.array([a['material_id'] for a in q])
  h=hh+h;s=ss+s;stems=[None]*len(hh)+stems
 if args.threshold_reference:
  crossings=[i for i,a in enumerate(s) if max(abs(a['velocity_m_s']))>=1e-3]
  if not crossings:raise SystemExit('No acceleration threshold yet: reference time is not defined')
  args.reference=h[crossings[0]]['time_s']
 keep=[i for i,r in enumerate(h) if r['time_s']>=args.reference-.75]
 h=[h[i] for i in keep];s=[s[i] for i in keep];stems=[stems[i] for i in keep]
 t=np.array([r['time_s'] for r in h]);tau=t-args.reference;speed=np.array([max(abs(a['velocity_m_s'])) for a in s]);peak=int(np.argmax(speed))
 rows=[]
 for r,a in zip(h,s):
  x=a['x_m'];w=a['weight_m'];damage=a['damage'];mature=(x>=5)&(x<=8);added=damage-initial(x);center=int(np.argmin(abs(x-6.5)))
  row=dict(step=r['step'],time_s=r['time_s'],relative_time_s=r['time_s']-args.reference,max_speed_m_s=float(max(abs(a['velocity_m_s']))),
   mature_absolute_slip_Dc=float(w[mature]@a['absolute_slip_m'][mature]/sum(w[mature])/6e-5),
   added_damage_m=float(w@added),peak_added_damage=float(max(added)),new_persistent_length_m=max(0.,float(w@a['formed'])-6.))
  for level in [.01,.1,.5,1.]:
   left=right=center
   while left>0 and damage[left-1]>=level:left-=1
   while right+1<len(x) and damage[right+1]>=level:right+=1
   row[f'left_a{level:g}_m']=float(x[left]);row[f'right_a{level:g}_m']=float(x[right]);row[f'left_a{level:g}_lower_m']=float(x[max(0,left-1)]);row[f'right_a{level:g}_upper_m']=float(x[min(len(x)-1,right+1)])
  for key in ['friction_J_m','radiation_J_m','maxwell_J_m','fracture_increment_J_m','bulk_increment_J_m','bond_increment_J_m','trapezoid_work_J_m','physical_remainder_J_m','continuous_history_error','continuous_numerical_resolution','publications']:row[key]=r.get(key,0.)
  rows.append(row)
 with (args.output/'event-history.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 with gzip.open(args.output/'event-gauss.csv.gz','wt') as f:
  writer=csv.writer(f);writer.writerow(['step','time_s','material_id']+COLUMNS)
  for r,a,stem in zip(h,s,stems):
   ids=[q['id'] for q in csv.DictReader(Path(str(stem)+'-surface.csv').open())] if stem else list(a.get('_ids',['prefix']*len(a['x_m'])))
   for q in range(len(a['x_m'])):writer.writerow([r['step'],r['time_s'],ids[q]]+[a[k][q] for k in COLUMNS])
 after=np.flatnonzero(speed>=.1*speed[peak]);last_high=int(after[-1]);decay_duration=float(t[-1]-t[last_high]);required=max(.005,5*6e-5/speed[peak]) if speed[peak] else np.inf
 summary=dict(reference_time_s=args.reference,window_start_s=float(t[0]),last_time_s=float(t[-1]),largest_observed_speed_m_s=float(speed[peak]),largest_observed_at_s=float(t[peak]),final_speed_m_s=float(speed[-1]),decay_duration_s=decay_duration,required_decay_duration_s=float(required),episode_complete=bool(speed[peak]>=1e-3 and speed[-1]<.1*speed[peak] and decay_duration>=required),last=rows[-1],note='Physical reference clock is explicit. Largest observed speed is not a completed peak unless decay criterion passes; no rupture-wave or speed claim.')
 crossing=np.flatnonzero(speed>=1e-3)
 if len(crossing):
  k=int(crossing[0]);summary['first_1e_3_m_s_bracket_s']=[float(t[max(0,k-1)]),float(t[k])]
 summary['maximum_geometry_y_m']=float(max(max(abs(a['y_m'])) for a in s))
 (args.output/'event-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Physical x grid is a plotting grid only. No constitutive state is evaluated on it.
 xgrid=np.linspace(4,max(max(a['x_m']) for a in s),400);V=np.array([np.interp(xgrid,a['x_m'],abs(a['velocity_m_s']),left=np.nan,right=np.nan) for a in s]);A=np.array([np.interp(xgrid,a['x_m'],a['damage'],left=np.nan,right=np.nan) for a in s]);DA=np.array([np.interp(xgrid,a['x_m'],a['damage']-initial(a['x_m']),left=np.nan,right=np.nan) for a in s])
 fig,ax=plt.subplots(2,2,figsize=(11.5,8));ax[0,0].semilogy(tau,np.maximum(speed,1e-12));ax[0,0].set(xlabel='t − t_ref (s)',ylabel='Maximum |V| (m/s)',title='First acceleration episode')
 ax[0,0].axvline(0,color='gray',ls=':');rapid=tau>=0
 if sum(rapid)<2:rapid=np.ones(len(t),dtype=bool)
 im=ax[0,1].pcolormesh(xgrid,tau[rapid],np.maximum(V[rapid],1e-12),norm=LogNorm(1e-7,max(1e-6,float(np.nanmax(V)))),shading='auto',cmap='magma');fig.colorbar(im,ax=ax[0,1],label='|V| (m/s)')
 im=ax[1,0].pcolormesh(xgrid,tau[rapid],np.maximum(DA[rapid],0),vmin=0,vmax=max(1e-4,float(np.nanmax(DA))),shading='auto',cmap='viridis');fig.colorbar(im,ax=ax[1,0],label='a − a_initial')
 for aa in [ax[0,1],ax[1,0]]:
  aa.plot([r['right_a1_m'] for r in rows if r['relative_time_s']>=0],tau[tau>=0],'c-',lw=2,label='Fully decohered right front (a=1)')
  aa.plot([r['right_a0.01_m'] for r in rows if r['relative_time_s']>=0],tau[tau>=0],'w--',lw=1,label='a=.01 process-zone front')
  aa.fill_betweenx(tau[tau>=0],[r['right_a1_m'] for r in rows if r['relative_time_s']>=0],[r['right_a1_upper_m'] for r in rows if r['relative_time_s']>=0],color='cyan',alpha=.2);aa.axvline(9,color='gray',ls=':');aa.set(xlabel='Physical x along Γ (m)',ylabel='t − t_ref (s)');aa.legend(fontsize=6.5,loc='lower left')
 ax[0,1].set_title('Slip activity: distinct from fracture advance');ax[1,0].set_title('Creation of damage, not a drawn tied reserve')
 progress=[r['mature_absolute_slip_Dc'] for r in rows]
 for key,label in [('right_a1_m','a=1'),('right_a0.5_m','a=.5'),('right_a0.01_m','a=.01')]:ax[1,1].plot(progress,[r[key] for r in rows],label=label)
 ax[1,1].fill_between(progress,[r['right_a1_m'] for r in rows],[r['right_a1_upper_m'] for r in rows],color='tab:blue',alpha=.2)
 ax[1,1].set(xlabel='Mean mature accumulated absolute slip / Dc',ylabel='Right front x (m)',title='Fronts aligned to physical event progress');ax[1,1].legend(fontsize=8)
 fig.suptitle(f't_ref = {args.reference:.6f} s; quasi-dynamic, no resolved waves; episode '+('decay criterion met' if summary['episode_complete'] else 'not yet complete'));fig.tight_layout();fig.savefig(args.output/'event-window.png',dpi=160);plt.close(fig)
 if args.gif:
  selected=np.unique(np.r_[np.searchsorted(progress,np.linspace(progress[0],progress[-1],22)),np.searchsorted(t,np.linspace(t[0],t[-1],10)),peak,len(h)-1]);selected=selected[selected<len(h)];frames=[]
  for i in selected:
   a=s[i];x=a['x_m'];fig,aa=plt.subplots(2,2,figsize=(10,6.6))
   aa[0,0].semilogy(x,np.maximum(abs(a['velocity_m_s']),1e-12));aa[0,0].set(ylim=(1e-9,max(1e-6,speed.max()*1.2)),ylabel='|V| (m/s)',title='Slip rate along the fault')
   aa[0,1].plot(x,a['damage']);aa[0,1].plot(x,initial(x),'k--',lw=.7,label='Initial');aa[0,1].set(ylim=(0,1.05),ylabel='Surface damage a',title='Cohesive zone / fully decohered material');aa[0,1].legend(fontsize=7)
   inset=aa[0,1].inset_axes([.5,.25,.45,.4]);inset.plot(x,a['damage']-initial(x));inset.set(xlim=(8.7,min(10.,max(x))),ylim=(0,max(1e-4,float(np.nanmax(DA))*1.1)));inset.tick_params(labelsize=6);inset.set_title('Added damage',fontsize=7)
   aa[1,0].semilogy(x,a['theta_normalized']);aa[1,0].set(ylim=(max(1e-7,min(np.min(b['theta_normalized']) for b in s)*.8),max(np.max(b['theta_normalized']) for b in s)*1.2),ylabel='Normalized age θ',title='RSF state, not phase')
   aa[1,1].plot(x,a['q_Pa'],label='Cohesive / tied reaction');aa[1,1].plot(x,a['resistance_Pa']-a['q_Pa']-a['damage']*25836.021365527624*a['velocity_m_s'],label='Friction');aa[1,1].plot(x,a['damage']*25836.021365527624*a['velocity_m_s'],label='Radiation');aa[1,1].set(ylabel='Resistance (Pa)',title='Current-pressure coupling retained');aa[1,1].legend(fontsize=7)
   for q in aa.flat:q.set(xlim=(4,xgrid[-1]),xlabel='Physical x (m)');q.axvspan(5,8,color='gray',alpha=.1);q.axvline(9,color='gray',ls=':',lw=.6)
   fig.suptitle(f't − {args.reference:.6f} s = {tau[i]:.9f} s; mature progress {progress[i]:.3f} Dc'+('' if summary['episode_complete'] else ' (episode incomplete)'));fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE));plt.close(fig)
  frames[0].save(args.output/'event-window.gif',save_all=True,append_images=frames[1:],duration=350,loop=0)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
