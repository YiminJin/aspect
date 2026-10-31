#!/usr/bin/env python3
"""Read native accepted exports only: SI histories, connected front brackets, plots."""
from pathlib import Path
import argparse,csv,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cohesive-rsf-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def read(run):
 out=run/'output' if (run/'output').exists() else run
 index=list(csv.DictReader((out/'cohesive-index.csv').open()));hist=[];surfaces=[];stems=[]
 for row in index:
  stem=out/Path(row['stem']).name
  d={r['quantity']:float(r['value']) for r in csv.DictReader(Path(str(stem)+'-diagnostics.csv').open())}
  ss=list(csv.DictReader(Path(str(stem)+'-surface.csv').open()))
  f={k:np.array([float(r[k]) for r in ss]) for k in ss[0] if k!='id'}
  x=f['x_m'];a=f['damage'];w=f['weight_m'];initial=np.minimum(1,np.minimum(x-4,np.maximum(0,9-x))).clip(0,1)
  d.update(time_s=float(row['time_s']),step=int(row['step']),growth_m=float(w@(a-initial)),max_growth=float(np.max(a-initial)),new_intact_damage_m=float(w@np.where(initial==0,a,0)),maximum_speed_m_s=float(np.max(abs(f['velocity_m_s']))))
  mature=(x>=5)&(x<=8)
  for name,key in [('mean_mature_slip_m','jump_m'),('mean_mature_theta','theta_normalized'),('mean_mature_pressure_Pa','pn_work_Pa')]:d[name]=float(w[mature]@f[key][mature]/sum(w[mature]))
  for level in [.01,.05,.1,.5,.99,1.]:
   center=int(np.argmin(abs(x-6.5)));right=center
   while right+1<len(a) and a[right+1]>=level:right+=1
   d[f'front_{level:g}_lower_m']=float(x[right]);d[f'front_{level:g}_upper_m']=float(x[min(right+1,len(a)-1)])
  hist.append(d);surfaces.append(f);stems.append(stem)
 meta=out.parent/'execution.json'
 if meta.exists():
  parent=json.loads(meta.read_text()).get('source_checkpoint')
  if parent and (Path(parent)/'cohesive-index.csv').exists():
   ph,ps,pt=read(Path(parent));own={r['step'] for r in hist}
   prior=[i for i,r in enumerate(ph) if r['step'] not in own and r['step']<hist[0]['step']]
   hist=[ph[i] for i in prior]+hist;surfaces=[ps[i] for i in prior]+surfaces;stems=[pt[i] for i in prior]+stems
 return hist,surfaces,stems

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--run',action='append',required=True,help='label=directory');p.add_argument('--output',type=Path,required=True);p.add_argument('--gif',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 allruns={};summary={}
 fig,ax=plt.subplots(2,3,figsize=(12,7))
 for item in a.run:
  label,path=item.split('=',1);h,s,stems=read(Path(path));allruns[label]=(h,s,stems)
  keys=sorted(set().union(*(r.keys() for r in h)))
  with (a.output/(label+'-history.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(h)
  def v(k):return np.array([r.get(k,0) for r in h])
  t=v('time_s');rate=v('maximum_speed_m_s');ax[0,0].semilogy(t,rate,label=label);ax[0,1].plot(t,v('mean_mature_theta'),label=label);ax[0,2].plot(t,v('growth_m'),label=label)
  ax[1,0].plot(t,v('front_1_lower_m'),label=label+' fully decohered');ax[1,0].plot(t,v('front_0.05_lower_m'),'--',label=label+' a≥.05')
  ax[1,1].plot(t,np.cumsum(v('friction_J_m')),label=label+' friction');ax[1,1].plot(t,np.cumsum(v('fracture_increment_J_m')),'--',label=label+' fracture');ax[1,1].plot(t,np.cumsum(v('radiation_J_m')),':',label=label+' radiation')
  ax[1,2].plot(t,np.cumsum(v('physical_remainder_J_m')),label=label+' signed');ax[1,2].plot(t,np.cumsum(abs(v('physical_remainder_J_m'))),'--',label=label+' absolute')
  slip=float(sum(v('friction_J_m')+v('radiation_J_m')));fract=float(sum(v('fracture_increment_J_m')));rem=float(sum(abs(v('physical_remainder_J_m'))))
  summary[label]=dict(accepted_steps=int(h[-1]['step']),final_time_s=float(t[-1]),maximum_speed_m_s=float(max(rate)),peak_amplification=float(max(rate)/1e-7),growth_m=float(v('growth_m')[-1]),intact_damage_m=float(v('new_intact_damage_m')[-1]),fully_decohered_front_bracket_m=[float(v('front_1_lower_m')[-1]),float(v('front_1_upper_m')[-1])],mature_slip_Dc=float(v('mean_mature_slip_m')[-1]/6e-5),friction_J_m=float(sum(v('friction_J_m'))),radiation_J_m=float(sum(v('radiation_J_m'))),fracture_J_m=fract,maxwell_J_m=float(sum(v('maxwell_J_m'))),absolute_physical_remainder_J_m=rem,remainder_over_sliding=rem/slip if slip else None,remainder_over_fracture=rem/fract if fract else None,max_BE_ledger_J_m=float(max(abs(v('ledger_error_J_m')))),max_source_scaled=float(max(v('source_scaled'))),max_surface_weak_Pa=float(max(v('weak_error_Pa'))),max_mechanical_weak_Pa=float(max(v('mechanical_weak_Pa'))),minimum_contact_Pa=float(min(v('minimum_normal_Pa')[1:])),maximum_RSS_MiB=float(max(v('peak_rank_memory_bytes'))/2**20),outer_rejections=float(v('clock_outer_rejections')[-1]))
  chosen=np.unique(np.linspace(0,len(h)-1,min(8,len(h))).astype(int))
  with (a.output/(label+'-profiles.csv')).open('w') as f:
   columns=['time_s','x_m','y_m','weight_m','damage','jump_m','velocity_m_s','theta_normalized','pn_work_Pa','q_Pa','resistance_Pa','tau_xy_minus_Pa','tau_xy_plus_Pa']
   writer=csv.writer(f);writer.writerow(columns)
   for i in chosen:
    for q in range(len(s[i]['x_m'])):writer.writerow([h[i]['time_s']]+[s[i][k][q] for k in columns[1:]])
 for a0,title,units in zip(ax.flat,['Maximum physical slip rate','Mature-patch state','Integrated added damage','Right connected fronts','Separate dissipative terms','Trapezoidal physical-work remainder'],['V (m/s)','Normalized age','∫ Δa ds (m)','x (m)','Energy (J/m)','Energy (J/m)']):
  a0.set(title=title,xlabel='Time (s)',ylabel=units);a0.legend(fontsize=7);a0.grid(alpha=.2)
 fig.tight_layout();fig.savefig(a.output/'histories.png',dpi=160);plt.close(fig)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 label=next(iter(allruns));h,s,stems=allruns[label]
 fig,axes=plt.subplots(2,3,figsize=(12,6.5))
 chosen=np.unique(np.linspace(0,len(h)-1,min(6,len(h))).astype(int))
 for i in chosen:
  ss=s[i];x=ss['x_m'];tag=f'{h[i]["time_s"]:.9g} s'
  for aa,key in zip(axes.flat,['jump_m','velocity_m_s','theta_normalized','pn_work_Pa','q_Pa','tau_xy_plus_Pa']):
   line,=aa.plot(x,ss[key],label=tag)
   if key=='pn_work_Pa':
    aa.plot(x,ss['pn_minus_Pa'],':',color=line.get_color(),lw=.7)
    aa.plot(x,ss['pn_plus_Pa'],':',color=line.get_color(),lw=.7)
 for aa,title,unit in zip(axes.flat,['Signed slip','Physical slip rate','Frictional state','Current normal reaction','Cohesive resistance','Raw upper-sided physical shear'],['j (m)','V (m/s)','Normalized age','pn (Pa)','q (Pa)','τxy (Pa)']):
  aa.set(xlabel='x (m)',ylabel=unit,title=title);aa.axvline(8,color='k',ls=':',lw=.6);aa.axvline(9,color='k',ls='--',lw=.6);aa.legend(fontsize=6)
 axes[1,0].ticklabel_format(axis='y',style='plain',useOffset=False)
 axes[1,0].text(.02,.03,'Solid: work normal; dotted: raw sides',transform=axes[1,0].transAxes,fontsize=7)
 axes[0,1].set_yscale('symlog',linthresh=1e-9);fig.tight_layout();fig.savefig(a.output/'profiles.png',dpi=160);plt.close(fig)
 selected=np.unique(np.linspace(0,len(h)-1,min(18,len(h))).astype(int));frames=[]
 # Equal physical stress scale across the entire animation, not autoscaled.
 fields=[]
 for i in selected:
  files=sorted(stems[i].parent.glob(stems[i].name+'-bulk-r*.csv'))
  fields.append(np.concatenate([np.genfromtxt(f,delimiter=',',names=True,dtype=None,encoding='utf8') for f in files]))
 lo=min(float(f['Txy_Pa'].min()) for f in fields);hi=max(float(f['Txy_Pa'].max()) for f in fields)
 for j,i in enumerate(selected):
  fig,axes=plt.subplots(2,2,figsize=(10,7));f=fields[j];ss=s[i]
  points=axes[0,0].scatter(f['x_m'],f['y_m'],c=f['Txy_Pa'],s=8,marker='s',vmin=lo,vmax=hi,cmap='RdBu_r');fig.colorbar(points,ax=axes[0,0],label='Physical τxy (Pa)')
  axes[0,0].plot(ss['x_m'],ss['y_m'],'k-',lw=.5);mask=ss['damage']>=.99;axes[0,0].scatter(ss['x_m'][mask],ss['y_m'][mask],c='k',s=8)
  axes[0,0].set(xlim=(0,16),ylim=(-6,6),xlabel='x (m)',ylabel='y (m)',title='Finite-wall specimen; black a≥.99');axes[0,0].set_aspect('equal')
  axes[0,1].plot(ss['x_m'],ss['damage'],label='Damage a');axes[0,1].axvspan(5,8,color='gray',alpha=.15,label='Initial mature seed');axes[0,1].set(xlim=(4,11.5),ylim=(0,1.05),xlabel='x (m)',ylabel='Surface damage');axes[0,1].legend(fontsize=8)
  inset=axes[0,1].inset_axes([.52,.23,.44,.4]);xx=ss['x_m']
  initial=np.minimum(1,np.minimum(xx-4,np.maximum(0,9-xx))).clip(0,1)
  inset.plot(xx,ss['damage']-initial);inset.axvline(9,color='k',ls=':',lw=.7)
  inset.set(xlim=(8.8,9.2),ylim=(0,max(.001,max(r['max_growth'] for r in h)*1.1)),title='New damage: a−a₀')
  inset.tick_params(labelsize=7);inset.title.set_fontsize(8)
  axes[1,0].semilogy(ss['x_m'],np.maximum(abs(ss['velocity_m_s']),1e-12));axes[1,0].set(xlim=(4,11.5),ylim=(1e-12,max(1e-6,summary[label]['maximum_speed_m_s']*2)),xlabel='x (m)',ylabel='|V| (m/s)')
  axes[1,1].plot(ss['x_m'],ss['q_Pa'],label='Cohesive / tied reaction');axes[1,1].plot(ss['x_m'],ss['resistance_Pa']-ss['q_Pa'],label='Friction + radiation');axes[1,1].set(xlim=(4,11.5),xlabel='x (m)',ylabel='Resistance (Pa)');axes[1,1].legend(fontsize=8)
  fig.suptitle(f'{label}: t={h[i]["time_s"]:.9f} s; imposed Δγ={1e-7*h[i]["time_s"]:.6g} (constant loading rate)');fig.tight_layout()
  if j in [0,len(selected)-1]:fig.savefig(a.output/('initial.png' if j==0 else 'final.png'),dpi=160)
  if a.gif:
   fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE))
  plt.close(fig)
 if frames:frames[0].save(a.output/'evolution.gif',save_all=True,append_images=frames[1:],duration=[700]+[350]*(len(frames)-2)+[1200],loop=0)
if __name__=='__main__':main()
