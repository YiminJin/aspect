#!/usr/bin/env python3
"""Plots of actual native histories and quadrature fields, in physical axes."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation,PillowWriter
from run_cohesive_interface_domain import OUT
from analyze_cohesive_interface_domain import case,at

COLORS=['#1764ab','#d65d00']
LABELS=['8 m period','16 m period']
def rows(file):
 with (OUT/file).open() as f:return list(csv.DictReader(f))
def values(rr,key):return np.array([float(r[key]) for r in rr])
def save(fig,name):
 fig.savefig(OUT/name,dpi=170,bbox_inches='tight');plt.close(fig)
def guides(ax):
 for x in [2.5,5.5]:ax.axvline(x,color='.45',lw=.7,ls=':')
 ax.set_xlabel('Physical x (m)');ax.grid(alpha=.2)

def main():
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 a,b=case('period-8',8,128,0),case('period-16',16,256,4)
 hs=[rows('period-8-history.csv'),rows('period-16-history.csv')]
 fig,ax=plt.subplots(2,1,figsize=(13,4.1),sharex=True)
 for i,(lo,hi) in enumerate([(0,8),(-4,12)]):
  ax[i].add_patch(Rectangle((lo,0),hi-lo,1,fc='#eaf0f7',ec=COLORS[i]))
  ax[i].plot([lo+.5,hi-.5],[.4,.4],color='.6',lw=2,label='Initially tied candidate')
  ax[i].plot([2.5,5.5],[.4,.4],color='#8b2020',lw=4,label='Mature seed: a = 1')
  ax[i].annotate('relative wall velocity 1 mm/s; hold after 2.45 s',(4,1.02),ha='center',fontsize=9)
  ax[i].set(title=LABELS[i],ylabel='y (m)',xlim=(-4.2,12.2),ylim=(-.15,1.4),aspect='equal')
  for x in [0,8]:ax[i].axvline(x,color='.5',ls=':',lw=.7)
 ax[0].legend(loc='lower left',ncol=2,fontsize=8);ax[1].set_xlabel('Physical x (m)')
 fig.suptitle('Horizontal-period control: same seed, wall distance, local mesh and loading\nΔx = 0.0625 m, Δy = 0.0125 m; no transfer of accepted histories')
 fig.tight_layout();save(fig,'specimen.png')

 fig,ax=plt.subplots(2,3,figsize=(13,7))
 keys=['integrated_new_damage_m','full_advance_m','integrated_abs_jump_m2','reaction_excess_N_m','absolute_remainder_J_m','bulk_excess_J_m']
 titles=['New damage integral ∫(a−a₀) dx (m)','Left fully decohered advance (m)','Integrated |jump| (m²)','Fault-induced reaction deficit (N/m)','Sum |physical work remainder| (J/m)','Bulk storage minus intact-strip storage (J/m)']
 for panel,key,title in zip(ax.flat,keys,titles):
  for i,h in enumerate(hs):panel.plot(values(h,'time_s'),values(h,key),color=COLORS[i],label=LABELS[i],lw=1.7,ls=['-','--'][i])
  panel.axvline(2.45,color='.5',ls=':',lw=.8);panel.set(xlabel='Time (s)',title=title);panel.grid(alpha=.2)
 ax[0,0].legend();ax[0,0].set_xlim(1.4,2.47);ax[0,1].set_xlim(1.4,2.47)
 ins=ax[0,0].inset_axes([.17,.55,.44,.37])
 for i,h in enumerate(hs):ins.plot(values(h,'time_s'),values(h,'integrated_new_damage_m'),color=COLORS[i],ls=['-','--'][i])
 ins.set(xlim=(1.75,2.02),ylim=(0,.04),title='Early growth');ins.tick_params(labelsize=7);ins.title.set_fontsize(8)
 fig.suptitle('Actual accepted evolution; vertical line marks loading → displacement hold\nExcess quantities compare domains; energy acceptance uses the raw, unsubtracted remainder')
 fig.tight_layout();save(fig,'histories.png')

 fig,ax=plt.subplots(3,3,figsize=(13,8.7),sharex=True)
 for k,t in enumerate([2.3,2.45,2.47]):
  for i,c in enumerate([a,b]):
   j=at(c,t);x=c['s']['x_m'][j];mask=(x>=0)&(x<=8)
   ax[0,k].plot(x[mask],c['s']['damage'][j,mask],color=COLORS[i],ls=['-','--'][i],label=LABELS[i])
   ax[1,k].plot(x[mask],1000*c['s']['jump_m'][j,mask],color=COLORS[i],ls=['-','--'][i])
   ax[2,k].plot(x[mask],c['s']['pn_Pa'][j,mask]/1000,color=COLORS[i],ls=['-','--'][i])
  ax[0,k].set(title=f't = {t:.2f} s',ylim=(-.03,1.04));ax[1,k].set_ylim(-.02,1.18)
  for panel in ax[:,k]:guides(panel)
 ax[0,0].set_ylabel('Surface damage a');ax[1,0].set_ylabel('Signed jump (mm)');ax[2,0].set_ylabel('Current normal pressure (kPa)');ax[0,0].legend()
 fig.suptitle('Common physical Γ profiles: dotted lines are original mature-seed endpoints\na is Gauss-owned surface damage; bulk φ remains provisional geometry')
 fig.tight_layout();save(fig,'profiles.png')

 rr=rows('physical-cells.csv')
 fig,ax=plt.subplots(3,3,figsize=(13,6.3),sharex=True,sharey=True)
 images=[]
 for row,t in enumerate([2.3,2.45,2.47]):
  maps=[]
  for name in ['period-8','period-16']:
   r=[v for v in rr if v['case']==name and float(v['time_s'])==t and 0<float(v['x_m'])<8]
   x=values(r,'x_m');y=values(r,'y_m');order=np.lexsort((x,y));maps.append(values(r,'tau_xy_Pa')[order].reshape(80,128)/1000)
  for k,z in enumerate([maps[0],maps[1],1000*(maps[1]-maps[0])]):
   im=ax[row,k].imshow(z,extent=[0,8,0,1],origin='lower',aspect='equal',cmap='coolwarm',vmin=-1 if k==2 else 0,vmax=1 if k==2 else 3.2)
   images.append(im);ax[row,k].plot([2.5,5.5],[.4,.4],'k-',lw=1);ax[row,k].set_ylabel(f't={t:.2f} s\ny (m)')
   if row==0:ax[row,k].set_title(['8 m period: τxy (kPa)','16 m period: τxy (kPa)','Difference (Pa); scale revised from measured data'][k])
 # Shared difference range is measured, with no clipping hidden in the plot.
 maximum=max(float(np.max(abs(im.get_array()))) for im in images[2::3]);maximum=max(maximum,1e-10)
 for im in images[2::3]:im.set_clim(-maximum,maximum)
 ax[0,2].set_title('16 m − 8 m: τxy (Pa)')
 for p in ax[-1]:p.set_xlabel('Physical x (m)')
 fig.colorbar(images[0],ax=ax[:,:2],label='Physical shear stress (kPa)',shrink=.6,pad=.02)
 fig.colorbar(images[2],ax=ax[:,2],label='Difference (Pa)',shrink=.6,pad=.02)
 fig.suptitle('Actual native bulk stress, Gauss6-weighted cell means on the common 8 m box\nBlack segment: initial mature seed; compare full-tensor weighted norms in CSV')
 save(fig,'physical-stress.png')

 fig,ax=plt.subplots(2,3,figsize=(12,5.9),sharex=True,sharey=True)
 for row,(key,label,limits) in enumerate([('tau_xy_Pa','Physical τxy',(0,3.2)),('normal_pressure_Pa','Horizontal-plane pn',(7.5,11.5))]):
  maps=[]
  for name in ['period-8','period-16']:
   r=[v for v in rr if v['case']==name and float(v['time_s'])==2.47 and .5<float(v['x_m'])<3]
   x=values(r,'x_m');y=values(r,'y_m');order=np.lexsort((x,y));maps.append(values(r,key)[order].reshape(80,40))
  difference=maps[1]-maps[0];bound=max(abs(difference).max(),1e-9)
  for k,z in enumerate([maps[0]/1000,maps[1]/1000,difference]):
   im=ax[row,k].imshow(z,extent=[.5,3,0,1],origin='lower',aspect='equal',cmap='coolwarm',vmin=-bound if k==2 else limits[0],vmax=bound if k==2 else limits[1])
   ax[row,k].plot([2.5,3],[.4,.4],'k-',lw=1.8);ax[row,k].axvline(2.5,color='.4',lw=.7,ls=':');ax[row,k].set_ylabel('y (m)')
   if row==0:ax[row,k].set_title(['8 m period','16 m period','16 m − 8 m'][k])
   fig.colorbar(im,ax=ax[row,k],label=label+(' (Pa)' if k==2 else ' (kPa)'),shrink=.7,pad=.02)
 for p in ax[-1]:p.set_xlabel('Physical x (m)')
 fig.suptitle('Near left tip at t = 2.47 s: actual native cell means\nBlack: initial mature seed; dotted: its original endpoint. pₙ = p₀ + p′ − τyy.')
 fig.tight_layout();save(fig,'tip-fields.png')

 double=np.loadtxt(b['p']/'derivative_checks.csv',delimiter=',',skiprows=1,ndmin=2);extended=np.loadtxt(b['p']/'derivative-extended.csv',delimiter=',',skiprows=1,ndmin=2)
 double=double[abs(double[:,0]-1.9425)<1e-9];extended=extended[abs(extended[:,0]-1.9425)<1e-9]
 fig,ax=plt.subplots(figsize=(7,4))
 ax.loglog(double[:,1],double[:,2],'o-',label='Original double residual FD');ax.loglog(extended[:,1],extended[:,2],'s-',label='Extended-precision reference FD');ax.axhline(1e-6,color='k',ls=':',label='Unchanged gate')
 ax.set(xlabel='Perturbation ε (m)',ylabel='Full active-column relative Frobenius difference',title='Preserved diagnostic stop at 1.9425 s; same perturbations and tangent');ax.grid(alpha=.2);ax.legend();fig.tight_layout();save(fig,'derivative-reference.png')

 times=np.unique(np.r_[np.arange(1.6,2.451,.025),2.47]);indices=[(int(np.argmin(abs(a['h']['time_s']-t))),int(np.argmin(abs(b['h']['time_s']-t)))) for t in times]
 fig,ax=plt.subplots(2,1,figsize=(10,5.8),sharex=True);lines=[]
 for i,c in enumerate([a,b]):
  lines.append([ax[0].plot([],[],color=COLORS[i],ls=['-','--'][i],label=LABELS[i])[0],ax[1].plot([],[],color=COLORS[i],ls=['-','--'][i])[0]])
 for p in ax:guides(p);p.set_xlim(-4,12)
 ax[0].set(ylabel='Surface damage a',ylim=(-.03,1.05));ax[1].set(ylabel='Signed jump (mm)',ylim=(-.03,1.2));ax[0].legend()
 title=fig.suptitle('');fig.tight_layout(rect=[0,0,1,.91])
 def frame(k):
  ia,ib=indices[k]
  assert abs(a['h']['time_s'][ia]-b['h']['time_s'][ib])<1e-9
  for i,(c,j) in enumerate([(a,ia),(b,ib)]):
   mask=c['s']['installed'][j]>0;x=c['s']['x_m'][j,mask]
   lines[i][0].set_data(x,c['s']['damage'][j,mask]);lines[i][1].set_data(x,1000*c['s']['jump_m'][j,mask])
  t=a['h']['time_s'][ia];title.set_text(f'Accepted surface histories: t = {t:.4f} s, imposed displacement = {min(t,2.45):.4f} mm\nAdded 16 m candidate records remain tied; dotted lines mark seed endpoints')
 FuncAnimation(fig,frame,frames=len(indices),interval=170).save(OUT/'evolution.gif',writer=PillowWriter(fps=6),dpi=100);plt.close(fig)
if __name__=='__main__':main()
