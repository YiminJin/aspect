#!/usr/bin/env python3
"""Compact physical-coordinate figures from exported native/reference data."""
import os,sys,csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/ci-boundary-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from assess_cohesive_interface_boundary import OUT,WORK,original,load,baseline,field,at,read,table
C=['#1764ab','#d65d00','#228550','#ad3483']
def rows(name):
 with (OUT/name).open() as f:return list(csv.DictReader(f))
def val(rr,key):return np.array([float(r[key]) for r in rr])
def save(fig,name):fig.savefig(OUT/name,dpi=170,bbox_inches='tight');plt.close(fig)
def seed(ax):
 for x in [2.5,5.5]:ax.axvline(x,color='.45',ls=':',lw=.7)
 ax.set_xlabel('Physical x (m)');ax.grid(alpha=.2)

def specimen():
 from matplotlib.patches import Rectangle
 fig=plt.figure(figsize=(12,6));grid=fig.add_gridspec(2,2,height_ratios=[1,1.4]);whole=fig.add_subplot(grid[0,:]);loading=fig.add_subplot(grid[1,0]);zoom=fig.add_subplot(grid[1,1])
 whole.add_patch(Rectangle((-4,0),16,1,fc='#eaf0f7',ec=C[0]));whole.plot([-3.5,11.5],[.4,.4],color='.5',lw=2);whole.plot([2.5,5.5],[.4,.4],'k-',lw=5);whole.scatter([-3.5,11.5],[.4,.4],marker='x',color='.4');whole.scatter([2.5,5.5],[.4,.4],marker='o',color='k',s=22)
 whole.annotate('+U(t), zero normal displacement',xy=(6,1.1),xytext=(2,1.5),arrowprops=dict(arrowstyle='->',color=C[0]),ha='center');whole.text(4,-.23,'Bottom fixed; top/bottom normal restraint; mean confinement p₀=10 kPa',ha='center',fontsize=9)
 whole.text(-4,.5,'periodic',rotation=90,ha='right',va='center');whole.text(12,.5,'periodic',rotation=90,ha='left',va='center');whole.text(4,.68,'Initial mature seed a=1',ha='center',fontsize=9);whole.text(-1,.68,'Initially tied a=0',ha='center',fontsize=9);whole.text(8.5,.68,'Initially tied a=0',ha='center',fontsize=9)
 whole.set(xlim=(-4.5,12.5),ylim=(-.4,1.8),aspect='equal',xlabel='Physical x (m)',ylabel='y (m)')
 t=np.linspace(0,2.58,400);loading.plot(t,np.minimum(t,2.55),color=C[0]);loading.axvline(2.2975,color='.5',ls=':',label='Retained checkpoint');loading.axvline(2.55,color=C[1],ls=':',label='Loading → hold');loading.set(xlabel='Time (s)',ylabel='Relative displacement (mm)',title='Same past; prescribed future continuation');loading.legend(fontsize=8);loading.grid(alpha=.2)
 for x in np.arange(1.5,3.5001,.0625):zoom.axvline(x,color='.8',lw=.3)
 for y in np.arange(0,1.0001,.0125):zoom.axhline(y,color='.8',lw=.3)
 zoom.plot([1.5,3.5],[.4,.4],color='.5',lw=2);zoom.plot([2.5,3.5],[.4,.4],'k-',lw=4);zoom.scatter([2.5],[.4],color='k');zoom.annotate('Initial mature tip',(2.5,.4),(2.6,.75),arrowprops=dict(arrowstyle='->'),fontsize=9);zoom.text(1.6,.16,'b=.4 m',fontsize=9);zoom.text(1.6,.68,'t=.6 m',fontsize=9)
 zoom.set(xlim=(1.5,3.5),ylim=(0,1),aspect='equal',xlabel='Physical x (m)',ylabel='y (m)',title='Unchanged mesh: Δx=.0625 m, Δy=.0125 m')
 fig.suptitle('Finite-height periodic straight-fault specimen: Γ fixed, φ provisional\nCandidate ends (crosses) are distinct from the initial mature tips (dots)')
 fig.tight_layout();save(fig,'specimen.png')

def reference():
 specimen()
 fig,ax=plt.subplots(2,2,figsize=(12,8));mode=rows('boundary-low-modes.csv');p=rows('same-memory-period-profiles.csv');frozen=rows('frozen-boundaries.csv')
 for i,(top,bottom,label) in enumerate([(.6,.4,'H=1, offset 0.1 m'),(.5,.5,'H=1, centred'),(.4,.6,'H=1, reflected'),(1.1,.9,'H=2, same offset'),(2.1,1.9,'H=4, same offset')]):
  r=[r for r in mode if float(r['top_m'])==top and float(r['bottom_m'])==bottom];ax[0,0].semilogx(val(r,'k_inv_m'),val(r,'k_normal_imag_Pa_m2')/1e6,label=label)
 ax[0,0].set(xlabel='Wavenumber k (1/m)',ylabel='k Im N(k) (MPa/m²)',title='Offset + impermeable walls create the 1/k normal term');ax[0,0].legend(fontsize=8);ax[0,0].grid(alpha=.2)
 for i,t in enumerate([2.3,2.45,2.47]):
  r=[r for r in p if float(r['time_s'])==t];ax[0,1].plot(val(r,'x_m'),val(r,'native_normal_difference_Pa'),color=C[i],label=f'Native: {t:.2f} s');ax[0,1].plot(val(r,'x_m'),val(r,'asymptotic_normal_difference_Pa'),'--',color='k',lw=.8)
 ax[0,1].plot([],[],'k--',label='Long-wave prediction');ax[0,1].set(ylabel='16 m − 8 m normal response (Pa)',title='Identical physical memory: pressure-ramp change');seed(ax[0,1]);ax[0,1].legend(fontsize=8)
 for name,label in [('L8-H1-offset','8×1'),('L16-H1-offset','16×1'),('L32-H1-offset','32×1'),('L16-H1-centred','16×1 centred'),('L16-H2-offset','16×2'),('L32-H4-offset','32×4')]:
  r=[r for r in frozen if r['case']==name and 0<=float(r['x_m'])<=8]
  ax[1,0].plot(val(r,'x_m'),val(r,'normal_Pa')/1000,label=label)
  if name!='L16-H1-centred':ax[1,1].plot(val(r,'x_m'),val(r,'tau_Pa')/1000,label=label)
 ax[1,0].set(ylabel='Current normal response (kPa)',title='Same Q1 memory and background Maxwell stress');ax[1,0].legend(ncol=2,fontsize=8);seed(ax[1,0])
 ax[1,1].set(ylabel='Work-adjoint shear response (kPa)',title='Moving walls is a physical stiffness change');ax[1,1].legend(fontsize=8);seed(ax[1,1])
 fig.suptitle('Fixed-memory analytical/reduced mixed strip; no constitutive update\nFor changed height, wall displacement is scaled to preserve the background stress')
 fig.tight_layout();save(fig,'boundary-mechanism.png')

def reflection():
 a=original(8);b=load('reflected',L=8,nx=128);fig,ax=plt.subplots(3,1,figsize=(10,7.4),sharex=True)
 for i,c in enumerate([a,b]):
  k=at(c,2.47);x=c['s']['x_m'][k]
  for panel,key,scale in [(ax[0],'damage',1),(ax[1],'jump_m',1000),(ax[2],'pn_Pa',.001)]:panel.plot(x,c['s'][key][k]*scale,color=C[i],label=['Original y=.4 m','Reflected y=.6 m'][i]);seed(panel)
 ax[0].set_ylabel('Surface damage a');ax[1].set_ylabel('Signed jump (mm)');ax[2].set_ylabel('Current normal pressure (kPa)');ax[0].legend()
 fig.suptitle('Native diagnostic at 2.47 s: same positive shear loading\nReflection swaps the pressure ramp and the developed flank; it does not replace the longer specimen')
 fig.tight_layout();save(fig,'native-reflection.png')

def final():
 a=baseline();b=load('event-half',a);hist=[rows('longer-history.csv'),rows('event-half-history.csv')]
 fig,ax=plt.subplots(2,3,figsize=(13,7));keys=['integrated_new_damage_m','full_advance_m','peak_jump_m','reaction_N_m','absolute_remainder_J_m','bulk_J_m'];labels=['New damage integral (m)','Left fully decohered advance (m)','Peak signed jump (m)','Wall reaction (N/m)','Sum |work remainder| (J/m)','Bulk elastic storage (J/m)']
 for panel,key,label in zip(ax.flat,keys,labels):
  for i,h in enumerate(hist):panel.plot(val(h,'time_s'),val(h,key),color=C[i],ls=['-','--'][i],label=['Selected Δt','Half Δt'][i])
  panel.axvline(2.55,color='.4',ls=':');panel.set(xlim=(2.30,2.58),xlabel='Time (s)',ylabel=label);panel.grid(alpha=.2)
 ax[0,0].legend();fig.suptitle('Finite-height L=16 m periodic specimen; vertical line: loading → hold\nSame physical seed, original y=.4 m, current pressure and single fracture inventory')
 fig.tight_layout();save(fig,'longer-histories.png')
 fig,ax=plt.subplots(2,2,figsize=(11.5,7.5))
 for k,h in enumerate(hist):
  t=val(h,'time_s');mask=t>=2.45-1e-10;U=val(h,'imposed_displacement_m')*1000
  ax[0,0].plot(U[mask],val(h,'reaction_N_m')[mask]/1000,color=C[k],ls=['-','--'][k],label=['Selected Δt','Half Δt'][k])
  start=np.argmin(abs(t-2.45));w=val(h,'absolute_remainder_J_m')-float(h[start]['absolute_remainder_J_m']);signed=val(h,'signed_remainder_J_m')-float(h[start]['signed_remainder_J_m'])
  ax[1,0].plot(t[mask],w[mask],color=C[k],label=['Selected sum |r|','Half sum |r|'][k]);ax[1,0].plot(t[mask],signed[mask],'--',color=C[k])
  for key,ls in [('fracture_J_m','-'),('sliding_J_m',':')]:
   denominator=val(h,key)-float(h[start][key]);take=mask&(denominator>1e-12);ax[1,1].plot(t[take],100*w[take]/denominator[take],color=C[k],ls=ls,label=['Selected','Half'][k]+(' / fracture' if key=='fracture_J_m' else ' / sliding'))
 h=hist[1];t=val(h,'time_s');start=np.argmin(abs(t-2.45));mask=t>=2.45-1e-10
 for key,label in [('work_J_m','External trapezoidal work'),('bulk_J_m','Δ bulk storage'),('bond_J_m','Δ bond storage'),('fracture_J_m','New fracture'),('friction_cumulative_J_m','Friction loss'),('radiation_cumulative_J_m','Radiation loss'),('Maxwell_J_m','Maxwell loss')]:
  z=val(h,key)-float(h[start][key]);ax[0,1].plot(t[mask],z[mask],label=label)
 ax[0,0].set(xlabel='Imposed relative displacement (mm)',ylabel='Wall reaction (kN/m)',title='Hold: reaction changes at fixed displacement');ax[0,0].legend(fontsize=8)
 ax[0,1].set(xlabel='Time (s)',ylabel='New-window energy (J/m)',title='Half-step physical storage/work/losses');ax[0,1].legend(fontsize=8,ncol=2)
 ax[1,0].set(xlabel='Time (s)',ylabel='Physical remainder (J/m)',title='Solid: sum |r|; dashed: signed sum');ax[1,0].legend(fontsize=8)
 ax[1,1].axhline(1,color='k',ls='--',lw=.8);ax[1,1].set(xlabel='Time (s)',ylabel='Sum |r| / new energy (%)',title='Raw remainder; no background correction');ax[1,1].legend(fontsize=8)
 for panel in [ax[0,1],*ax[1]]:panel.axvline(2.55,color='.5',ls=':');panel.grid(alpha=.2)
 fig.suptitle('Physical energy accuracy is separate from exact backward-Euler ledger closure\nThe initial 3 J/m seed inventory is excluded from new fracture; hold external work is zero')
 fig.tight_layout();save(fig,'longer-energy-work.png')
 times=[2.45,2.50,2.55,2.58];fig,ax=plt.subplots(3,4,figsize=(14,8),sharex=True)
 for k,t in enumerate(times):
  for i,c in enumerate([a,b]):
   j=at(c,t);x=c['s']['x_m'][j]
   for r,key,scale in [(0,'damage',1),(1,'jump_m',1000),(2,'pn_Pa',.001)]:ax[r,k].plot(x,c['s'][key][j]*scale,color=C[i],ls=['-','--'][i]);seed(ax[r,k])
  ax[0,k].set_title(f't = {t:.2f} s');ax[0,k].set_ylim(-.03,1.04)
  for level in [.01,.05,.5,1.]:ax[0,k].axhline(level,color='.7',lw=.5,ls=':')
 for r,label in enumerate(['Surface damage a','Signed jump (mm)','Current normal pressure (kPa)']):ax[r,0].set_ylabel(label)
 fig.suptitle('Actual accepted surface histories: blue selected Δt, orange half Δt\nDotted lines: initial mature seed. Damage/front geometry is distinct from slip.')
 fig.tight_layout();save(fig,'longer-profiles.png')
 fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True,sharey=True)
 take=b['h']['time_s']>=2.30-1e-10;tt=b['h']['time_s'][take];centres=(np.arange(256)+.5)*.0625-4;wg=np.array([5/18,4/9,5/18])
 damage=b['s']['damage'][take].reshape(-1,256,3)@wg;velocity=b['s']['V_m_s'][take].reshape(-1,256,3)@wg
 im=ax[0].pcolormesh(centres,tt,damage,shading='nearest',vmin=0,vmax=1,cmap='magma');fig.colorbar(im,ax=ax[0],label='Weighted cell-mean a')
 im=ax[1].pcolormesh(centres,tt,velocity*1000,shading='nearest',cmap='viridis');fig.colorbar(im,ax=ax[1],label='Weighted slip rate (mm/s)')
 hh=hist[1];front=val(hh,'continuous_left_full_x_m');times_h=val(hh,'time_s');mask=times_h>=2.30-1e-10
 ax[0].plot(front[mask],times_h[mask],'c-',lw=1.2,label='Continuous fully decohered left front');ax[0].legend(fontsize=8,loc='upper right')
 for panel in ax:
  panel.axhline(2.55,color='w',ls=':',lw=1);panel.set(ylabel='Time (s)',ylim=(2.30,2.58),yticks=[2.30,2.35,2.40,2.45,2.50,2.55,2.58]);seed(panel)
 fig.suptitle('Half-step accepted trajectory: developed advance is separate from process-zone expansion\nInitial seed endpoints dotted vertically; loading ends at the white horizontal line')
 fig.tight_layout();save(fig,'longer-space-time.png')
 # Native full Gauss6 pressure at selected common physical times; plot cell means.
 fig,ax=plt.subplots(2,2,figsize=(12,4.7),sharex=True,sharey=True);maps=[];cellrows=[]
 for t in times:
  selected=b if t in [2.55,2.58] else a
  _,z=read(field(selected,t));z[:,0]-=4;q=z.reshape(-1,36,10);w=q[:,:,2];area=w.sum(axis=1);means=(q*w[:,:,None]).sum(axis=1)/area[:,None];order=np.lexsort((means[:,0],means[:,1]));maps.append(means[order,9].reshape(80,256)/1000)
  for value,volume in zip(means,area):cellrows.append(dict(case=selected['name'],time_s=t,x_m=value[0],y_m=value[1],area_m2=volume,tau_xy_Pa=value[7],pprime_Pa=value[8],normal_pressure_Pa=value[9]))
 lo=min(m.min() for m in maps);hi=max(m.max() for m in maps)
 for panel,t,z in zip(ax.flat,times,maps):
  im=panel.imshow(z,extent=[-4,12,0,1],origin='lower',aspect='equal',cmap='coolwarm',vmin=lo,vmax=hi);panel.plot([2.5,5.5],[.4,.4],'k-',lw=1.8);panel.set(title=f't = {t:.2f} s',xlabel='Physical x (m)',ylabel='y (m)')
 fig.colorbar(im,ax=ax,label='Horizontal-plane pn (kPa)',shrink=.7,pad=.02);fig.suptitle('Native bulk physical normal pressure: p₀ + p′ − τyy\nCell means; black segment is the initial mature seed. Last two panels use the half-step replay.')
 save(fig,'longer-pressure.png')
 table(OUT/'longer-physical-cells.csv',cellrows)
 # Each GIF frame is one accepted state, without interpolating internal variables.
 indices=np.unique(np.r_[np.searchsorted(b['h']['time_s'],np.linspace(2.35,2.58,40)),at(b,2.55),at(b,2.58)]);indices=np.clip(indices,0,len(b['h']['time_s'])-1)
 fig,ax=plt.subplots(4,1,figsize=(10,8.6),sharex=True);lines=[];animation=[]
 for panel in ax:seed(panel);panel.set_xlim(-4,12)
 keys=[('damage',1),('jump_m',1000),('V_m_s',1000),('pn_Pa',.001)]
 for panel,(key,scale) in zip(ax,keys):
  lines.append(panel.plot([],[],color=C[0])[0]);values=b['s'][key]*scale;panel.set_ylim(min(0,float(values.min()))-.03,max(.05,float(values.max()))*1.05)
 ax[0].set_ylabel('Surface damage a');ax[1].set_ylabel('Signed jump (mm)');ax[2].set_ylabel('Slip rate (mm/s)');ax[3].set_ylabel('Normal pressure (kPa)');ax[3].set_ylim(.95*b['s']['pn_Pa'].min()/1000,1.05*b['s']['pn_Pa'].max()/1000);ax[3].axhline(10,color='.6',lw=.7,ls=':');title=fig.suptitle('');fig.tight_layout(rect=[0,0,1,.92])
 def frame(k):
  j=indices[k];t=b['h']['time_s'][j];x=b['s']['x_m'][j]
  for line,(key,scale) in zip(lines,keys):line.set_data(x,b['s'][key][j]*scale)
  title.set_text(f'Finite L=16 m, H=1 m specimen: t={t:.5f} s; imposed displacement={min(t,2.55):.5f} mm\n'+('Displacement hold; Maxwell/RSF/cohesive histories remain active' if t>2.55 else 'Continued loading; initial seed endpoints dotted'))
 FuncAnimation(fig,frame,frames=len(indices),interval=170).save(OUT/'longer-evolution.gif',writer=PillowWriter(fps=6),dpi=100);plt.close(fig)
 for j in indices:
  for q,x in enumerate(b['s']['x_m'][j]):animation.append(dict(time_s=b['h']['time_s'][j],x_m=x,weight_m=b['weights'][q],damage=b['s']['damage'][j,q],jump_m=b['s']['jump_m'][j,q],velocity_m_s=b['s']['V_m_s'][j,q],normal_pressure_Pa=b['s']['pn_Pa'][j,q]))
 table(OUT/'animation-profiles.csv',animation)

if __name__=='__main__':
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 {'reference':reference,'reflection':reflection,'final':final}[sys.argv[1]]()
