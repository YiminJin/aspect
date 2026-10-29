#!/usr/bin/env python3
"""Audit plots only: no invalid candidate trajectory or scalar-transect claim."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pranger_feasibility import OUT,LENGTH,GAMMA_REF,make_double,geometry


def read(name):return list(csv.DictReader((OUT/name).open()))
def values(rows,key):return np.array([float(r[key]) for r in rows])


plt.rcParams.update({'font.size':10,'axes.titlesize':11})
data=read('mixed-spectrum.csv');pole=read('oblique-pole.csv');transfer=read('operator-multipliers.csv')
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
chosen=[r for r in transfer if float(r['k_times_length'])<=3]
for key,label,style in [('chosen_explicit_multiplier','Chosen explicit: 1 − (kℓr)²','-'),
                         ('identity_multiplier','Same RSF, no interaction','--'),
                         ('unchosen_implicit_multiplier','Implicit averaging: NOT selected',':')]:
    axes[0,0].plot(values(chosen,'k_times_length'),values(chosen,key),style,label=label)
axes[0,0].axhline(0,color='.5',lw=.7)
axes[0,0].set(xlabel='kℓr',ylabel='Fourier multiplier',title='Operators agree only at low order')
axes[0,0].legend(fontsize=8)
for law,label in [('explicit_state','Explicit state interaction'),('same_RSF_local','Same rate/state law, local'),('direct_effect_only','State off; direct effect retained')]:
    rows=[r for r in data if r['state']=='double_steady' and float(r['wave_normal_deg'])==45. and r['law']==law]
    axes[0,1].plot(values(rows,'k_times_length'),values(rows,'maximum_frozen_growth_s1'),'o-',label=label)
axes[0,1].axhline(0,color='k',lw=.7)
axes[0,1].set_xscale('symlog',linthresh=1)
axes[0,1].set_xlim(left=0)
axes[0,1].set(xlabel='kℓr',ylabel='Determined spectral growth [s⁻¹]',title='Finite-h steady control can look stabilized')
axes[0,1].legend(fontsize=8)
for law,label,style in [('explicit_state','Explicit, kℓr = 10','-'),('same_RSF_local','Local, same rate/state law','--'),('direct_effect_only','State off, a > 0',':')]:
    rows=[r for r in pole if float(r['k_times_length'])==10. and r['law']==law]
    axes[1,0].loglog(values(rows,'offset_below_critical_deg'),values(rows,'maximum_frozen_growth_s1'),style+'o',label=label)
axes[1,0].set(xlabel='Degrees below 30.04149248° wave normal',ylabel='Frozen-pencil growth [s⁻¹]',title='Reversed-memory oblique pole survives')
axes[1,0].legend(fontsize=8)
null=read('mature-null-mode.csv')
for growth in [1.,1000.,1e6]:
    rows=[r for r in null if float(r['arbitrary_growth_s1'])==growth]
    axes[1,1].semilogy(values(rows,'k_times_length'),np.maximum(values(rows,'mixed_relative_residual'),1e-20),'o-',label=f'Assigned λ = {growth:g} s⁻¹')
axes[1,1].axhline(1e-12,color='.5',ls=':',label='Verification allowance')
axes[1,1].set_xscale('symlog',linthresh=1)
axes[1,1].set_xlim(left=0)
axes[1,1].set(xlabel='kℓr',ylabel='Relative mixed-equation residual',title='Mature limit: λ is arbitrary, not predicted')
axes[1,1].text(.98,.80,'All recorded residuals = 0;\nshown at 10⁻²⁰.',transform=axes[1,1].transAxes,ha='right')
axes[1,1].legend(fontsize=8)
fig.suptitle('One common equivalent rate/state, two changing Coulomb systems\nMixed pressure and 3-D tensor metric retained; no added viscosity or phase propagation')
fig.savefig(OUT/'mechanical-gate.png',dpi=170);plt.close(fig)

base=make_double(.99,0,mature=True)
m=np.array([geometry(base['psi'][i],s)[0] for i,s in enumerate([1,-1])])
amplitude=1e-7
rates=np.array([amplitude/(m[0,0,0]-m[1,0,0]),-amplitude/(m[0,0,0]-m[1,0,0])])
x=np.linspace(-.02,.02,25);y=np.linspace(-.02,.02,25);X,Y=np.meshgrid(x,y)
wave=2*np.pi/.02;normal=(X+Y)/np.sqrt(2)
strain=amplitude*np.cos(wave*normal)
# u = transverse a *[-2 amplitude sin(k r)/k], so symgrad u=diag(strain,-strain).
velocity=-2*amplitude*np.sin(wave*normal)/wave
with (OUT/'mature-null-field.csv').open('w') as f:
    writer=csv.writer(f)
    writer.writerow(['x_m','y_m','delta_Wxx_s1','delta_Wyy_s1','delta_velocity_x_m_s','delta_velocity_y_m_s'])
    writer.writerows(zip(X.ravel(),Y.ravel(),strain.ravel(),-strain.ravel(),
                         -velocity.ravel()/np.sqrt(2),velocity.ravel()/np.sqrt(2)))
fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
image=axes[0].pcolormesh(1000*X,1000*Y,strain,shading='nearest',cmap='RdBu_r',vmin=-amplitude,vmax=amplitude)
axes[0].quiver(1000*X[::2,::2],1000*Y[::2,::2],-velocity[::2,::2]/np.sqrt(2),velocity[::2,::2]/np.sqrt(2),color='k')
axes[0].set(aspect='equal',xlabel='x [mm]',ylabel='y [mm]',title='Analytic 45° mode: div-free velocity')
fig.colorbar(image,ax=axes[0],label='δWxx = −δWyy [s⁻¹]')
axes[1].bar(['System +','System −'],rates,color=['tab:blue','tab:orange'])
axes[1].axhline(0,color='k',lw=.8)
axes[1].set(ylabel='Conjugate rate variation [s⁻¹]',title='Redistribution changes flow direction')
axes[1].text(.5,.86,'δq = δμ = δΘ = δτ = δp = δφ = 0\nBoth base rates stay positive\nState operator sees no perturbation',transform=axes[1].transAxes,ha='center',va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
fig.suptitle('Exact mature specialization Q = 0: an uncontrolled linear mode\nAnalytic diagnostic, not a simulated growth event; neither Coulomb plane is locked')
fig.savefig(OUT/'mature-null-mode.png',dpi=170);plt.close(fig)
