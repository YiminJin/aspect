#!/usr/bin/env python3
"""Compact figures from actual mixed-symbol and native quadrature exports."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coulomb_localization_analysis import WORK,OUT,read,write

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':170,'axes.grid':True,'grid.alpha':.2})
points=read(OUT/'point-symbols.csv')
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
phase=np.array([.2,.6,.75,.8,.9,.95,.99]);curves=[]
for active,label,color in [(1,'Single system','#b34328'),(3,'Two systems (sampled states)','#326fa2')]:
    value=[]
    for phi in phase:
        r=[r for r in points if float(r['phi'])==phi and float(r['mu'])==.1 and int(r['active'])==active]
        v=min(float(r['rate_mixed_min_over_G']) for r in r);value.append(v)
        curves.append({'phi':phi,'systems':label,'minimum_rate_symbol_over_G':v})
    ax[0,0].plot(phase,value,'o-',label=label,color=color)
ax[0,0].axhline(0,color='k',lw=.8);ax[0,0].axvline(.7676112792918935,color='#b34328',ls=':',lw=1.2)
ax[0,0].set(xlim=(.58,1),ylim=(-.012,.043),xlabel=r'Phase $\phi$',ylabel=r'Minimum mixed coefficient $\Lambda/G$',
             title='Continuous-rate loss at developed damage')
ax[0,0].legend(fontsize=9)
ax[0,0].text(.595,-.0105,r'Single-system threshold: $\phi=0.76761$')
write('phase-symbol-summary.csv',curves)

angle=read(OUT/'angular-symbol.csv');theta=np.degrees([float(r['wave_normal_rad']) for r in angle])
for key,label,ls in [('fixed_pressure_over_G','Fixed pressure','--'),('actual_mixed_over_G','Actual mixed','-'),('rate_mixed_over_G','Continuous rate',':')]:
    ax[0,1].plot(theta,[float(r[key]) for r in angle],ls,label=label)
roots=read(OUT/'mixed-null-modes.csv')
for r in roots:
    if r['operator']=='actual finite step':ax[0,1].plot(np.degrees(float(r['wave_normal_rad'])),0,'kx',ms=6)
ax[0,1].axhline(0,color='k',lw=.8)
ax[0,1].set(xlim=(163,170),ylim=(-.004,.014),xlabel='Wave-normal angle in physical axes (degrees)',
             ylabel=r'Transverse coefficient / $G$',title=r'$\phi=.8$, actual step: pressure changes the conclusion')
ax[0,1].legend(fontsize=9)

forward=[r for r in points if r['kind']=='forward' and float(r['phi'])==.8 and float(r['normal_sign'])==1 and int(r['step'])==9]
forward.sort(key=lambda r:float(r['dt']));dt=[float(r['dt']) for r in forward]
ax[1,0].semilogx(dt,[float(r['algorithmic_mixed_min_over_G']) for r in forward],'o-',label='Actual algorithmic mixed tangent')
ax[1,0].axhline(float(forward[-1]['rate_mixed_min_over_G']),color='#b34328',ls='--',label='Continuous-rate limit')
ax[1,0].set(xlabel=r'Constitutive interval $\Delta t$ (s)',ylabel=r'Minimum $\Lambda/G$',
             title='Same incoming state; decreasing timestep does not cure loss')
ax[1,0].legend(fontsize=9)
write('point-timestep.csv',[{k:r[k] for k in ['dt','phi','active','algorithmic_mixed_min_over_G','algorithmic_fixed_P_min_over_G','rate_mixed_min_over_G']} for r in forward])

coupled=read(OUT/'coupled-symbol.csv');k=[float(r['k_times_ell']) for r in coupled]
ax[1,1].semilogx(k,[float(r['mixed_transverse_over_G']) for r in coupled],label='Formal free-phase elimination')
ax[1,1].axhline(float(coupled[0]['fixed_phi_over_G']),ls='--',color='#b34328',label='Admissible fixed-phase mechanical mode')
ax[1,1].axhline(0,lw=.8,color='k')
ax[1,1].set(xlabel=r'Wavenumber $k\ell$',ylabel=r'Mixed coefficient / $G$',
             title='Phase gradient loses influence on short mechanical waves')
ax[1,1].legend(fontsize=9)
ax[1,1].text(.03,-.016,'Free-phase curve is diagnostic only.\nPositive obstacle force permits '+r'$\delta\phi=0$.',fontsize=9)
fig.suptitle('Zero-dilation Coulomb law: full incompressible mixed-symbol audit',fontsize=14)
fig.savefig(OUT/'mechanical-symbol.png');plt.close(fig)

fig,ax=plt.subplots(2,3,figsize=(13,8.3),layout='constrained')
colors=['#4a6c9b','#cc8b29','#b34328'];native=read(OUT/'native-histories.csv');conditioning=read(OUT/'conditioning.csv');profiles=read(OUT/'normal-profiles.csv')
table=[];maps=[]
for col,(n,color) in enumerate(zip([4,8,16],colors)):
    name=f'p08-n{n}-half';data=read(WORK/'runs'/name/'coulomb-0020.csv')
    x=np.array([float(r['x_m']) for r in data]);y=np.array([float(r['y_m']) for r in data])
    value=2*np.hypot([float(r['Wxx_s-1']) for r in data],[float(r['Wxy_s-1']) for r in data])
    scatter=ax[0,col].scatter(x,y,c=value,s=10 if n==4 else (4 if n==8 else 1.7),cmap='magma',vmin=0,vmax=.09,rasterized=True)
    for z in np.linspace(0,.25,n+1):
        ax[0,col].axhline(z,color='k',lw=.35,alpha=.3);ax[0,col].axvline(z,color='k',lw=.35,alpha=.3)
    ax[0,col].grid(False);ax[0,col].set(aspect='equal',xlim=(0,.25),ylim=(0,.25),xlabel='x (m)',ylabel='y (m)',title=f'{n} × {n}: actual Gauss6 slip rate')
    curve=[r for r in profiles if r['case']==name and abs(float(r['time_s'])-.125)<1e-12 and r['slip_rate_s-1']]
    ax[1,0].plot([float(r['normal_coordinate_m']) for r in curve],[float(r['slip_rate_s-1']) for r in curve],color=color,label=f'{n} × {n}')
    hist=next(r for r in native if r['case']==name and abs(float(r['time_s'])-.125)<1e-12)
    cond=next(r for r in conditioning if r['case']==name)
    ax[1,2].plot(.25/n,float(hist['slip_participation_width_m']),'o',color=color)
    table.append({**hist,**{k:cond[k] for k in ['Riesz_sigma_min','Riesz_condition_2']}})
    maps.extend({'mesh_n':n,'time_s':.125,'x_m':x[i],'y_m':y[i],'JxW_m2':r['JxW_m2'],'equivalent_slip_rate_s-1':value[i]} for i,r in enumerate(data))
fig.colorbar(scatter,ax=ax[0,:],label=r'$\sqrt{2 W:W}$ (s$^{-1}$)',shrink=.75,pad=.015)
for suffix,marker,ls,label in [('half','o','-',r'$\phi=.8$, half step'),('full','x',':',r'$\phi=.8$, full step')]:
    data=[r for r in conditioning if r['case'].startswith('p08-') and r['case'].endswith(suffix) and abs(float(r['time_s'])-.125)<1e-12]
    data.sort(key=lambda r:float(r['h_m']))
    ax[1,1].plot([float(r['h_m']) for r in data],[float(r['Riesz_sigma_min']) for r in data],marker+ls,label=label)
data=[r for r in conditioning if r['case'].startswith('p02-')]
data.sort(key=lambda r:float(r['h_m']))
ax[1,1].plot([float(r['h_m']) for r in data],[float(r['Riesz_sigma_min']) for r in data],'s--',label=r'$\phi=.2$ controls*')
ax[1,0].set(xlim=(-.23,.05),xlabel='Coordinate along selected wave normal (m)',ylabel=r'Averaged slip rate (s$^{-1}$)',
            title=r'Common physical Gaussian: $\sigma=\ell/2$')
ax[1,0].legend(fontsize=9)
ax[1,1].set(xlabel=r'Cell size $\Delta x=\Delta y$ (m)',ylabel='Minimum Riesz-scaled mixed singular value',title='Conditioning worsens under refinement')
ax[1,1].legend(fontsize=8)
ax[1,2].set(xlabel=r'Cell size $\Delta x=\Delta y$ (m)',ylabel='Participation width (m)',ylim=(0,.21),title='Broad slip: no collapsing band observed here')
ax[1,2].axhline(.046875,color='k',ls=':',label=r'Fixed phase length $\ell$')
ax[1,2].legend(fontsize=9)
fig.suptitle(r'Native fixed-phase block at $t=.125$ s; $\phi=.8$, $\Delta t=.00625$ s, four ranks',fontsize=14)
fig.supxlabel('*Low-phase controls use the full step to .1 s, then the fine case cuts back; these are not a matched temporal-convergence pair.',fontsize=9)
fig.savefig(OUT/'native-refinement.png');plt.close(fig)
write('native-final-comparison.csv',table);write('native-slip-maps.csv',maps)
print('Two compact PNGs and their plotted CSVs written')
