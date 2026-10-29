#!/usr/bin/env python3
"""Two compact physical plots from accepted native Gauss-point data."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/coulomb-native-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coulomb_native_checks import WORK,OUT,read,column,write,save

hist=read(OUT/'histories.csv');work=read(OUT/'work.csv')
fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
for name,color,label in [('homogeneous','black','Affine block'),('nonuniform-r1','tab:blue','Nonuniform block')]:
    rows=[r for r in hist if r['case']==name];t=column(rows,'time_s')
    axes[0,0].plot(t,column(rows,'p0_Pa')/1000,color=color,label=label)
    axes[0,1].fill_between(t,column(rows,'min_pprime_Pa'),column(rows,'max_pprime_Pa'),color=color,alpha=.15)
    axes[0,1].plot(t,column(rows,'mean_pprime_Pa'),color=color)
    axes[1,0].plot(t,column(rows,'min_all_plane_normal_Pa')/1000,color=color,label=label)
    axes[1,0].plot(t,column(rows,'min_pn_plus_Pa')/1000,color=color,ls='--')
    axes[1,0].plot(t,column(rows,'min_pn_minus_Pa')/1000,color=color,ls=':')
    axes[2,0].plot(t,column(rows,'mean_tauxy_Pa'),color=color,label=label+' σxy')
    axes[2,0].plot(t,column(rows,'mean_tauxx_Pa')-10000,color=color,ls='--',label=label+' σxx')
rows=[r for r in hist if r['case']=='nonuniform-r1'];t=column(rows,'time_s')
axes[1,1].stackplot(t,*[column(rows,f'activity_{i}_area_fraction') for i in range(4)],
                    labels=['Stick','+ system only','− system only','Both systems'],colors=['.8','tab:orange','tab:green','tab:purple'])
reference=read(OUT/'point-reference.csv')
axes[2,0].plot(column(reference,'time_s'),column(reference,'tauxy_Pa'),'o',mfc='none',color='tab:red',ms=3,label='Point σxy')
energy=[r for r in work if r['case']=='nonuniform-r1'];t=column(energy,'time_s')
storage=np.cumsum(column(energy,'energy_change'));diss=np.cumsum(sum(column(energy,k) for k in ['Db','Dc','Df']))
endpoint=np.cumsum(column(energy,'reaction_endpoint_work_J_m'));trap=np.cumsum(column(energy,'reaction_trapezoidal_work_J_m'))
axes[2,1].plot(t,endpoint,label='Endpoint reaction work',color='black')
axes[2,1].plot(t,trap,label='Trapezoidal reaction work',color='black',ls='--')
axes[2,1].plot(t,storage,label='Recoverable storage change')
axes[2,1].plot(t,diss,label='Physical dissipation (BE)')
axes[2,1].plot(t,endpoint-storage-diss,label='Endpoint numerical remainder')
titles=['Prescribed background confinement','Solved perturbation: range and mean',
        'Min normal pressures at actual Gauss points','Nonuniform block: active-system area fractions',
        'Mean physical Cauchy stress; fixed-phase point control','Nonuniform work and dissipation; fracture storage fixed']
ylabels=['p₀ [kPa]','p′ [Pa]','Normal pressure [kPa]','Area fraction','Physical stress [Pa]','Energy / work [J/m]']
for ax,title,ylabel in zip(axes.flat,titles,ylabels):
    ax.set_title(title,fontsize=10);ax.set_xlabel('Time [s]');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    ax.axvline(.1,color='.5',lw=.7);ax.axvline(.25,color='.5',lw=.7)
axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8,title='Solid: all planes; dashed/dotted: ± candidates',title_fontsize=7)
axes[1,1].legend(fontsize=7,loc='lower left');axes[2,0].legend(fontsize=7);axes[2,1].legend(fontsize=7)
fig.suptitle('Fresh confined 4×4 block, φ=0.2 fixed, μ=0.1, cᵣ=0, m=4\nShear → normal-deviatoric loading at 0.10 s → displacement hold at 0.25 s',fontsize=11)
fig.savefig(OUT/'response.png',dpi=160);plt.close(fig)

rows=read(OUT/'quadrature-step16.csv');x=column(rows,'x_m');y=column(rows,'y_m');xx=np.unique(x);yy=np.unique(y)
def grid(values):
    result=np.zeros((len(yy),len(xx)));result[np.searchsorted(yy,y),np.searchsorted(xx,x)]=values;return result
fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
values=[column(rows,'pprime_Pa'),column(rows,'tauxx_Pa')-column(rows,'total_pressure_Pa'),column(rows,'tauxy_Pa'),
        np.minimum(column(rows,'pn_plus_Pa'),column(rows,'pn_minus_Pa')),column(rows,'w_plus_s-1'),column(rows,'w_minus_s-1')]
labels=['Solved p′ [Pa]','Physical σxx [Pa]','Physical σxy [Pa]','Min candidate pn [Pa]','+ system slip speed [1/s]','− system slip speed [1/s]']
for k,(ax,value,label) in enumerate(zip(axes.flat,values,labels)):
    im=ax.pcolormesh(xx,yy,grid(value),shading='nearest',cmap='RdBu_r' if k<3 else 'viridis')
    fig.colorbar(im,ax=ax,shrink=.8);ax.set_title(label,fontsize=10)
    for coordinate in np.linspace(0,.25,5):ax.axvline(coordinate,color='.6',lw=.5);ax.axhline(coordinate,color='.6',lw=.5)
    ax.set_aspect('equal');ax.set_xlim(0,.25);ax.set_ylim(0,.25);ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
    if k>=4:
        theta=column(rows,'theta_plus_rad' if k==4 else 'theta_minus_rad');speed=value
        for i in range(0,len(rows),13):
            if speed[i]>1e-10 and np.isfinite(theta[i]):
                dx=.006*np.cos(theta[i]);dy=.006*np.sin(theta[i]);ax.plot([x[i]-dx,x[i]+dx],[y[i]-dy,y[i]+dy],color='white',lw=1)
fig.suptitle('Accepted nonuniform block at t=0.20 s; p₀=10000 Pa; actual Gauss6 samples\nWhite lines: active slip-plane tangents, not phase-front or crack angles. φ remains 0.2 everywhere.',fontsize=11)
fig.savefig(OUT/'fields.png',dpi=160);plt.close(fig)

parameters=[('G',1e6,'Pa','Shear modulus'),('eta',1e8,'Pa s','Maxwell viscosity'),('Gc',1.,'J/m²','Retained fracture parameter'),
            ('ell',.046875,'m','Retained phase length'),('m',4.,'1','Retained degradation parameter'),('strength_drop',2000.,'Pa','Native Cohesions input; distinct from c_r'),
            ('mu',.1,'1','Constant friction coefficient'),('c_r',0.,'Pa','Residual cohesion'),('p0',10000.,'Pa','Prescribed mean confinement'),
            ('phi',.2,'1','Exactly fixed phase'),('dt',.0125,'s','Constant accepted timestep'),('cells',16,'1','Fixed uniform mesh'),
            ('native_DoFs',617,'1','Includes 162 velocity, 25 pressure, 25 phase and static fields'),('DG_coefficients',432,'1','3 components × 9 modes × 16 cells'),
            ('quadrature_points',576,'1','36 Gauss6 points per cell'),('reference_residual_strength',995.0371902099893,'Pa','Pure shear at p′=0, Q=0'),
            ('reference_intact_peak',2955.3324556233765,'Pa','Intact threshold; not exercised by fixed phi=.2')]
write('parameters.csv',[dict(parameter=k,value=v,unit=u,meaning=m) for k,v,u,m in parameters])
save('plot-definitions.json',{'stress':'Cauchy sigma=tau-(p0+pprime)I. sigma_xy=tau_xy. No hypothetical confinement: p0 is explicit native input.',
     'sampling':'Actual accepted Gauss6 quadrature samples; time means use physical JxW. No material return on averaged histories.',
     'activity':'JxW-weighted fractions, plane speed >1e-10/s is active. The native law does not use this display threshold.',
     'directions':'Plane tangents, modulo pi. Undefined maximizers have explicit false flags and NaN angles. No phase-front inference.',
     'energy':'Per unit out-of-plane thickness; trapezoidal work uses saved full reactions, BE dissipation remains endpoint-integrated. Numerical remainders are not physical dissipation.',
     'pressure_reference':'Physical prescribed mean pprime=0. All velocity boundaries constrained; reactions contain p0 and the initial -p0I stress.',
     'phase':'Uniform phi=.2 fixed exactly. No phase KKT or phase-stability qualification.'})
print('Two PNGs and compact parameter/plot definitions exported')
