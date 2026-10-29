#!/usr/bin/env python3
"""Only measured CSV evidence; no specimen or phase-front illustrations."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reconcile_feng_rsf import OUT


def read(name):return list(csv.DictReader((OUT/name).open()))
def col(rows,key):return np.array([float(r[key]) for r in rows])
plt.rcParams.update({'font.size':10,'axes.titlesize':11})

fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
summary=read('loading-summary.csv')
for phi in [.2,.8,.99]:
    rows=[r for r in summary if float(r['phi'])==phi]
    axes[0,0].loglog(col(rows,'dt_s'),col(rows,'max_global_violation_Pa'),'o-',label=f'φ={phi:g}')
axes[0,0].set(xlabel='Δt [s]',ylabel='Peak global-yield violation [Pa]',title='Printed update: refinement does not remove the violation')
axes[0,0].legend();axes[0,0].invert_xaxis()
rows=[r for r in read('loading-histories.csv') if float(r['phi'])==.8 and int(r['steps'])==2048]
axes[0,1].plot(col(rows,'t_s'),col(rows,'normal_deg'),label='Printed stationary branch')
axes[0,1].axvline(next(float(r['first_failed_endpoint_s']) for r in summary if float(r['phi'])==.8 and int(r['steps'])==2048),color='r',ls='--',label='First rejected endpoint')
axes[0,1].set(xlim=(.8,1.),xlabel='Loading time [s]',ylabel='Plane normal from +x [degrees]',title='Smooth branch motion is not global admissibility')
axes[0,1].legend(fontsize=8)
rows=[r for r in read('continuous-viability.csv') if int(r['branch'])==0 and float(r['phi'])<1]
axes[1,0].bar([str(r['phi']) for r in rows],col(rows,'other_Fdot_Pa_s'))
axes[1,0].set(yscale='log',ylim=(1e-4,30),xlabel='Pre-existing phase obstacle φ',ylabel='Unselected-branch Ḟ [Pa/s]',title='First activation: either single flow violates the other branch')
for i,r in enumerate(rows):axes[1,0].text(i,float(r['other_Fdot_Pa_s'])*1.3,f"{float(r['other_Fdot_Pa_s']):.4g}",ha='center',fontsize=9)
axes[1,0].text(.03,.06,'At activation: A = 0, Y = 0, phase force = +8 Pa\nThe obstacle prevents spontaneous phase growth.',transform=axes[1,0].transAxes,fontsize=9,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
rows=[r for r in read('endpoint-formulas.csv') if float(r['Qref_xy_Pa'])==-.05]
r=rows[0]
axes[1,1].axis('off')
axes[1,1].text(0,.96,'Reversed-memory endpoint: two different issues',va='top',weight='bold')
axes[1,1].text(0,.82,'Printed arcsine shortcut:\n  assumed shear sign is inconsistent;\n  stationarity residual = −0.01436 Pa.\n\nAll branches of the same stationary equation:\n  two globally admissible endpoints remain.\n\nAn orientation tie at one endpoint cannot select\nbetween two different endpoint states.\nNo conjugate average or tolerance change is used.',va='top',linespacing=1.5)
fig.suptitle('Endpoint discrepancy is explained; a general continuation closure is not established\nDiagnostic paths after the first failure are not accepted physical trajectories')
fig.savefig(OUT/'endpoint-reconciliation.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
angles=read('kinetic-angles.csv')
for variant,style in [('RSF+radiation','-'),('state off','--')]:
    rows=[r for r in angles if r['case']=='reversed memory branch 0' and r['variant']==variant]
    axes[0,0].plot(col(rows,'wave_normal_deg'),col(rows,'max_growth_s1'),style,label=variant)
axes[0,0].axhline(0,color='k',lw=.7)
for angle in [30.04149248,59.95850752]:axes[0,0].axvline(angle,color='.7',ls=':',lw=.8)
axes[0,0].set(xlabel='Perturbation wave normal [degrees]',ylabel='Maximum Re(λ) [s⁻¹]',title='Solved pressure, state and changing direction retained')
axes[0,0].legend(fontsize=8)
wavelength=read('wavelengths.csv')
for variant in ['RSF+radiation','radiation only']:
    rows=[r for r in wavelength if r['variant']==variant and abs(float(r['wave_normal_deg'])-90)<.1]
    axes[0,1].loglog(col(rows,'wavelength_m'),col(rows,'mixed_QZ_max_growth_s1'),'o-',label=variant)
axes[0,1].set(xlabel='Perturbation wavelength [m]',ylabel='Maximum Re(λ) [s⁻¹]',title='Bounded growth; no wavelength cutoff is supplied')
axes[0,1].legend(fontsize=8)
timesteps=read('pressure-lag-timesteps.csv')
for ax,variant in zip(axes[1],['RSF+radiation','radiation only']):
    for label,style in [('coupled','o-'),('lagged','s--')]:
        rows=[r for r in timesteps if r['variant']==variant and r['pressure_update']==label]
        ax.semilogx(col(rows,'dt_s'),col(rows,'effective_growth_s1'),style,label=label+' pressure')
    ax.axhline(float(rows[0]['continuous_growth_s1']),color='k',ls=':',label='Continuous mixed limit')
    ax.set(xlabel='Δt [s]',ylabel='log(spectral radius) / Δt [s⁻¹]',title=variant+': previous-pressure memory included')
    ax.legend(fontsize=8)
fig.suptitle('Kinetic regularity on a selected smooth branch is distinct from endpoint selection\nReversed-memory frozen state; pressure lag is a timestep approximation, not a new continuum cure')
fig.savefig(OUT/'kinetics-pressure.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
history=read('point-histories.csv')
comparisons=[]
for coarse,fine in [(.0125,.00625),(.00625,.003125)]:
    current=[r for r in history if r['pressure_update']=='coupled' and float(r['dt_s'])==coarse]
    reference={round(float(r['t_s']),10):r for r in history if r['pressure_update']=='coupled' and float(r['dt_s'])==fine}
    matched=[reference[round(float(r['t_s']),10)] for r in current]
    stress=np.column_stack([col(current,'tau_xx_Pa'),col(current,'tau_xy_Pa')])
    refstress=np.column_stack([col(matched,'tau_xx_Pa'),col(matched,'tau_xy_Pa')])
    # tau_yy=-tau_xx, tau_zz=0: the common full-tensor factor sqrt(2) cancels.
    comparisons.append(dict(coarse_dt_s=coarse,fine_dt_s=fine,
        relative_full_tensor_time_l2=np.linalg.norm(stress-refstress)/np.linalg.norm(refstress),
        relative_speed_time_l2=np.linalg.norm(col(current,'V_m_s')-col(matched,'V_m_s'))/np.linalg.norm(col(matched,'V_m_s')),
        max_stress_component_difference_Pa=np.max(abs(stress-refstress)),
        final_shear_difference_Pa=abs(stress[-1,1]-refstress[-1,1]),
        final_work_relative_difference=abs(float(current[-1]['work_J_m3'])-float(matched[-1]['work_J_m3']))/float(matched[-1]['work_J_m3'])))
with (OUT/'response-comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
for dt,style in [(.0125,'-'),(.00625,'--'),(.003125,':')]:
    rows=[r for r in history if r['pressure_update']=='coupled' and float(r['dt_s'])==dt]
    axes[0,0].plot(col(rows,'t_s'),col(rows,'tau_xy_Pa'),style,label=f'Δt={dt:g} s')
    axes[0,1].semilogy(col(rows,'t_s'),col(rows,'V_m_s'),style,label=f'Δt={dt:g} s')
for ax in axes[0]:ax.axvline(.15,color='.6',ls=':')
axes[0,0].set(xlabel='Time [s]',ylabel='Physical τxy [Pa]',title='Mature point: imposed shear reverses at 0.15 s')
axes[0,0].legend(fontsize=8)
axes[0,1].set(xlabel='Time [s]',ylabel='Slip speed V ≥ 0 [m/s]',title='Fully coupled pressure: accepted endpoint sequence')
axes[0,1].text(.02,.04,'Lagged free-direction trials stop at\n0.025 / 0.0125 / 0.009375 s;\nno matched complete lagged path is claimed.',transform=axes[0,1].transAxes,fontsize=8)
interface=read('interface-histories.csv')
rows=[r for r in interface if r['pressure_update']=='coupled' and float(r['dt_s'])==.25]
axes[1,0].plot(col(rows,'t_s'),col(rows,'reference_traction_Pa'),color='k',label='Exact state + current pressure')
axes[1,0].plot(col(rows,'t_s'),col(rows,'traction_Pa'),'--',label='BE reference')
axes[1,0].set(xlabel='Time [s]',ylabel='Signed interface traction [Pa]',title='Separate coherent-plane velocity steps / reversal / stop')
axes[1,0].legend(fontsize=8)
rows=[r for r in read('point-timesteps.csv') if r['pressure_update']=='coupled'];x=np.arange(len(rows));bottom=np.zeros(len(rows))
last=[next(v for v in reversed(history) if v['pressure_update']=='coupled' and v['dt_s']==r['dt_s']) for r in rows]
for key,label in [('storage_J_m3','Storage'),('friction_J_m3','Friction'),('radiation_J_m3','Radiation'),('Maxwell_J_m3','Maxwell'),('BE_J_m3','BE remainder')]:
    values=col(last,key);axes[1,1].bar(x,values,bottom=bottom,label=label);bottom+=values
axes[1,1].set(xticks=x,xticklabels=[f'{float(r["dt_s"]):g}' for r in rows],xlabel='Δt [s]',ylabel='Energy / work density [J/m³]',title='Point work partition at 0.3 s; numerical loss kept separate')
axes[1,1].legend(fontsize=8,ncol=2)
fig.suptitle('Small independent kinetic controls; no phase evolution or propagation\nTensor histories retained; no accumulated-slip storage is added to mature RSF')
fig.savefig(OUT/'point-and-interface.png',dpi=170);plt.close(fig)
