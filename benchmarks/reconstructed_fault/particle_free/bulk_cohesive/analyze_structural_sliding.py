#!/usr/bin/env python3
"""Compact indexed evidence and plots; all values come from exported solves."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from structural_sliding import OUT
from structural_rsf import P0,ETA_RAD,V0
from reconcile_feng_rsf import friction
DELIVERY=OUT/'delivery'


def load(name):
    with (OUT/name).open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v)
            except ValueError:pass
    return rows


def write(name,rows):
    with (DELIVERY/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def val(rows,k):return np.array([r[k] for r in rows])
def save(fig,name):fig.savefig(DELIVERY/name,dpi=160);plt.close(fig)


def main():
    DELIVERY.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    old=load('sliding_histories.csv');new=load('refined_histories.csv');hist=old+new
    # Public plot data: final fine histories plus reference and coarse resolution curves.
    selected=[r for r in hist if (r['mode']=='band' and (r['dt_s']==.002 or r['cells']==512)) or (r['mode']!='band' and r['peak_phi']==1.)]
    profiles=load('sliding_profiles.csv')+load('refined_profiles.csv')
    prof=[r for r in profiles if r['cells']==512 and r['dt_s']==.001]
    write('01_sliding_history.csv',selected);write('02_sliding_profiles.csv',prof)
    comparison=load('sliding_comparison.csv')+load('refined_comparison.csv');write('03_resolution.csv',comparison)
    angular=load('pressure_state_controls.csv');write('04_mixed_angular.csv',angular)
    waves=load('interface_spectrum.csv');write('05_mixed_wavelength.csv',waves)
    oblique=load('oblique_history.csv');write('06_oblique_nonlinear.csv',oblique)
    birth=load('intact_admission.csv');write('07_intact_admission.csv',birth)
    snapshots=[]
    for peak in [.8,1.]:
        for mode in ['band','interface','coherent']:
            data=[r for r in hist if r['peak_phi']==peak and r['mode']==mode and r['dt_s']==.001 and (mode!='band' or r['cells']==512)]
            for target in [2.,2.1,4.,4.5,6.]:
                r=min(data,key=lambda r:abs(r['time_s']-target));snapshots.append(r)
    write('08_response_landmarks.csv',snapshots)
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for peak,color,label in [(.8,'C0','developed (peak φ = .8)'),(1.,'C1','mature center (peak φ = 1)')]:
        data=[r for r in hist if r['peak_phi']==peak and r['mode']=='band' and r['cells']==512 and r['dt_s']==.001]
        ax[0,0].plot(val(data,'time_s'),val(data,'traction_Pa'),color=color,label=label)
        ax[0,1].plot(1000*val(data,'slip_m'),val(data,'traction_Pa'),color=color,label=label)
        ax[0,2].plot(val(data,'time_s'),val(data,'Theta_min'),color=color,label=label)
        p=[r for r in prof if r['peak_phi']==peak and abs(r['time_s']-4)<1e-9]
        ax[1,0].plot(val(p,'y_m'),1000*val(p,'local_V_m_s'),color=color,label=label)
        ax[1,1].plot(val(data,'time_s'),1000*val(data,'rate_width_m'),color=color,label=label)
        denom=val(data,'friction_J_m2')+val(data,'radiation_J_m2');use=denom>1e-5
        ax[1,2].plot(val(data,'time_s')[use],100*val(data,'absolute_step_remainders_J_m2')[use]/denom[use],color=color,label=label)
    reference=[r for r in hist if r['peak_phi']==1. and r['mode']=='interface' and r['dt_s']==.001]
    coherent=[r for r in hist if r['peak_phi']==1. and r['mode']=='coherent' and r['dt_s']==.001]
    ax[0,0].plot(val(reference,'time_s'),val(reference,'traction_Pa'),'k--',label='mature interface RSF')
    ax[0,1].plot(1000*val(reference,'slip_m'),val(reference,'traction_Pa'),'k--',label='mature interface RSF')
    ax[0,1].plot(1000*val(coherent,'slip_m'),val(coherent,'traction_Pa'),':',color='.5',label='coherent-profile cohesive reference')
    ax[0,2].plot(val(reference,'time_s'),val(reference,'Theta_min'),'k--',label='interface state')
    ax[0,0].set(xlabel='Time (s)',ylabel='Physical shear traction (Pa)',title='Velocity increase, hold and reversal');ax[0,0].legend(fontsize=8)
    ax[0,1].set(xlabel='Signed integrated slip (mm)',ylabel='Physical shear traction (Pa)',title='Same Maxwell block / boundary motion');ax[0,1].legend(fontsize=7)
    ax[0,2].set(xlabel='Time (s)',ylabel='Minimum local frictional age Θ',yscale='log',title='Distributed state is not one interface age')
    ax[1,0].set(xlabel='Cross-band y (m)',ylabel='Local RSF speed (mm/s)',title='Speed concentration at end of loading')
    ax[1,1].set(xlabel='Time (s)',ylabel='Rate participation width (mm)',title='Developed / mature-center profiles')
    ax[1,2].set(xlabel='Time (s)',ylabel='Σ|step work remainders| / sliding loss (%)',title='Trapezoidal work; separate from BE ledger');ax[1,2].axhline(1,color='k',ls=':');ax[1,2].set_yscale('log')
    for a in [ax[0,0],ax[0,2],ax[1,1],ax[1,2]]:
        for t in [2,4,4.5]:a.axvline(t,color='.6',ls=':',lw=.7)
    save(fig,'sliding_response.png')

    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for mode,label in [('full','full pressure + state'),('fixed_age_control','state perturbation suppressed'),('fixed_pn_control','pressure perturbation suppressed'),('both_fixed_control','both suppressed')]:
        data=[r for r in angular if r['mode']==mode and r['angle_deg']<=45]
        ax[0,0].plot(val(data,'angle_deg'),val(data,'growth_s_inv'),label=label)
    ax[0,0].axhline(0,color='k',lw=.7);ax[0,0].set(xlabel='Wavevector angle from fault tangent (degrees)',ylabel='Largest active growth rate (1/s)',title='Fixed physical fault normal; oblique disturbances');ax[0,0].legend(fontsize=8)
    for mode,label in [('bulk_oblique','bulk oblique, angle 2.29°'),('closed_interface','closed interface RSF')]:
        data=[r for r in waves if r['mode']==mode]
        ax[0,1].semilogx(val(data,'wavelength_m'),val(data,'growth_s_inv'),label=label)
    ax[0,1].axhline(0,color='k',lw=.7);ax[0,1].set(xlabel='Wavelength (m)',ylabel='Largest active growth rate (1/s)',title='Different mechanical operators, same RSF parameters');ax[0,1].legend(fontsize=8)
    for dt in [.002,.001,.0005]:
        data=[r for r in oblique if r['cells']==128 and r['dt_s']==dt]
        ax[1,0].plot(val(data,'time_s'),100*val(data,'Theta_cosine'),label=f'Δt={dt:g} s')
    ax[1,0].plot(val(data,'time_s'),100*val(data,'linear_Theta_cosine'),'k--',label='continuous linear prediction')
    ax[1,0].set(xlabel='Time (s)',ylabel='State perturbation amplitude (%)',title='Nonlinear mixed solve reproduces growth');ax[1,0].legend(fontsize=8)
    ax[1,1].plot(val(data,'time_s'),val(data,'pressure_range_Pa'),label='solved pressure range')
    ax[1,1].plot(val(data,'time_s'),val(data,'normal_pressure_range_Pa'),label='actual-plane pressure range')
    ax[1,1].set(xlabel='Time (s)',ylabel='Peak-to-peak pressure change (Pa)',title='No pressure lag or frozen normal stress');ax[1,1].legend(fontsize=8)
    save(fig,'mixed_response.png')

    fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for mode,label in [('existing_support','ray with existing support I > 0'),('new_uniform_ray','entirely new ray, I → 0')]:
        data=[r for r in birth if r['topology']==mode and r['phi']==1e-24]
        ax[0].semilogx(val(data,'dt_s'),val(data,'driving_Pa'),'o-',label=label)
        ax[1].semilogx(val(data,'dt_s'),val(data,'Q_xy_Pa'),'o-',label=label)
    ax[0].axhline(8,color='k',ls='--',label='fracture contribution = 8 Pa');ax[0].legend(fontsize=7)
    ax[0].set(xlabel='Trial Δt (s)',ylabel='One-sided mechanical phase driving (Pa)',title='Same 4 kPa trial shear; old A=K=0')
    ax[1].set(xlabel='Trial Δt (s)',ylabel='Limiting cohesive traction Qxy (Pa)',title='Intact quotient is not stored in A=0')
    for dt in [.05,.003125,.00078125]:
        data=[r for r in birth if r['topology']=='new_uniform_ray' and r['dt_s']==dt]
        ax[2].semilogx(val(data,'phi'),val(data,'driving_Pa'),'o-',ms=3,label=f'Δt={dt:g} s')
    ax[2].set(xlabel='Positive trial φ → 0',ylabel='Phase driving (Pa)',title='Resolve the limit; no numerical phase floor');ax[2].legend(fontsize=8)
    save(fig,'intact_admission.png')

    # Saved physical times only, no inferred fronts or interpolation in histories.
    fig,ax=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    times=[1,2,2.1,3,4,4.5,5,6]
    vlimit=1050*max(abs(val(prof,'local_V_m_s')))
    def frame(index):
        for a in ax:a.clear()
        for peak,color in [(.8,'C0'),(1.,'C1')]:
            p=[r for r in prof if r['peak_phi']==peak and abs(r['time_s']-times[index])<1e-9]
            ax[0].plot(val(p,'y_m'),1000*val(p,'local_V_m_s'),color=color,label=f'peak φ={peak:g}')
            ax[1].semilogy(val(p,'y_m'),val(p,'Theta'),color=color)
        ax[0].set(xlim=(-.1,.1),ylim=(-vlimit,vlimit),xlabel='y (m)',ylabel='Signed local speed (mm/s)');ax[0].legend()
        ax[1].set(xlim=(-.1,.1),ylim=(1e-4,1.1),xlabel='y (m)',ylabel='Frictional age Θ')
        fig.suptitle(f'Fixed phase, preserved histories — t={times[index]:g} s')
    animation=FuncAnimation(fig,frame,frames=len(times));animation.save(DELIVERY/'sliding_profiles.gif',writer=PillowWriter(fps=1));plt.close(fig)
    partition=[]
    for peak in [.8,1.]:
        p=[r for r in prof if r['peak_phi']==peak and abs(r['time_s']-4)<1e-9]
        x=val(p,'y_m');mass=np.full(len(x),x[1]-x[0]);mass[[0,-1]]*=.5
        chi=val(p,'chi_inv_m');v=val(p,'local_V_m_s');mu=np.array([(friction(abs(vv)/V0,rr['Theta'])[0] if vv!=0 else 0.) for vv,rr in zip(v,p)])
        velocity=mass@(chi*v)
        partition.append(dict(peak_phi=peak,time_s=4.,traction_Pa=p[0]['physical_shear_Pa'],interface_V_m_s=velocity,
            cohesive_Pa=float(mass@(val(p,'cohesive_traction_Pa')*chi*v)/velocity),
            friction_Pa=float(mass@(P0*mu*chi*abs(v))/velocity),radiation_Pa=float(mass@(ETA_RAD*chi*v*v)/velocity)))
    metrics=dict(traction_partition=partition,max_accumulated_engineering_shear=max(abs(2*r['K_xy']) for r in prof),
        at_four_seconds=[r for r in snapshots if r['mode'] in ['band','interface'] and abs(r['time_s']-4)<1e-8])
    (DELIVERY/'response_summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
