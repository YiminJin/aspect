#!/usr/bin/env python3
"""Read-only physics/ledger assessment and plots of common-path time tests."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss

from total_energy_step import Energy, HERE, N, coefficients, kkt
from total_energy_time_comparison import make_energy

OUT=HERE/'total_energy_time_evidence'
energy=Energy()
mass_row=np.asarray(energy.mass.sum(axis=1))


def load(name):
    path=OUT/name
    return (json.loads((path/'summary.json').read_text()),
            json.loads((path/'records.json').read_text()),
            dict(np.load(path/'fields.npz')))


def compare(name,reference):
    s,r,f=load(name); sr,rr,fr=load(reference)
    energy=make_energy(s.get('aligned_bulk_quadrature',False))
    stride=sr['steps']//s['steps'];assert stride*s['steps']==sr['steps']
    assert np.array_equal(f['time'],fr['time'][::stride])
    result=dict(reference=reference)
    for key in ['phi','tau','slip','Z','strain','inelastic','history']:
        difference=f[key]-fr[key][::stride]
        result[key+'_max_common_time_error']=float(np.max(abs(difference)))
        result[key+'_final_error']=float(np.max(abs(difference[-1])))
    q=f['Z']/f['I'];qr=fr['Z'][::stride]/fr['I'][::stride]
    result['q_max_common_time_error_Pa']=float(np.max(abs(q-qr)))
    result['q_final_error_Pa']=float(np.max(abs(q[-1]-qr[-1])))
    slip_errors=[]
    for node in [0,energy.ns//2,energy.ns]:
        limit=min(f['slip'][-1,node],fr['slip'][-1,node])
        samples=np.linspace(0,limit,1001)
        slip_errors.append(float(max(abs(np.interp(samples,f['slip'][:,node],q[:,node])
                -np.interp(samples,fr['slip'][:,node],fr['Z'][:,node]/fr['I'][:,node])))))
    result['q_max_common_local_slip_error_Pa']=max(slip_errors)
    dphi=f['phi']-fr['phi'][::stride]
    result['phi_max_common_time_L2_error']=float(max(
        np.sqrt(np.dot(energy.wb,(energy.B@v)**2)) for v in dphi))
    # Separate incoming-state alignment: it is NOT the primary common-time
    # comparison and never replaces it as a convergence claim.
    anchor_index=np.arange(s['steps'])*stride+1
    result['phi_max_equal_anchor_load_error']=float(np.max(abs(f['phi'][1:]-fr['phi'][anchor_index])))
    result['release_total_difference_J_m']=s['sums']['phase_release']-sr['sums']['phase_release']
    result['prescribed_work_difference_J_m']=s['sums']['prescribed_external_work']-sr['sums']['prescribed_external_work']
    return result


def checks(name):
    s,records,f=load(name)
    energy=make_energy(s.get('aligned_bulk_quadrature',False))
    assert s['healing_max']==0 and s['min_I']>0 and s['phase_kkt_max']<=1e-11
    assert np.min(np.diff(f['phi'],axis=0))>=0 and np.max(f['phi'])<1
    assert s['mechanical_kkt_max']<1e-9
    smooth_work_corrections=[]
    for j,r in enumerate(records):
        for key,tol in [('phase_path_error',1e-9),('temporal_quadrature_error',1e-9),
                        ('residual_strain_reconstruction_error',1e-12),
                        ('bulk_identity_defect',1e-8),('fault_identity_defect',1e-8)]:
            assert abs(r[key])<tol,(name,key,r[key])
        gap=(r['prescribed_external_work']-r['bulk_storage_change']-r['fault_storage_change']
             -r['fracture_expenditure']-r['bulk_viscous']-r['fault_viscous']-r['friction']
             -r['phase_release'])
        assert abs(gap-r['mechanical_work_error'])<1e-8
        assert abs(r['mechanical_work_error']-(r['exchange_defect']-r['free_bulk_work']))<1e-8
        selected=s['scheme']
        assert r[selected+'_slip_condition_error']<1e-9
        assert abs(r['free_bulk_endpoint_residual'] if selected=='endpoint' else r['free_bulk_residual'])<1e-9
        if selected=='mean':assert abs(gap)<1e-8
        V=np.array(r['slip_rate']); assert np.min(V)>=0
        force=np.array(r[selected+'_slip_force'])
        assert np.max(abs(force[V>0]),initial=0)<1e-9
        assert np.max(force[V==0],initial=0)<1e-9
        # Exact work against the original smooth boundary velocity, evaluated
        # on the discrete stress path. Separate this temporal kinematic
        # approximation from the scheme's mechanical equilibrium defect.
        p=energy.profile(f['phi'][j+1]);oldp=energy.profile(f['phi'][j])
        jump_tau=f['tau'][j]-(energy.T@f['Z'][j])*(p['chi']-oldp['chi'])
        mode=2*np.pi*np.cos(2*np.pi*energy.y)
        rate=r['prescribed_rate']+mode*r['free_bulk_velocity_amplitude']-p['chi']*(energy.T@V)
        beta,kap,mean,eta_bar,*_=coefficients(r['dt'],1e8)
        assert max(abs(f['tau'][j+1]-beta*jump_tau-kap*rate))<1e-10
        assert max(abs(f['Z'][j+1]-beta*f['Z'][j]-kap*V))<1e-10
        for temporal,visc,scheme in [(beta,kap,'endpoint'),(mean,eta_bar,'mean')]:
            tau=temporal*jump_tau+visc*rate
            zs=temporal*(energy.S@f['Z'][j])+visc*(energy.S@V)
            force=np.asarray(energy.T.T@(energy.wb*p['chi']*tau)
                             -energy.S.T@(energy.ws*(zs/p['isurf']+1000.)))
            assert max(abs(force-np.array(r[scheme+'_slip_force'])))<1e-9
            if scheme==s['scheme']:
                assert abs(math.fsum(energy.wb*mode*tau))<1e-9
                assert np.max(abs(force[V>0]),initial=0)<1e-9
                assert np.max(force[V==0],initial=0)<1e-9
        reconstruction=f['strain'][j+1]-f['inelastic'][j+1]-(f['tau'][j+1]+(energy.T@f['Z'][j+1])*p['chi'])/1e6
        assert max(abs(reconstruction))<1e-12
        if s['phase_evolves']:
            anchor=SimpleNamespace(tau=f['tau'][j],Z=f['Z'][j],profile=oldp)
            gradient=energy.objective(f['phi'][j+1],anchor)[1]
            assert kkt(gradient,f['phi'][j+1],f['phi'][j])<=1e-11
        correction=[]
        for order in [16,24]:
            z,w=leggauss(order);t=r['dt']*(z+1)/2
            moment=float(np.dot(r['dt']*w/2,(t-r['dt']/2)*np.expm1(-t/100.)))
            correction.append(s['amplitude']*moment*np.dot(energy.wb,jump_tau-1e8*rate))
        assert abs(correction[0]-correction[1])<1e-9
        smooth_work_corrections.append(float(correction[-1]))
    # Stored fracture and stresses independently reproduce the cumulative ledger.
    fracture=energy.fracture(f['phi'][-1])-energy.fracture(f['phi'][0])
    eb=np.dot(energy.wb,f['tau'][-1]**2-f['tau'][0]**2)/(2e6)
    ec=np.dot(energy.ws,(energy.S@f['Z'][-1])**2/(energy.S@f['I'][-1])
                         -(energy.S@f['Z'][0])**2/(energy.S@f['I'][0]))/(2e6)
    sums=s['sums']
    assert abs(fracture-sums['fracture_expenditure'])<1e-8
    assert abs(eb-sums['bulk_storage_change'])<1e-8
    assert abs(ec-sums['fault_storage_change'])<1e-8
    result=dict(summary=s,
        endpoint_slip_condition_error_Pa=max(condition(r,'endpoint') for r in records),
        mean_slip_condition_error_Pa=max(condition(r,'mean') for r in records),
        largest_release_rate_J_m_s=max(r['phase_release']/r['dt'] for r in records),
        cumulative_absolute_mechanical_work_error_J_m=sum(abs(r['mechanical_work_error']) for r in records),
        endpoint_bulk_impulse_imbalance_Pa_m_s=sum(abs(r['free_bulk_endpoint_residual'])*r['dt'] for r in records),
        stick_node_range=[min(r['stick_nodes'] for r in records),max(r['stick_nodes'] for r in records)],
        final_phase_change_max=float(max(f['phi'][-1]-f['phi'][0])),
        end_center_q_difference_max_Pa=float(max(abs(f['Z'][:,0]/f['I'][:,0]-f['Z'][:,energy.ns//2]/f['I'][:,energy.ns//2]))),
        final_fracture_J_m=fracture)
    finalp=energy.profile(f['phi'][-1]);direct,_=energy.direct(f['phi'][-1])
    result['final_surface_source_representation_error']=float(max(abs(direct/finalp['isurf']-1)))
    result['free_bulk_velocity_amplitude_max_m_s']=max(abs(r['free_bulk_velocity_amplitude']) for r in records)
    result['final_q_minmax_Pa']=[float(min(f['Z'][-1]/f['I'][-1])),float(max(f['Z'][-1]/f['I'][-1]))]
    result['final_slip_minmax_m']=[float(min(f['slip'][-1])),float(max(f['slip'][-1]))]
    result['smooth_loading_work_correction_J_m']=sum(smooth_work_corrections)
    result['work_against_smooth_loading_J_m']=sums['prescribed_external_work']+sum(smooth_work_corrections)
    result['smooth_loading_ledger_error_J_m']=sums['mechanical_work_error']+sum(smooth_work_corrections)
    peak=max(records,key=lambda r:r['phase']['max_increment'])
    j=records.index(peak)
    change=f['phi'][j+1]-f['phi'][j]
    index=int(np.argmax(change))
    result['largest_increment']=dict(time=peak['time'],anchor_time=peak['phase_anchor_time'],
        anchor_displacement=peak['phase_anchor_displacement'],
        location=[index//(N+1)/N,index%(N+1)/N],increment=float(change[index]),
        release_J_m=peak['phase_release'])
    return result


def condition(r,scheme):
    force=np.array(r[scheme+'_slip_force'])/mass_row;V=np.array(r['slip_rate'])
    return float(max(np.max(abs(force[V>0]),initial=0),np.max(force[V==0],initial=0)))


def savefig(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=160)
    fig.savefig(OUT/(name+'.pdf'));plt.close(fig)


def main():
    names=[f'frozen_{scheme}{n}' for scheme in ['mean','endpoint'] for n in [16,32,256,512]]
    names+=['frozen_mean8_checked','frozen_endpoint8']
    names += [f'evolving_mean{n}' for n in [8,16,32,64,128]]
    # One further onset refinement, if explicitly run to resolve a concrete jump.
    if (OUT/'evolving_mean256/summary.json').exists():names+=['evolving_mean256']
    frozen_ref='frozen_endpoint512'
    evolving_ref='evolving_mean256' if 'evolving_mean256' in names else 'evolving_mean128'
    old_evolving_ref=evolving_ref
    names += sorted(p.parent.name for p in OUT.glob('aligned_*/summary.json'))
    primary='aligned_' if (OUT/'aligned_evolving_mean128/summary.json').exists() else ''
    if primary:
        frozen_ref='aligned_frozen_endpoint512'
        evolving_ref='aligned_evolving_mean256' if (OUT/'aligned_evolving_mean256/summary.json').exists() else 'aligned_evolving_mean128'
    assessment={name:checks(name) for name in names}
    comparisons={}
    for name in names:
        if name.startswith('aligned_'):
            reference=evolving_ref if 'evolving' in name else frozen_ref
        else:
            reference=old_evolving_ref if 'evolving' in name else 'frozen_endpoint512'
        comparisons[name]=compare(name,reference)
    result=dict(prototype_only=True,production_changed=False,loading='gamma(t)=.256*t, 0<=t<=1 s',
        frozen_reference=frozen_ref,evolving_reference=evolving_ref,
        reference_notes='Frozen reference additionally checked against endpoint256 and mean512; '
                        'evolving reference sensitivity reported, not assumed negligible.',
        checks=assessment,comparisons=comparisons)
    (OUT/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for scheme in ['mean','endpoint']:
        for n in [8,16,32]:
            name=primary+f'frozen_{scheme}{n}'
            if scheme=='mean':
                if primary:name+=('_refined' if n==8 else '_checked' if n==32 else '')
                elif n==8:name+='_checked'
            s,r,f=load(name);label=f'{scheme}, dt={s["dt"]:g}'
            axes[0,0].plot(f['time'],f['tau'][:,np.argmax(abs(f['tau'][-1]))]/1e3,label=label)
            axes[0,1].plot(f['time'],f['slip'][:,energy.ns//2],label=label)
            axes[1,0].plot(f['time'][1:],[condition(v,'endpoint') for v in r],label=label)
            axes[1,1].plot(f['time'][1:],np.cumsum([v['mechanical_work_error'] for v in r]),label=label)
    s,r,f=load(frozen_ref)
    axes[0,0].plot(f['time'],f['tau'][:,np.argmax(abs(f['tau'][-1]))]/1e3,'k--',label='endpoint512 reference')
    axes[0,1].plot(f['time'],f['slip'][:,energy.ns//2],'k--')
    for ax,ylabel in zip(axes.flat,['Selected bulk shear stress (kPa)','Center cumulative slip (m)',
                                  'Endpoint slip condition error (Pa)','Cumulative work error (J/m)']):
        ax.set(xlabel='Time (s)',ylabel=ylabel);ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7);savefig(fig,'frozen-mechanical-comparison')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    _,_,ref=load(evolving_ref)
    tail=int(np.argmax(ref['phi'][-1]-ref['phi'][0]));core=(N//2)*(N+1)+N//2
    evolution_names=[n for n in names if n.startswith(primary+'evolving')]
    evolution_names.sort(key=lambda n:load(n)[0]['steps'])
    for i,name in enumerate(evolution_names):
        s,r,f=load(name);label=f'dt={s["dt"]:g}'
        color=plt.get_cmap('tab10')(i)
        axes[0,0].plot(f['time'],f['phi'][:,tail],label=label,color=color)
        axes[0,0].plot(f['time'],f['phi'][:,core],':',alpha=.35,color=color)
        axes[0,1].plot(f['slip'][:,energy.ns//2],f['Z'][:,energy.ns//2]/f['I'][:,energy.ns//2],label=label,color=color)
        axes[0,2].plot(f['time'][1:],np.cumsum([v['phase_release'] for v in r]),label=label,color=color)
        axes[1,0].plot(f['time'][1:],[v['phase_release'] for v in r],label=label,color=color)
        axes[1,1].plot(f['time'][1:],[abs(v['free_bulk_endpoint_residual']) for v in r],label=label,color=color)
        axes[1,2].plot(np.linspace(0,1,N+1),(f['phi'][-1]-f['phi'][0]).reshape(N+1,N+1)[N//2],label=label,color=color)
    for ax,(x,y) in zip(axes.flat,[('Time (s)','Phase at greatest-change node (core dotted)'),
        ('Local slip (m)','Center cohesive traction (Pa)'),('Time (s)','Cumulative D_relax (J/m)'),
        ('Time (s)','Instantaneous D_relax per step (J/m)'),('Time (s)','Endpoint free-mode imbalance (Pa m)'),
        ('Transverse y (m)','Final phase change at x=.5 m')]):
        ax.set(xlabel=x,ylabel=y);ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);savefig(fig,'evolving-phase-release')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for i,name in enumerate(evolution_names):
        s,r,f=load(name);label=f'dt={s["dt"]:g}'
        color=plt.get_cmap('tab10')(i)
        axes[0].plot(f['slip'][:,energy.ns//2],f['phi'][:,core],label=label,color=color)
        axes[1].plot(f['slip'][:,energy.ns//2],f['phi'][:,tail],label=label,color=color)
        axes[2].plot(f['time'],f['Z'][:,energy.ns//2]/f['I'][:,energy.ns//2],label=label,color=color)
        # Endpoint curves coincide by translational symmetry; retain their
        # values in records and quantify the difference instead of hiding it.
        axes[2].plot(f['time'],f['Z'][:,0]/f['I'][:,0],':',alpha=.3,color=color)
    for ax,x,y in zip(axes,['Center slip (m)','Center slip (m)','Time (s)'],
                     ['Core phase','Tail phase','Cohesive traction (Pa), endpoint dotted']):
        ax.set(xlabel=x,ylabel=y);ax.grid(alpha=.25)
    axes[0].legend(fontsize=7);savefig(fig,'local-phase-and-traction')
    fig,ax=plt.subplots(figsize=(6,4))
    for scheme in ['mean','endpoint']:
        ns=[8,16,32,256,512]
        keys=[primary+f'frozen_{scheme}{n}'+(('_refined' if n==8 else '_checked' if n==32 else '') if primary and scheme=='mean' else '_checked' if not primary and scheme=='mean' and n==8 else '') for n in ns]
        ys=[comparisons[key]['tau_max_common_time_error'] for key in keys]
        ax.loglog(1/np.array(ns),np.maximum(ys,1e-10),'o-',label=scheme)
    ax.set(xlabel='dt (s)',ylabel='Maximum common-time bulk stress error (Pa)')
    ax.legend();ax.grid(alpha=.25);savefig(fig,'frozen-reference-errors')
    if primary:
        fig,axes=plt.subplots(1,2,figsize=(9,3.4))
        for name,label in [('frozen_endpoint32','Unsplit bulk Q3'),
                           ('aligned_frozen_endpoint32','Q3 on intersection partition')]:
            s,r,f=load(name);x=np.linspace(0,1,energy.ns+1)
            axes[0].plot(x,f['Z'][-1]/f['I'][-1],'.-',label=label)
            axes[1].plot(x,f['slip'][-1],'.-',label=label)
        axes[0].set(xlabel='Along-fault x (m)',ylabel='Cohesive traction (Pa)')
        axes[1].set(xlabel='Along-fault x (m)',ylabel='Cumulative slip (m)')
        axes[0].legend(fontsize=8)
        for ax in axes:ax.grid(alpha=.25)
        savefig(fig,'surface-sampling-artifact')
    print('PASS numerical ledgers, irreversible bounds, KKT and saved-field checks:',len(names),'paths.')
    for name in names:
        s=assessment[name]['summary'];c=comparisons[name]
        print(name,'jump',s['max_phase_increment'],'Drelax',s['sums']['phase_release'],
              'phi_error',c['phi_max_common_time_error'],'stress_error',c['tau_max_common_time_error'],
              'q_error',c['q_max_common_time_error_Pa'],'slip_error',c['slip_max_common_time_error'])


if __name__=='__main__':main()
