#!/usr/bin/env python3
"""Unaligned common-time fields and separate split/continuous energy ledgers."""
import json
from pathlib import Path

import numpy as np

import single_pass_comparison as test
from assess_native_endpoint import STRESS, STORAGE
from assess_coupled_branch import totals

E = test.E


def read(path):
    return json.loads(path.read_text())


def split_totals(rows):
    keys=['phase_energy_change','D_relax','bulk_change','cohesive_change','fracture_change',
          'bulk_viscous','cohesive_viscous','friction','bulk_work','lifting_work','external_work',
          'bulk_profile_change','cohesive_profile_change','external_remainder_unadjusted',
          'fixed_phase_mechanical_remainder','mechanical_power','friction_residual_power',
          'mechanical_power_absolute','friction_residual_power_absolute']
    return {k:dict(signed=float(sum(row[k] for row in rows)),
                   absolute=float(sum(abs(row[k]) for row in rows)),
                   maximum=float(max(abs(row[k]) for row in rows))) for k in keys}


def load(path):
    fields = dict(np.load(path/'fields.npz'))
    history = read(path/'history.json')
    # Older retained arrays predate the redundant NPZ time member. Use their
    # actual accepted clock, never a shifted or reconstructed physical history.
    times = np.r_[0., [row['time'] for row in history]]
    if 'time' in fields:
        assert np.array_equal(fields['time'], times)
    else:
        fields['time'] = times
    assert len(fields['phase']) == len(times)
    return dict(path=path, fields=fields,
                initial=dict(np.load(path/'initialization.npz')),
                history=history, summary=read(path/'summary.json'))


def sampled_stress(r, fields, j, s):
    phi = fields['phase'][j]
    pp = ((1-s)*phi[:-1,None]+s*phi[1:,None]).ravel()
    R = fields['R'][j]
    derivative = r.n*((-3+4*s)*R[:-2:2,None]+(4-8*s)*R[1::2,None]
                     +(-1+4*s)*R[2::2,None]).ravel()
    I, _ = r.direct(phi)
    chi = test.old.core.h(pp)/I
    return test.old.G*derivative-fields['Z'][j]*chi, chi


def comparison(r, coarse, fine, slip_scale, rate_scale):
    a, b = coarse['fields'], fine['fields']
    lookup = {float(t):i for i,t in enumerate(b['time'])}
    z,w = np.polynomial.legendre.leggauss(24)
    s=(z+1)/2;weight=np.tile(w/(2*r.n),r.n)
    rows=[]
    for j,hist in enumerate(coarse['history'],1):
        k=lookup[hist['time']];previous=lookup[float(a['time'][j-1])]
        ref=fine['history'][k-1]
        ta,_=sampled_stress(r,a,j,s);tb,chi=sampled_stress(r,b,k,s);delta=ta-tb
        rate=(ref['slip']-(fine['history'][previous-1]['slip'] if previous else 0.))/(hist['time']-a['time'][j-1])
        rows.append(dict(time=hist['time'],phase=float(max(abs(a['phase'][j]-b['phase'][k]))),
            q=abs(hist['q']-ref['q']),slip_fraction=abs(hist['slip']-ref['slip'])/slip_scale,
            interval_rate_fraction=abs(hist['V']-rate)/rate_scale,
            whole_stress_fraction=float(np.sqrt(weight@(delta*delta)))/STRESS,
            localized_stress_fraction=float(np.sqrt((weight*chi)@(delta*delta)/(weight@chi)))/STRESS,
            bulk_storage_fraction=abs(hist['bulk']-ref['bulk'])/STORAGE,
            centroid_difference=hist['centroid']['phase']-ref['centroid']['phase']))
    targets=dict(phase=.001,q=1.,slip_fraction=.001,interval_rate_fraction=.005,
                 whole_stress_fraction=.01,localized_stress_fraction=.01,bulk_storage_fraction=.01)
    maxima={k:float(max(row[k] for row in rows)) for k in targets}
    return dict(maximum=maxima,worst_time={k:max(rows,key=lambda row:row[k])['time'] for k in targets},
                passes={k:bool(maxima[k]<=targets[k]) for k in targets},targets=targets,by_time=rows)


def onset(fields, index=None):
    d=fields['phase']-fields['phase'][0]
    growth=np.max(d,axis=1) if index is None else d[:,index]
    hit=np.flatnonzero(growth>1e-6)
    return None if not len(hit) else [float(fields['time'][hit[0]-1]),float(fields['time'][hit[0]])]


def controls(data):
    f,h,s=data['fields'],data['history'],data['summary']
    dt=s['dt'];beta=np.exp(-test.old.G*dt/test.old.ETA)
    kappa=-test.old.ETA*np.expm1(-test.old.G*dt/test.old.ETA)
    error=max(abs((f['R'][1:]-beta*f['R'][:-1]-kappa/test.old.G*f['velocity']).ravel()))
    zerror=max(abs(f['Z'][1:]-beta*f['Z'][:-1]-kappa*np.array([v['V'] for v in h])))
    assert s['completed'] and s['last_accepted']==.625 and error<1e-16 and zerror<1e-10
    assert np.min(np.diff(f['phase'],axis=0))>=-4.922e-12 and np.min(f['phase'])>=0 and np.max(f['phase'])<1
    assert max(v['force'] for v in h)<1e-7 and max(v['contact'] for v in h)<1e-7
    incoming='incoming_KKT' if 'incoming_KKT' in h[0] else 'KKT'
    assert max(v[incoming] for v in h)<=test.TARGET
    displacement=sum(dt*.256*(v['time']-dt/2) for v in h)
    assert abs(displacement-.05)<1e-14
    result=dict(summary=s,maximum_subproblem_KKT=max(v[incoming] for v in h),
        maximum_force=max(v['force'] for v in h),maximum_contact=max(v['contact'] for v in h),
        recurrence=float(error),Z_recurrence=float(zerror),minimum_I=min(v['I'] for v in h),
        maximum_phase_increment=max(v['phase_increment'] for v in h),
        full_onset=onset(f),core_onset=onset(f,len(f['y'])//2),onset_threshold=1e-6,
        imposed_displacement_m=float(displacement),
        final=h[-1],continuous_path_energy=totals(read(data['path']/'energy.json')))
    if (data['path']/'split-energy.json').exists():
        ledger=read(data['path']/'split-energy.json')
        # Independently check the corrected arithmetic guard against actual
        # incoming energy; the first run's guard originally used scratch state.
        for row in ledger:
            initial=sum(row[k]-row[k+'_change'] for k in ['bulk','cohesive','fracture'])
            assert row['D_relax']>=-32*np.finfo(float).eps*max(1.,abs(initial))
        result.update(split_path_energy=split_totals(ledger),
            maximum_split_identity_error=max(v['split_identity_error'] for v in ledger),
            maximum_continuous_identity_error=max(v['identity_error'] for v in read(data['path']/'energy.json')),
            maximum_endpoint_phase_defect=max(v['endpoint_phase_defect'] for v in h),
            maximum_endpoint_mass_scaled_defect_Pa=max(v['endpoint_mass_scaled_defect_Pa'] for v in h))
    return result


def checks(r, data):
    """Compare cached endpoint mechanics with the original dense single-pass kernel.

    Both evaluate a frozen entry; neither advances any trajectory's history.
    Also repeat split-path integration at twice its time quadrature.
    """
    from scipy.linalg import eigh
    records=[]
    for t in [.5,.5625,.625]:
        f=data['fields'];j=int(round(t/data['summary']['dt']));hist=data['history'][j-1]
        pa,pb=f['phase'][j-1],f['phase'][j]
        Ra,Za=f['R'][j-1],float(f['Z'][j-1]);dt=data['summary']['dt']
        p=test.endpoint.SharedFE();p.incoming(pa,Ra,Za,0.,t,dt);p.evaluate(pb)
        cached=p.last[3].copy();cached_R=p.last[4].copy();cached_Z=p.last[5]
        rr=p.r;rr.D=rr.D.toarray();rr.r=Ra.copy();rr.Z=Za;rr.phi=pb.copy();rr.slip=0.
        dense=rr.mechanical(t,dt)
        errors=dict(R=float(max(abs(rr.r-cached_R))),Z=abs(rr.Z-cached_Z),
                    q=abs(dense['q']-cached['q']),V=abs(dense['V']-cached['V']))
        assert errors['R']<1e-11 and errors['Z']<1e-6 and errors['q']<1e-7 and errors['V']<1e-9
        args=(r,pa,pb,Ra,Za,f['R'][j],float(f['Z'][j]),f['velocity'][j-1],hist['V'],t,dt)
        a=test.split_ledger(*args,order=16);b=test.split_ledger(*args,order=32)
        r.r=Ra.copy();r.Z=Za
        gradient=r.objective(pb)[1]
        free=(pb>pa+1e-13)|(abs(gradient)<=10*test.TARGET)
        ids=np.flatnonzero(free)
        curvature=None
        if len(ids):
            H=r.phase_hessian(pb);M=test.old.mass(r)
            curvature=float(eigh(test.old.A*H[np.ix_(ids,ids)],M[np.ix_(ids,ids)],
                                 subset_by_index=[0,0],eigvals_only=True)[0])
        records.append(dict(time=t,cached_mechanics_errors=errors,
                            incoming_phase_mass_curvature_Pa=curvature,
                            time_quadrature_uncertainty=max(abs(a[k]-b[k]) for k in a)))
    return records


def assess():
    single=[load(E/f'dt-1-{d}') for d in [1024,2048]]
    coupled=[load(test.endpoint.E/f'dt-1-{d}') for d in [1024,2048]]
    for a in single+coupled:
        assert all(np.array_equal(a['initial'][k],coupled[0]['initial'][k]) for k in a['initial'])
    r=test.endpoint.SharedFE().r
    slip_scale=coupled[-1]['history'][-1]['slip'];rate_scale=max(v['V'] for v in coupled[-1]['history'])
    output=dict(initial_fields_identical=True,global_reflection_or_recentering=False,
        stress_scale_Pa=STRESS,storage_scale_J_per_m=STORAGE,slip_scale_m=slip_scale,rate_scale_m_per_s=rate_scale,
        endpoint_reference_uncertainty=read(test.endpoint.E/'temporal-comparison.json')['maximum'],
        single_against_reference=[comparison(r,a,coupled[-1],slip_scale,rate_scale) for a in single],
        same_timestep=[comparison(r,a,b,slip_scale,rate_scale) for a,b in zip(single,coupled)],
        single_halving=comparison(r,single[0],single[1],slip_scale,rate_scale),
        single_controls=[controls(a) for a in single],coupled_controls=[controls(a) for a in coupled],
        isolated_checks=checks(r,single[0]))
    test.old.write(E/'assessment.json',output)
    print(json.dumps({k:[{v:a[v] for v in ['maximum','passes']} for a in output[k]]
                      for k in ['single_against_reference','same_timestep']},indent=2))
    plot(r,single,coupled)


def plot(r,single,coupled):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for kind,group in [('Single pass',single),('Endpoint',coupled)]:
        for a in group:
            h=a['history'];f=a['fields'];d=round(1/a['summary']['dt']);label=f'{kind}, 1/{d} s'
            ts=[v['time'] for v in h]
            axes[0,0].plot(ts,f['phase'][1:,len(r.y)//2],label=label)
            axes[0,1].plot(ts,[v['q'] for v in h],label=label)
            axes[0,2].plot(ts,[v['slip'] for v in h],label=label)
            axes[1,0].plot([v['slip'] for v in h],[v['q'] for v in h],label=label)
            # Rates over the same 1/64-s physical windows, never different dt averages.
            edges=np.arange(0,41)/64;inds=[int(round(t*d)) for t in edges]
            slips=np.r_[0.,[v['slip'] for v in h]]
            axes[1,1].plot((edges[1:]+edges[:-1])/2,np.diff(slips[inds])*64,label=label)
            axes[1,2].plot(f['y'],f['phase'][-1],label=label)
    for ax,title,xlabel,ylabel in zip(axes.flat,
        ['Core phase','Cohesive traction','Cumulative slip','Traction versus slip','Common-window slip rate','Final unaligned phase'],
        ['Time (s)']*3+['Slip (m)','Time (s)','Normal coordinate y (m)'],
        ['φ','q (Pa)','Slip (m)','q (Pa)','Mean V (m/s)','φ']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.savefig(E/'physical-comparison.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    edges=np.arange(536,553)/1024
    for name,group in [('Single',single),('Endpoint',coupled)]:
        for a in group:
            d=round(1/a['summary']['dt']);h=a['history'];f=a['fields'];label=f'{name}, 1/{d} s'
            axes[0].plot(f['time'],f['phase'][:,len(r.y)//2]-f['phase'][0,len(r.y)//2],label=label)
            slips=np.r_[0.,[v['slip'] for v in h]];ids=[round(t*d) for t in edges]
            axes[1].plot((edges[1:]+edges[:-1])/2,np.diff(slips[ids])*1024,label=label)
    axes[0].set(xlim=(.5234375,.5390625),ylim=(0,.012),title='Core damage onset',xlabel='Time (s)',ylabel='Growth above initial φ')
    axes[1].set(title='Same 1/1024-s windows',xlabel='Window midpoint (s)',ylabel='Mean V (m/s)')
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.savefig(E/'onset-comparison.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for kind,group in [('Single pass',single),('Endpoint',coupled)]:
        for a in group:
            d=round(1/a['summary']['dt']);label=f'{kind}, 1/{d} s';f=a['fields']
            for ax,t in zip(axes.flat,[.53125,.5625,.59375,.625]):
                j=round(t*d);ax.plot(f['y'],f['phase'][j],label=label);ax.set(title=f't={t:g} s',xlabel='y (m)',ylabel='φ');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Common-time phase profiles; no alignment');fig.savefig(E/'phase-profiles.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    z,_=np.polynomial.legendre.leggauss(24);s=(z+1)/2
    yq=(r.y[:-1,None]+r.dy*s).ravel()
    for kind,group in [('Single pass',single),('Endpoint',coupled)]:
        for a in group:
            d=round(1/a['summary']['dt']);label=f'{kind}, 1/{d} s'
            for ax,t in zip(axes.flat,[.53125,.5625,.59375,.625]):
                tau,_=sampled_stress(r,a['fields'],round(t*d),s)
                ax.plot(yq,tau,label=label,linewidth=.8)
                ax.set(title=f't={t:g} s',xlabel='y (m)',ylabel='Shear stress (Pa)');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Independent 24-point stress sampling; six-point assembly unchanged');fig.savefig(E/'stress-profiles.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for kind,group in [('Single pass',single),('Endpoint',coupled)]:
        for a in group:
            d=round(1/a['summary']['dt']);label=f'{kind}, 1/{d} s';rows=read(a['path']/'energy.json');ts=[v['time'] for v in rows]
            for ax,key in zip(axes.flat,['external_remainder','mechanical_power_absolute','friction_residual_power_absolute','phase_power_absolute']):
                title={'external_remainder':'External energy remainder','mechanical_power_absolute':'Absolute mechanical residual work',
                       'friction_residual_power_absolute':'Absolute friction residual work','phase_power_absolute':'Absolute phase residual work'}[key]
                ax.plot(ts,np.cumsum([v[key] for v in rows]),label=label);ax.set(title=title,xlabel='Time (s)',ylabel='Cumulative (J/m)');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Common reconstructed continuous-path audit; separate contributions');fig.savefig(E/'continuous-energy.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for a in single:
        d=round(1/a['summary']['dt']);rows=read(a['path']/'split-energy.json');ts=[v['time'] for v in rows]
        for ax,key in zip(axes.flat,['D_relax','external_remainder_unadjusted','mechanical_power_absolute','friction_residual_power_absolute']):
            title={'D_relax':'Measured phase release','external_remainder_unadjusted':'Unadjusted external energy remainder',
                   'mechanical_power_absolute':'Absolute mechanical residual work','friction_residual_power_absolute':'Absolute friction residual work'}[key]
            ax.plot(ts,np.cumsum([v[key] for v in rows]),label=f'1/{d} s');ax.set(title=title,xlabel='Time (s)',ylabel='Cumulative (J/m)');ax.grid(alpha=.25)
    axes[0,0].legend();fig.suptitle('Actual single-pass split path: measured release is not constitutive dissipation');fig.savefig(E/'split-energy.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for a in single:
        d=round(1/a['summary']['dt']);h=a['history'];ts=[v['time'] for v in h]
        axes[0].semilogy(ts,[max(v['incoming_KKT'],1e-16) for v in h],label=f'Incoming, 1/{d}')
        axes[0].semilogy(ts,[max(v['endpoint_phase_defect'],1e-16) for v in h],'--',label=f'Endpoint defect, 1/{d}')
        axes[1].plot(ts,[v['endpoint_mass_scaled_defect_Pa'] for v in h],label=f'1/{d} s')
    axes[0].axhline(test.TARGET,color='k',linestyle=':',label='Subproblem target')
    axes[0].set(title='Convergence and splitting are separate',xlabel='Time (s)',ylabel='Scaled projected defect')
    axes[1].set(title='Completed-state phase defect',xlabel='Time (s)',ylabel='Nodal-measure scaled defect (Pa)')
    labels=[];cost=[]
    for name,group in [('Single',single),('Endpoint\nretained',coupled)]:
        for a in group:labels.append(f"{name}\n1/{round(1/a['summary']['dt'])}");cost.append(a['summary']['runtime_seconds'])
    axes[2].bar(labels,cost);axes[2].set(title='Recorded prototype cost',ylabel='Whole run (s)')
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=8)
    fig.savefig(E/'defect-and-cost.png',dpi=145);plt.close(fig)


if __name__=='__main__':
    assess()
