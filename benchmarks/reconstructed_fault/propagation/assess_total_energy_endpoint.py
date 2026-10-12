#!/usr/bin/env python3
"""Read-only evolving endpoint/mean comparison; never advances saved histories."""
import json
import hashlib
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from assess_total_energy_time import load, compare, checks
from total_energy_step import HERE, ROOT, N, G, ETA
from total_energy_time_comparison import make_energy

OUT = HERE/'total_energy_endpoint_evidence'
COUNTS = [64, 128, 256]


def name(scheme, n):
    return f'aligned_evolving_{scheme}{n}'


def rates_comparison(coarse, fine):
    s, r, f = load(coarse)
    sf, rf, ff = load(fine)
    stride = sf['steps']//s['steps']
    assert stride*s['steps'] == sf['steps']
    v = np.array([x['slip_rate'] for x in r])
    vf = np.array([x['slip_rate'] for x in rf])
    # Two distinct comparisons: approximate endpoint samples, and the actual
    # interval deposits divided by interval length. Neither is relabeled.
    v_end = vf[stride-1::stride]
    v_average = vf.reshape(s['steps'], stride, -1).mean(axis=1)
    return dict(endpoint_sample_max_difference_m_s=float(np.max(abs(v-v_end))),
                common_interval_average_max_difference_m_s=float(np.max(abs(v-v_average))),
                final_endpoint_sample_difference_m_s=float(np.max(abs(v[-1]-v_end[-1]))),
                meanings=dict(coarse=s['scheme'], fine=sf['scheme'],
                    endpoint='last solved rate in interval; endpoint approximation for endpoint mechanics',
                    interval='actual accumulated-slip increment / common interval duration'))


def field_locations(coarse, fine):
    s, r, f = load(coarse)
    sf, rf, ff = load(fine)
    stride = sf['steps']//s['steps']
    energy = make_energy(True)
    result = {}
    for field in ['phi','tau','slip']:
        difference = abs(f[field]-ff[field][::stride])
        j,i = np.unravel_index(np.argmax(difference),difference.shape)
        if field=='phi':location=[i//(N+1)/N,i%(N+1)/N]
        elif field=='tau':location=[float(energy.x[i]),float(energy.y[i])]
        else:location=[i/energy.ns,.5]
        result[field] = dict(time_s=float(f['time'][j]), location_m=location,
                             difference=float(difference[j,i]))
    return result


def native_rank_audit():
    """Rank bound for the proposed pullback, not a native eigenvalue test."""
    folder=ROOT/'build-propagation-integration/output-irreversibility_dt03125'
    prm=folder/'parameters.prm'; log=folder/'log.txt'
    text=prm.read_text(); lines=log.read_text()
    cells=int(re.search(r'Number of active cells: ([\d,]+)',lines)[1].replace(',',''))
    dofs=[int(v.replace(',','')) for v in re.search(
        r'Number of degrees of freedom: [\d,]+ \(([^)]+)\)',lines)[1].split('+')]
    degree=int(re.search(r'set Stokes velocity polynomial degree\s*=\s*(\d+)',text)[1])
    assert degree==2 and cells==1024 and dofs[:2]==[8450,1089]
    n=32; boundary=2*(4*(2*n+1)-4); free=dofs[0]-boundary
    divfree=free-dofs[1]; rank=3*cells
    result=dict(kind='Analytical rank bound for a proposed projected-parent-stress energy; NOT a defect of current Stokes',
        measured_config=dict(active_cells=cells,velocity_degree=degree,velocity_dofs=dofs[0],
            pressure_dofs=dofs[1],mapped_stress_interpolator='cell average',dimension=2),
        derived=dict(all_Dirichlet_velocity_boundary_dofs=boundary,free_velocity_dofs=free,
            divergence_free_dimension_lower_bound=divfree,per_cell_symmetric_stress_components=3,
            projected_stress_image_rank_upper_bound=rank,
            projected_energy_velocity_kernel_dimension_lower_bound=divfree-rank),
        assumptions=['Fixed mesh/particles, one cell average per stress component',
            'Linear incident-cell averaging and FE evaluation cannot increase rank',
            'Matching mechanical Hessian from projected stress energy, with fixed profile and delta V=0',
            'Use ALL pressure equations as upper bound, without assuming their independence'],
        native_model_implemented=False,native_eigenvalue_run=False,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [prm,log,ROOT/'source/particle/interpolator/cell_average.cc',ROOT/'source/simulator/parameters.cc']})
    (OUT/'native-entry-rank-audit.json').write_text(json.dumps(result,indent=2)+'\n')


def state_update_errors():
    """Saved-field engineering R=(tau+chi*Z)/G recurrence, without mutation."""
    energy=make_energy(True)
    errors={}
    for scheme in ['mean','endpoint']:
        for n in COUNTS+([512] if scheme=='endpoint' else []):
            key=name(scheme,n);s,r,f=load(key)
            previous=np.zeros_like(f['tau'][0]);error=0.
            for j,record in enumerate(r):
                profile=energy.profile(f['phi'][j+1])
                actual=(f['tau'][j+1]+profile['chi']*(energy.T@f['Z'][j+1]))/G
                kappa=-ETA*np.expm1(-record['dt']*G/ETA)
                expected=record['beta']*previous+kappa/G*(
                    record['prescribed_rate']+2*np.pi*np.cos(2*np.pi*energy.y)
                    *record['free_bulk_velocity_amplitude'])
                error=max(error,float(max(abs(actual-expected))))
                previous=actual
            assert error<1e-12,(key,error)
            errors[key]=error
    return errors


def native_state_audit():
    """Uniform-Maxwell R encoding; not a native implementation/run.

    In engineering shear, R=(tau+chi*Z)/G. In full tensors the same
    physical variable is tau/(2G)+(Z/G)*chi*S. Cancelling the matching
    bulk/fault Maxwell terms gives R_new=beta*R_old+kappa/G*strain_rate.
    This identity avoids projecting particle stresses, but requires an
    authoritative accepted R field. It is not an inverse of that projector.
    """
    paths=[ROOT/'source/material_model/phase_field_fault/constitutive.cc',
           ROOT/'source/material_model/phase_field_fault/history.cc',
           ROOT/'source/reconstructed_fault/surface_system_bulk_work.cc',
           ROOT/'source/simulator/assemblers/reconstructed_fault_stokes.cc',
           ROOT/'include/aspect/material_model/phase_field_fault.h']
    result=dict(kind='Saved-field algebra/source audit; NOT a native FE or checkpoint test',
        accepted_R_update_error=state_update_errors(),
        exact_scope=['Uniform identical bulk/fault G and eta; fixed shear tensor',
            'Unloaded R0=0 and native FE velocity: R is symgrad of an FE vector',
            'No rotation, transport, mesh changes or temperature/composition-dependent coefficients'],
        proposed_encoding='d_R_new=beta*d_R_old+(kappa/G)*u_accepted',
        native_existing_hook='benchmark_retained_stress; plugin-owned, default off, not serialized',
        native_state_implemented=False,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'native-entry-state-audit.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    native_rank_audit()
    native_state_audit()
    results = {}
    for scheme in ['mean', 'endpoint']:
        for n in COUNTS + ([512] if scheme=='endpoint' and
                (HERE/'total_energy_time_evidence/aligned_evolving_endpoint512/summary.json').exists() else []):
            key = name(scheme, n)
            checked = checks(key)
            s, records, f = load(key)
            work = np.array([r['mechanical_work_error'] for r in records])
            release = np.array([r['phase_release'] for r in records])
            boundary_work = sum(abs(r['prescribed_external_work']) for r in records)
            fast = np.array([.18 <= r['time'] <= .4 for r in records])
            checked.update(
                cumulative_absolute_phase_release_J_m=float(sum(abs(release))),
                cumulative_absolute_exchange_defect_J_m=sum(abs(r['exchange_defect']) for r in records),
                cumulative_absolute_free_mode_work_J_m=sum(abs(r['free_bulk_work']) for r in records),
                maximum_absolute_mechanical_work_error_J_m=float(max(abs(work))),
                absolute_mechanical_error_over_absolute_boundary_work=float(sum(abs(work))/boundary_work),
                absolute_mechanical_error_over_fracture=float(sum(abs(work))/s['sums']['fracture_expenditure']),
                fast_interval=dict(time_window_s=[.18, .4],
                    phase_release_J_m=float(sum(release[fast])),
                    absolute_phase_release_J_m=float(sum(abs(release[fast]))),
                    signed_mechanical_error_J_m=float(sum(work[fast])),
                    absolute_mechanical_error_J_m=float(sum(abs(work[fast])))),
                final_normal_boundary_phi=[float(x) for x in f['phi'][-1].reshape(N+1,N+1)[N//2,[0,-1]]],
                initial_source_error=float(max(abs(make_energy(True).direct(f['phi'][0])[0]
                    /make_energy(True).profile(f['phi'][0])['isurf']-1))),
                velocity_meaning=('interval rate selected by exact mean forces' if scheme=='mean'
                    else 'endpoint-force approximation held fixed over interval'),
                maximum_actual_slip_rate_m_s=max(max(r['slip_rate']) for r in records))
            results[key] = checked
    pairs = {}
    for scheme in ['mean', 'endpoint']:
        for a,b in zip(COUNTS[:-1], COUNTS[1:]):
            key = f'{scheme}{a}_vs_{b}'
            pairs[key] = compare(name(scheme,a), name(scheme,b))
            pairs[key]['rates'] = rates_comparison(name(scheme,a), name(scheme,b))
            pairs[key]['worst_locations'] = field_locations(name(scheme,a), name(scheme,b))
    for n in COUNTS:
        key = f'mean_vs_endpoint{n}'
        pairs[key] = compare(name('mean',n), name('endpoint',n))
        pairs[key]['rates'] = rates_comparison(name('mean',n), name('endpoint',n))
        pairs[key]['worst_locations'] = field_locations(name('mean',n), name('endpoint',n))
    if name('endpoint',512) in results:
        for scheme in ['mean','endpoint']:
            key=f'{scheme}256_vs_endpoint512'
            pairs[key]=compare(name(scheme,256),name('endpoint',512))
            pairs[key]['rates']=rates_comparison(name(scheme,256),name('endpoint',512))
            pairs[key]['worst_locations']=field_locations(name(scheme,256),name('endpoint',512))
    # Common physical times are the primary comparison. The inherited anchor
    # alignment is explicitly a separate lag diagnostic, not a replacement.
    result = dict(prototype_only=True, production_changed=False,
        settings='unchanged corrected intersection Q3, instantaneous total-energy phase residual, a=.256, t=1',
        checks=results, comparisons=pairs, fast_window_s=[.18,.4])
    (OUT/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    fig, axes = plt.subplots(2,3,figsize=(13,7))
    energy = make_energy(True)
    _,_,ref = load(name('endpoint',256))
    core = N//2*(N+1)+N//2
    tail = N//2*(N+1)+5  # y=.3125, fixed before comparing trajectories
    stress_index = int(np.argmax(abs(ref['tau'][-1])))
    for scheme in ['mean','endpoint']:
        for n in COUNTS:
            s,r,f = load(name(scheme,n))
            label = f'{scheme}, dt={s["dt"]:g}'
            style = '--' if scheme=='mean' else '-'
            color = plt.get_cmap('viridis')((COUNTS.index(n)+1)/4)
            q=f['Z']/f['I']
            axes[0,0].plot(f['time'], f['phi'][:,tail],style,color=color,label=label)
            axes[0,1].plot(f['time'], f['tau'][:,stress_index]/1000,style,color=color)
            axes[0,2].plot(f['time'], q[:,energy.ns//2],style,color=color)
            axes[1,0].plot(f['time'][1:], [x['slip_rate'][energy.ns//2] for x in r],style,color=color)
            axes[1,1].plot(f['time'],f['slip'][:,energy.ns//2],style,color=color)
            axes[1,2].plot(f['slip'][:,energy.ns//2],q[:,energy.ns//2],style,color=color)
    if name('endpoint',512) in results:
        s,r,f=load(name('endpoint',512));q=f['Z']/f['I']
        axes[0,0].plot(f['time'],f['phi'][:,tail],'k:',label='endpoint, 512-step check')
        axes[0,1].plot(f['time'],f['tau'][:,stress_index]/1000,'k:')
        axes[0,2].plot(f['time'],q[:,energy.ns//2],'k:')
        axes[1,0].plot(f['time'][1:],[x['slip_rate'][energy.ns//2] for x in r],'k:')
        axes[1,1].plot(f['time'],f['slip'][:,energy.ns//2],'k:')
        axes[1,2].plot(f['slip'][:,energy.ns//2],q[:,energy.ns//2],'k:')
    labels=[('Time (s)','Tail phase at y=.3125 m'),('Time (s)','Selected bulk stress (kPa)'),
        ('Time (s)','Center/endpoint cohesive traction (Pa)'),('Time (s)','Actual solved slip rate (m/s)'),
        ('Time (s)','Center/endpoint accumulated slip (m)'),('Local slip (m)','Cohesive traction (Pa)')]
    for ax,(x,y) in zip(axes.flat,labels):ax.set(xlabel=x,ylabel=y);ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    save(fig,'physical-histories')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for scheme in ['mean','endpoint']:
        for n in COUNTS:
            s,r,f=load(name(scheme,n));t=f['time'][1:]
            style='--' if scheme=='mean' else '-';color=plt.get_cmap('viridis')((COUNTS.index(n)+1)/4)
            label=f'{scheme}, {n} steps'
            axes[0,0].plot(f['time'],f['phi'][:,tail],style,color=color,label=label)
            axes[0,1].plot(t,[x['phase_release'] for x in r],style,color=color)
            axes[0,2].plot(t,[x['mechanical_work_error'] for x in r],style,color=color)
            axes[1,0].plot(t,np.cumsum([abs(x['phase_release']) for x in r]),style,color=color)
            axes[1,1].plot(t,np.cumsum([abs(x['mechanical_work_error']) for x in r]),style,color=color)
            axes[1,2].plot(t,np.cumsum([x['mechanical_work_error'] for x in r]),style,color=color)
    if name('endpoint',512) in results:
        s,r,f=load(name('endpoint',512));t=f['time'][1:]
        axes[0,0].plot(f['time'],f['phi'][:,tail],'k:',label='endpoint, 512-step check')
        axes[0,1].plot(t,[x['phase_release'] for x in r],'k:')
        axes[0,2].plot(t,[x['mechanical_work_error'] for x in r],'k:')
        axes[1,0].plot(t,np.cumsum([abs(x['phase_release']) for x in r]),'k:')
        axes[1,1].plot(t,np.cumsum([abs(x['mechanical_work_error']) for x in r]),'k:')
        axes[1,2].plot(t,np.cumsum([x['mechanical_work_error'] for x in r]),'k:')
    for ax in axes[0]:ax.set_xlim(.18,.4)
    for ax,y in zip(axes.flat,['Fast tail phase','D_relax per step (J/m)','Signed mechanical gap per step (J/m)',
        'Cumulative |D_relax| (J/m)','Cumulative |mechanical gap| (J/m)','Cumulative signed mechanical gap (J/m)']):
        ax.set(xlabel='Time (s)',ylabel=y);ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7);save(fig,'fast-tail-and-separate-energy')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for scheme in ['mean','endpoint']:
        for n in COUNTS:
            s,r,f=load(name(scheme,n));style='--' if scheme=='mean' else '-'
            color=plt.get_cmap('viridis')((COUNTS.index(n)+1)/4);label=f'{scheme}, {n}'
            for ax,time in zip(axes,[.25,.375,1.]):
                index=int(round(time*n))
                ax.plot(np.linspace(0,1,N+1),f['phi'][index].reshape(N+1,N+1)[N//2],style,color=color,label=label)
                ax.set(title=f't={time:g} s',xlabel='Normal y (m)',ylabel='Phase');ax.grid(alpha=.25)
    if name('endpoint',512) in results:
        s,r,f=load(name('endpoint',512))
        for ax,time in zip(axes,[.25,.375,1.]):
            ax.plot(np.linspace(0,1,N+1),f['phi'][int(time*512)].reshape(N+1,N+1)[N//2],
                    'k:',label='endpoint, 512-step check')
    axes[0].legend(fontsize=7);save(fig,'spatial-phase')
    for key,checked in results.items():
        s=checked['summary']
        print(key, 'phi',s['final_phi_max'],'increment',s['max_phase_increment'],
            'Drelax',s['sums']['phase_release'],'work signed/absolute',s['sums']['mechanical_work_error'],
            checked['cumulative_absolute_mechanical_work_error_J_m'], 'work fraction',
            checked['absolute_mechanical_error_over_absolute_boundary_work'])
    for key,value in pairs.items():print(key,json.dumps(value))


def save(fig,name):
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/(name+'.'+suffix),dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
