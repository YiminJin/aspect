"""Plot the captured normal split and check projection/ownership, without solves."""
import argparse
import json
import warnings
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='genfromtxt: Empty input file')
        return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8'))


def perturbation(a, field):
    """Remove only the explicitly exported reference, not the broad trend."""
    if field in ('weak_normal_sum_Pa', 'existing_production_weak_normal_Pa'):
        return a[field]-a['weak_reference_normal_Pa']
    return a[field]


def roughness(x, y):
    """Neighbouring-point chord departure: proxy, not proven numerical error."""
    xi=(x[1:-1]-x[:-2])/(x[2:]-x[:-2])
    return y[1:-1]-((1-xi)*y[:-2]+xi*y[2:])


def inspect(path):
    a=read(path)
    assert np.array_equal(a['vertex_id'], np.arange(len(a)))
    assert len(np.unique(a['accepted_step'])) == 1
    mass=a['normal_row_mass']
    assert np.all(mass>0)
    closure=a['weak_pressure_Pa']+a['weak_minus_n_tau_n_Pa']+a['weak_reference_normal_Pa']-a['existing_production_weak_normal_Pa']
    errors={}
    pairs=[('weak_pressure_Pa','pressure_load'),('weak_minus_n_tau_n_Pa','deviatoric_normal_load'),('weak_reference_normal_Pa','reference_load')]
    pairs += [(f'weak_d_{c}_Pa',f'd_{c}_load') for c in ('history','strain','slip') if f'weak_d_{c}_Pa' in a.dtype.names]
    for field,load in pairs:
        y=a['mass_diagonal']*a[field]
        y[1:]+=a['mass_left'][1:]*a[field][:-1]
        y[:-1]+=a['mass_right'][:-1]*a[field][1:]
        errors[field]=float(np.max(abs(y-a[load])/mass))
    info=dict(step=int(a['accepted_step'][0]), time_s=float(a['time_s'][0]), elapsed_s=float(a['elapsed_s'][0]),
              closure_max_Pa=float(np.max(abs(closure))), closure_RMS_Pa=float(np.sqrt(np.dot(mass,closure**2)/mass.sum())),
              projection_residual_over_mass_Pa=errors)
    return a,info


def matched_offset_zoom(q, profile, dest, step, lo, hi):
    """Unsmoothed QPs in fixed 2 m strips, not a line through mixed offsets.

    Identity is unchanged between accepted states. The exported r is retained
    so that the finite strip width cannot be mistaken for exact centerline data.
    """
    q=q[(q['down_dip_s_m']>=lo*1000)&(q['down_dip_s_m']<=hi*1000)]
    boundaries=profile['down_dip_s_m']/1000
    boundaries=np.sort(boundaries[(boundaries>=lo)&(boundaries<=hi)])
    cells=sorted(set(q['cell'].tolist()))
    labels={cell:i for i,cell in enumerate(cells)}
    fig,axes=plt.subplots(4,2,figsize=(15,12),sharex=True,layout='constrained')
    metadata=[];info={}
    for row,target in enumerate((-50.,0.,50.)):
        strip=q[abs(q['r_m']-target)<=1.]
        strip=np.sort(strip,order='down_dip_s_m')
        for field,label in [('p','p'),('minus_n_tau_n','−n·τ·n'),('perturbation_normal','p − n·τ·n')]:
            axes[row,0].scatter(strip['down_dip_s_m']/1000,strip[field]/1000,s=10,label=label)
        axes[row,0].set_ylabel('kPa');axes[row,0].set_title(f'r = {target:g} ± 1 m; {len(strip)} actual QPs')
        for rank in np.unique(strip['rank']):
            a=strip[strip['rank']==rank]
            axes[row,1].scatter(a['down_dip_s_m']/1000,a['r_m']-target,s=20,label=f'rank {rank}')
        for a in strip:
            label=labels[a['cell']]
            metadata.append([step,target,label]+[a[n] for n in q.dtype.names])
        # Labels mark sampled cells, not inferred physical cell boundaries.
        for cell in np.unique(strip['cell']):
            a=strip[strip['cell']==cell];i=len(a)//2
            axes[row,1].annotate(f'C{labels[cell]}',(a['down_dip_s_m'][i]/1000,a['r_m'][i]-target),
                                 fontsize=5,rotation=65,xytext=(1,3),textcoords='offset points')
        axes[row,1].set(ylim=(-1.25,1.6),ylabel='Actual r − target (m)',title='Cell labels and MPI ownership; full IDs in CSV')
        info[str(target)]=dict(samples=len(strip),r_min=float(min(strip['r_m'])),r_max=float(max(strip['r_m'])),
            ranks=list(map(int,np.unique(strip['rank']))),cell_count=len(np.unique(strip['cell'])),
            range_Pa={f:[float(min(strip[f])),float(max(strip[f]))] for f in ('p','minus_n_tau_n','perturbation_normal')})
    axes[3,0].scatter(q['down_dip_s_m']/1000,q['r_m'],c=q['rank'],s=2,cmap='tab20')
    axes[3,0].set(ylabel='Normal offset (m)',title='All native QPs; colour = MPI rank')
    axes[3,1].scatter(q['down_dip_s_m']/1000,q['cell_diameter']/np.sqrt(2),s=2,label='Square-cell edge h')
    axes[3,1].set(ylabel='h (m)',title='Bulk-cell size (not a refinement-boundary inference)')
    for ax in axes.flat:
        for s in boundaries:ax.axvline(s,color='grey',ls=':',alpha=.5,lw=.7)
        ax.grid(alpha=.15)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
    for ax in axes[-1]:ax.set_xlabel('Down-dip distance (km); dotted = actual fault vertices')
    fig.suptitle(f'Matched offset strips: {lo}–{hi} km, accepted step {step}\nNo smoothing or normal interpolation; cell-label lookup in companion CSV')
    fig.savefig(dest/f'offset_zoom_{step}_{lo}_{hi}.png',dpi=190);plt.close(fig)
    with (dest/f'offset_zoom_{step}_{lo}_{hi}.csv').open('w',newline='') as stream:
        out=csv.writer(stream);out.writerow(['export_step','target_r_m','cell_label']+list(q.dtype.names));out.writerows(metadata)
    info['cell_edge_range_m']=[float(min(q['cell_diameter'])/np.sqrt(2)),float(max(q['cell_diameter'])/np.sqrt(2))]
    info['fault_element_lengths_m']=list(map(float,np.unique(np.round(np.diff(boundaries)*1000,6))))
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--compare-control',type=Path)
    args=parser.parse_args()
    paths=sorted(args.output.glob('normal_profile_*.csv'),key=lambda p:int(p.stem.split('_')[-1]))
    if not paths: raise ValueError('No accepted diagnostic profiles')
    records=[inspect(p) for p in paths]
    initial,final=records[0][0],records[-1][0]
    summary=dict(states=[r[1] for r in records], windows={}, raw={}, control=None)
    component_fields=['weak_d_history_Pa','weak_d_strain_Pa','weak_d_slip_Pa']
    has_components=all(f in final.dtype.names for f in component_fields)
    summary['stress_component_status']='captured' if has_components else 'not exported; cannot infer history/current split from total stress'
    dest=args.output/'normal_plots_revised';dest.mkdir(exist_ok=True)
    names=['weak_pressure_Pa','weak_minus_n_tau_n_Pa','weak_normal_sum_Pa','existing_production_weak_normal_Pa']
    labels=['Pressure pʷ','Deviatoric dʷ = −(n·τ·n)ʷ','pʷ + dʷ','Production σₙ − reference']
    for lo,hi in [(22,35),(60,90),(70,71),(79,80)]:
        mask=(final['down_dip_s_m']>=lo*1000)&(final['down_dip_s_m']<=hi*1000)
        order=np.argsort(final['down_dip_s_m'][mask]); x=final['down_dip_s_m'][mask][order]/1000
        a,b=initial[mask][order],final[mask][order]
        fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,layout='constrained')
        for field,label in zip(names,labels):
            axes[0,0].plot(x,perturbation(a,field)/1e3,label=label)
            axes[0,1].plot(x,perturbation(b,field)/1e3,label=label,ls='--' if field==names[-1] else '-')
            axes[2,0].plot(x,perturbation(b,field)-perturbation(a,field),label=label)
        # Identical scales show the small five-state increment honestly.
        limits=[axes[0,c].get_ylim() for c in (0,1)]
        for ax in axes[0]:ax.set_ylim(min(l[0] for l in limits),max(l[1] for l in limits))
        for field,load,label in [('weak_pressure_Pa','pressure_load_over_row_mass','Pressure'),('weak_minus_n_tau_n_Pa','deviatoric_load_over_row_mass','Deviatoric')]:
            axes[1,0].plot(x,b[field]/1e3,label=label+' consistent')
            axes[1,0].plot(x,b[load]/1e3,'--',label=label+' f/m')
            axes[1,1].plot(x[1:-1],roughness(x,b[field]),label=label+' consistent')
            axes[1,1].plot(x[1:-1],roughness(x,b[load]),'--',label=label+' f/m')
        axes[2,1].plot(x,b['I_h'],label='I_h final')
        axes[2,1].plot(x,a['I_h'],'--',label='I_h baseline')
        titles=['First accepted diagnostic state','Final accepted diagnostic state','Consistent projection versus row-scaled load',
                'Chord-departure roughness proxy (Pa)','Final minus baseline (Pa)','Completed I_h (m)']
        for ax,title in zip(axes.flat,titles):
            ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.legend(fontsize=7)
        for ax in axes[0]:ax.set_ylabel('kPa (reference removed from σₙ)')
        axes[1,0].set_ylabel('kPa')
        for ax in axes[-1]:ax.set_xlabel('Down-dip distance (km)')
        fig.suptitle(f'BP5 captured normal traction, {lo}–{hi} km\n'
                     f'Accepted steps {int(initial["accepted_step"][0])}–{int(final["accepted_step"][0])}; '
                     f'elapsed {initial["elapsed_s"][0]:.9g}–{final["elapsed_s"][0]:.9g} s')
        fig.savefig(dest/f'normal_{lo}_{hi}.png',dpi=180);plt.close(fig)
        p=roughness(x,b[names[0]]);d=roughness(x,b[names[1]])
        weights=b['normal_row_mass'][1:-1]
        fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
        dots=ax.scatter(p/1e3,d/1e3,c=x[1:-1],s=18)
        extent=max(np.max(abs(p)),np.max(abs(d)))/1e3
        ax.plot([-extent,extent],[extent,-extent],'k--',lw=.8,label='Exact cancellation: d roughness = −p roughness')
        ax.set(xlabel='Pressure chord departure (kPa)',ylabel='Deviatoric chord departure (kPa)',
               title=f'{lo}–{hi} km; step {int(final["accepted_step"][0])}\nRoughness proxy, not proven numerical error')
        ax.legend(fontsize=7);ax.grid(alpha=.2);fig.colorbar(dots,ax=ax,label='Down-dip distance (km)')
        fig.savefig(dest/f'roughness_scatter_{lo}_{hi}.png',dpi=180);plt.close(fig)
        summary['windows'][f'{lo}-{hi}']=dict(
            work_weighted_chord_RMS_Pa={field:float(np.sqrt(np.dot(weights,roughness(x,b[field])**2)/weights.sum()))
                                        for field in names+['pressure_load_over_row_mass','deviatoric_load_over_row_mass']},
            pressure_deviatoric_roughness_correlation=float(np.corrcoef(p,d)[0,1]) if np.std(p)*np.std(d)>0 else None,
            increments_max_Pa={field:float(np.max(abs(perturbation(b,field)-perturbation(a,field)))) for field in names} if len(records)>1 else None,
            perturbation_range_Pa={field:[float(np.min(perturbation(b,field))),float(np.max(perturbation(b,field)))] for field in names})
        if has_components:
            closure=sum(b[f] for f in component_fields)-b['weak_minus_n_tau_n_Pa']
            values=[b['weak_d_history_Pa'],b['weak_d_strain_Pa']+b['weak_d_slip_Pa']]
            summary['windows'][f'{lo}-{hi}']['components']=dict(
                closure_max_Pa=float(max(abs(closure))),
                work_weighted_chord_RMS_Pa={label:float(np.sqrt(np.dot(weights,roughness(x,v)**2)/weights.sum()))
                                          for label,v in zip(('incoming_history','current_step'),values)})
            fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
            axes[0].plot(x,b['weak_minus_n_tau_n_Pa']/1e3,label='Total dʷ')
            axes[0].plot(x,values[0]/1e3,'--',label='Incoming βτ_old normal contribution')
            axes[0].set_ylabel('kPa');axes[0].legend()
            for field,label in zip(component_fields[1:],('Current bulk strain','Current crack-strain subtraction')):
                axes[1].plot(x,b[field],label=label)
            axes[1].plot(x,values[1],'--',label='Current step sum')
            axes[1].set(xlabel='Down-dip distance (km)',ylabel='Pa');axes[1].legend()
            fig.savefig(dest/f'history_split_{lo}_{hi}.png',dpi=180);plt.close(fig)
    # Raw QPs remain a scatter in (s,r), never a falsely connected 1-D curve.
    raw_baseline=None
    for step in sorted({int(initial['accepted_step'][0]),int(final['accepted_step'][0])}):
        raw=[]; identities=set()
        for path in sorted(args.output.glob(f'normal_qp_{step}_rank*.csv')):
            q=read(path)
            for row in q:
                identity=(str(row['cell']),int(row['qp']),int(row['fault']))
                if identity in identities:raise ValueError(f'Duplicate MPI sample {identity}')
                identities.add(identity)
            if len(q):raw.append(q)
        if not raw:continue
        q=np.concatenate(raw)
        q=np.sort(q,order=['cell','qp','fault'])
        if raw_baseline is None:
            raw_baseline=q
        else:
            for field in ('cell','qp','fault','segment','xi','x','y','r_m','rank'):
                np.testing.assert_array_equal(raw_baseline[field],q[field],err_msg=f'Raw matching changed: {field}')
            summary['matched_QP_weight_change']=dict(
                max_absolute=float(max(abs(q['work_weight']-raw_baseline['work_weight']))),
                max_relative=float(max(abs(q['work_weight']/raw_baseline['work_weight']-1))))
            summary['matched_QP_increments']={}
            for lo,hi in ((22,35),(60,90),(70,71),(79,80)):
                m=(q['down_dip_s_m']>=lo*1000)&(q['down_dip_s_m']<=hi*1000)
                w=q['work_weight'][m]
                summary['matched_QP_increments'][f'{lo}-{hi}']={f:dict(
                    max_Pa=float(max(abs(q[f][m]-raw_baseline[f][m]))),
                    work_RMS_Pa=float(np.sqrt(np.dot(w,(q[f][m]-raw_baseline[f][m])**2)/w.sum())))
                    for f in ('p','minus_n_tau_n','perturbation_normal')}
        explicit=-(q['n_x']**2*q['tau_xx']+2*q['n_x']*q['n_y']*q['tau_xy']+q['n_y']**2*q['tau_yy'])
        summary['raw'][step]=dict(samples=len(q),duplicate_owners=0,
            tensor_contraction_max_error_Pa=float(np.max(abs(explicit-q['minus_n_tau_n']))),
            normal_closure_max_Pa=float(np.max(abs(q['p']+q['minus_n_tau_n']+q['reference_normal']-q['total_normal']))))
        summary['raw'][step]['offset_windows']={f'{lo}-{hi}':matched_offset_zoom(q,final,dest,step,lo,hi)
                                                for lo,hi in ((70,71),(79,80))}
        for lo,hi in [(22,35),(60,90)]:
            m=(q['down_dip_s_m']>=lo*1000)&(q['down_dip_s_m']<=hi*1000)
            a=q[m];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            for ax,field in zip(axes.flat,['p','minus_n_tau_n','perturbation_normal','cell_diameter']):
                factor=1 if field=='cell_diameter' else 1e-6
                dots=ax.scatter(a['down_dip_s_m']/1000,a['r_m'],c=a[field]*factor,s=3,cmap='viridis',rasterized=True)
                ax.set(xlabel='Down-dip distance (km)',ylabel='Signed normal offset (m)',title=field)
                fig.colorbar(dots,ax=ax,label='m' if factor==1 else 'MPa')
            fig.suptitle(f'Unsmoothed owned production QPs, step {step}; geometry/refinement association, not causation')
            fig.savefig(dest/f'raw_{step}_{lo}_{hi}.png',dpi=180);plt.close(fig)
    if args.compare_control:
        path=args.compare_control/f'normal_profile_{int(initial["accepted_step"][0])}.csv'
        c=read(path)
        for field in ['accepted_step','time_s','vertex_id','down_dip_s_m']:np.testing.assert_array_equal(initial[field],c[field])
        summary['control']={key:float(np.max(abs(initial[key]-c[key]))) for key in
                            ['V_m_per_s','incoming_Theta_s','committed_Theta_s','cumulative_slip_m','I_h','existing_production_weak_normal_Pa']}
        # Compare history fingerprints separately from current constitutive
        # traction. Sums/squared sums are checks, not a proof of pointwise identity.
        step=int(initial['accepted_step'][0])
        check_name=f'normal_checks_{step}.csv'
        if (args.output/check_name).is_file() and (args.compare_control/check_name).is_file():
            left,right=read(args.output/check_name),read(args.compare_control/check_name)
            summary['control_history_fingerprints']={key:float(np.max(abs(left[key]-right[key])))
                                                     for key in left.dtype.names}
        else:
            summary['control_history_fingerprints']='missing: complete accepted diagnostic output required'
        for name in ['normal_restored_fault.csv']+[p.name for p in sorted(args.output.glob('normal_restored_particles_rank*.csv'))]:
            left,right=read(args.output/name),read(args.compare_control/name)
            if left.shape!=right.shape or left.dtype.names!=right.dtype.names:
                raise ValueError(f'Restored inventory layout differs: {name}')
            summary.setdefault('control_restored_inventory',{})[name]={
                key:float(np.max(abs(left[key]-right[key]))) for key in left.dtype.names}
    summary['global_load_closure_Pa_m']={}
    for path in sorted(args.output.glob('normal_totals_*.csv')):
        a=read(path)
        summary['global_load_closure_Pa_m'][path.stem]=list(map(float,a['owned_QP_integral_Pa_m']-a['assembled_row_sum_Pa_m']))
    # Selection made after accepted k-1 controls step k. Never pair k's
    # outgoing-state controller record with its already completed mechanics.
    selection=args.output/'timestep_selection.csv'
    if selection.is_file():
        rows=read(selection);summary['timestep_selection']=[]
        for a,info in records:
            previous=rows[rows['accepted_step']==info['step']-1]
            if not len(previous):
                summary['timestep_selection'].append(dict(step=info['step'],status='preceding selection unavailable'))
                continue
            row=previous[-1]
            fields=['convection','fault','state_predictor','ceiling','first_cap','growth_cap']
            bounds={key:float(row[key]) for key in fields}
            selected=float(row['selected'])
            controls=[key for key,value in bounds.items() if abs(value-selected)<=1e-12*max(1,abs(selected))]
            summary['timestep_selection'].append(dict(step=info['step'],bounds_s=bounds,selected_s=selected,
                controlling_candidates=controls,termination_reduced=bool(row['termination_reduced']),
                predictor_max_node=int(a['vertex_id'][np.argmax(a['predicted_weighted_log_change'])])))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
