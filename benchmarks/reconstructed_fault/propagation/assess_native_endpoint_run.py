#!/usr/bin/env python3
"""Compare actual accepted native fields with the identical prototype schedule."""
import argparse
import re
from pathlib import Path
import numpy as np
import native_endpoint as native
import asymmetric_branch as old

E=native.E


def assess(case, live=False, prototype_directory=None):
    provenance=old.read(E/f'{case}-start.json' if live else E/f'{case}-run.json')
    output=old.ROOT/'build-propagation-integration/runs'/f'output-native-endpoint-{case}'
    metrics=np.genfromtxt(output/'metrics.csv',delimiter=',',names=True)
    prototype=prototype_directory or E/'prefix-2-fine-1024-onset'
    f=dict(np.load(prototype/'fields.npz'));h=old.read(prototype/'history.json')
    phase0=np.genfromtxt(output/'phase-0.csv',delimiter=',',names=True)
    reference0=np.interp(phase0['y'],f['y'],f['phase'][0])
    assert max(abs(phase0['phase']-reference0))<1e-15
    rows=[];phases=[];surfaces=[];stress_profiles=[]
    z,w=np.polynomial.legendre.leggauss(12);previous_native_slip=np.zeros(65)
    for row in metrics:
        step=int(row['step']);time=float(row['time'])
        phase=np.genfromtxt(output/f'phase-{step}.csv',delimiter=',',names=True)
        surface=np.genfromtxt(output/f'surface-{step}.csv',delimiter=',',names=True)
        j=np.flatnonzero(f['time']==time)[0]
        p=f['phase'][j];reference=np.interp(phase['y'],f['y'],p)
        delta=phase['phase']-reference
        worst=int(np.argmax(abs(delta)))
        slices=[]
        for x in np.unique(phase['x']):
            section=phase[phase['x']==x];section=np.sort(section,order='y');slices.append(section['phase'])
        slices=np.array(slices);mean=slices.mean(axis=0);y=f['y'];dy=np.diff(y)
        centroid=float(np.sum(dy*((2*y[:-1]+y[1:])*mean[:-1]+(y[:-1]+2*y[1:])*mean[1:])/6)/np.trapezoid(mean,y))
        tangential=float(np.max(np.ptp(slices,axis=0)))
        if step:
            a=h[j-1];previous_time=metrics[step-1]['time']
            j0=int(np.flatnonzero(f['time']==previous_time)[0]);reference_previous=h[j0-1]['slip'] if j0 else 0.
            common_rate=(a['slip']-reference_previous)/(time-previous_time)
            rate_native=(surface['slip']-previous_native_slip)/(time-previous_time)
            previous_native_slip=surface['slip'].copy()
            result=dict(step=step,time=time,phase=float(max(abs(delta))),
                phase_worst_location=[float(phase['x'][worst]),float(phase['y'][worst])],
                tangential_phase_variation=tangential,phase_centroid=centroid,
                centroid_difference=centroid-a['centroid']['phase'],
                q=float(max(abs(surface['q']-a['q']))),slip=float(max(abs(surface['slip']-a['slip']))),
                interval_rate=float(max(abs(rate_native-common_rate))),
                tangential_q_variation=float(np.ptp(surface['q'])),
                tangential_slip_variation=float(np.ptp(surface['slip'])),
                normalization_relative_error=float(max(abs(surface['I']/a['I']-1))),
                bulk_storage=abs(row['bulk']-a['bulk']),cohesive_storage=abs(row['cohesive']-a['cohesive']),
                fracture_storage=abs(row['fracture']-a['fracture']))
            stress_file=output/f'stress-{step}.csv'
            if stress_file.exists():
                data=np.genfromtxt(stress_file,delimiter=',',names=True)
                rms=[];local=[];normal=[]
                for x in np.unique(data['x']):
                    section=data[data['x']==x];section=np.sort(section,order='y')
                    yy=section['y'];cell=np.minimum(511,np.floor(512*yy).astype(int));s=512*yy-cell
                    R=f['R'][j];derivative=512*((-3+4*s)*R[2*cell]+(4-8*s)*R[2*cell+1]+(-1+4*s)*R[2*cell+2])
                    pp=(1-s)*p[cell]+s*p[cell+1];chi=old.core.h(pp)/a['I']
                    reference_stress=1e6*derivative-f['Z'][j]*chi
                    d=section['tau_xy']-reference_stress;weights=np.tile(w/1024,512)
                    # Positive embedded J2 of the difference reduces to d^2
                    # in pure shear, preserving the retained stress scale.
                    # Include native normal modes rather than hiding them.
                    xx,yy=section['tau_xx'],section['tau_yy']
                    squared=d*d+.5*(xx*xx+yy*yy+(xx+yy)**2)
                    rms.append(float(np.sqrt(weights@squared)))
                    local.append(float(np.sqrt((weights*chi)@squared/(weights@chi))))
                    normal.append(float(max(abs(section['tau_xx']))+max(abs(section['tau_yy']))))
                result.update(whole_stress=max(rms),localized_stress=max(local),normal_stress_max=max(normal))
                stress_profiles.append((time,data))
            rows.append(result)
        phases.append((time,y,mean,slices.min(axis=0),slices.max(axis=0)));surfaces.append((time,surface))
    summary=dict(provenance=provenance,live_snapshot=live,
                 prototype_directory=str(prototype),
                 completed=bool(not live and provenance['returncode']==0 and metrics[-1]['time']==.53125),
                 initialization_phase_error=float(max(abs(phase0['phase']-reference0))),
                 maximum_phase_growth=float(max(phases[-1][4]-f['phase'][0])),
                 maximum_healing=float(max(metrics['healing'])),
                 maximum_FE_recurrence=float(max(metrics['FE_recurrence'])),
                 maximum={key:max(abs(r[key]) if key=='centroid_difference' else r[key]
                                  for r in rows if key in r)
                          for key in set().union(*(r.keys() for r in rows))
                          if key not in ['step','time','phase_worst_location']},
                 rows=rows,global_reflection=False,aligned=False,
                 stress_comparison_times=[r['time'] for r in rows if 'whole_stress' in r],
                 native_stress_sampling='12 off-assembly normal points in three tangential sections; RMS uses positive embedded J2 of the full deviatoric stress difference')
    maximum=summary['maximum'];finest=old.read(E/'dt-1-2048/history.json')
    targets=dict(phase=.001,q=1.,slip=.001*finest[-1]['slip'],
                 interval_rate=.005*max(a['V'] for a in finest),
                 whole_stress=.01*1332.521975,localized_stress=.01*1332.521975,
                 bulk_storage=.01*.887807407)
    summary['targets']=targets
    summary['passes']={key:bool(maximum.get(key,float('inf'))<=value) for key,value in targets.items()}
    summary['accepted_interval_targets_pass']=all(summary['passes'].values())
    summary['resolved_phase_growth']=summary['maximum_phase_growth']>1e-6
    core=int(np.argmin(abs(f['y']-.5)))
    summary['core_phase_growth']=float(phases[-1][4][core]-f['phase'][0,core])
    summary['resolved_core_growth']=summary['core_phase_growth']>1e-6
    summary['first_core_growth_time']=next((float(t) for t,y,mean,lo,hi in phases
        if hi[core]-f['phase'][0,core]>1e-6),None)
    if summary['completed']:
        assert summary['resolved_core_growth'],'A completed onset comparison must include accepted core growth.'
    summary['qualification_interval']=[0.,float(metrics[-1]['time'])]
    # Compare the independently re-evaluated, published-state KKT residual
    # with each solve's actual fixed entry target. No new tolerance is added.
    entry_targets={};step=None
    for line in (E/f'{case}.log').read_text().splitlines():
        match=re.search(r'\*\*\* Timestep (\d+):',line)
        if match:step=int(match.group(1))
        match=re.search(r'Endpoint phase target: initial=([^,]+), target=(\S+)',line)
        if match:entry_targets[step]=float(match.group(2))*64.00000000008
    summary['published_KKT']=[dict(step=int(row['step']),residual=float(row['KKT']),
        target=entry_targets[int(row['step'])],
        passes=bool(row['KKT']<=entry_targets[int(row['step'])])) for row in metrics if row['step']]
    if (output/'path-energy.csv').exists():
        ledger=np.atleast_1d(np.genfromtxt(output/'path-energy.csv',delimiter=',',names=True))
        summary['energy_comparison_times']=ledger['time'].tolist()
        summary['energy_scope']='Every recorded physical step; full-prefix fixtures also include pre-growth loading. Absolute work channels integrate absolute intermediate residual power, separately from absolute signed step work.'
        summary['growing_path_energy']={key:dict(signed=float(sum(ledger[key])),absolute=float(sum(abs(ledger[key]))),
            maximum=float(max(abs(ledger[key])))) for key in ledger.dtype.names if key!='time'}
        reference_ledger=old.read(prototype/'energy.json')
        summary['matching_growing_prototype_energy']={key:dict(signed=float(sum(a[key] for a in reference_ledger if a['time'] in ledger['time'])),
            absolute=float(sum(abs(a[key]) for a in reference_ledger if a['time'] in ledger['time'])))
            for key in ['external_work','bulk_work','lifting_work',
                        'bulk_change','cohesive_change','fracture_change',
                        'bulk_viscous','cohesive_viscous','friction',
                        'mechanical_power','friction_residual_power','phase_power',
                        'mechanical_power_absolute','friction_residual_power_absolute',
                        'phase_power_absolute','bulk_profile_power','normalization_power',
                        'fracture_power','signed_profile_cost','external_remainder']}
        # Keep the observer's stated sign convention. Compare integration
        # independently rather than masking a path error by energy cancellation.
        refined=output/'path-energy-refined.csv'
        if refined.exists():
            fine=np.atleast_1d(np.genfromtxt(refined,delimiter=',',names=True))
            summary['time_quadrature_uncertainty']={}
            for row in fine:
                coarse=ledger[ledger['time']==row['time']]
                assert len(coarse)==1
                summary['time_quadrature_uncertainty'][str(float(row['time']))]={
                    key:float(abs(row[key]-coarse[0][key])) for key in fine.dtype.names if key!='time'}
        summary['external_remainder_conventions']={
            'native':'storage change - boundary reaction work + losses',
            'prototype':'boundary reaction work - storage change - losses'}
    old.write(E/f'{case}-comparison.json',summary)
    plot(case,metrics,phases,surfaces,stress_profiles,f,h)
    print(old.json.dumps({k:summary[k] for k in ['completed','maximum_phase_growth','maximum','maximum_healing','passes']}))


def plot(case,metrics,phases,surfaces,stress_profiles,f,h):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    last=metrics[-1]['time'];j=np.flatnonzero(f['time']==last)[0]
    tt=f['time'][:j+1]
    axes[0,0].plot(tt,np.max(f['phase'][:j+1]-f['phase'][0],axis=1),label='matching prototype')
    axes[0,0].plot(metrics['time'],[max(p[4]-f['phase'][0]) for p in phases],'o',label='fresh native full field')
    axes[0,0].set(xlim=(.527 if last>=.527 else max(0.,last-.05),last),xlabel='t (s)',ylabel='maximum damage growth')
    axes[0,1].plot([a['slip'] for a in h[:j]],[a['q'] for a in h[:j]],label='prototype')
    for index,label in [(0,'native endpoint'),(32,'native center')]:
        axes[0,1].plot([s['slip'][index] for _,s in surfaces],[s['q'][index] for _,s in surfaces],'o',label=label)
    onset=[a for a in h[:j] if a['time']>=.52734375]
    if onset:
        axes[0,1].set(xlim=(onset[0]['slip']-.00005,onset[-1]['slip']+.00005),
                      ylim=(min(a['q'] for a in onset)-.2,max(a['q'] for a in onset)+.2),
                      title='Growing-phase interval')
    axes[0,1].set(xlabel='cumulative slip (m)',ylabel='q (Pa)')
    for t,y,p,low,high in phases[-3:]:
        line=axes[0,2].plot(y,p-f['phase'][0],label=f'native x mean {t:g} s')[0]
        axes[0,2].fill_between(y,low-f['phase'][0],high-f['phase'][0],color=line.get_color(),alpha=.2)
    axes[0,2].plot(f['y'],f['phase'][j]-f['phase'][0],'--',label='prototype final')
    axes[0,2].set(xlabel='y (m)',ylabel='phase change')
    if stress_profiles:
        t,data=stress_profiles[-1]
        for x in np.unique(data['x']):
            d=np.sort(data[data['x']==x],order='y');axes[1,0].plot(d['y'],d['tau_xy'],label=f'native x={x:g}')
        axes[1,0].set(xlabel='y (m)',ylabel='off-quadrature shear stress (Pa)')
    window=metrics['time']>=.52734375
    if not np.any(window):
        window=metrics['time']>=metrics['time'][max(0,len(metrics)-3)]
    for key,style in [('bulk','-'),('cohesive','--'),('fracture',':')]:
        axes[1,1].plot(metrics['time'][window],metrics[key][window]-metrics[key][window][0],
                       style,marker='o',markersize=3,label=f'native {key}')
    axes[1,1].set(xlabel='t (s)',ylabel='storage change (J/m)',
                  title=f"Relative to {metrics['time'][window][0]:g} s")
    data=old.read(E/f'{case}-comparison.json')
    for key,style in [('mechanical_residual_work','-'),('friction_residual_work','--'),('phase_residual_work',':')]:
        if 'growing_path_energy' in data:
            output=old.ROOT/'build-propagation-integration/runs'/f'output-native-endpoint-{case}'
            en=np.atleast_1d(np.genfromtxt(output/'path-energy.csv',delimiter=',',names=True));axes[1,2].plot(en['time'],np.cumsum(en[key]),style,marker='o',markersize=3,label=key)
    axes[1,2].set(xlabel='t (s)',ylabel='separate signed residual works (J/m)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
    fig.savefig(E/f'{case}-comparison.png',dpi=160);fig.savefig(E/f'{case}-comparison.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case')
    parser.add_argument('--live',action='store_true',help='Read accepted outputs of a running case; never report a completed run.')
    parser.add_argument('--prototype',type=Path,help='Matching schedule trajectory directory.')
    args=parser.parse_args();assess(args.case,args.live,args.prototype)
