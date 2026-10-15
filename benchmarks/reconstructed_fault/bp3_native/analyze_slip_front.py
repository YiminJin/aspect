#!/usr/bin/env python3
"""Bounded local acceleration/front assessment using the existing BP3 metrics."""
import csv
import json
from pathlib import Path

import numpy as np

from analyze_section7b import FIELDS, read, rms, oscillations
from analyze_section7cd import metric, normal_diagnostics
from analyze_managed_particles import check

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'build-bp3-native/slip-front'
VP, DC = 1e-9, .008


def assess(directory):
    journal = read(directory/'coupled-checks.csv')
    moving = read(directory/'moving-particles.csv')
    accepted = read(directory/'accepted_steps.csv')
    profiles = [read(directory/f'coupled-profile-{int(step)}.csv') for step in journal['step']]
    initial = read(directory/'diagnostic-initial-state.csv')
    s = profiles[0]['s_m']
    expected_theta = DC/VP*np.exp(np.log(10)*np.exp(-((s/s[-1]-.35)/.25)**2))
    assert np.allclose(initial['theta_s'], expected_theta, rtol=4e-15, atol=0)
    assert np.array_equal(initial['theta_s'], profiles[0]['theta_s'])
    a = .010+.015*initial['fraction']
    mu = a*np.arcsinh(VP/(2e-6)*np.exp((.6+.015*np.log(1e-6*expected_theta/DC))/a))
    assert np.max(np.abs(initial['background_shear_Pa']-(50e6*mu+4624440*VP))) < 2e-8
    assert np.array_equal(initial['fraction'], np.clip(s/3000., 0., 1.))
    assert np.all(profiles[0]['slip_m']==0.)
    history = []
    for index, p in enumerate(profiles):
        assert np.array_equal(s, p['s_m'])
        assert np.array_equal(p['I_h_m'], profiles[0]['I_h_m'])
        # These are M^{-1} times the assembled constant-normal load, not the
        # underlying exactly fixed nodal property. Allow mass-solve roundoff.
        assert np.max(np.abs(p['background_Pa']-50e6)) < 64*np.finfo(float).eps*50e6
        assert np.allclose(p['Omega'], p['V_m_s']*p['theta_s']/DC, rtol=4e-15)
        rough = {key: oscillations(s, p[key], VP if key=='V_m_s' else 50e6)
                 for key in ['V_m_s', 'raw_normal_Pa', 'filtered_normal_Pa', 'pressure_Pa', 'deviatoric_Pa']}
        fronts = {}
        for factor in [2., 5., 10.]:
            v = p['V_m_s']/VP
            crossings = [float(s[j]+(s[j+1]-s[j])*(factor-v[j])/(v[j+1]-v[j]))
                         for j in range(len(s)-1) if (v[j]-factor)*(v[j+1]-factor)<0.]
            fronts[str(factor)] = dict(crossings_s_m=crossings,
                                      nodes_above=int(np.sum(v>=factor)))
        predicted_log_change = actual_log_change = 0.
        if index:
            old = profiles[index-1]
            dt = journal['dt_s'][index]
            x = old['V_m_s']*dt/DC
            predicted = old['theta_s']*np.exp(-x)+DC/old['V_m_s']*(-np.expm1(-x))
            predicted_log_change = float(np.max(np.abs(np.log(predicted/old['theta_s']))))
            actual_log_change = float(np.max(np.abs(np.log(p['theta_s']/old['theta_s']))))
        history.append(dict(step=int(journal['step'][index]), time_s=float(journal['time_s'][index]),
                            dt_s=float(journal['dt_s'][index]),
                            max_V_m_s=float(np.max(p['V_m_s'])),
                            peak_s_m=float(s[np.argmax(p['V_m_s'])]),
                            ranges={key:[float(np.min(p[key])),float(np.max(p[key]))] for key in FIELDS},
                            roughness=rough, fronts=fronts,
                            predicted_log_state_change=predicted_log_change,
                            actual_log_state_change=actual_log_change,
                            crossings=int(moving['crossings'][index]), births=int(moving['births'][index]),
                            removals=int(moving['removed'][index]),
                            residual=float(accepted['normalized_nonlinear_residual'][index])))
    # Disturbance correlation is descriptive, not a causal attribution. A
    # constant event series has no defined correlation, rather than a zero one.
    dt = journal['dt_s'][1:]
    predictors = dict(crossings=moving['crossings'][1:], births=moving['births'][1:], removals=moving['removed'][1:],
                      dt_s=dt, log_dt_change=np.r_[0., np.diff(np.log(dt))])
    correlations = {}
    for region in ['lower_200m', 'interior']:
        for key in ['V_m_s', 'raw_normal_Pa', 'filtered_normal_Pa']:
            growth = np.diff([h['roughness'][key][region] for h in history])
            correlations[region+'/'+key] = {
                name: float(np.corrcoef(growth, values)[0,1]) if np.std(values)>0 and np.std(growth)>0 else None
                for name, values in predictors.items()}
    arrivals = {}
    for factor in [2., 5., 10.]:
        times = []
        for j in range(len(s)):
            v = np.array([p['V_m_s'][j]/VP for p in profiles])
            hits = np.flatnonzero(v>=factor)
            if len(hits):
                k = hits[0]
                t = journal['time_s'][k]
                if k:
                    t = journal['time_s'][k-1]+(t-journal['time_s'][k-1])*(factor-v[k-1])/(v[k]-v[k-1])
                times.append(dict(s_m=float(s[j]), time_s=float(t), accepted_bracket_s=float(journal['dt_s'][k])))
        arrivals[str(factor)] = times
    # Existing cuts include both one-sided values at cell faces. Separate
    # pressure continuity from constitutive normal-stress trace jumps.
    face_jumps = {}
    for step in sorted(set([0, int(journal['step'][np.argmax([h['max_V_m_s'] for h in history])]),
                            int(journal['step'][-1])])):
        groups = {}
        with (directory/f'bulk-normal-cuts-{step}.csv').open() as stream:
            for row in csv.DictReader(stream):
                groups.setdefault((int(row['station']),round(float(row['r_m']),8)),[]).append(row)
        face_jumps[str(step)] = {}
        for station in [1,2,3]:
            faces = [rows for (j,r),rows in groups.items() if j==station and len({q['cell'] for q in rows})>1]
            face_jumps[str(step)][str(station)] = {
                key:max(max(float(q[key]) for q in rows)-min(float(q[key]) for q in rows) for rows in faces)
                for key in ['pressure_Pa','deviatoric_Pa','normal_Pa']}
    return dict(checks=check(directory), history=history, arrivals=arrivals, correlations=correlations,
                face_jumps=face_jumps,
                projected_background_normal_roundoff_Pa=float(max(np.max(np.abs(p['background_Pa']-50e6)) for p in profiles)),
                maximum_predicted_log_state_change=max(h['predicted_log_state_change'] for h in history),
                maximum_actual_log_state_change=max(h['actual_log_state_change'] for h in history),
                peak=max(history,key=lambda h:h['max_V_m_s']),
                initial_background_shear_range_Pa=[float(min(initial['background_shear_Pa'])),float(max(initial['background_shear_Pa']))],
                final_normal=normal_diagnostics(profiles[-1])), profiles


def main():
    result, all_profiles = {}, {}
    for case in ['primary', 'comparison']:
        directory = BASE/'runs'/case
        if (directory/'coupled-checks.csv').exists():
            result[case], all_profiles[case] = assess(directory)
    if len(result)==2:
        for filename in ['initial-mesh.csv','initial-phase.csv','mechanical-quadrature.csv',
                         'initial-particle-identity.csv','diagnostic-initial-state.csv']:
            assert (BASE/'runs/primary'/filename).read_bytes()==(BASE/'runs/comparison'/filename).read_bytes(),filename
        left, right = [all_profiles[c] for c in ['primary', 'comparison']]
        # Adaptive schedules differ. Interpolate recorded fields in time only,
        # and retain brackets plus a linear/quadratic sampling indicator. This
        # is not an interpolation error bound or a convergence estimate.
        def at_time(profiles, time):
            times = np.array([p['time_s'][0] for p in profiles])
            k = min(np.searchsorted(times,time), len(times)-1)
            if times[k]==time:
                return profiles[k], 0., {key:0. for key in FIELDS}
            assert k>0 and times[k-1]<time<times[k]
            fraction = (time-times[k-1])/(times[k]-times[k-1])
            value = {key:(1-fraction)*profiles[k-1][key]+fraction*profiles[k][key] for key in FIELDS}
            value['s_m'] = profiles[k]['s_m']
            selected = np.arange(max(0,min(k-1,len(times)-3)),max(0,min(k-1,len(times)-3))+3)
            indicator = {}
            for key in FIELDS:
                quadratic = np.zeros_like(value[key])
                for j in selected:
                    coefficient = np.prod([(time-times[i])/(times[j]-times[i]) for i in selected if i!=j])
                    quadratic += coefficient*profiles[j][key]
                indicator[key] = float(np.max(np.abs(value[key]-quadratic)))
            return value, float(times[k]-times[k-1]), indicator
        end = min(left[-1]['time_s'][0],right[-1]['time_s'][0])
        a, _, _ = at_time(left,end)
        b, _, _ = at_time(right,end)
        scales = dict(V_m_s=VP, theta_s=DC/VP, Omega=1.,
                      slip_m=max(DC,np.max(np.abs(a['slip_m']))))
        result['comparison_endpoint_s'] = float(end)
        result['final_comparison'] = {
            key:metric(a['s_m'], a[key], np.interp(a['s_m'],b['s_m'],b[key]),scales.get(key,50e6)) for key in FIELDS}
        result['initial_comparison'] = {
            key:metric(left[0]['s_m'], left[0][key], np.interp(left[0]['s_m'],right[0]['s_m'],right[0][key]),scales.get(key,50e6)) for key in FIELDS}
        result['common_times'] = []
        for time in sorted(set([0.,end]+[p['time_s'][0] for p in left if p['time_s'][0]<=end])):
            a, bracket_a, indicator_a = at_time(left,time)
            b, bracket_b, indicator_b = at_time(right,time)
            scale = dict(V_m_s=max(VP,np.max(a['V_m_s'])), theta_s=max(np.max(a['theta_s']),np.max(b['theta_s'])),
                         Omega=max(1.,np.max(a['Omega'])),slip_m=max(DC,np.max(np.abs(a['slip_m']))))
            result['common_times'].append(dict(time_s=float(time), brackets_s=[bracket_a,bracket_b],
                sampling_indicator_abs=[indicator_a,indicator_b],
                fields={key:metric(a['s_m'],a[key],b[key],scale.get(key,50e6)) for key in FIELDS},
                max_abs_log_V_ratio=float(np.max(np.abs(np.log(b['V_m_s']/a['V_m_s'])))),
                max_abs_log_theta_ratio=float(np.max(np.abs(np.log(b['theta_s']/a['theta_s']))))))
    records = ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'
    result['runs'] = {p.stem:json.loads(p.read_text()) for p in records.glob('bp3-slip-front-*.json')}
    result['total_simulation_seconds'] = sum(r['seconds'] for r in result['runs'].values())
    (BASE/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    endpoint = []
    disturbances = []
    for case, profiles in all_profiles.items():
        selected = sorted(set([0,int(np.argmax([p['V_m_s'].max() for p in profiles])),len(profiles)-1]))
        for step in selected:
            p = profiles[step]
            for i in np.flatnonzero(p['s_m']>=p['s_m'][-1]-200):
                endpoint.append(dict(case=case,step=step,time_s=float(p['time_s'][0]),s_m=float(p['s_m'][i]),
                                     **{key:float(p[key][i]) for key in FIELDS}))
        for h in result[case]['history']:
            row = {key:h[key] for key in ['step','time_s','dt_s','max_V_m_s','crossings','births','removals',
                                          'predicted_log_state_change','actual_log_state_change']}
            row['case'] = case
            row.update({key+'/'+region:h['roughness'][key][region] for key in ['V_m_s','raw_normal_Pa',
                        'filtered_normal_Pa','pressure_Pa','deviatoric_Pa'] for region in ['lower_200m','interior']})
            disturbances.append(row)
    for name, rows in [('endpoint-profiles.csv',endpoint),('disturbances.csv',disturbances)]:
        with (BASE/name).open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Compact scientific artifacts: accepted-step fields, not rendered snapshots.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,3,figsize=(12,9))
    for key, ax in zip(FIELDS,axes.flat):
        for case, profiles in all_profiles.items():
            for time_index,time in enumerate([0.,8e6,12e6,end]):
                if time<=end:
                    p,_,_ = at_time(profiles,time)
                    ax.plot(p['s_m'],p[key],label=f'{case} t={time:.4g}s',
                            color=f'C{time_index}',linestyle='-' if case=='primary' else '--')
        ax.set_title(key); ax.set_xlabel('Down-dip distance [m]'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(BASE/'fields.pdf'); plt.close(fig)
    fig, axes = plt.subplots(len(all_profiles),2,figsize=(10,4*len(all_profiles)),squeeze=False)
    maximum = max(np.max([p['V_m_s']/VP for p in profiles]) for profiles in all_profiles.values())
    for row,(case,profiles) in enumerate(all_profiles.items()):
        t=np.array([p['time_s'][0] for p in profiles]); s=profiles[0]['s_m']
        heat = axes[row,0].pcolormesh(s,t/1e6,np.log10([p['V_m_s']/VP for p in profiles]),
                                     shading='nearest',vmin=-.1,vmax=np.log10(maximum))
        fig.colorbar(heat,ax=axes[row,0],label='log10(V/Vp)')
        contours = axes[row,0].contour(s,t/1e6,np.array([p['V_m_s']/VP for p in profiles]),
                                       levels=[2,5,10],colors='white',linewidths=.5)
        axes[row,0].clabel(contours,fmt='%g Vp',fontsize=7)
        axes[row,0].set_title(case+' log10(V/Vp)'); axes[row,0].set_ylabel('Time [1e6 s]')
        axes[row,0].set_xlabel('Down-dip distance [m]')
        axes[row,1].plot(t[1:]/1e6,np.diff(t)/1e6)
        axes[row,1].set_title(case+' accepted dt'); axes[row,1].set_xlabel('Time [1e6 s]'); axes[row,1].set_ylabel('dt [1e6 s]')
    fig.tight_layout(); fig.savefig(BASE/'front-and-timestep.pdf'); plt.close(fig)


if __name__=='__main__':
    main()
