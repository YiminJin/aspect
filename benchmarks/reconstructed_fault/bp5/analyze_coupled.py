"""Common-time, incoming-history comparisons; no change to simulation data."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_short import OUT, BP3
from analyze_mechanical_modes import table
from analyze_mechanical_width import records
from analyze_length_coupled import raw_qps, rms
from run_mechanical_width import VirtualProfile, SN, NORMAL
from check_first_cycle_restart import convergence, difference

REGIONS=[('whole',0,100000/SN),('top',0,2000),('transition',13000,20000),
         ('interior',30000,35000),('deep',40000,80000),('bottom',100000/SN-2000,100000/SN),
         ('top_patch_edge',4000,6000),('probe_patch_start',6000,8000),
         ('probe_patch_end',25000,27000),('bottom_patch_edge',100000/SN-6000,100000/SN-4000)]


def check(label):
    run=OUT/label
    accepted=np.atleast_1d(table(run/'accepted_steps.csv'))
    manifest=json.loads((run/'launch.json').read_text())
    mesh=manifest['mesh']
    profile=VirtualProfile(100,24.4140625 if mesh=='candidate' else 12.20703125)
    bg=np.loadtxt(OUT/'fixtures'/mesh/'prestress.txt',skiprows=1)
    assert np.all(accepted['fresh_linear_checks_passed']==1)
    assert np.max(accepted['Theta_relative_error'])<1e-12
    assert np.all(accepted['free']+accepted['lower_active']==1156)
    log=run/'run.log'
    execution=json.loads((run/'execution.json').read_text())
    if not execution['expected_completion']:
        # Audit only genuinely accepted states; retain the complete failed log
        # separately. A passing prefix never qualifies the interrupted run.
        text=log.read_text()
        marker=f"*** Timestep {int(accepted[-1]['step'])+1}:"
        assert marker in text
        log=run/'accepted_prefix.log'
        log.write_text(text[:text.index(marker)])
    conv=convergence(log)
    results=[]
    surface=np.atleast_1d(table(run/'surface.csv'))
    surface_x=(100000-surface['y'])/SN
    order=np.argsort(surface_x)
    for state in accepted:
        step=int(state['step'])
        if not (run/f'state_work_{step}.csv').exists(): continue  # Restart metadata prefix is not a new solve.
        data=np.atleast_1d(table(run/f'state_work_{step}.csv'))
        weak=np.atleast_1d(table(run/f'work_weak_{step}.csv'))
        assert np.all(data['Theta_in']>0) and np.all(data['Theta_out']>0)
        x=data['V']*float(state['dt'])/.1
        expected=data['Theta_in'] if step==0 else data['Theta_in']*np.exp(-x)-.1/data['V']*np.expm1(-x)
        assert max(abs(expected/data['Theta_out']-1))<1e-12
        if (run/f'state_work_{step-1}.csv').exists():
            previous=np.atleast_1d(table(run/f'state_work_{step-1}.csv'))
            np.testing.assert_array_equal(data['Theta_in'],previous['Theta_out'])
        # Separate O(1e9 Pa m) weak terms lose digits when recombined. Compare
        # in Pa using the actual row measure, as in the existing work observer.
        recombination=max(abs(data['weak_residual']-(data['weak_q']-data['weak_friction']-data['weak_damping']))/weak['weight'])
        assert recombination<1e-5
        raw=np.concatenate([raw_qps(p) for p in sorted(run.glob(f'work_qp_{step}_rank*.csv'))])
        assert len(set(zip(raw['cell'],raw['qp'])))==len(raw)
        active=raw['source_active']==1
        assert np.all(abs(raw['r'][~active])>=profile.r[-1]-1e-8)
        h=profile.m*raw['phi']*(1+raw['phi'])/(1-raw['phi'])**2
        assert max(abs(raw['chi'][active]-h[active]/raw['Ih'][active]))<1e-12
        full_weight=raw['JxW']*h/np.interp(raw['xd'],surface_x[order],surface['Ih'][order])
        j=raw['segment'].astype(int);xi=raw['xi']
        def interp(c): return (1-xi)*bg[j,c]+xi*bg[j+1,c]
        background=interp(2)-interp(4)-interp(5)/interp(6)
        tauN=.75*raw['tau_xx']+.25*raw['tau_yy']-SN*raw['tau_xy']
        shear=-.5*SN*raw['tau_xx']+.5*SN*raw['tau_yy']-.5*raw['tau_xy']
        assert max(abs(raw['sigma_n']-(50e6+raw['p']-tauN)))<1e-5
        assert max(abs(raw['q']-(background+shear)))<1e-5
        weight=raw['JxW']*raw['chi']
        reconstructed=np.zeros((len(weak),3))
        for end,basis in [(0,1-xi),(1,xi)]:
            for c,value in enumerate([np.ones(len(raw)),raw['q'],raw['sigma_n']]):
                np.add.at(reconstructed[:,c],j+end,weight*basis*value)
        endpoints=(weak['xd']<1500)|(weak['xd']>100000/SN-1500)
        mass_error=max(abs(reconstructed[endpoints,0]/weak['weight'][endpoints]-1))
        traction_error=max(np.max(abs(reconstructed[endpoints,1]-weak['q'][endpoints])/weak['weight'][endpoints]),
                           np.max(abs(reconstructed[endpoints,2]-weak['sigma'][endpoints])/weak['weight'][endpoints]))
        assert mass_error<1e-11 and traction_error<1e-5
        result=dict(step=step,time=float(state['time']),dt=float(state['dt']),
             state_error=float(max(abs(expected/data['Theta_out']-1))),Theta_min=float(min(data['Theta_out'])),
             V_min=float(min(data['V'])),V_max=float(max(data['V'])),max_Vdt_Dc=float(max(x)) if step else 0.,
             lower_active=int(state['lower_active']),free=int(state['free']),surface_RMS_Pa=float(state['surface_RMS_Pa']),
             endpoint_mass_error=float(mass_error),endpoint_traction_error_Pa=float(traction_error),
             recombined_weak_terms_error_Pa=float(recombination),
             omitted_FE_tail_fraction_in_exported_windows=float(sum(full_weight[~active])/sum(full_weight)),
             top_continuation_QPs=int(sum(active & (raw['xd']<0))),
             bottom_continuation_QPs=int(sum(active & (raw['xd']>100000/SN))),
             final_convergence=conv[step])
        result['raw_windows']={}
        for name,lo,hi in REGIONS[1:]:
            mask=(raw['xd']>=lo)&(raw['xd']<=hi)
            if name=='top': mask=raw['xd']<=hi
            if name=='bottom': mask=raw['xd']>=lo
            if not np.any(mask): continue
            result['raw_windows'][name]={k:[float(min(raw[k][mask])),float(max(raw[k][mask]))]
                for k in ('q','p','tau_xx','tau_yy','tau_xy','sigma_n')}
            result['raw_windows'][name]['strain_mismatch_rms']=rms(raw['elastic_norm'][mask],weight[mask])
            result['raw_windows'][name]['continuation_QPs']=int(np.count_nonzero(mask&((raw['xd']<0)|(raw['xd']>100000/SN))))
        results.append(result)
    (run/'checks.json').write_text(json.dumps(results,indent=2)+'\n')
    return accepted,results


def compare(a,b,kind='spatial'):
    ca,checks_a=check(a);cb,checks_b=check(b)
    pairs=[]
    for sa in ca:
        matches=cb[abs(cb['time']-sa['time'])<1e-6]
        if len(matches)==1 and (OUT/b/f"state_work_{int(matches[0]['step'])}.csv").exists():
            pairs.append((int(sa['step']),int(matches[0]['step']),float(sa['time'])))
    assert pairs
    initial={};rows=[];profiles=[];restart={}
    for ka,kb,t in pairs:
        da=np.atleast_1d(table(OUT/a/f'state_work_{ka}.csv'))
        db=np.atleast_1d(table(OUT/b/f'state_work_{kb}.csv'))
        wa=np.atleast_1d(table(OUT/a/f'work_weak_{ka}.csv'))
        wb=np.atleast_1d(table(OUT/b/f'work_weak_{kb}.csv'))
        np.testing.assert_array_equal(da['xd'],db['xd'])
        x=da['xd'];w=(wa['weight']+wb['weight'])/2
        fields={name:(da[name],db[name]) for name in ('V','Theta_in','Theta_out','slip')}
        fields.update(q=(wa['q']/wa['weight'],wb['q']/wb['weight']),
                      sigma=(wa['sigma']/wa['weight'],wb['sigma']/wb['weight']))
        if kind=='restart':
            restart[str(t)]={name:difference(v,z) for name,(v,z) in fields.items()}
        # This is only a fixed-state sensitivity scale. Native weak friction is
        # taken from production, never replaced by mu of nodal plotted fields.
        direct=.004+.036*np.clip((x-15000)/3000,0,1)
        load_scale=abs(fields['sigma'][1])*direct+32038120320/(2*3464)*db['V']
        for name,(v,z) in fields.items():
            if name not in initial:
                if t==0: initial[name]=(v.copy(),z.copy())
                else:
                    start=np.atleast_1d(table(OUT/a/'state_work_2.csv'))
                    start_weak=np.atleast_1d(table(OUT/a/'work_weak_2.csv'))
                    value=start_weak[name]/start_weak['weight'] if name in ('q','sigma') else start[name]
                    initial[name]=(value.copy(),value.copy())
            oldv,oldz=initial[name]
            for region,lo,hi in REGIONS:
                keep=(x>=lo-1e-7)&(x<=hi+1e-7)
                delta=v[keep]-z[keep];weights=w[keep]
                increment=(v-oldv-z+oldz)[keep]
                growth=rms((z-oldz)[keep],weights)
                i=np.flatnonzero(keep)[np.argmax(abs(delta))]
                rows.append(dict(time=t,step_a=ka,step_b=kb,region=region,field=name,
                    RMS=rms(delta,weights),relative_total=rms(delta,weights)/rms(z[keep],weights) if rms(z[keep],weights) else 0.,
                    maximum=float(max(abs(delta))),xd_at_maximum=float(x[i]),
                    maximum_local_relative=float(max(abs(delta)/np.maximum(abs(z[keep]),1e-30))) if name in ('V','Theta_in','Theta_out') else None,
                    initial_RMS=rms((oldv-oldz)[keep],weights),increment_RMS=rms(increment,weights),
                    fine_increment_RMS=growth,relative_increment=rms(increment,weights)/growth if growth else None,
                    fixed_state_rate_sensitivity=float(max(abs(delta)*(db['weak_friction']/db['weak_sigma'])[keep]/load_scale[keep]))
                      if name=='sigma' else float(max(abs(delta)/load_scale[keep])) if name=='q' else None))
            profiles.extend(dict(time=t,xd=float(x[i]),field=name,a=float(v[i]),b=float(z[i]),difference=float(v[i]-z[i])) for i in range(len(x)))
    records(OUT/f'{kind}_differences.csv',rows);records(OUT/f'{kind}_profiles.csv',profiles)
    if kind=='restart': (OUT/'restart_equivalence.json').write_text(json.dumps(restart,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for region,lo,hi in REGIONS:
        if region=='interior' or region=='deep': continue
        fig,axes=plt.subplots(len(pairs),4,figsize=(13,3*len(pairs)),squeeze=False)
        for row,(ka,kb,t) in enumerate(pairs):
            for label,k,style in [(a,ka,'-'),(b,kb,'--')]:
                d=np.atleast_1d(table(OUT/label/f'state_work_{k}.csv'));order=np.argsort(d['xd'])
                wv=np.atleast_1d(table(OUT/label/f'work_weak_{k}.csv'))
                vals=[d['V']*1e9,d['Theta_out'],wv['q']/wv['weight']*1e-6,wv['sigma']/wv['weight']*1e-6]
                for col,(value,title) in enumerate(zip(vals,['V/Vp','Outgoing Theta (s)','Work q (MPa)','Work sigma (MPa)'])):
                    axes[row,col].plot(d['xd'][order]/1000,value[order],style,label=label)
                    axes[row,col].set(xlim=(lo/1000,hi/1000),xlabel='Down dip (km)',ylabel=title,title=f't={t:g} s')
            axes[row,0].legend(fontsize=7)
        fig.tight_layout();fig.savefig(OUT/f'{kind}_{region}.png',dpi=130);plt.close(fig)
    print(kind,'matched times:',[p[2] for p in pairs])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check');p.add_argument('--compare',nargs=2);p.add_argument('--kind',default='spatial')
    args=p.parse_args()
    if args.check: print(json.dumps(check(args.check)[1],indent=2))
    else: compare(*args.compare,kind=args.kind)
