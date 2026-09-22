"""Accepted-state and actual-source checks for the opt-in length-scale replay."""
import argparse
import json
import numpy as np
from analyze_mechanical_modes import table
from analyze_mechanical_width import records
from length_coupled import HERE,OUT
from run_mechanical_width import SN,NORMAL,VirtualProfile


def rms(v,w):
    return float(np.sqrt(np.dot(w,v*v)/sum(w)))


def raw_qps(path):
    with path.open() as stream:
        names=stream.readline().strip().split(',')
        return np.loadtxt(stream,delimiter=',',ndmin=1,
            dtype=[(name,'U64' if name=='cell' else 'f8') for name in names])


def check(label):
    run=OUT/label
    accepted=np.atleast_1d(table(run/'accepted_steps.csv'))
    assert np.all(accepted['fresh_linear_checks_passed']==1)
    assert np.max(accepted['Theta_relative_error'])<1e-12
    assert np.all(accepted['free']+accepted['lower_active']==1156)
    assert accepted[0]['max_committed_stress_Pa']==0
    bg=np.loadtxt(HERE/'fixtures/modified_bp3_dc024_ell50/prestress.txt',skiprows=1)
    reference='target_cells_reference.txt' in (run/'run.prm').read_text()
    profile=VirtualProfile(50,6.103515625 if reference else 12.20703125)
    completion=np.concatenate([np.atleast_1d(table(f)) for f in run.glob('ih_bottom_completion_rank*.csv')])
    endpoint_completion=[]
    for row in completion[completion['outside']>0]:
        origin=np.array([row['x'],row['y']]);extent=profile.extent
        total=profile.integrate(origin,-extent,extent)
        inside=profile.integrate(origin,max(-extent,-origin[1]/NORMAL[1]),
            min(extent,(100000-origin[1])/NORMAL[1]))
        error=row['completed']/total-1
        assert abs(error)<1e-6
        assert abs(row['inside']+row['outside']-row['completed'])<1e-9
        endpoint_completion.append(dict(id=int(row['id']),xd=(100000-origin[1])/SN,
            inside_fraction=inside/total,completed_relative_error=error,
            inside_relative_error=row['inside']/inside-1))
    (run/'endpoint_completion_checks.json').write_text(json.dumps(endpoint_completion,indent=2)+'\n')
    surface=np.atleast_1d(table(run/'surface.csv'))
    sx=(100000-surface['y'])/SN;order=np.argsort(sx)
    result=[]
    for state in accepted:
        step=int(state['step'])
        raw=np.concatenate([raw_qps(f) for f in sorted(run.glob(f'work_qp_{step}_rank*.csv'))])
        weak=np.atleast_1d(table(run/f'work_weak_{step}.csv'))
        assert len(set(zip(raw['cell'],raw['qp'])))==len(raw)
        active=raw['source_active']==1
        # Q1 interpolation spreads the finite profile slightly beyond its
        # configured association radius. Measure those tails; distinguish them
        # from a hole in the admitted band or a missing continuation wedge.
        assert np.all(abs(raw['r'][~active])>=profile.r[-1]-1e-8)
        assert np.all((raw['xd'][~active]>=-1e-8)&(raw['xd'][~active]<=100000/SN+1e-8))
        h=profile.m*raw['phi']*(1+raw['phi'])/(1-raw['phi'])**2
        assert np.max(abs(raw['chi'][active]-h[active]/raw['Ih'][active]))<1e-12
        full_weight=raw['JxW']*h/np.interp(raw['xd'],sx[order],surface['Ih'][order])
        j=raw['segment'].astype(int);xi=raw['xi']
        def interp(c): return (1-xi)*bg[j,c]+xi*bg[j+1,c]
        background=interp(2)-interp(4)-interp(5)/interp(6)
        tauN=.75*raw['tau_xx']+.25*raw['tau_yy']-SN*raw['tau_xy']
        shear=-.5*SN*raw['tau_xx']+.5*SN*raw['tau_yy']-.5*raw['tau_xy']
        assert np.max(abs(raw['sigma_n']-(50e6+raw['p']-tauN)))<1e-5
        assert np.max(abs(raw['q']-(background+shear)))<1e-5
        reconstructed=np.zeros((len(weak),3))
        weight=raw['JxW']*raw['chi']
        for end,basis in [(0,1-xi),(1,xi)]:
            for c,value in enumerate([np.ones(len(raw)),raw['q'],raw['sigma_n']]):
                np.add.at(reconstructed[:,c],j+end,weight*basis*value)
        # The exported top/bottom windows contain the whole support of these
        # endpoint test functions, including the physical-box continuation wedges.
        endpoint=(weak['xd']<1500)|(weak['xd']>100000/SN-1500)
        mass_error=max(abs(reconstructed[endpoint,0]/weak['weight'][endpoint]-1))
        traction_error=max(np.max(abs(reconstructed[endpoint,1]-weak['q'][endpoint])/weak['weight'][endpoint]),
                           np.max(abs(reconstructed[endpoint,2]-weak['sigma'][endpoint])/weak['weight'][endpoint]))
        assert mass_error<1e-11 and traction_error<1e-5
        record=dict(step=step,time=float(state['time']),exported_QPs=len(raw),
                    unassociated_positive_phase=int(sum(~active)),
                    omitted_FE_tail_fraction=float(sum(full_weight[~active])/sum(full_weight)),
                    endpoint_mass_relative_error=float(mass_error),endpoint_traction_error_Pa=float(traction_error),
                    surface_RMS_Pa=float(state['surface_RMS_Pa']),nonlinear_relative=float(state['normalized_nonlinear_residual']),
                    min_V=float(state['min_free_V']),max_V=float(state['max_V']),lower_active=int(state['lower_active']))
        for name,mask in [('top',raw['xd']<2000),('transition',(raw['xd']>13000)&(raw['xd']<20000)),
                          ('interior',(raw['xd']>59000)&(raw['xd']<61000)),('bottom',raw['xd']>100000/SN-2000)]:
            assert np.any(mask)
            record[name]=dict(q_min=float(min(raw['q'][mask])),q_max=float(max(raw['q'][mask])),
                sigma_min=float(min(raw['sigma_n'][mask])),sigma_max=float(max(raw['sigma_n'][mask])),
                pressure_min=float(min(raw['p'][mask])),pressure_max=float(max(raw['p'][mask])),
                strain_mismatch_rms=rms(raw['elastic_norm'][mask],weight[mask]),
                source_measure=float(sum(weight[mask])),
                omitted_FE_tail_fraction=float(sum(full_weight[mask&~active])/sum(full_weight[mask])),
                continuation_QPs=int(np.count_nonzero(mask&((raw['xd']<0)|(raw['xd']>100000/SN)))))
        result.append(record)
    (run/'coupled_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def compare(a,b):
    checks={name:check(name) for name in [a,b]}
    ca=np.atleast_1d(table(OUT/a/'accepted_steps.csv'));cb=np.atleast_1d(table(OUT/b/'accepted_steps.csv'))
    np.testing.assert_allclose(ca['time'],cb['time'],rtol=1e-12,atol=1e-6)
    rows=[];initial={};profiles=[];matched=[]
    regions=[('whole',0,100000/SN+1),('top',0,2000),('transition',13000,20000),
             ('interior',25000,35000),('deep',40000,80000),('bottom',100000/SN-2000,100000/SN+1),
             ('top_patch_edge',4000,6000),('transition_patch_start',9000,11000),
             ('transition_patch_end',22000,24000),('bottom_patch_edge',100000/SN-6000,100000/SN-4000)]
    for step in ca['step'].astype(int):
        pa=np.atleast_1d(table(OUT/a/f'profiles/fault_{step}.csv'))
        pb=np.atleast_1d(table(OUT/b/f'profiles/fault_{step}.csv'))
        wa=np.atleast_1d(table(OUT/a/f'work_weak_{step}.csv'));wb=np.atleast_1d(table(OUT/b/f'work_weak_{step}.csv'))
        np.testing.assert_array_equal(pa['xd_m'],pb['xd_m'])
        w=(wa['weight']+wb['weight'])/2;x=pa['xd_m']
        fields={'V':(pa['V_m_per_s'],pb['V_m_per_s']),'Theta':(pa['Theta_s'],pb['Theta_s']),
                'slip':(pa['slip_m'],pb['slip_m']),'q_Q1':(pa['q_weak_Pa'],pb['q_weak_Pa']),
                'sigma_Q1':(pa['sigma_n_weak_Pa'],pb['sigma_n_weak_Pa']),
                'q_work_average':(wa['q']/wa['weight'],wb['q']/wb['weight']),
                'sigma_work_average':(wa['sigma']/wa['weight'],wb['sigma']/wb['weight'])}
        for field,(v,z) in fields.items():
            if step==0: initial[field]=(v.copy(),z.copy())
            oldv,oldz=initial[field]
            matched.extend(dict(step=int(step),xd=float(x[j]),field=field,candidate=float(v[j]),reference=float(z[j]),
                difference=float(v[j]-z[j]),initial_difference=float(oldv[j]-oldz[j]),
                increment_difference=float(v[j]-oldv[j]-z[j]+oldz[j])) for j in range(len(x)))
            for region,lo,hi in regions:
                # Physical endpoints can round to -7e-12 m; include them in
                # windows instead of accidentally dropping the endpoint row.
                keep=(x>=lo-1e-7)&(x<=hi+1e-7);weights=w[keep];delta=v[keep]-z[keep]
                signal=rms(z[keep],weights);growth=rms(z[keep]-oldz[keep],weights)
                change=rms((v-oldv-z+oldz)[keep],weights)
                i=np.flatnonzero(keep)[np.argmax(abs(delta))]
                rows.append(dict(step=step,time=float(pa['time_s'][0]),region=region,field=field,
                    difference_rms=rms(delta,weights),relative_total=rms(delta,weights)/signal if signal else None,
                    maximum_local_relative=float(np.max(abs(delta/z[keep]))) if field in ('V','Theta') else None,
                    initial_difference_rms=rms((oldv-oldz)[keep],weights),
                    increment_difference_rms=change,fine_increment_rms=growth,
                    relative_increment=change/growth if growth else None,max_difference=float(max(abs(delta))),
                    xd_at_max_difference=float(x[i])))
        order=np.argsort(x)
        for p,label in [(pa,a),(pb,b)]:
            y=p['V_m_per_s'][order];s=x[order]
            chord=y[1:-1]-((s[2:]-s[1:-1])*y[:-2]+(s[1:-1]-s[:-2])*y[2:])/(s[2:]-s[:-2])
            for region,lo,hi in regions:
                mask=(s[1:-1]>=lo-1e-7)&(s[1:-1]<=hi+1e-7)
                if np.any(mask):
                    k=np.flatnonzero(mask)[np.argmax(abs(chord[mask]))]
                    profiles.append(dict(step=step,mesh=label,region=region,maximum_chord_over_Vp=float(max(abs(chord[mask]))/1e-9),
                        xd=float(s[1:-1][k])))
    records(OUT/'spatial_differences.csv',rows);records(OUT/'velocity_chord.csv',profiles)
    records(OUT/'matched_fault_profiles.csv',matched)
    (OUT/'coupled_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,limits in [('whole',(0,100000/SN/1000)),('transition',(13,20)),('top',(0,2)),('bottom',(100000/SN/1000-2,100000/SN/1000)),
                        ('top_patch',(3.5,6.5)),('transition_patch',(9,24)),
                        ('bottom_patch',(100000/SN/1000-6.5,100000/SN/1000-3.5))]:
        fig,axes=plt.subplots(len(ca),4,figsize=(13,3*len(ca)),squeeze=False)
        for row,step in enumerate(ca['step'].astype(int)):
            for label,style in [(a,'-'),(b,'--')]:
                p=np.atleast_1d(table(OUT/label/f'profiles/fault_{step}.csv'));order=np.argsort(p['xd_m'])
                for col,(field,scale,title) in enumerate([('V_m_per_s',1e9,'V/Vp'),('Theta_s',1.,'Committed Theta (s)'),
                    ('q_weak_Pa',1e-6,'Current q (MPa), Q1'),('sigma_n_weak_Pa',1e-6,'Current sigma_n (MPa), Q1')]):
                    axes[row,col].plot(p['xd_m'][order]/1000,p[field][order]*scale,style,label=label)
                    axes[row,col].set(xlim=limits,xlabel='Down dip (km)',ylabel=title,title=f'step {step}')
                    axes[row,col].grid(alpha=.25)
            axes[row,0].legend(fontsize=7)
        fig.tight_layout();fig.savefig(OUT/f'spatial_{name}.png',dpi=150);plt.close(fig)
    print('Spatial/source diagnostics written to',OUT)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label');p.add_argument('--compare',nargs=2)
    args=p.parse_args()
    if args.label: print(json.dumps(check(args.label),indent=2))
    else: compare(*args.compare)
