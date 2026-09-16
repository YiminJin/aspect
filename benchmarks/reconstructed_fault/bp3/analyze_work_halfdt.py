"""Two-level temporal comparison of revised BP3 mechanics; diagnostics only.

Friction uses Q1 interpolation of the PRECEDING nodal state in mechanics.
The counterfactual uses the committed new nodal state at the same accepted V
and sigma, with no history publication, projection back, or mechanical solve.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_uniform_sliding import read,cat,write,records

HERE=Path(__file__).resolve().parent
BASE=HERE/'work-replay-50-local4'
FINE=HERE/'work-replay-halfdt-50-local4'
OUT=FINE/'comparison'
YEAR=31557600.
REGIONS={'junction':(39000,41000),'15km':(13000,16500),'18km':(16500,20000)}


def mu(v,theta):
    # This exact constitutive expression is used only in the fully VS region
    # around 40 km: the surface material fraction there is identically one.
    return .025*np.arcsinh(v/(2e-6)*np.exp((.6+.015*np.log(theta*1e-6/.008))/.025))


def friction(root,k,f,raw):
    old=read(root/f'fault_{max(0,k-1)}.csv')
    mask=(raw['source_active']==1)&(raw['xd']>38000)&(raw['xd']<42000)
    j=raw['segment'][mask].astype(int);z=raw['xi'][mask];n=len(f['V'])
    v=(1-z)*f['V'][j]+z*f['V'][j+1]
    np.testing.assert_allclose(v,raw['V'][mask],rtol=2e-14,atol=1e-25)
    lag=(1-z)*old['Theta'][j]+z*old['Theta'][j+1]
    new=(1-z)*f['Theta'][j]+z*f['Theta'][j+1]
    sigma=raw['sigma_n'][mask];weights=raw['JxW'][mask]*raw['chi'][mask]
    load=lambda values:np.bincount(j,weights=weights*(1-z)*values,minlength=n)+np.bincount(j+1,weights=weights*z*values,minlength=n)
    lag_load=load(mu(v,lag)*sigma);new_load=load(mu(v,new)*sigma);mass=load(np.ones(len(j)))
    # Recorded friction_traction is the consistent Q1 representation. Undo
    # that mass solve to compare the actual weak friction integral.
    recorded=f['mass_diagonal']*f['friction_traction']
    recorded[:-1]+=f['mass_upper'][:-1]*f['friction_traction'][1:]
    recorded[1:]+=f['mass_upper'][:-1]*f['friction_traction'][:-1]
    near=(f['xd']>=39800)&(f['xd']<=40100)
    error=float(max(abs(lag_load[near]-recorded[near])/mass[near]))
    assert error<1e-5,('Lagged friction does not reproduce production weak load',root,k,error)
    rows=[]
    for target in (39900.,39950.,40000.):
        node=int(np.argmin(abs(f['xd']-target)))
        rows.append(dict(case=root.name,step=k,time_yr=float(f['time'][0]/YEAR),xd=float(f['xd'][node]),
            V=float(f['V'][node]),Theta_lagged=float(old['Theta'][node]),Theta_updated=float(f['Theta'][node]),
            aging_x=float(f['V'][node]*f['dt'][node]/.008) if k else 0.,
            node_mu_lagged=float(mu(f['V'][node],old['Theta'][node])),node_mu_updated=float(mu(f['V'][node],f['Theta'][node])),
            weak_friction_lagged_Pa=float(lag_load[node]/mass[node]),
            weak_friction_updated_Pa=float(new_load[node]/mass[node]),
            weak_change_Pa=float((new_load[node]-lag_load[node])/mass[node]),
            production_reproduction_error_Pa=error))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    def parameters(tree,prefix=''):
        values={}
        for key,value in tree.items():
            if isinstance(value,dict) and 'value' in value:values[prefix+key]=value['value']
            elif isinstance(value,dict):values.update(parameters(value,prefix+key+'/'))
        return values
    settings={label:parameters(json.loads((root/'parameters.json').read_text())) for label,root in [('base',BASE),('half',FINE)]}
    differences={key:[settings['base'].get(key),settings['half'].get(key)]
                 for key in settings['base'].keys()|settings['half'].keys()
                 if settings['base'].get(key)!=settings['half'].get(key)}
    assert set(differences)=={'Output directory','Termination criteria/Termination criteria'}
    fine_clock=read(FINE/'accepted_steps.csv');coarse_clock=read(BASE/'accepted_steps.csv')
    log=(FINE/'run.log').read_text();mapping=[dict(baseline_step=0,half_step=0,time=0.)]+json.loads((FINE/'shared_times.json').read_text())
    if not args.partial:
        assert json.loads((FINE/'execution.json').read_text())['status']==0
        np.testing.assert_array_equal(fine_clock['step'],np.arange(21))
        assert 'BP3 REPLAY COMPLETE: audited accepted step=20' in log
        assert '*** Timestep 21:' not in log
    linear=re.findall(r'Fault linear solve: iterations=(\d+), estimated=[^,]+, fresh=([^,]+), target=([^,]+)',log)
    assert linear and all(float(a)<=float(b) for _,a,b in linear)
    history=[];nonlinear=[];nodes=[];stress=[];weak=[];friction_rows=[];matched=[];localization=[]
    initial=read(BASE/'fault_0.csv');previous=None
    for k in fine_clock['step'].astype(int):
        if not (FINE/f'fault_{k}.csv').exists():continue
        f=read(FINE/f'fault_{k}.csv');h=read(FINE/f'history_{k}.csv')
        assert h['Theta_reference_relative_error'][0]<1e-12
        for name in ('Ih','x','y','C'):np.testing.assert_array_equal(f[name],initial[name])
        assert np.all(f['V'][f['prescribed']==1]==1e-9)
        if k==0:
            for name in ('V','Theta','slip','q','friction_traction'):
                np.testing.assert_allclose(f[name],initial[name],rtol=1e-12,atol=1e-25)
        else:
            np.testing.assert_allclose(f['slip'],previous['slip']+f['dt']*f['V'],rtol=1e-13,atol=1e-15)
            expected=coarse_clock['dt'][(k+1)//2]/2
            assert abs(float(f['dt'][0])-expected)<=1e-12*expected
        section=log.split(f'*** Timestep {k}:',1)[1].split('*** Timestep ',1)[0]
        residual=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',section)[-1]
        assert max(map(float,residual))<1e-8
        nonlinear.append(dict(step=int(k),bulk=float(residual[0]),surface=float(residual[1])))
        history.append(dict(step=int(k),time=float(f['time'][0]),theta_error=float(h['Theta_reference_relative_error'][0])))
        previous=f
    for pair in mapping:
        k=pair['half_step'];b=pair['baseline_step']
        if not (FINE/f'fault_{k}.csv').exists():continue
        f=read(FINE/f'fault_{k}.csv');c=read(BASE/f'fault_{b}.csv')
        assert abs(f['time'][0]-c['time'][0])<=1e-5
        matched.append(pair);profiles=dict(xd=f['xd'])
        for key in ('V','Theta','slip','q','friction_traction'):
            profiles[key+'_half']=f[key];profiles[key+'_base']=c[key]
        center=(f['xd'][:-1]+f['xd'][1:])/2
        gradient={label:np.diff(values['slip'])/np.diff(values['xd']) for label,values in [('half',f),('base',c)]}
        write(OUT/f'profiles_{b}.csv',profiles)
        write(OUT/f'slip_gradient_{b}.csv',dict(xd=center,**gradient))
        for target in (15000.,18000.,25000.,39900.,39950.,40000.):
            j=int(np.argmin(abs(f['xd']-target)))
            row=dict(baseline_step=b,half_step=k,time_yr=float(f['time'][0]/YEAR),xd=float(f['xd'][j]))
            for key in ('V','Theta','slip'):
                row[key+'_base']=float(c[key][j]);row[key+'_half']=float(f[key][j])
            last=int(np.argmin(abs(center-39975.)))
            row['last_gradient_base']=float(gradient['base'][last]);row['last_gradient_half']=float(gradient['half'][last])
            nodes.append(row)
        raw={label:cat(root.glob(f'work_qp_{step}_rank*.csv'),('cell',))
             for label,root,step in [('half',FINE,k),('base',BASE,b)]}
        for label,root,step,values in [('half',FINE,k,f),('base',BASE,b,c)]:
            d=raw[label];friction_rows+=friction(root,step,values,d)
            np.testing.assert_allclose(d['sigma_n'],50e6+d['p']-d['tauN'],rtol=1e-13,atol=1e-7)
            w=read(root/f'work_weak_{step}.csv')
            for name,(lo,hi) in REGIONS.items():
                mask=(d['xd']>=lo)&(d['xd']<=hi);s=(w['xd']>=lo)&(w['xd']<=hi)
                r=dict(case=label,baseline_step=b,step=step,time_yr=float(values['time'][0]/YEAR),region=name)
                for field in ('p','tau_xx','tau_yy','tau_xy','tauN','sigma_n','q'):
                    r[field+'_min']=float(min(d[field][mask]));r[field+'_max']=float(max(d[field][mask]))
                stress.append(r)
                r=dict(case=label,baseline_step=b,step=step,time_yr=float(values['time'][0]/YEAR),region=name)
                for field in ('p','tauN','sigma','q'):
                    mean=w[field][s]/w['weight'][s];r[field+'_min']=float(min(mean));r[field+'_max']=float(max(mean))
                weak.append(r)
        # All saved physical QPs are unchanged. Compare tensor/normal traction
        # directly there, without substituting published old history fields.
        a,z=raw['half'],raw['base']
        for key in ('x','y','JxW','phi'):np.testing.assert_array_equal(a[key],z[key])
        # Identical degradation laws in the two materials can still acquire
        # last-bit mixture arithmetic differences after particle projection.
        # Compare localization at roundoff, while geometry and phase stay exact.
        chi_error=float(max(abs(a['chi']-z['chi'])))
        chi_allowance=16*np.finfo(float).eps*max(abs(z['chi']))
        np.testing.assert_allclose(a['chi'],z['chi'],rtol=0,atol=chi_allowance)
        localization.append(dict(baseline_step=b,max_chi_difference=chi_error,allowance=float(chi_allowance)))
        mask=(a['xd']>=39000)&(a['xd']<=41000)
        write(OUT/f'current_stress_{b}.csv',dict(x=a['x'][mask],y=a['y'][mask],xd=a['xd'][mask],r=a['r'][mask],
            **{key+'_'+label:d[key][mask] for label,d in raw.items() for key in ('p','tauN','sigma_n','tau_xx','tau_xy')}))
        if b==10:
            for name,limits in [('junction',(39,40.2)),('transitions',(13,20))]:
                fig,axes=plt.subplots(3,2,figsize=(12,10))
                for ax,key in zip(axes.flat,('V','Theta','slip','q','friction_traction','gradient')):
                    displayed=[]
                    for label,color in [('base','tab:blue'),('half','tab:orange')]:
                        x=center if key=='gradient' else f['xd'];y=gradient[label] if key=='gradient' else profiles[key+'_'+label]
                        order=np.argsort(x);mask=(x>=limits[0]*1000)&(x<=limits[1]*1000)
                        displayed.extend(y[mask]);ax.plot(x[order]/1000,y[order],label=label,color=color)
                    lo=min(displayed);hi=max(displayed);pad=.05*(hi-lo) if hi>lo else .01*max(abs(hi),1e-20)
                    ax.set(xlim=limits,ylim=(lo-pad,hi+pad),xlabel='Down dip (km)',ylabel=key)
                    for mark in (15,18,40):
                        if limits[0]<=mark<=limits[1]:ax.axvline(mark,color='k',ls=':')
                    ax.legend()
                fig.suptitle('29.24190894 yr: same revised mechanics, half real timesteps')
                fig.tight_layout();fig.savefig(OUT/f'final_{name}.png',dpi=150);plt.close(fig)
            # Compare actual mechanical weak traction, not raw pressure alone.
            fig,axes=plt.subplots(3,2,figsize=(12,10))
            for col,(name,limits) in enumerate([('junction',(39,41)),('transitions',(13,20))]):
                for label,root,step in [('base',BASE,b),('half',FINE,k)]:
                    w=read(root/f'work_weak_{step}.csv');order=np.argsort(w['xd'])
                    mask=(w['xd']>=limits[0]*1000)&(w['xd']<=limits[1]*1000)
                    order=order[mask[order]]
                    for row,(field,sign) in enumerate([('p',1),('tauN',-1),('sigma',1)]):
                        axes[row,col].plot(w['xd'][order]/1000,sign*w[field][order]/w['weight'][order]/1e6,label=label)
                        axes[row,col].set(xlim=limits,xlabel='Down dip (km)',ylabel=('minus ' if sign<0 else '')+field+' (MPa)')
                        for mark in (15,18,40):
                            if limits[0]<=mark<=limits[1]:axes[row,col].axvline(mark,color='k',ls=':')
                        axes[row,col].legend()
            fig.suptitle('Current constitutive stress: J chi N-weighted means, 29.24190894 yr')
            fig.tight_layout();fig.savefig(OUT/'final_weak_stress.png',dpi=150);plt.close(fig)
    for name,rows in [('nodes',nodes),('raw_stress',stress),('weak_stress',weak),('friction_lag',friction_rows),('history',history)]:
        records(OUT/f'{name}.csv',rows)
    summary=dict(complete=not args.partial,matched_times=matched,accepted_states=len(nonlinear),nonlinear=nonlinear,
        resolved_parameter_changes=differences,
        localization_comparison=localization,
        fresh_linear_checks=len(linear),total_Krylov=sum(int(i) for i,_,_ in linear),
        worst_fresh_ratio=max(float(a)/float(b) for _,a,b in linear),
        maximum_lower_active=int(max(fine_clock['lower_active'])),
        minimum_line_search_alpha=float(min(fine_clock['min_alpha'])),
        maximum_friction_reproduction_error_Pa=max(r['production_reproduction_error_Pa'] for r in friction_rows),
        friction_diagnostic='Same accepted V and sigma, interpolate preceding versus newly committed nodal Theta at actual production QPs; no mechanical feedback.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('nonlinear','matched_times')},indent=2))


if __name__=='__main__':main()
