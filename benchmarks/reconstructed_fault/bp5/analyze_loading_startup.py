"""Loading-only qualification: unchanged native work and history timing."""
import argparse
import json
import math
import numpy as np
from run_loading_startup import OUT
from check_startup_30km import table, raw_qps
from check_first_cycle_restart import convergence


def aging(v, theta, dt):
    return theta*np.exp(-v*dt/.1)-.1/v*np.expm1(-v*dt/.1)


def metric(z, w):
    return dict(max=float(np.max(abs(z))), RMS=float(np.sqrt(np.dot(w,z*z)/sum(w))))


def check(case):
    path=OUT/case
    assert json.loads((path/'execution.json').read_text())['passed']
    conv=convergence(path/'run.log')
    clock=table(path/'accepted_steps.csv')
    init=table(OUT/'startup/steady_initialization.csv')
    f=init['strengthening_fraction']
    assert np.all((f>=0)&(f<=1))
    np.testing.assert_allclose(init['Theta0'],1e8*.8**(1-f),rtol=3e-16,atol=0)
    assert max(abs(init['weak_error_Pa']))<1e-5
    assert np.all(clock['fresh_linear_checks_passed']==1)
    assert max(clock['Theta_relative_error'])<1e-12
    assert np.all(clock['free']+clock['lower_active']==len(init))
    ratio=.03/(.004*(1-f)+.04*f)
    initial=table(OUT/'startup/state_work_0.csv')
    initial_weak=table(OUT/'startup/work_weak_0.csv')
    def raw_state(folder,k):
        raw=np.concatenate([raw_qps(p) for p in sorted(folder.glob(f'work_qp_{k}_rank*.csv'))])
        return np.sort(raw,order=['cell','qp'])
    initial_raw=raw_state(OUT/'startup',0)
    previous=table(OUT/'startup/state_work_1.csv') if case!='startup' else None
    if case!='startup':
        assert not (path/'steady_initialization.csv').exists(), 'Restart repeated initialization'
        assert 'projected-state initialization:' not in (path/'run.log').read_text()
    results={}
    for k in conv:
        s=table(path/f'state_work_{k}.csv')
        w=table(path/f'work_weak_{k}.csv')
        assert max(abs(w['bg']-init['background_load'])/w['weight'])<1e-5
        assert not (path/'weak_initialization.csv').exists()
        assert min(s['V'])>=1e-20 and min(s['Theta_out'])>0
        predicted=realized=0.
        if k==0:
            np.testing.assert_array_equal(s['Theta_in'],init['Theta0'])
            np.testing.assert_array_equal(s['Theta_out'],s['Theta_in'])
            assert np.all(s['slip']==0) and clock[0]['max_committed_stress_Pa']==0
            assert clock[0]['free']==len(init) and clock[0]['lower_active']==0
            assert max(abs(s['V']/1e-9-1))<=.005
        else:
            np.testing.assert_array_equal(s['Theta_in'],previous['Theta_out'])
            dt=s['dt'][0]
            assert max(abs(aging(s['V'],s['Theta_in'],dt)/s['Theta_out']-1))<1e-12
            expected=np.array([math.fma(float(dt),float(v),float(old))
                               for v,old in zip(s['V'],previous['slip'])])
            np.testing.assert_array_equal(s['slip'],expected)
            predicted=float(max(ratio*abs(np.log(aging(previous['V'],s['Theta_in'],dt)/s['Theta_in']))))
            realized=float(max(ratio*abs(np.log(s['Theta_out']/s['Theta_in']))))
            assert predicted<=.02+1e-12

        # Independent initialization audit at actual bulk QPs: interpolate
        # Theta, not log Theta. Later particles/material mixtures advect, so
        # the initial mixture must NOT be used to reconstruct later friction.
        raw=raw_state(path,k)
        for name in ('cell','qp','x','y','source_active','segment','xi','phi'):
            np.testing.assert_array_equal(raw[name],initial_raw[name])
        profile_change={}
        for name in ('Ih','chi'):
            np.testing.assert_allclose(raw[name],initial_raw[name],rtol=1e-12,atol=0)
            profile_change[name]=float(max(abs(raw[name]-initial_raw[name])))
        window=(s['xd']>=27000)&(s['xd']<=36000)
        friction_error=None
        if k==0:
            raw=raw[(raw['source_active']==1)&(raw['chi']>0)]
            j=raw['segment'].astype(int); xi=raw['xi']
            chem=(1-xi)*init['projected_chemical'][j]+xi*init['projected_chemical'][j+1]
            # One-chemical-field production composition-fraction rule.
            fq=np.clip(chem,0,1)
            a=.004*(1-fq)+.04*fq
            theta=(1-xi)*s['Theta_in'][j]+xi*s['Theta_in'][j+1]
            mu=a*np.arcsinh(raw['V']/(2e-6)*np.exp((.6+.03*np.log(1e-6*theta/.1))/a))
            loads=np.zeros(len(s))
            for end,shape in ((0,1-xi),(1,xi)):
                loads+=np.bincount(j+end,weights=raw['JxW']*raw['chi']*shape*raw['sigma_n']*mu,minlength=len(s))
            friction_error=float(max(abs(loads[window]-s['weak_friction'][window])/w['weight'][window]))
            assert friction_error<1e-5, (k, friction_error)
        assert max(abs(w['weight']/initial_weak['weight']-1))<1e-12
        regions={}
        for name,mask in dict(weakening=(s['xd']>2000)&(s['xd']<28000),
                              transition=window,deep=(s['xd']>35000)&(s['xd']<110000),
                              top=s['xd']<2000,bottom=s['xd']>113000).items():
            delta_q=w['q']/w['weight']-initial_weak['q']/initial_weak['weight']
            delta_sigma=w['sigma']/w['weight']-initial_weak['sigma']/initial_weak['weight']
            regions[name]=dict(V_over_Vp=[float(min(s['V'][mask]/1e-9)),float(max(s['V'][mask]/1e-9))],
                R=[float(min(s['V'][mask]*s['Theta_out'][mask]/.1)),float(max(s['V'][mask]*s['Theta_out'][mask]/.1))],
                slip_deficit_m=metric(1e-9*s['time'][0]-s['slip'][mask],w['weight'][mask]),
                shear_change_Pa=metric(delta_q[mask],w['weight'][mask]),
                normal_change_Pa=metric(delta_sigma[mask],w['weight'][mask]))
        results[k]=dict(time=float(s['time'][0]),dt=float(s['dt'][0]),
            predicted_state_change=predicted,realized_state_change=realized,
            native_QP_friction_error_Pa=friction_error,profile_change=profile_change,regions=regions)
        previous=s
    if case=='startup':
        first=table(path/'first_update_maxwell.csv')[0]
        assert first['old_FE_max_Pa']==0 and first['error_Pa']<=1e-6+1e-11*first['scale_Pa']
    else:
        # Copied checkpoint histories/mesh are byte-identical; only pending clock
        # differs. The first resumed mechanics uses precisely the saved Theta.
        source=json.loads((path/'checkpoint_source.json').read_text())
        assert source['all_other_uncompressed_bytes_identical']
    result=dict(passed=True,convergence=conv,states=results,
                initialization_weak_error_Pa=float(max(abs(init['weak_error_Pa']))))
    (path/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(case=case,passed=True,convergence=conv),indent=2))


def compare():
    a,b=OUT/'startup',OUT/'half'
    for p in (a,b):assert json.loads((p/'checks.json').read_text())['passed']
    x,y=table(a/'state_work_2.csv'),table(b/'state_work_3.csv')
    old=table(a/'state_work_1.csv'); first_half=table(b/'state_work_2.csv')
    np.testing.assert_array_equal(x['Theta_in'],first_half['Theta_in'])
    assert x['time'][0]==y['time'][0]
    wa,wb,w0=[table(p)['weight'] for p in (a/'work_weak_2.csv',b/'work_weak_3.csv',a/'work_weak_1.csv')]
    fields={}
    for name in ('V','slip','Theta_in','Theta_out','weak_q','weak_sigma','weak_friction'):
        u,v,initial=x[name],y[name],old[name]
        if name.startswith('weak_'):u=u/wa;v=v/wb;initial=initial/w0
        if name=='Theta_in':initial=old['Theta_out']
        err=metric(u-v,wa); change=metric(v-initial,wa)
        fields[name]=dict(error=err,evolving_change=change,
                         error_over_change_RMS=err['RMS']/change['RMS'] if change['RMS'] else None)
    logv=np.log(x['V']/y['V']); logtheta=np.log(x['Theta_out']/y['Theta_out'])
    fast=(x['V']>=1e-10)|(y['V']>=1e-10)
    screens=dict(log_V_fast=float(max(abs(logv[fast])))<=.02,
                 log_Theta=float(max(abs(logtheta)))<=1e-3,
                 slip_over_Dc=float(max(abs(x['slip']-y['slip']))/.1)<=1e-4)
    result=dict(passed=all(screens.values()),screens=screens,metrics=fields,
                log_V=metric(logv,wa),log_V_fast=metric(logv[fast],wa[fast]),
                log_Theta=metric(logtheta,wa),slip_over_Dc=metric((x['slip']-y['slip'])/.1,wa),
                begin_time=float(old['time'][0]),end_time=float(x['time'][0]),
                incoming_state_note='Identical at branch start; the last half-step consumes its own preceding update.')
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=OUT/'startup'; initial=table(base/'work_weak_0.csv')
    for label,window in [('whole',None),('transition',(27,36))]:
        fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
        for k in range(4):
            s=table(base/f'state_work_{k}.csv'); w=table(base/f'work_weak_{k}.csv')
            order=np.argsort(s['xd'])
            if window:
                order=order[(s['xd'][order]>=window[0]*1000)&(s['xd'][order]<=window[1]*1000)]
            xx=s['xd'][order]/1000
            t=s['time'][0]/86400; name=f'{t:.3f} d (step {k})'
            series=[s['V']/1e-9,s['Theta_out']/1e8,s['V']*s['Theta_out']/.1,
                    (1e-9*s['time'][0]-s['slip'])*1e3,
                    (w['q']/w['weight']-initial['q']/initial['weight'])/1e3,
                    (w['sigma']/w['weight']-initial['sigma']/initial['weight'])/1e3]
            for ax,z in zip(axes.flat,series):ax.plot(xx,z[order],label=name)
            axes[0,1].plot(xx,s['Theta_in'][order]/1e8,':',color=axes[0,1].lines[-1].get_color())
        for ax,title in zip(axes.flat,['V/Vp','Theta / 1e8 s (dotted: incoming)','V Theta_out / Dc',
                                      'Slip deficit (mm)','Native weak shear change (kPa)','Native weak normal change (kPa)']):
            ax.set_ylabel(title);ax.grid(alpha=.25)
            for pos in (30,33):ax.axvline(pos,color='gray',lw=.7,ls='--')
            if window:ax.set_xlim(*window)
        axes[0,0].legend(fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('Down-dip distance (km)')
        fig.suptitle('Loading-driven R_VW=0.8: accepted split histories; native work averages')
        fig.tight_layout();fig.savefig(OUT/f'loading-{label}.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['startup','half','plot'])
    args=p.parse_args()
    if args.action=='plot':plot()
    else:
        check(args.action)
        if args.action=='half':compare()
