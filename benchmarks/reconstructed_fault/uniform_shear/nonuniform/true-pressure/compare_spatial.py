#!/usr/bin/env python3
"""One 32-to-64 comparison of the matched true-pressure responses."""
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from verify_pilot import read, stats

base=Path(__file__).resolve().parent
fields=('sigma','p','tauN','V','slip')


def inner(x,a,b):
    return float(np.sum(np.diff(x)*(2*a[:-1]*b[:-1]+a[:-1]*b[1:]
                       +a[1:]*b[:-1]+2*a[1:]*b[1:])/6)/(x[-1]-x[0]))


def response(k,suffix):
    a,b=(read(base/f'{name}{suffix}-surface-{k}.csv') for name in ('pilot','homogeneous'))
    assert np.array_equal(a['s'],b['s'])
    return a['s'],{name:a[name]-b[name] for name in fields}


def main():
    start=time.monotonic()
    records={name:json.loads((base/f'{name}-verification.json').read_text())
             for name in ('pilot','homogeneous','pilot64','homogeneous64')}
    assert all([s['time_s'] for s in r['steps']]==[0.,.5,1.] for r in records.values())
    assert len({r['resources']['executable_sha256'] for r in records.values()})==1
    assert len({r['resources']['plugin_sha256'] for r in records.values()})==1
    assert max(r['support_half_width_m'] for r in records.values())-min(
               r['support_half_width_m'] for r in records.values())<1e-12
    result=dict(coarse='reused accepted 32x128 pair',fine='64x256 pair only',
                comparison='bump minus matched homogeneous, then fine minus coarse',
                exact_piecewise_Q1_integration=True,initialization={},steps=[])
    for suffix,label in (('','coarse'),('64','fine')):
        a,b=(base/f'{name}{suffix}' for name in ('pilot','homogeneous'))
        assert (a/'phase_0.csv').read_bytes()==(b/'phase_0.csv').read_bytes()
        surface=read(a/'surface_0.csv')
        phase=read(a/'phase_0.csv')
        # Each mesh solves the same prescribed initial data; its frozen FE
        # profile is not reset to another mesh's realized initialization.
        center=np.unique(phase[phase['y']==0]['phi'])
        result['initialization'][label]=dict(
            Theta=records['pilot'+suffix]['initial_Theta'],
            initial_C=stats(surface['x'],surface['C']),
            Ih=stats(surface['x'],surface['Ih']),
            phi_center_range=[float(min(center)),float(max(center))],
            omitted_fraction=records['pilot'+suffix]['omitted_fraction_max'])
    fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False)
    for k,t in enumerate((0.,.5,1.)):
        cx,c=response(k,''); fx,f=response(k,'64')
        x=np.unique(np.r_[cx,fx])
        coarse={n:np.interp(x,cx,c[n]) for n in fields}
        fine={n:np.interp(x,fx,f[n]) for n in fields}
        change={n:fine[n]-coarse[n] for n in fields}
        row=dict(time_s=t,fields={},signed_components={})
        for n in fields:
            cs,fs,ds=(stats(x,a[n]) for a in (coarse,fine,change))
            ca=coarse[n]-cs['mean']; fa=fine[n]-fs['mean']
            row['fields'][n]=dict(coarse=cs,fine=fs,fine_minus_coarse=ds,
                change_rms_over_fine_signal_rms=ds['rms']/fs['rms'] if fs['rms'] else None,
                change_anomaly_over_fine_anomaly=ds['anomaly_rms']/fs['anomaly_rms'] if fs['anomaly_rms'] else None,
                anomaly_profile_correlation=inner(x,ca,fa)/(cs['anomaly_rms']*fs['anomaly_rms'])
                    if cs['anomaly_rms']*fs['anomaly_rms'] else None)
        for label,a in (('coarse',coarse),('fine',fine),('change',change)):
            assert max(abs(a['sigma']-a['p']+a['tauN']))<1e-9
            p,q=a['p'],-a['tauN']
            ps,qs=stats(x,p),stats(x,q)
            pa,qa=p-ps['mean'],q-qs['mean']
            row['signed_components'][label]=dict(delta_p=ps,minus_delta_tauN=qs,
                component_anomaly_cross_moment_Pa2=inner(x,pa,qa),
                component_anomaly_correlation=inner(x,pa,qa)/(ps['anomaly_rms']*qs['anomaly_rms'])
                    if ps['anomaly_rms']*qs['anomaly_rms'] else None,
                identity_max_error_Pa=float(max(abs(a['sigma']-p-q))))
        columns=[x]; names=['s']
        for label,a in (('coarse',coarse),('fine',fine),('change',change)):
            for n in fields:
                columns.append(a[n]); names.append(label+'_'+n)
            columns.append(-a['tauN']); names.append(label+'_minus_tauN')
        np.savetxt(base/f'spatial-common-{k}.csv',np.column_stack(columns),delimiter=',',
                   comments='',header=','.join(names))
        for label,a,style in (('32',coarse,'--'),('64',fine,'-')):
            axes[0,k].plot(x,a['sigma']-stats(x,a['sigma'])['mean'],style,label=label)
            axes[1,k].plot(x,a['p'],style,label=label+' delta p')
            axes[1,k].plot(x,-a['tauN'],style,label=label+' -delta tau:N')
        axes[0,k].set_title(f't={t:g} s: mean-removed delta sigma_n')
        axes[1,k].set_title('Signed components (means retained)')
        for ax in axes[:,k]:
            ax.set(xlabel='fault coordinate (m)',ylabel='Pa'); ax.grid(alpha=.25); ax.legend()
        result['steps'].append(row)
    fig.tight_layout(); fig.savefig(base/'spatial-normal-feedback.png',dpi=160)
    result['analysis_seconds']=time.monotonic()-start
    (base/'spatial-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
