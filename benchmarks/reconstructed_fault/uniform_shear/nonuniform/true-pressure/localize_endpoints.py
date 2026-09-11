#!/usr/bin/env python3
"""Saved-output endpoint localization at t=.5 s; no solver or field mutation."""
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from verify_pilot import read

base=Path(__file__).resolve().parent
names=('sigma','p','minus_tauN')


def moments(x,y,a,b):
    z=np.r_[a,x[(x>a)&(x<b)],b]
    v=np.interp(z,x,y); h=np.diff(z)
    first=np.sum(h*(v[:-1]+v[1:])/2)
    second=np.sum(h*(v[:-1]**2+v[:-1]*v[1:]+v[1:]**2)/3)
    return float(first),float(second)


def response(suffix):
    a,b=(read(base/f'{n}{suffix}-surface-1.csv') for n in ('pilot','homogeneous'))
    assert np.array_equal(a['s'],b['s'])
    return a['s'],dict(sigma=a['sigma']-b['sigma'],p=a['p']-b['p'],
                      minus_tauN=-a['tauN']+b['tauN'])


def crossing(x,y,fraction):
    target=fraction*y[0]
    for i in range(len(y)-1):
        if y[i]>=target and y[i+1]<target:
            return float(x[i]+(x[i+1]-x[i])*(y[i]-target)/(y[i]-y[i+1]))
    raise ValueError('No crossing in the measured endpoint lobe')


def main():
    start=time.monotonic()
    cx,c=response(''); fx,f=response('64')
    assert cx[0]==fx[0]==0 and cx[-1]==fx[-1]
    L=cx[-1]; hc=cx[1]-cx[0]; hf=fx[1]-fx[0]
    assert np.allclose(np.diff(cx),hc,rtol=0,atol=1e-14)
    assert np.allclose(np.diff(fx),hf,rtol=0,atol=1e-14)
    x=np.unique(np.r_[cx,fx]); difference={n:np.interp(x,fx,f[n])-np.interp(x,cx,c[n]) for n in names}
    result=dict(time_s=.5,L_m=float(L),h_coarse_m=float(hc),h_fine_m=float(hf),
                comparison='fine minus coarse of bump-minus-homogeneous fields',
                cuts={},endpoint_similarity={},topology={})
    rows=[]
    # Both masks use a common physical interval. Coarse-element cuts remove
    # m coarse / 2m fine elements; fine-element cuts also expose half-coarse cuts.
    for grid,h in (('coarse',hc),('fine',hf)):
        table=[]
        for m in range(1,int(round(L/(2*h)))):
            a=m*h; b=L-a; row=dict(elements_each_end=m,width_m=float(a),fields={})
            for name in names:
                e=difference[name]; first,total=moments(x,e,0,L)
                mean=first/L
                _,left=moments(x,e,0,a); _,right=moments(x,e,b,L)
                interior_first,inside=moments(x,e,a,b)
                assert abs(left+inside+right-total)<1e-12*total
                _,anomaly_total=moments(x,e-mean,0,L)
                _,anomaly_inside=moments(x,e-mean,a,b)
                fine_first,fine_square=moments(fx,f[name],a,b)
                coarse_first,coarse_square=moments(cx,c[name],a,b)
                interior_rms=np.sqrt(inside/(b-a))
                fine_rms=np.sqrt(fine_square/(b-a))
                fine_anomaly=np.sqrt(max(0,fine_square/(b-a)-(fine_first/(b-a))**2))
                local_anomaly=np.sqrt(max(0,inside/(b-a)-(interior_first/(b-a))**2))
                metrics=dict(total_rms=float(np.sqrt(total/L)),
                    endpoint_squared_error_fraction=float((left+right)/total),
                    left_squared_error_fraction=float(left/total),right_squared_error_fraction=float(right/total),
                    endpoint_rms_global_denominator=float(np.sqrt((left+right)/L)),
                    interior_rms_global_denominator=float(np.sqrt(inside/L)),
                    interior_conditional_rms=float(interior_rms),
                    fine_interior_rms=float(fine_rms),
                    interior_error_over_fine_signal=float(interior_rms/fine_rms),
                    interior_local_anomaly_error=float(local_anomaly),
                    fine_interior_anomaly_rms=float(fine_anomaly),
                    interior_anomaly_error_over_fine_anomaly=float(local_anomaly/fine_anomaly),
                    globally_mean_removed_endpoint_fraction=float(1-anomaly_inside/anomaly_total))
                row['fields'][name]=metrics
                rows.append([grid,m,a,name]+list(metrics.values()))
            table.append(row)
        result['cuts'][grid]=table
    header='grid,elements_each_end,width_m,field,'+','.join(metrics)
    with (base/'endpoint-cuts.csv').open('w') as out:
        out.write(header+'\n')
        for row in rows: out.write(','.join(map(str,row))+'\n')
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    u=np.linspace(0,5,501)
    normalized=[u]; columns=['distance_over_h']
    for j,side in enumerate(('left','right')):
        result['endpoint_similarity'][side]={}
        for i,name in enumerate(names):
            values={}; metrics={}
            for label,z,v,h,style in (('32',cx,c[name],hc,'--'),('64',fx,f[name],hf,'-')):
                distance=z if side=='left' else L-z[::-1]
                value=v if side=='left' else v[::-1]
                values[label]=np.interp(u,distance/h,value)
                normalized.append(values[label]); columns.append(side+'_'+name+'_'+label)
                metrics[label]=dict(amplitude_Pa=float(value[0]),
                    first_interior_value_Pa=float(value[1]),
                    half_height_width_m=crossing(distance,value,.5),
                    first_zero_width_m=crossing(distance,value,0),
                    half_height_width_over_h=crossing(distance,value,.5)/h,
                    first_zero_width_over_h=crossing(distance,value,0)/h)
                axes[i,j].plot(distance[:6]/h,value[:6],style+'o',label=label)
            metrics['amplitude_ratio_64_over_32']=metrics['64']['amplitude_Pa']/metrics['32']['amplitude_Pa']
            metrics['half_height_width_ratio_64_over_32']=metrics['64']['half_height_width_m']/metrics['32']['half_height_width_m']
            metrics['first_zero_width_ratio_64_over_32']=metrics['64']['first_zero_width_m']/metrics['32']['first_zero_width_m']
            # Exact integration on integer normalized knots (the FE support
            # coordinates), not the dense plotting samples.
            knots=np.arange(3,dtype=float)
            ev=np.interp(knots,u,values['64'])-np.interp(knots,u,values['32'])
            metrics['normalized_0_to_2_profile_rms_over_coarse_amplitude']=np.sqrt(moments(knots,ev,0,2)[1]/2)/abs(metrics['32']['amplitude_Pa'])
            result['endpoint_similarity'][side][name]=metrics
            axes[i,j].set(title=side+' '+name,xlabel=('s/h_Gamma' if side=='left' else '(L-s)/h_Gamma'),ylabel='Pa')
            axes[i,j].legend(); axes[i,j].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(base/'endpoint-scaled-profiles.png',dpi=160)
    np.savetxt(base/'endpoint-scaled-profiles.csv',np.column_stack(normalized),delimiter=',',comments='',header=','.join(columns))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in names:
        cut=result['cuts']['coarse']
        axes[0].plot([r['elements_each_end'] for r in cut],
                     [100*r['fields'][name]['endpoint_squared_error_fraction'] for r in cut],'-o',label=name)
        axes[1].semilogy([r['elements_each_end'] for r in cut],
                        [r['fields'][name]['interior_error_over_fine_signal'] for r in cut],'-o',label=name)
    axes[0].set(ylabel='Endpoint share of squared error (%)')
    axes[1].set(ylabel='Interior RMS error / fine interior RMS signal')
    for ax in axes: ax.set(xlabel='Coarse elements excluded at EACH end'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(base/'endpoint-error-partition.png',dpi=160)
    for suffix,label in (('','coarse'),('64','fine')):
        weak=read(base/f'pilot{suffix}/surface_weak_1.csv')
        mass=np.diag(weak['Mdiag'])+np.diag(weak['Moff'][:-1],1)+np.diag(weak['Moff'][:-1],-1)
        n=len(weak)
        assert mass[0,-1]==mass[-1,0]==0 and len(weak['Moff'][:-1])==n-1
        result['topology'][label]=dict(independent_surface_nodes=n,segments=n-1,
            endpoint_mass_coupling=float(mass[0,-1]),
            endpoint_row_mass=[float(sum(mass[0])),float(sum(mass[-1]))],
            adjacent_row_mass=[float(sum(mass[1])),float(sum(mass[-2]))])
    result['analysis_seconds']=time.monotonic()-start
    (base/'endpoint-localization.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
