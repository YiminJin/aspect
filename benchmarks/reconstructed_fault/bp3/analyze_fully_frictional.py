"""Compare the fresh all-frictional replay with the saved constrained clock."""
import argparse
import json
import os
from pathlib import Path
import re

os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-fully-frictional-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_uniform_sliding import read, cat, select, records, write
from analyze_free_trace import weak

HERE=Path(__file__).resolve().parent
BASE=HERE/'work-replay-50-local4'
RUN=HERE/'fully-frictional-seven-local4'
VP=1e-9
YEAR=31557600.
L=1e5/(np.sqrt(3)/2)


def samples(root,k):
    q=cat(root.glob(f'work_qp_{k}_rank*.csv'),('cell',))
    q=select(q,np.lexsort((q['y'],q['x'])))
    q['weight']=q['JxW']*q['chi'];q['shape1']=q['xi']
    q['minus_tauN']=-q['tauN']
    np.testing.assert_allclose(q['sigma_n'],50e6+q['p']-q['tauN'],rtol=2e-14,atol=1e-7)
    return q


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through',type=int,default=7,choices=range(8))
    parser.add_argument('--run',type=Path,default=HERE/'fully-frictional-seven-local4')
    args=parser.parse_args()
    global RUN
    RUN=args.run.resolve()
    last=args.through
    if last==7:
        ex=json.loads((RUN/'execution.json').read_text())
        assert ex['status']==0 and ex['fresh_linear_passed'] and ex['first_update_passed']
    out=RUN/('analysis' if last==7 else f'analysis-through-{last}');out.mkdir(exist_ok=True)
    log=(RUN/'run.log').read_text()
    nodes=[]; regions=[]; stresses=[]; junction=[]; checks=[]; balances=[]; differences=[]
    previous={};initial={};initial_weak={};final={}
    nodal_regions={'VW_core':(0,13000),'shallow':(0,15000),'transition':(15000,18000),
                   'junction':(39500,40500),'deep':(45000,100000),'top':(0,2500),'bottom':(L-2500,L)}
    raw_regions={'junction':(39800,40200),'deep_entry':(37000,43000),'15km':(13000,16500),'18km':(16500,20000),
                 'deep_control':(59000,61000),'top':(-2000,2500),'bottom':(L-2500,L+2000)}
    for k in range(last+1):
        pair={tag:read(root/f'fault_{k}.csv') for tag,root in [('constrained',BASE),('frictional',RUN)]}
        a,b=pair.values()
        for key in ('xd','x','y','time','dt','Ih'):
            np.testing.assert_array_equal(a[key],b[key])
        if k==0:
            np.testing.assert_array_equal(a['Theta'],b['Theta'])
            for rank in range(4):
                p,r=[read(root/f'mature_history_0_rank{rank}.csv') for root in (BASE,RUN)]
                for key in p:np.testing.assert_array_equal(p[key],r[key])
        raw_pair={tag:samples(root,k) for tag,root in [('constrained',BASE),('frictional',RUN)]}
        a,b=raw_pair.values()
        for key in ('x','y','JxW','source_active','segment','xi','phi','Ih'):
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_allclose(a['chi'],b['chi'],rtol=1e-12,atol=1e-22)
        section=log.split(f'*** Timestep {k}:',1)[1].split('*** Timestep ',1)[0]
        residual=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',section)[-1]
        assert all(float(v)<1e-8 for v in residual),residual
        for tag,root in [('constrained',BASE),('frictional',RUN)]:
            f=pair[tag];q=raw_pair[tag];n=len(f['V']);t=f['time'][0]
            w=read(root/f'work_weak_{k}.csv');old=previous.get(tag,f)
            if tag=='frictional':np.testing.assert_array_equal(f['prescribed'],0.)
            if k:
                dt=f['dt'][0];x=f['V']*dt/.008
                expected=old['Theta']*np.exp(-x)-.008/f['V']*np.expm1(-x)
                np.testing.assert_allclose(f['Theta'],expected,rtol=1e-12)
                np.testing.assert_allclose(f['slip'],old['slip']+dt*f['V'],rtol=3e-15,atol=1e-17)
            else:
                initial[tag]=f;initial_weak[tag]=w
                np.testing.assert_array_equal(f['slip'],0.)
            for key in ('x','y','Ih'):np.testing.assert_array_equal(f[key],initial[tag][key])
            j=q['segment'].astype(int);xi=q['xi'];mask=q['source_active']==1
            np.testing.assert_allclose(q['V'][mask],((1-xi)*f['V'][j]+xi*f['V'][j+1])[mask],rtol=4e-14,atol=1e-28)
            # Exact fixed-state RSF check on the deep a=.025 plateau,
            # including the now-free bottom continued-source QPs.
            qq=select(q,mask&(q['chi']>0)&(q['xd']>=37000));jj=qq['segment'].astype(int);zz=qq['xi']
            theta=(1-zz)*old['Theta'][jj]+zz*old['Theta'][jj+1]
            mu=.025*np.arcsinh(qq['V']/(2e-6)*np.exp((.6+.015*np.log(1e-6*theta/.008))/.025))
            r=weak(qq,qq['q']-mu*qq['sigma_n']-4624440.*qq['V'],n)
            for node in (0,1,2,794,795,796,797):
                err=abs(r[node]-f['weak_residual'][node]);assert err<.003,(tag,k,node,err)
                checks.append(dict(case=tag,step=k,node=node,weak_reproduction_error_Pa_m=err))
                balances.append(dict(case=tag,step=k,node=node,xd=f['xd'][node],
                                     native_mass=w['weight'][node],residual_Pa=f['weak_residual'][node]/w['weight'][node],
                                     q_Pa=w['q'][node]/w['weight'][node],sigma_Pa=w['sigma'][node]/w['weight'][node]))
            v=f['V'][[797,796,795]]/VP
            junction.append(dict(case=tag,step=k,time_years=t/YEAR,V39900=v[0],V39950=v[1],V40000=v[2],
                                 V40050=f['V'][794]/VP,chord_notch=.5*(v[0]+v[2])-v[1],
                                 below_both=max(0.,min(v[0],v[2])-v[1]),
                                 slip_gradient=(f['slip'][795]-f['slip'][796])/50.))
            for target in (0,5000,10000,15000,18000,25000,39900,39950,40000,40050,45000,60000,80000,100000,L):
                i=int(np.argmin(abs(f['xd']-target)))
                nodes.append(dict(case=tag,step=k,time_years=t/YEAR,xd=f['xd'][i],target=target,node=i,
                                  V_over_Vp=f['V'][i]/VP,Theta=f['Theta'][i],slip=f['slip'][i],
                                  deficit_m=VP*t-f['slip'][i],q_Pa=w['q'][i]/w['weight'][i],
                                  dq_Pa=w['q'][i]/w['weight'][i]-initial_weak[tag]['q'][i]/initial_weak[tag]['weight'][i],
                                  sigma_Pa=w['sigma'][i]/w['weight'][i],p_Pa=w['p'][i]/w['weight'][i]))
            order=np.argsort(f['xd']);xd=f['xd'][order]
            # Endpoint-half-length weights are purely an along-fault reporting
            # measure, not a replacement for production work quadrature.
            cell_weights=np.zeros(n);h=np.diff(xd);cell_weights[:-1]+=h/2;cell_weights[1:]+=h/2
            weights=np.empty(n);weights[order]=cell_weights
            for name,(lo,hi) in nodal_regions.items():
                m=(f['xd']>=lo-1e-7)&(f['xd']<=hi+1e-7);ww=weights[m]
                deficit=VP*t-f['slip'][m];delta_q=w['q'][m]/w['weight'][m]-initial_weak[tag]['q'][m]/initial_weak[tag]['weight'][m]
                regions.append(dict(case=tag,step=k,region=name,time_years=t/YEAR,
                                    V_min_over_Vp=min(f['V'][m])/VP,V_max_over_Vp=max(f['V'][m])/VP,
                                    V_mean_over_Vp=np.sum(ww*f['V'][m])/sum(ww)/VP,
                                    mean_deficit_m=np.sum(ww*deficit)/sum(ww),
                                    mean_delta_q_Pa=np.sum(ww*delta_q)/sum(ww),min_delta_q_Pa=min(delta_q),max_delta_q_Pa=max(delta_q)))
            for name,(lo,hi) in raw_regions.items():
                m=(q['xd']>=lo)&(q['xd']<=hi)
                for key in ('p','tau_xx','tau_yy','tau_xy','minus_tauN','sigma_n','q','elastic_norm'):
                    values=q[key][m];ww=q['weight'][m];mean=np.sum(ww*values)/sum(ww)
                    stresses.append(dict(case=tag,step=k,region=name,field=key,
                                         min=min(values),max=max(values),span=np.ptp(values),
                                         mean=mean,RMS_variation=np.sqrt(np.sum(ww*(values-mean)**2)/sum(ww)),
                                         unassociated_positive_phase_QPs=int(np.sum(q['source_active'][m]==0))))
            if k in (0,last):
                write(out/f'{tag}_raw_{k}.csv',q)
                write(out/f'{tag}_nodes_{k}.csv',dict(xd=f['xd'],V=f['V'],Theta=f['Theta'],slip=f['slip'],
                    q=w['q']/w['weight'],sigma=w['sigma']/w['weight'],p=w['p']/w['weight']))
            previous[tag]=f;final[tag]=(f,w)
        for name,(lo,hi) in raw_regions.items():
            a,b=raw_pair.values();m=(a['xd']>=lo)&(a['xd']<=hi)
            for key in ('p','tau_xx','tau_yy','tau_xy','sigma_n','q'):
                d=b[key][m]-a[key][m]
                differences.append(dict(step=k,region=name,field=key,max_abs=max(abs(d)),RMS=np.sqrt(np.mean(d*d))))
    for name,rows in [('junction',junction),('nodes',nodes),('regions',regions),('raw_stress',stresses),
                      ('checks',checks),('weak_balances',balances),('matching_differences',differences)]:
        records(out/f'{name}.csv',rows)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for tag,(f,w) in final.items():
        xd=f['xd']/1000;t=f['time'][0]
        axes[0,0].plot(xd,f['V']/VP,label=tag)
        m=(xd>=39.5)&(xd<=40.5);axes[0,1].plot(xd[m],f['V'][m]/VP,'-o',label=tag)
        axes[0,2].plot(xd,(VP*t-f['slip'])*1000,label=tag)
        dq=w['q']/w['weight']-initial_weak[tag]['q']/initial_weak[tag]['weight']
        axes[1,0].plot(xd,dq/1000,label=tag)
        for ax,region in zip(axes[1,1:],('top','bottom')):
            values=[r for r in stresses if r['case']==tag and r['region']==region and r['field']=='sigma_n']
            times=[r['time_years'] for r in junction if r['case']==tag]
            ax.plot(times,[r['span']/1000 for r in values],'-o',label=tag)
    titles=['Final V / Vp','40-km neighbourhood: V / Vp','Final slip deficit [mm]',
            'Change in native weak shear traction [kPa]','Top raw normal-stress span [kPa]','Bottom raw normal-stress span [kPa]']
    for i,(ax,title) in enumerate(zip(axes.flat,titles)):
        ax.set_title(title);ax.grid();ax.legend();ax.set_xlabel('Time [yr]' if i>=4 else 'Down-dip distance [km]')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    summary=dict(passed=True,last_step=last,final_time_years=previous['frictional']['time'][0]/YEAR,
                 max_weak_reproduction_error_Pa_m=max(r['weak_reproduction_error_Pa_m'] for r in checks),
                 final_junction=[r for r in junction if r['step']==last],
                 final_regions=[r for r in regions if r['step']==last])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
