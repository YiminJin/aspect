"""Matched-clock history, jump and current-stress audit of the seven-step replay."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-trace-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_uniform_sliding import read, cat, select, records, write
from analyze_free_trace import weak

HERE = Path(__file__).resolve().parent
BASE = HERE/'work-replay-50-local4'
RUN = HERE/'trace-replay-seven-local4'
VP = 1e-9
YEAR = 31557600.


def raw(root, step):
    q = cat(root.glob(f'work_qp_{step}_rank*.csv'), ('cell',))
    q = select(q, (q['source_active']==1)&(q['chi']>0))
    q = select(q, np.lexsort((q['y'], q['x'])))
    q['weight'] = q['JxW']*q['chi']
    q['shape1'] = q['xi'].copy()
    if root==RUN: q['shape1'][q['segment']==794] = 0.
    q['minus_tauN'] = -q['tauN']
    np.testing.assert_allclose(q['sigma_n'], 50e6+q['p']-q['tauN'], rtol=2e-14, atol=1e-7)
    return q


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through',type=int,default=7,choices=range(8))
    last=parser.parse_args().through
    if last==7:
        execution = json.loads((RUN/'execution.json').read_text())
        assert execution['status']==0 and execution['first_update_passed'] and execution['fresh_linear_passed']
    out = RUN/('analysis' if last==7 else f'analysis-through-{last}'); out.mkdir(exist_ok=True)
    histories=[]; stress=[]; nodes=[]; budgets=[]; checks=[]; matches=[]; traces=[]
    previous={}; initial={}; curves={}
    for k in range(last+1):
        fields={tag:read(root/f'fault_{k}.csv') for tag,root in [('shared',BASE),('split',RUN)]}
        for key in ('time','dt','xd','x','y','Ih'):
            np.testing.assert_array_equal(fields['shared'][key],fields['split'][key])
        if k==0:
            np.testing.assert_array_equal(fields['shared']['Theta'],fields['split']['Theta'])
            for rank in range(4):
                a,b=[read(root/f'mature_history_0_rank{rank}.csv') for root in (BASE,RUN)]
                for key in ('id','H_inert','tau_xx','tau_yy','tau_xy'):
                    np.testing.assert_array_equal(a[key],b[key])
        samples={tag:raw(root,k) for tag,root in [('shared',BASE),('split',RUN)]}
        for key in ('x','y','JxW','segment','xi','phi','Ih','chi'):
            if key=='chi':
                # Re-evaluating the FE material mixture after different particle
                # motion changes the last bits even on the uniform plateau.
                np.testing.assert_allclose(samples['shared'][key],samples['split'][key],rtol=1e-12,atol=1e-22)
            else:
                np.testing.assert_array_equal(samples['shared'][key],samples['split'][key])
        for tag,root in [('shared',BASE),('split',RUN)]:
            f=fields[tag]; q=samples[tag]; old=previous.get(tag,f)
            n=len(f['V']); j=q['segment'].astype(int); xi=q['shape1']
            np.testing.assert_allclose(q['V'],(1-xi)*f['V'][j]+xi*f['V'][j+1],rtol=4e-14,atol=1e-28)
            deep=795 if tag=='shared' else 794
            np.testing.assert_array_equal(f['prescribed'][:deep+1],1.)
            np.testing.assert_array_equal(f['V'][:deep+1],VP)
            np.testing.assert_allclose(f['slip'][:deep+1],VP*f['time'][0],rtol=3e-15,atol=1e-17)
            if k:
                dt=f['dt'][0]; x=f['V']*dt/.008
                expected=old['Theta']*np.exp(-x)-.008/f['V']*np.expm1(-x)
                np.testing.assert_allclose(f['Theta'],expected,rtol=1e-12)
                np.testing.assert_allclose(f['slip'],old['slip']+dt*f['V'],rtol=3e-15,atol=1e-17)
            else:
                np.testing.assert_array_equal(f['slip'],0.)
                initial[tag]=f
            # Correctly timed friction: committed preceding nodal state, with
            # the deep-side trace selected by the same one-sided basis.
            theta=(1-xi)*old['Theta'][j]+xi*old['Theta'][j+1]
            mu=.025*np.arcsinh(q['V']/(2e-6)*np.exp((.6+.015*np.log(1e-6*theta/.008))/.025))
            # Only the 37--43 km window is on this single material plateau.
            local=(q['xd']>=37000)&(q['xd']<=43000)
            qq=select(q,local); mm=mu[local]
            loads={'shear':qq['q'],'friction':-mm*qq['sigma_n'],'damping':-4624440.*qq['V']}
            loads['R']=sum(loads.values())
            for node in (794,795,796,797):
                for segment in (node-1,node):
                    mask=qq['segment']==segment
                    for term,values in loads.items():
                        load=weak(select(qq,mask),values[mask],n)[node]
                        budgets.append(dict(case=tag,step=k,node=node,segment=segment,term=term,load_Pa_m=load))
                reproduced=weak(qq,loads['R'],n)[node]
                # Compare unreplaced physical rows, including prescribed rows.
                error=abs(reproduced-f['weak_residual'][node])
                assert error<.003,(tag,k,node,error)
                checks.append(dict(case=tag,step=k,node=node,weak_load_error_Pa_m=error))
            v=f['V'][[797,796,795]]/VP
            chord=.5*(v[0]+v[2])-v[1]
            jump=f['slip'][795]-f['slip'][deep]
            histories.append(dict(case=tag,step=k,time_years=f['time'][0]/YEAR,
                V39900=v[0],V39950=v[1],V40000minus=v[2],V40000plus=1.,
                chord_notch=chord,below_both=max(0.,min(v[0],v[2])-v[1]),
                rate_jump_over_Vp=v[2]-1.,slip_jump_m=jump,
                slip_jump_fraction=jump/(VP*f['time'][0]) if k else 0.,
                slip_gradient=(f['slip'][795]-f['slip'][796])/50.,
                Theta39950=f['Theta'][796],Theta40000minus=f['Theta'][795],Theta40000plus=f['Theta'][deep],
                theta_peak=f['Theta'][796]-.5*(f['Theta'][797]+f['Theta'][795])))
            for side,node in [('free',795),('deep',deep)]:
                traces.append(dict(case=tag,step=k,side=side,xd=40000.,storage_node=node,
                                   V=f['V'][node],Theta=f['Theta'][node],slip=f['slip'][node]))
            native=read(root/f'work_weak_{k}.csv')
            for node in range(790,806):
                nodes.append(dict(case=tag,step=k,node=node,
                                  **{key:float(f[key][node]) for key in ('xd','V','Theta','slip','prescribed','weak_residual')},
                                  **{'weak_'+key:float(native[key][node]/native['weight'][node]) for key in ('p','tauN','q','sigma')},
                                  native_weight=float(native['weight'][node])))
            for name,lo,hi in [('junction',39800.,40200.),('junction_wide',37000.,43000.),('transition',13000.,20000.),('deep_control',59000.,61000.)]:
                mask=(q['xd']>=lo)&(q['xd']<=hi)
                if not np.any(mask):continue
                for key in ('p','tau_xx','tau_yy','tau_xy','minus_tauN','sigma_n','q'):
                    a=q[key][mask];w=q['weight'][mask];mean=np.sum(w*a)/sum(w)
                    stress.append(dict(case=tag,step=k,window=name,field=key,min_Pa=min(a),max_Pa=max(a),
                                       min_xd=q['xd'][mask][np.argmin(a)],max_xd=q['xd'][mask][np.argmax(a)],
                                       span_Pa=np.ptp(a),mean_Pa=mean,RMS_variation_Pa=np.sqrt(np.sum(w*(a-mean)**2)/sum(w))))
            if tag=='split':
                # The last Jacobian export is a mechanical input, not a new
                # accepted-state stress diagnostic; use it to verify Theta timing.
                paths=[]
                for path in root.glob(f'state_qp_{k}_rank*.csv'):
                    with path.open() as stream:
                        next(stream)
                        if next(stream,None) is not None:paths.append(path)
                qp=cat(paths,('cell',)); jj=qp['segment'].astype(int)
                zz=qp['xi'].copy();zz[jj==794]=0.
                expected=(1-zz)*old['Theta'][jj]+zz*old['Theta'][jj+1]
                np.testing.assert_allclose(qp['Theta'],expected,rtol=3e-14)
                np.testing.assert_array_equal(qp['V'][jj==794],VP)
                z=qp['V']/(2e-6)*np.exp((.6+.015*np.log(1e-6*expected/.008))/.025)
                np.testing.assert_allclose(qp['friction'],.025*np.arcsinh(z)*qp['sigma'],rtol=4e-14)
                dt=f['dt'][0] if k else 4e6
                kappa=-1e26*np.expm1(-dt*32038120320./1e26)
                tangent=kappa*qp['chi']+qp['sigma']*.025*z/(qp['V']*np.sqrt(1+z*z))+4624440.
                np.testing.assert_allclose(qp['Kfixed'],tangent,rtol=5e-13)
                np.testing.assert_array_equal(qp['Kstate0'],0.)
                np.testing.assert_array_equal(qp['Kstate1'],0.)
            curves[(tag,k)]=f
            previous[tag]=f
        a,b=samples['shared'],samples['split']
        mask=(a['xd']>=39800)&(a['xd']<=40200)
        for field in ('p','tau_xx','tau_yy','tau_xy','sigma_n','q'):
            d=b[field][mask]-a[field][mask]
            matches.append(dict(step=k,field=field,max_abs_difference_Pa=max(abs(d)),RMS_difference_Pa=np.sqrt(np.mean(d*d))))
        if k in (0,1,2,7):
            window=(a['xd']>=39500)&(a['xd']<=40500)
            data={key:a[key][window] for key in ('x','y','xd','chi','Ih','JxW')}
            for tag,q in samples.items():
                for field in ('V','p','tau_xx','tau_yy','tau_xy','sigma_n','q'):
                    data[tag+'_'+field]=q[field][window]
            write(out/f'matched_current_QP_{k}.csv',data)
    for name,data in [('histories',histories),('endpoint_traces',traces),('raw_stress',stress),('nodes',nodes),('element_budgets',budgets),('checks',checks),('matching_differences',matches)]:
        records(out/f'{name}.csv',data)
    fig,ax=plt.subplots(2,2,figsize=(11,8))
    for tag in ('shared','split'):
        color='tab:blue' if tag=='shared' else 'tab:orange'
        h=[r for r in histories if r['case']==tag];t=[r['time_years'] for r in h]
        ax[0,0].plot(t,[r['chord_notch'] for r in h],'-o',label=tag,color=color)
        ax[0,1].plot(t,[1000*r['slip_jump_m'] for r in h],'-o',label=tag,color=color)
        f=curves[(tag,last)];m=(f['xd']>=39700)&(f['xd']<=40300)
        # Plot each side separately: never connect distinct traces as one Q1 edge.
        for side in (m&(f['xd']<=40000.00001),m&(f['xd']>40000.00001)):
            ax[1,0].plot(f['xd'][side]/1000,f['V'][side]/VP,'-o',color=color,label=tag if np.any(side& (f['xd']<40000)) else None)
        if tag=='split':
            ax[1,0].plot([40.,f['xd'][794]/1000],[1.,1.],'-s',color='tab:orange')
        s=[r for r in stress if r['case']==tag and r['window']=='junction' and r['field']=='sigma_n']
        ax[1,1].plot(t,[r['span_Pa']/1000 for r in s],'-o',label=tag,color=color)
    for a,title in zip(ax.flat,['Within-free chord notch / Vp','Slip(40-) - slip(40+) [mm]','Final V / Vp','Current normal-stress span, 39.8–40.2 km [kPa]']):
        a.set_title(title);a.grid();a.legend()
    for a in (ax[0,0],ax[0,1],ax[1,1]):a.set_xlabel('Time [yr]')
    ax[1,0].set_xlabel('Down-dip distance [km]')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    result=dict(passed=True,steps=last+1,time_years=histories[-1]['time_years'],
                max_weak_reproduction_error_Pa_m=max(c['weak_load_error_Pa_m'] for c in checks),
                final=[r for r in histories if r['step']==last])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
