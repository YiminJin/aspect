"""Compare committing work mechanics and the saved mature volume baseline.

Native weak averages, common parent/domain FE observations, physical-QP raw
samples and saved Float32 VTU pressure are deliberately distinct diagnostics.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_uniform_sliding import read, cat, records, write
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy

HERE=Path(__file__).resolve().parent
RUN=HERE/'work-replay-50-local4'
BASE=HERE/'mature-fault-50-local4'
YEAR=31557600.
L=100000/(np.sqrt(3)/2)
REGIONS={'top':(-2000,2500),'15km':(13000,16500),'18km':(16500,20000),
         '40km':(37000,43000),'interior':(59000,61000),'bottom':(L-2500,L+2000)}


def pressure_pattern(root,k):
    # Full saved physical-node set; the earlier helper only kept the junction.
    values={}
    for path in sorted(root.glob(f'bulk_{k}_*.vtu')):
        reader=vtkXMLUnstructuredGridReader();reader.SetFileName(str(path));reader.Update()
        grid=reader.GetOutput();xy=vtk_to_numpy(grid.GetPoints().GetData())
        pressure=vtk_to_numpy(grid.GetPointData().GetArray('delta_pressure'))
        for point,value in zip(xy,pressure):
            key=tuple(float(x) for x in point);value=float(value)
            if key in values:
                assert abs(values[key]-value)<=1e-3+5e-7*abs(value),'Inconsistent duplicated VTU node'
            values[key]=value
    return values


def baseline_weak(k,n):
    d=cat(BASE.glob(f'history_surface_step{k}_rank*.csv'))
    result={key:np.bincount(d['node'].astype(int),weights=d[key],minlength=n)
            for key in ('weight','p','particle_q','particle_sigma','fe_q','fe_sigma')}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    out=RUN/'comparison';out.mkdir(exist_ok=True)
    accepted=read(RUN/'accepted_steps.csv');reference=read(BASE/'accepted_steps.csv')
    log=(RUN/'run.log').read_text().replace('\0','')
    linear=[tuple(map(float,x)) for x in re.findall(r'Fault linear solve: iterations=\d+, estimated=[^,]+, fresh=([^,]+), target=([^,]+)',log)]
    assert linear and all(a<=b for a,b in linear)
    if not args.partial:
        assert json.loads((RUN/'execution.json').read_text())['status']==0
        assert abs(accepted['time'][-1]-reference['time'][-1])<1e-5
        assert 'BP3 WORK REPLAY FIRST UPDATE PASSED' in log
    regions=[];mechanics=[];raw_results=[];pressure_results=[];history=[];nonlinear=[];nodes=[]
    previous=None;initial=None;matched=[]
    reached=np.flatnonzero(abs(accepted['time']-reference['time'][-1])<1e-5)
    primary_final_step=int(accepted['step'][reached[0]]) if len(reached) else None
    for k,time in zip(accepted['step'].astype(int),accepted['time']):
        if not (RUN/f'fault_{k}.csv').exists():continue
        f=read(RUN/f'fault_{k}.csv');h=read(RUN/f'history_{k}.csv')
        assert h['Theta_reference_relative_error'][0]<1e-12
        np.testing.assert_array_equal(f['C'],np.zeros(len(f['V'])))
        assert np.all(f['V'][f['prescribed']==1]==1e-9)
        if initial is None:initial=f
        for key in ('x','y','Ih'):np.testing.assert_array_equal(f[key],initial[key])
        if previous is not None:
            np.testing.assert_allclose(f['slip'],previous['slip']+f['dt']*f['V'],rtol=1e-13,atol=1e-15)
        section=log.split(f'*** Timestep {k}:',1)[1].split('*** Timestep ',1)[0]
        nr=re.findall(r'Relative nonlinear residuals .*?: ([^,\n]+), ([^\n]+)',section)[-1]
        assert all(float(v)<1e-8 for v in nr)
        nonlinear.append(dict(step=int(k),time=float(time),bulk=float(nr[0]),surface=float(nr[1])))
        history.append(dict(step=int(k),time=float(time),Theta_error=float(h['Theta_reference_relative_error'][0]),
                            stress_max_Pa=float(h['committed_particle_stress_max_Pa'][0])))
        raw=cat(RUN.glob(f'work_qp_{k}_rank*.csv'),('cell',))
        np.testing.assert_allclose(raw['sigma_n'],50e6+raw['p']-raw['tauN'],rtol=1e-13,atol=1e-7)
        for name,(lo,hi) in REGIONS.items():
            mask=(raw['xd']>=lo)&(raw['xd']<=hi)
            d=dict(step=int(k),time_yr=float(time/YEAR),region=name,samples=int(sum(mask)),
                   missing_positive_source=int(sum(mask & (raw['source_active']==0))))
            for field in ('p','sigma_n','tauN','q','elastic_norm'):
                d[field+'_min']=float(min(raw[field][mask]));d[field+'_max']=float(max(raw[field][mask]))
            raw_results.append(d)
        old=np.where(abs(reference['time']-time)<1e-5)[0]
        previous=f
        if not len(old):continue
        b=int(old[0]);matched.append(dict(new_step=int(k),baseline_step=b,time=float(time)))
        # A two-ULP end-time remainder generated an extra near-zero-dt state.
        # Preserve/check it above, but never replace the actual clock-matched
        # step-10 trajectory with this different Maxwell interval.
        if primary_final_step is not None and k>primary_final_step:
            matched.pop();continue
        g=read(BASE/f'fault_{b}.csv');w=read(RUN/f'work_weak_{k}.csv');c=read(RUN/f'common_fe_weak_{k}.csv')
        z=baseline_weak(b,len(f['V']));xd=f['xd']
        np.testing.assert_array_equal(xd,g['xd'])
        profiles=dict(xd=xd,V_new=f['V'],V_baseline=g['V'],Theta_new=f['Theta'],Theta_baseline=g['Theta'],
            slip_new=f['slip'],slip_baseline=g['slip'],
            p_common_new=c['p']/c['weight'],p_common_baseline=z['p']/z['weight'],
            sigma_common_new=c['sigma']/c['weight'],sigma_common_baseline=z['fe_sigma']/z['weight'],
            minus_tauN_common_new=-c['tauN']/c['weight'],
            minus_tauN_common_baseline=z['fe_sigma']/z['weight']-50e6-z['p']/z['weight'],
            sigma_native_new=w['sigma']/w['weight'],sigma_native_baseline=z['particle_sigma']/z['weight'],
            q_common_new=c['q']/c['weight'],q_common_baseline=z['fe_q']/z['weight'])
        write(out/f'profiles_{k}.csv',profiles)
        for target in (0.,15000.,18000.,25000.,39900.,39950.,40000.,L):
            j=int(np.argmin(abs(xd-target)))
            for label,data in [('new',f),('baseline',g)]:
                nodes.append(dict(step=int(k),baseline_step=b,time_yr=float(time/YEAR),target_xd=target,
                    actual_xd=float(xd[j]),node=j,case=label,V=float(data['V'][j]),Theta=float(data['Theta'][j]),
                    slip=float(data['slip'][j]),sigma_common=float(profiles['sigma_common_'+label][j]),
                    p_common=float(profiles['p_common_'+label][j])))
        centers=(xd[:-1]+xd[1:])/2
        grad_new=np.diff(f['slip'])/np.diff(xd);grad_old=np.diff(g['slip'])/np.diff(xd)
        write(out/f'slip_gradient_{k}.csv',dict(xd=centers,new=grad_new,baseline=grad_old))
        for name,(lo,hi) in REGIONS.items():
            mask=(xd>=lo)&(xd<=hi);elements=(centers>=lo)&(centers<=hi)
            for label,values in [('new',f),('baseline',g)]:
                d=dict(step=int(k),baseline_step=b,time_yr=float(time/YEAR),region=name,case=label)
                for key in ('V','Theta','slip'):
                    d[key+'_min']=float(min(values[key][mask]));d[key+'_max']=float(max(values[key][mask]))
                d['max_abs_slip_gradient']=float(max(abs((grad_new if label=='new' else grad_old)[elements])))
                mechanics.append(d)
            for field in ('p_common','minus_tauN_common','sigma_common','sigma_native','q_common'):
                a=profiles[field+'_new'][mask];v=profiles[field+'_baseline'][mask]
                amp=lambda x:float(max(abs(x-(50e6 if field.startswith('sigma') else 0.))))
                regions.append(dict(step=int(k),baseline_step=b,time_yr=float(time/YEAR),region=name,field=field,
                    new_min=float(min(a)),new_max=float(max(a)),baseline_min=float(min(v)),baseline_max=float(max(v)),
                    new_ptp=float(np.ptp(a)),baseline_ptp=float(np.ptp(v)),
                    new_amplitude=amp(a),baseline_amplitude=amp(v)))
        if (RUN/f'bulk_{k}.pvtu').exists() and (BASE/f'bulk_{b}.pvtu').exists():
            a=pressure_pattern(RUN,k);v=pressure_pattern(BASE,b)
            assert a.keys()==v.keys()
            points=np.array(list(a));pa=np.array(list(a.values()));pv=np.array([v[p] for p in a])
            s=(100000-points[:,1])*np.sqrt(3)/2+(100000*(.5+.25/(np.sqrt(3)/2))-points[:,0])*.5
            normal=(100000*(.5+.25/(np.sqrt(3)/2))-points[:,0])*np.sqrt(3)/2-(100000-points[:,1])*.5
            for name,(lo,hi) in REGIONS.items():
                mask=(s>=lo)&(s<=hi)&(abs(normal)<1200.)
                pressure_results.append(dict(step=int(k),baseline_step=b,region=name,count=int(sum(mask)),
                    new_min=float(min(pa[mask])),new_max=float(max(pa[mask])),
                    baseline_min=float(min(pv[mask])),baseline_max=float(max(pv[mask])),
                    difference_RMS=float(np.sqrt(np.mean((pa[mask]-pv[mask])**2)))))
        # Full-resolution along-fault curves, not smoothing or initial-error subtraction.
        fig,axes=plt.subplots(3,2,figsize=(12,10))
        fields=[('V','m/s'),('Theta','s'),('slip','m'),('sigma_common','Pa'),('p_common','Pa'),('sigma_native','Pa')]
        order=np.argsort(xd)
        for ax,(field,unit) in zip(axes.flat,fields):
            for suffix,label in [('baseline','old mature'),('new','work replay')]:
                ax.plot(xd[order]/1000,profiles[field+'_'+suffix][order],label=label,lw=1)
            for mark in (15,18,40):ax.axvline(mark,color='k',ls=':',lw=.5)
            ax.set(xlabel='Down-dip km',ylabel=f'{field} ({unit})');ax.legend(fontsize=7)
            if field in ('V','Theta'):ax.set_yscale('log')
        fig.suptitle(f'{time/YEAR:.6g} yr; native averages use different measures')
        fig.tight_layout();fig.savefig(out/f'profiles_{k}.png',dpi=130);plt.close(fig)
        if not args.partial and k==primary_final_step:
            for name,limits in [('transitions',(13,20)),('junction',(37,43)),('top',(0,2.5)),('bottom',((L-2500)/1000,L/1000))]:
                fig,axes=plt.subplots(3,2,figsize=(12,9))
                for ax,(field,unit) in zip(axes.flat,fields):
                    for suffix,label in [('baseline','old mature'),('new','work replay')]:
                        ax.plot(xd[order]/1000,profiles[field+'_'+suffix][order],label=label,lw=1)
                    for mark in (15,18,40):
                        if limits[0]<=mark<=limits[1]:ax.axvline(mark,color='k',ls=':',lw=.7)
                    ax.set(xlabel='Down-dip km',ylabel=f'{field} ({unit})',xlim=limits)
                    mask=(xd/1000>=limits[0])&(xd/1000<=limits[1])
                    low=min(profiles[field+'_new'][mask].min(),profiles[field+'_baseline'][mask].min())
                    high=max(profiles[field+'_new'][mask].max(),profiles[field+'_baseline'][mask].max())
                    if field in ('V','Theta'):
                        ax.set_yscale('log');ax.set_ylim(low*.8,high*1.2)
                    elif high>low:ax.set_ylim(low-.05*(high-low),high+.05*(high-low))
                    ax.legend(fontsize=7)
                fig.suptitle(f'{name}, {time/YEAR:.6g} yr; native measures differ')
                fig.tight_layout();fig.savefig(out/f'final_{name}.png',dpi=140);plt.close(fig)
    for name,rows in [('regions',regions),('mechanics',mechanics),('raw_qp',raw_results),('pressure_common_vertices',pressure_results),('history',history),('selected_nodes',nodes)]:
        if rows:records(out/(name+'.csv'),rows)
    summary=dict(complete=not args.partial,accepted_states=len(nonlinear),matched_times=matched,
        nonlinear=nonlinear,fresh_linear_checks=len(linear),worst_fresh_ratio=max(a/b for a,b in linear),
        first_update_marker='BP3 WORK REPLAY FIRST UPDATE PASSED' in log,
        first_update=read(RUN/'first_update_maxwell.csv') if (RUN/'first_update_maxwell.csv').exists() else None,
        sampling_caveat='Common FE uses the same parent/domain rule on each trajectory; advected parents differ. Native weak measures differ. Raw old extrema are particle-history samples; new raw are bulk QPs. VTU pressure is common physical nodes, Float32.')
    old_initial=read(BASE/'fault_0.csv')
    summary['initial_comparison']={field:dict(
        max_absolute_change=float(max(abs(initial[field]-old_initial[field]))),
        new_range=[float(min(initial[field])),float(max(initial[field]))],
        baseline_range=[float(min(old_initial[field])),float(max(old_initial[field]))])
        for field in ('Theta','Ih','V','C','slip')}
    summary['accepted_clock_max_change_s']=max(abs(m['time']-reference['time'][m['baseline_step']]) for m in matched)
    summary['primary_final_step']=primary_final_step
    summary['end_time_remainder_states']=[{key:float(accepted[key][j]) for key in ('step','time','dt')}
        for j in range(len(accepted['time'])) if primary_final_step is not None and accepted['step'][j]>primary_final_step]
    summary['weak_stress_observer_error_Pa']=max(map(float,re.findall(r'frozen weak stress discrepancy=([^ ]+) Pa',log)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.tolist())+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('nonlinear','matched_times')},indent=2,default=lambda x:x.tolist()))


if __name__=='__main__':main()
