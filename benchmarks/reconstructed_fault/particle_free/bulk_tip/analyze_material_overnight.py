#!/usr/bin/env python3
"""Midpoint temporal screen against retained BE controls, with actual time locations."""
import argparse,json,os
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import analyze_synthetic_growth as growth
from analyze_time_update import read,frame,compare,balance
ROOT=growth.ROOT
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/material_overnight'
growth.OUT=OUT

def read_points(path):
    # Explicit formats avoid Python type inference on millions of diagnostic
    # rows. CellId is the sole textual column; all other columns are numeric.
    with path.open() as stream:
        names=stream.readline().strip().split(',')
    dtype=[(name,'U32' if name=='cell' else 'f8') for name in names]
    return np.atleast_1d(np.loadtxt(path,delimiter=',',skiprows=1,dtype=dtype)).view(np.recarray)

def fast_frame(directory,step):
    arrays=[read_points(p) for p in sorted(directory.glob(f'tip-points-{step:04d}-r*.csv'))]
    values=np.concatenate(arrays).view(np.recarray)
    return values[np.lexsort((values.q,values.cell))]

def fast_balance(directory,pattern):
    values=np.concatenate([read_points(p) for p in sorted(directory.glob(pattern))]).view(np.recarray)
    return values[np.lexsort((values.i,values.cell))]

growth.frame=fast_frame
growth.balance=fast_balance

def reduce_case(label,reuse=False,accepted_end=.06,allow_stopped=False):
    cached=OUT/f'{label}-analysis.json'
    if reuse and cached.exists():
        result=json.loads(cached.read_text());directory=growth.RUNS/label/'output'
        assert result['command']==json.loads((directory.parent/'command.json').read_text())
        rows=read(directory/'bulk-accepted.csv')
        rows=rows[rows.time<=accepted_end+1e-10]
        assert len(rows)==result['cost']['intervals'] and abs(rows.time[-1]-accepted_end)<1e-10
        saved={0:fast_frame(directory,0)}
        for t in (.04,accepted_end):
            step=int(rows.step[np.argmin(abs(rows.time-t))]);saved[t]=fast_frame(directory,step)
        return result,rows,saved
    result,rows,saved=growth.analyze(label,accepted_end,allow_stopped)
    directory=growth.RUNS/label/'output'
    power=read(directory/'tip-power.csv');power=power[power.time<=accepted_end+1e-10] if 'time' in power.dtype.names else power[:len(rows)]
    assert np.array_equal(rows.step,power.step)
    result['method']=result['command']['arguments']['time_update'] or 'backward Euler'
    for name,start in [('energy_full',0),('energy_window',.04)]:
        mask=rows.time>start+1e-10
        for field in ('friction','radiation'):
            values=power[field][mask]
            result[name][field]=dict(signed=float(sum(values)),absolute=float(sum(abs(values))))
        result[name]['dissipation_channel_defect']=float(max(abs(rows.Df[mask]-power.friction[mask]-power.radiation[mask])))
    (OUT/f'{label}-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result,rows,saved

def temporal_compare(first,second):
    a,ra,fa=first;b,rb,fb=second
    for name in fa[0].dtype.names:
        assert np.array_equal(fa[0][name],fb[0][name],equal_nan=True) if fa[0][name].dtype.kind=='f' else np.array_equal(fa[0][name],fb[0][name])
    out={'same_initial_gauss_state':True,'endpoints':{}}
    for t in (.04,.06):
        pa=fa[min(fa,key=lambda k:abs(k-t))];pb=fb[min(fb,key=lambda k:abs(k-t))]
        v=compare(pa,pb)
        for prefix,names in [('stress_tensor',['tauxx','tauyy','tauzz','tauxy']),('cohesive_tensor',['Axx','Ayy','Azz','Axy'])]:
            metric=np.array([1.,1.,1.,2.]);diff=np.stack([pa[n]-pb[n] for n in names],axis=1);truth=np.stack([pb[n] for n in names],axis=1)
            error=np.sqrt(np.sum(pb.weight*np.sum(diff**2*metric,axis=1)));reference=np.sqrt(np.sum(pb.weight*np.sum(truth**2*metric,axis=1)))
            v[prefix]=dict(L2=float(error),relative_L2=float(error/reference) if reference else None)
        norm=np.sqrt(sum(pb.weight*(pb.phi-fb[0].phi)**2))
        v['phase_increment_relative_L2']=float(np.sqrt(sum(pb.weight*(pa.phi-pb.phi)**2))/norm)
        sections=[]
        for case,rows in [(a,ra),(b,rb)]:
            step=int(rows.step[np.argmin(abs(rows.time-t))])
            sections.append(read(growth.RUNS/case['label']/'output'/f'tip-sections-{step:04d}.csv'))
        sa,sb=sections
        for key in ('slip','age'):
            error=np.sqrt(sum(sb.ds*(sa[key]-sb[key])**2));norm=np.sqrt(sum(sb.ds*sb[key]**2))
            v[key]=dict(max_abs=float(max(abs(sa[key]-sb[key]))),relative_L2=float(error/norm))
        out['endpoints'][str(t)]=v
    out['added_fracture_relative_change']=abs(a['energy_full']['added_fracture']/b['energy_full']['added_fracture']-1)
    # Each case is observed at its physical stage time. Interpolate in time,
    # never relabel a midpoint rate as an endpoint rate or extrapolate it.
    observations=[]
    for case,rows in [(a,ra),(b,rb)]:
        d=growth.RUNS/case['label']/'output';times=read(d/'tip-time-location.csv').stage_time
        rates=np.stack([read(d/f'tip-sections-{int(s):04d}.csv').V for s in rows.step])
        observations.append((times,rates))
    out['rates_at_common_times']={}
    weights=sb.ds
    for t in (.04,.05,.0575,.05875):
        values=[]
        for times,rates in observations:
            assert min(times)<=t<=max(times)
            values.append(np.array([np.interp(t,times,rates[:,j]) for j in range(rates.shape[1])]))
        x,y=values;norm=np.sqrt(sum(weights*y*y))
        out['rates_at_common_times'][str(t)]=dict(relative_L2=float(np.sqrt(sum(weights*(x-y)**2))/norm),max_abs=float(max(abs(x-y))))
    out['rate_comparison']='Piecewise-linear time reconstruction at recorded physical stage times, no extrapolation; different from matching endpoint storage.'
    return out

def screen(cases,comparison):
    fine=cases[-1][0]
    ratios={k:fine[k]['absolute_error_over_added_fracture'] for k in ('energy_full','energy_window')}
    improving=all(fine[k]['physical_work_error']['absolute']<cases[0][0][k]['physical_work_error']['absolute'] for k in ratios)
    phase=comparison['endpoints']['0.06']['phase_increment_relative_L2']
    fracture=comparison['added_fracture_relative_change']
    intact=all(max(s['max_boundary_phase_increment'],s['max_remote_phase_increment'])<=1e-6 for c,_,_ in cases for s in c['steps'])
    classifications=[tuple(c['steps'][-1]['regions'][region]['max_delta_phi']>1e-4
                           for region in ('old_central_band','axial_intact','transverse_intact')) for c,_,_ in cases]
    same_response=len(set(classifications))==1
    gates=all(c['gates']['mechanical_residual']<=1e-9 and c['gates']['phase_KKT']<=1e-7
              and all(abs(row.ledger_error)<=1e-10*max(1.,abs(row.Wext),row.bulk+row.cohesive+row.fracture) for row in rows)
              for c,rows,_ in cases)
    passed=all(x<=.1 for x in ratios.values()) and improving and phase<=.01 and fracture<=.01 and intact and same_response and gates
    return dict(passed=passed,absolute_remainder_ratios=ratios,improving=improving,phase_increment_relative_L2=phase,fracture_relative_change=fracture,remote_boundary_integrity=intact,same_response=same_response,original_gates_passed=gates,declared_limits={'remainder_ratio':.1,'state_change_approx':.01})

def plots(cases):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    meshes={.125/2**r['command']['arguments']['refinement'] for r,_,_ in cases}
    for result,rows,saved in cases:
        h=.125/2**result['command']['arguments']['refinement']
        tag=result['method']+f" dt={rows.dt[0]:g}"+(f", h={h:g}" if len(meshes)>1 else '')
        axes[0,0].plot(rows.time,[s['max_delta_phi'] for s in result['steps']],label=tag)
        axes[0,1].plot(rows.time,rows.fracture-result['initial_fracture'],label=tag)
        error=rows.Wext-rows.energy_change-rows.Db-rows.Dc-rows.Df
        line,=axes[0,2].plot(rows.time,np.cumsum(error),label=tag)
        axes[0,2].plot(rows.time,np.cumsum(abs(error)),ls='--',color=line.get_color())
        axes[1,0].plot(rows.time,[s['max_slip'] for s in result['steps']],label=tag)
        axes[1,1].plot(rows.time,[s['regions']['axial_intact']['max_delta_phi'] for s in result['steps']],label=tag)
        added=rows.fracture-result['initial_fracture']
        ratio=np.full(len(rows),np.nan)
        np.divide(np.cumsum(abs(error)),added,out=ratio,where=added>1e-12)
        axes[1,2].plot(rows.time,ratio,label=tag)
    titles=['Phase increment','Added SIPG fracture','Physical work remainder\nsolid signed, dashed absolute','Cumulative section slip','Initially intact axial phase','Absolute remainder / added fracture']
    units=['phase','J/m','J/m','m','phase','ratio']
    for ax,title,unit in zip(axes.flat,titles,units):
        ax.set(title=title,xlabel='endpoint time (s)',ylabel=unit);ax.grid(alpha=.2);ax.legend(fontsize=6)
    axes[0,2].set_yscale('symlog',linthresh=1e-8)
    axes[1,2].set(yscale='log',ylim=(1e-5,1e3));axes[1,2].axhline(.1,color='k',ls=':')
    fig.suptitle('Synthetic specimen: storage at endpoints; dissipation/work use each method’s stage')
    for ext in ['png','pdf']:fig.savefig(OUT/f'temporal-comparison.{ext}',dpi=170)
    plt.close(fig)
    # Phase and increment images for midpoint pair only, with common ranges.
    midpoint=[c for c in cases if c[0]['method']=='midpoint']
    fig,axes=plt.subplots(len(midpoint),2,figsize=(10,3.5*len(midpoint)),squeeze=False,layout='constrained')
    vmax=max(c[0]['steps'][-1]['max_delta_phi'] for c in midpoint)
    for row,(r,rows,saved) in enumerate(midpoint):
        p=saved[max(saved)];seed=saved[0];order=np.lexsort((p.x,p.y));x=np.unique(p.x);y=np.unique(p.y)
        for col,values in enumerate([p.phi,p.phi-seed.phi]):
            z=values[order].reshape(len(y),len(x));im=axes[row,col].pcolormesh(x,y,z,shading='nearest',vmin=0,vmax=1 if col==0 else vmax,cmap='viridis' if col==0 else 'magma',rasterized=True)
            axes[row,col].set(xlabel='x (m)',ylabel='y (m)',aspect='equal',title=f"{r['label']}, t={rows.time[-1]:g} s"+(' (prefix)' if not r.get('trajectory_complete',True) else '')+': '+('phase' if col==0 else 'increment'))
            for edge in [2,6]:axes[row,col].axvline(edge,color='white',ls=':',lw=.6)
            fig.colorbar(im,ax=axes[row,col],shrink=.7)
    for ext in ['png','pdf']:fig.savefig(OUT/f'midpoint-phase.{ext}',dpi=170)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('labels',nargs=2);p.add_argument('--reuse-completed',action='store_true');args=p.parse_args()
    cases=[reduce_case(label,args.reuse_completed) for label in args.labels]
    comparison=temporal_compare(*cases);comparison['screen']=screen(cases,comparison)
    (OUT/'midpoint-comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
    # Reuse completed BE reductions; read their small histories/frames only for plotting.
    be=[]
    for label in ['synthetic-growth-be-dt0025-local','synthetic-growth-be-dt00125-local']:
        r=json.loads((ROOT/f'benchmarks/reconstructed_fault/particle_free/evidence/synthetic_growth/{label}.json').read_text());r['method']='backward Euler';d=growth.RUNS/label/'output';rows=read(d/'bulk-accepted.csv');be.append((r,rows,{}))
    plots(be+cases)
    from growth_spatial_analysis import trace_plot
    trace_plot(cases,OUT)
    print(json.dumps(comparison['screen'],indent=2))
