#!/usr/bin/env python3
"""One selected synthetic spatial sensitivity, with independently equilibrated seeds."""
import argparse,json,os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from analyze_material_overnight import OUT,reduce_case,growth,fast_frame
from analyze_time_update import read
from growth_spatial_analysis import spatial_comparison,CellPolynomials,weighted_error,trace_plot

def accepted_prefix(label,end):
    """Observe complete accepted exports, without claiming a completed trajectory."""
    directory=growth.RUNS/label/'output'
    command=json.loads((directory.parent/'command.json').read_text())
    assert not command['arguments']['fixed_phase'] and not command['arguments']['resume_from']
    rows=read(directory/'bulk-accepted.csv');rows=rows[rows.time<=end+1e-10]
    assert len(rows) and abs(rows.time[-1]-end)<1e-10 and end>=.04
    initial=float(read(directory/'tip-initial.csv').fracture[0])
    result=dict(label=label,command=command,initial_fracture=initial,
                method=command['arguments']['time_update'] or 'backward Euler',
                energy_full=growth.energy(rows,initial,0),
                observation='Accepted prefix only; original target/status retained in command.')
    saved={0:fast_frame(directory,0)}
    for t in sorted(set([.04,end])):
        step=int(rows.step[np.argmin(abs(rows.time-t))]);assert min(abs(rows.time-t))<1e-10
        p=fast_frame(directory,step)
        h=.125/2**command['arguments']['refinement']
        assert len(p)==36*round(8/h)*round(4/h) and len(p)==len(saved[0])
        assert np.array_equal(p.cell,saved[0].cell) and np.array_equal(p.q,saved[0].q)
        saved[t]=p
    return result,rows,saved

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('coarse');parser.add_argument('fine')
    parser.add_argument('--accepted-prefix',type=float,help='Explicit shared accepted end >=.04; no completed fine trajectory or .06 qualification is claimed')
    args=parser.parse_args()
    if args.accepted_prefix is None:
        coarse=reduce_case(args.coarse,reuse=True);fine=reduce_case(args.fine);end=.06;stem='spatial'
    else:
        end=args.accepted_prefix
        coarse=accepted_prefix(args.coarse,end);fine=accepted_prefix(args.fine,end);stem='spatial-prefix'
    a,ra,fa=coarse;b,rb,fb=fine
    result={'comparison_kind':'One spatial sensitivity, not an order estimate; same physical obstacle re-equilibrated at each mesh.',
            'observation_end':end,'accepted_prefix_only':args.accepted_prefix is not None,
            'labels':[a['label'],b['label']],
            'initial_fracture':[a['initial_fracture'],b['initial_fracture']],
            'added_fracture':[a['energy_full']['added_fracture'],b['energy_full']['added_fracture']],'states':{}}
    hc=.125/2**a['command']['arguments']['refinement'];hf=.125/2**b['command']['arguments']['refinement']
    assert abs(hc/hf-2)<1e-12
    final_diff=None
    for t in sorted(set([.04,end])):
        ca=fa[min(fa,key=lambda k:abs(k-t))];fi=fb[min(fb,key=lambda k:abs(k-t))]
        comparison,diff=spatial_comparison(ca,fa[0],fi,fb[0],hc,hf)
        surfaces=[];stage_times=[]
        for r,rows in [(a,ra),(b,rb)]:
            step=int(rows.step[np.argmin(abs(rows.time-t))]);surfaces.append(read(growth.RUNS/r['label']/'output'/f'tip-sections-{step:04d}.csv'))
            clock=read(growth.RUNS/r['label']/'output/tip-time-location.csv')
            stage_times.append(float(clock.stage_time[np.argmin(abs(clock.step-step))]))
        assert abs(stage_times[0]-stage_times[1])<1e-12,'Spatial rate comparison requires matching stage clocks.'
        sa,sb=surfaces;indices=np.floor(sa.s/hc).astype(int);fine_indices=np.floor(sb.s/hc).astype(int)
        fields=['slip','age','V','force']
        for key in fields:
            mapped=np.empty_like(sb.s)
            for cell in np.unique(indices):
                old=indices==cell;new=fine_indices==cell
                xi=2*(sa.s[old]-(cell+.5)*hc)/hc
                coefficient=np.polynomial.legendre.legfit(xi,sa[key][old],5)
                mapped[new]=np.polynomial.legendre.legval(2*(sb.s[new]-(cell+.5)*hc)/hc,coefficient)
            comparison[key]=weighted_error(mapped,sb[key],sb.ds)
        comparison['surface_measure']='Q5 observation interpolant of section samples, fine ds weights; no raw DoF matching.'
        comparison['surface_times']=dict(slip_age_endpoint=t,rate_resistance_stage=stage_times[0])
        result['states'][str(t)]=comparison
        if t==end:final_diff=diff
    (OUT/f'{stem}-comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    trace_plot([coarse,fine],OUT,stem+'-native-traces')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    p=fb[max(fb)];order=np.lexsort((p.x,p.y));x=np.unique(p.x);y=np.unique(p.y)
    for ax,values,title in zip(axes,[p.phi,p.phi-fb[0].phi,final_diff],['Fine endpoint phase','Fine increment','Coarse − fine increment']):
        z=values[order].reshape(len(y),len(x))
        limit=float(max(abs(values)))
        options=dict(vmin=-limit,vmax=limit) if ax is axes[-1] else dict(vmin=0,vmax=1 if ax is axes[0] else limit)
        c=ax.pcolormesh(x,y,z,shading='nearest',cmap='coolwarm' if ax is axes[-1] else 'viridis',rasterized=True,**options)
        ax.set(xlabel='x (m)',ylabel='y (m)',title=title,aspect='equal');fig.colorbar(c,ax=ax,shrink=.7)
    fig.suptitle(f'Accepted observations at {end:g} s'+(' — prefix only' if args.accepted_prefix is not None else ''))
    for suffix in ['png','pdf']:fig.savefig(OUT/f'{stem}-phase.{suffix}',dpi=170)
    print(json.dumps({'observation_end':end,'accepted_prefix_only':args.accepted_prefix is not None,
                      'phase_increment_relative_L2':result['states'][str(end)]['increment']['relative_L2'],
                      'added_fracture':result['added_fracture'],'initial_fracture':result['initial_fracture']},indent=2))
