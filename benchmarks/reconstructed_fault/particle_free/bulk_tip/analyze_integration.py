#!/usr/bin/env python3
"""Reduce solver-integration controls against their actual zero-clock obstacles.

No simulation inputs or old evidence are modified. Mesh comparisons use common
coarse-cell means, not invented arbitrary-point native stress interpolation.
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('labels', nargs='+')
p.add_argument('--figures', action='store_true')
p.add_argument('--name', default='integration-summary')
a = p.parse_args()
out = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/tip_solver'
out.mkdir(exist_ok=True, parents=True)

def read(file):
    return np.atleast_1d(np.genfromtxt(file, delimiter=',', names=True, dtype=None, encoding='utf8')).view(np.recarray)

def points(directory, step):
    q = np.concatenate([read(f) for f in sorted(directory.glob(f'tip-points-{step:04d}-r*.csv'))]).view(np.recarray)
    return q[np.lexsort((q.y,q.x))]

def tensor_norm(q):
    return np.sqrt(np.sum(q.weight*(q.tauxx**2+q.tauyy**2+q.tauzz**2+2*q.tauxy**2)))

def accum(values):
    return dict(signed=float(np.sum(values)), absolute=float(np.sum(np.abs(values))))

cases, data = {}, {}
for label in a.labels:
    run = ROOT/'build-particle-free/bulk-tip/runs'/label
    cmd = json.loads((run/'command.json').read_text())
    result = {k:cmd.get(k) for k in ('exit_code','seconds','binary_sha256','checker_sha256')}
    result.update(dt=cmd['arguments']['dt'],h=.25/2**cmd['arguments']['refinement'])
    cases[label] = result
    d = run/'output'
    if not (d/'bulk-accepted.csv').exists():continue
    steps = read(d/'bulk-accepted.csv');last = steps[-1]
    q = points(d,int(last.step))
    # Restarts contain only their new window. Use the original accepted frame
    # at the incoming clock, and subtract its stored physical energy explicitly.
    initial_directory=d
    initial_step=0
    incoming_storage=0.
    result['window_initial_time']=0.
    if not list(d.glob('tip-points-0000-r*.csv')):
        initial_directory=Path(cmd['arguments']['resume_from'])
        old_steps=read(initial_directory/'bulk-accepted.csv')
        incoming_time=float(steps[0].time-steps[0].dt)
        match=old_steps[np.abs(old_steps.time-incoming_time)<1e-12]
        assert len(match)==1, 'Missing exact incoming accepted energy/frame'
        initial_step=int(match[0].step)
        incoming_storage=float(match[0].bulk+match[0].cohesive)
        result['window_initial_time']=incoming_time
    initial = points(initial_directory,initial_step)
    sec = read(d/f'tip-sections-{int(last.step):04d}.csv');power = read(d/'tip-power.csv')
    localized = 'localized' in str(cmd['arguments'].get('fixture',''))
    center,half = (2.,.8) if localized else (1.,.5)
    growth = q.phi-initial.phi
    remote = np.abs(q.x-center)>half+.5
    tip = np.abs(np.abs(q.x-center)-half)<.25
    Y = np.zeros(len(q));valid = q.phi<1
    Y[valid] = 80*(1+3*q.phi[valid])*q.qcoh[valid]**2/(2e6*(1-q.phi[valid])**3)
    remainder = steps.Wext-steps.energy_change-steps.Db-steps.Dc-power.friction-power.radiation
    added = steps.energy_change-np.diff(steps.bulk+steps.cohesive,prepend=incoming_storage)
    result.update(last_time=float(last.time),steps=len(steps),phase_min=float(q.phi.min()),phase_max=float(q.phi.max()),
        minimum_damage_increment=float(growth.min()),damage_increment=float(q.weight@growth),
        initial_damage_integral=float(initial.weight@initial.phi),stress_L2=float(tensor_norm(q)),
        slip_rate_max=float(np.max(np.abs(sec.V))),slip_max=float(np.max(np.abs(sec.slip))),
        mechanical_residual_max=float(np.max(steps.mechanical_residual)),phase_KKT_max=float(np.max(steps.phase_KKT)),
        ledger_max=float(np.max(np.abs(steps.ledger_error))),adjoint_max=float(np.max(np.abs(power.power_adjoint_error))),
        section_rate_defect_max=float(np.max(np.abs(sec.integral_rate_defect))),
        remote_driving_over_intact_threshold=float(np.max(Y[remote])/40) if np.any(remote) else None,
        remote_damage_increment=float(np.sum(q.weight[remote]*growth[remote])),
        tip_damage_increment=float(np.sum(q.weight[tip]*growth[tip])),
        tip_driving_max=float(np.max(Y[tip])),
        phase_extents={str(t):dict(initial=([float(initial.x[initial.phi>t].min()),float(initial.x[initial.phi>t].max())] if np.any(initial.phi>t) else None),
                                  final=([float(q.x[q.phi>t].min()),float(q.x[q.phi>t].max())] if np.any(q.phi>t) else None)) for t in (.001,.01,.1,.5,.99)})
    if (d/'tip-initial.csv').exists():result['initial_fracture']=float(read(d/'tip-initial.csv').fracture[0])
    terms=['stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work','ledger_error']
    result['energy']={k:accum(steps[k]) for k in terms}
    result['energy'].update(physical_remainder=accum(remainder),fracture_added=float(added.sum()),
        work=accum(steps.Wext),friction=float(power.friction.sum()),radiation=float(power.radiation.sum()),
        bulk_viscosity=float(steps.Db.sum()),cohesive_viscosity=float(steps.Dc.sum()))
    mask=steps.time>.04+1e-12
    result['active_window']=dict(start=.04,remainder=accum(remainder[mask]),fracture_added=float(added[mask].sum()))
    positive=negative=0.;multiple=0;maximum_bands=0
    for row in steps:
        frame=points(d,int(row.step));ss=read(d/f'tip-sections-{int(row.step):04d}.csv')
        local=row.dt*frame.weight*frame.w*ss.force[frame.section.astype(int)]
        positive+=float(np.maximum(local,0).sum());negative+=float(np.minimum(local,0).sum())
        for k in np.unique(frame.section):
            samples=frame[frame.section==k];samples=samples[np.argsort(samples.y)]
            active=np.abs(samples.w)>max(1e-10,1e-6*np.max(np.abs(samples.w)))
            count=int(np.sum(active & ~np.r_[False,active[:-1]]));maximum_bands=max(maximum_bands,count);multiple+=count>1
    result['signed_work']=dict(positive=positive,negative=negative,maximum_bands=maximum_bands,section_steps_with_multiple_bands=multiple,
                               diagnostic_threshold='max(1e-10/s, 1e-6 section max |w|)')
    if (d/'tip-solver.csv').exists():
        counts=read(d/'tip-solver.csv')
        result['solver']={k:int(np.sum(counts[k])) for k in ('nonlinear_iterations','krylov_iterations','stokes_solves','active_releases','search_shifts')}
        if 'sparse_entries' in counts.dtype.names:result['solver']['sparse_entries_max']=int(np.max(counts.sparse_entries))
    data[label]=(steps,q,initial,sec,remainder,added,Y)

comparisons={}
for i,name in enumerate(data):
    for other in list(data)[i+1:]:
        f,g=data[name][1],data[other][1]
        if abs(cases[name]['last_time']-cases[other]['last_time'])>1e-12:continue
        if abs(f.x.max()-g.x.max())>.1:continue
        cols=['phi','tauxx','tauyy','tauzz','tauxy','Axy','signed_shear']
        def means(q):
            nx=round((4 if q.x.max()>2 else 2)/.25)
            ids=np.floor(q.x/.25).astype(int)+nx*np.floor(q.y/.25).astype(int)
            weights=np.bincount(ids,weights=q.weight)
            return np.array([np.bincount(ids,weights=q.weight*q[c])/weights for c in cols])
        af,ag=means(f),means(g);err=ag-af
        compare={c:float(np.sqrt(.0625*np.sum(err[j]**2))) for j,c in enumerate(cols)}
        compare['initial_phase_L2_cellmeans']=float(np.sqrt(.0625*np.sum((means(data[name][2])[0]-means(data[other][2])[0])**2)))
        compare['damage_increment_L2_cellmeans']=float(np.sqrt(.0625*np.sum(((ag[0]-means(data[other][2])[0])-(af[0]-means(data[name][2])[0]))**2)))
        for field in ('V','slip','age'):
            def section_means(sec):
                ids=np.floor(sec.s/.25).astype(int)
                return np.bincount(ids,weights=sec.ds*sec[field])/np.bincount(ids,weights=sec.ds)
            sf,sg=section_means(data[name][3]),section_means(data[other][3])
            compare[field+'_relative_L2_cellmeans']=float(np.linalg.norm(sg-sf)/max(np.linalg.norm(sg),1e-30))
        compare['stress_relative_L2']=float(np.sqrt(np.sum(err[1:4]**2)+2*np.sum(err[4]**2))/max(np.sqrt(np.sum(ag[1:4]**2)+2*np.sum(ag[4]**2)),1e-30))
        if cases[name]['window_initial_time']!=cases[other]['window_initial_time']:
            compare.pop('initial_phase_L2_cellmeans');compare.pop('damage_increment_L2_cellmeans')
        if cases[name]['window_initial_time']==cases[other]['window_initial_time'] and len(f)==len(g) and np.max(np.abs(f.x-g.x))<1e-12:
            compare['initial_phase_max_difference']=float(np.max(np.abs(data[name][2].phi-data[other][2].phi)))
        comparisons[name+':'+other]=compare
summary=dict(cases=cases,common_coarse_cell_comparisons=comparisons)
(out/(a.name+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
if a.figures and data:
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[k for k in data if 'localized' in k and 'unloaded' not in k][:3] or list(data)[-3:]
    pretty=lambda k: f"h={cases[k]['h']:g}, dt={cases[k]['dt']:g}"
    max_increment=max(float(np.max(data[k][1].phi-data[k][2].phi)) for k in selected)
    fig,axes=plt.subplots(len(selected),2,figsize=(10,3*len(selected)),squeeze=False,constrained_layout=True)
    for row,k in zip(axes,selected):
        steps,q,initial,sec,remainder,added,Y=data[k]
        for ax,value,title in zip(row,[q.phi,q.phi-initial.phi],['Damage','Increment from equilibrated initial obstacle']):
            # Display native samples as flat nearest-sample tiles, without smoothing.
            # Round coordinates only to merge affine-mapping ulps in this image.
            xx,yy=np.round(q.x,13),np.round(q.y,13);ux,uy=np.unique(xx),np.unique(yy)
            assert len(ux)*len(uy)==len(q)
            field=np.empty((len(uy),len(ux)));field[np.searchsorted(uy,yy),np.searchsorted(ux,xx)]=value
            sc=ax.pcolormesh(ux,uy,field,shading='nearest',cmap='magma',vmin=0,vmax=.8 if title=='Damage' else max_increment)
            fig.colorbar(sc,ax=ax,shrink=.8)
            ax.set_xlim(0,4 if q.x.max()>2 else 2);ax.set_ylim(0,2 if q.y.max()>1 else 1)
            ax.set(title=pretty(k)+'\n'+('Damage' if title=='Damage' else 'Increment from initial equilibrium'),xlabel='x (m)',ylabel='y (m)',aspect='equal')
            ax.title.set_fontsize(10)
    fig.savefig(out/(a.name+'-damage.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for k in selected:
        steps,q,initial,sec,remainder,added,Y=data[k]
        axes[0,0].plot(sec.s,sec.V,label=pretty(k));axes[0,1].plot(sec.s,sec.slip,label=pretty(k))
        color=axes[0,0].lines[-1].get_color()
        axes[1,0].plot(steps.time,np.cumsum(remainder),color=color,label=pretty(k)+' signed')
        axes[1,0].plot(steps.time,np.cumsum(np.abs(remainder)),ls='--',color=color,label=pretty(k)+' absolute')
        axes[1,1].plot(steps.time,np.cumsum(added),label=pretty(k))
    for ax,title,unit in zip(axes.flat,['Integrated fault rate','Cumulative slip','Physical work remainder','Added fracture'],['m/s','m','J/m','J/m']):
        ax.set(title=title,ylabel=unit);ax.legend(fontsize=7)
    axes[0,0].set_xlabel('section x (m)');axes[0,1].set_xlabel('section x (m)')
    axes[1,0].set_xlabel('time (s)');axes[1,1].set_xlabel('time (s)')
    fig.savefig(out/(a.name+'-slip-energy.png'),dpi=160);plt.close(fig)
