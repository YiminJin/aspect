#!/usr/bin/env python3
"""Reduce fresh efficiency/probe windows, using their actual accepted ancestry."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/tip_efficiency'
def read(p):
    return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding='utf8')).view(np.recarray)
def frame(d,step):
    q=np.concatenate([read(f) for f in sorted(d.glob(f'tip-points-{step:04d}-r*.csv'))]).view(np.recarray)
    return q[np.lexsort((q.y,q.x))]
def accumulate(x):return dict(signed=float(np.sum(x)),absolute=float(np.abs(x).sum()))
def reduce(label,until=None):
    run=ROOT/'build-particle-free/bulk-tip/runs'/label;d=run/'output'
    command=json.loads((run/'command.json').read_text())
    result={k:command.get(k) for k in ('exit_code','seconds','binary_sha256','checker_sha256','arguments')}
    if not (d/'bulk-accepted.csv').exists():return result,None
    rows=read(d/'bulk-accepted.csv');power=read(d/'tip-power.csv')
    if until is not None:rows=rows[rows.time<=until+1e-12]
    power=power[np.isin(power.step,rows.step)]
    parent=rows[0].time-rows[0].dt
    incoming=Path(command['arguments']['resume_from']) if command['arguments']['resume_from'] else d
    if parent>1e-12:
        old=read(incoming/'bulk-accepted.csv');old=old[np.abs(old.time-parent)<1e-12]
        assert len(old)==1,('No matching parent',label,parent)
        old=old[0];initial=frame(incoming,int(old.step));old_fracture=float(old.fracture)
    else:
        initial=frame(d,0);old_fracture=float(read(d/'tip-initial.csv').fracture[0])
    result.update(parent_time=float(parent),last_time=float(rows[-1].time),accepted_steps=len(rows),
                  maximum_residuals={k:float(np.max(np.abs(rows[k]))) for k in ('mechanical_residual','phase_KKT','ledger_error')})
    q=frame(d,int(rows[-1].step));sec=read(d/f'tip-sections-{int(rows[-1].step):04d}.csv')
    remainder=rows.Wext-rows.energy_change-rows.Db-rows.Dc-power.friction-power.radiation
    result['energy']={k:accumulate(rows[k]) for k in ('Wext','Db','Dc','stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work','ledger_error')}
    result['energy'].update(remainder=accumulate(remainder),friction=accumulate(power.friction),radiation=accumulate(power.radiation),
        added_fracture=float(rows[-1].fracture-old_fracture))
    result['damage_increment']=float(q.weight@(q.phi-initial.phi))
    result['phase_extents']={str(t):([float(q.x[q.phi>t].min()),float(q.x[q.phi>t].max())] if np.any(q.phi>t) else None) for t in (.001,.01,.1,.5,.99)}
    result['H_extents']={str(t):([float(q.x[q.H>t].min()),float(q.x[q.H>t].max())] if np.any(q.H>t) else None) for t in (.5,.9,.99)}
    result['max_slip']=float(np.max(np.abs(sec.slip)))
    for filename in ('tip-solver','tip-profile'):
        if (d/(filename+'.csv')).exists():
            data=read(d/(filename+'.csv'));data=data[np.isin(data.step,rows.step)];result[filename]={k:(float(data[k].sum()) if k=='step_seconds' else int(data[k].sum())) for k in data.dtype.names if k!='step'}
    if (d/'tip-timers.csv').exists():
        result['timers']={str(r.section):dict(seconds=float(r.seconds),calls=int(r.calls)) for r in read(d/'tip-timers.csv')}
    if (d/'tip-remote.csv').exists():
        result['remote']=[{k:float(r[k]) for k in r.dtype.names} for r in read(d/'tip-remote.csv')]
    return result,(rows,q,sec,initial)
def force_sensitivity(reference,candidate):
    """Independent endpoint RSF check for the unchanged constant benchmark law.

    Thresholds classify errors for reporting only; every section remains in the
    numerical model and the zero-threshold maximum is always retained.
    """
    result={}
    for threshold in (0.,1e-10,1e-8,1e-6):
        mask=np.maximum(np.abs(reference.V),np.abs(candidate.V))>threshold
        result[str(threshold)]=dict(sections=int(mask.sum()),maximum_force_difference=float(np.max(np.abs(reference.force[mask]-candidate.force[mask])))) if np.any(mask) else None
    def law(q):
        return 15*np.arcsinh(q.resolved_V*np.exp((.6+.02*np.log(q.age*.001/.01))/.015)/.002)+1000*q.resolved_V
    result['independent_law_error']=float(max(np.max(np.abs(law(reference)-reference.force)),np.max(np.abs(law(candidate)-candidate.force))))
    i=int(np.argmax(np.abs(reference.force-candidate.force)))
    result['worst_reference']={k:float(reference[k][i]) for k in ('s','V','resolved_V','age','force')}
    result['worst_candidate']={k:float(candidate[k][i]) for k in ('s','V','resolved_V','age','force')}
    if reference.resolved_V[i]*candidate.resolved_V[i]>0:
        result['log_rate_force_difference']=float(15*np.log(abs(candidate.resolved_V[i]/reference.resolved_V[i])))
    return result
def compare(first,second):
    r,x,s,_=first;c,y,t,_=second
    assert abs(r[-1].time-c[-1].time)<1e-12 and len(x)==len(y)
    assert np.max(np.abs(x.x-y.x))+np.max(np.abs(x.y-y.y))<1e-12
    result=dict(point_max={},point_L2={},section_max={},section_relative_L2={})
    for k in ('phi','tauxx','tauyy','tauzz','tauxy','Axx','Ayy','Azz','Axy','qcoh','w','signed_shear','absolute_shear'):
        diff=y[k]-x[k];result['point_max'][k]=float(np.max(np.abs(diff)));result['point_L2'][k]=float(np.sqrt(x.weight@(diff*diff)))
    for k in ('V','age','slip','force'):
        diff=t[k]-s[k];result['section_max'][k]=float(np.max(np.abs(diff)))
        result['section_relative_L2'][k]=float(np.sqrt(s.ds@(diff*diff))/max(1e-30,np.sqrt(s.ds@(s[k]*s[k]))))
    result['force_sensitivity']=force_sensitivity(s,t)
    result['storage_difference']={k:float(c[-1][k]-r[-1][k]) for k in ('bulk','cohesive','fracture')}
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('labels',nargs='+');p.add_argument('--name',required=True);p.add_argument('--until',type=float,help='Restrict physical reductions; command seconds and timer snapshots still describe the whole run');a=p.parse_args()
    results={};data={}
    for label in a.labels:results[label],data[label]=reduce(label,a.until)
    comparisons={}
    for i,k in enumerate(data):
        for j in list(data)[i+1:]:
            if data[k] is not None and data[j] is not None and abs(data[k][0][-1].time-data[j][0][-1].time)<1e-12 and len(data[k][1])==len(data[j][1]) and np.max(np.abs(data[k][1].x-data[j][1].x))+np.max(np.abs(data[k][1].y-data[j][1].y))<1e-12:
                comparisons[k+':'+j]=compare(data[k],data[j])
    OUT.mkdir(parents=True,exist_ok=True);(OUT/(a.name+'.json')).write_text(json.dumps(dict(cases=results,comparisons=comparisons),indent=2)+'\n')
    print(json.dumps(dict(cases=results,comparisons=comparisons),indent=2))
