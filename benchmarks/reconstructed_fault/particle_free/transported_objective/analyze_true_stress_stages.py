#!/usr/bin/env python3
"""Native intermediate-energy ledger; independent accepted-field confirmation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,re
from compare_bounded_profile import load,energy,reduce,ROOT,RUNS
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/direct_true_stress'
result={}
for group,names,reference in [('constant_cutback',['true-cutback-control'],'true-cutback-control'),('layered_restart',['true-l-prefix','true-l-restart4'],'true-l-p')]:
    cmd,rows,meta,states=load(reference);terms=[]
    for name in names:
        log=(RUNS/name/'run.log').read_text();transport={};reaction={}
        for line in log.splitlines():
            if 'Direct transport energy:' not in line and 'Direct source energy:' not in line:continue
            row={k:float(v) for k,v in re.findall(r'(\w+)=([^\s]+)',line)};step=int(row['step'])
            if 'Direct transport energy:' in line:transport[step]=row
            elif row['publication']==1:reaction[step]=row
        for step,t in transport.items():
            r=reaction[step];observed=energy(states[step],meta['G'],cmd['arguments']['eta'])['storage']
            assert abs(observed-r['projected'])<1e-18,(name,step,observed,r)
            tol=512*2.220446049250313e-16*max(abs(t['advected']),abs(r['local']),1e-30)
            assert r['loss']>=-tol and t['advected']-t['rotated_projected']>=-tol,(name,t,r,tol)
            terms.append(dict(case=name,step=step,dt=t['dt'],accepted=t['accepted'],advected=t['advected'],rotated_projected=t['rotated_projected'],
                reaction_local=r['local'],reaction_projected=r['projected'],observed_storage=observed,
                transport_change=t['advected']-t['accepted'],rotation_projection_change=t['rotated_projected']-t['advected'],
                unprojected_reaction_change=r['local']-t['rotated_projected'],source_projection_loss=r['loss'],observer_energy_error=observed-r['projected']))
    reduced,_=reduce(reference);scale=reduced['power_scale']
    sums={}
    for key in ['transport_change','rotation_projection_change','unprojected_reaction_change','source_projection_loss']:
        sums[key]={'signed':sum(t[key] for t in terms),'absolute':sum(abs(t[key]) for t in terms)}
    result[group]=dict(steps=terms,sums=sums,active_work_scale=scale,source_projection_absolute_over_active=sums['source_projection_loss']['absolute']/scale,
        interpretation='Native q8 operation ledger. Transport includes physical boundary exchange and numerical DG/time error; its net change is not physical dissipation. Rotation is pointwise norm preserving; its projection can lose energy. Reaction local-minus-projected is numerical source-projection loss, not additional Maxwell dissipation. Endpoint/trapezoidal continuum work residuals are independently reduced elsewhere.')
(OUT/'stage-energy.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{'sums':v['sums'],'projection_over_active':v['source_projection_absolute_over_active']} for k,v in result.items()},indent=2))
