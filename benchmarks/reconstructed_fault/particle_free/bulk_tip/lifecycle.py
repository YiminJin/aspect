#!/usr/bin/env python3
"""Compare published physical state across MPI and restart, without interpolation."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[4]
runs=root/'build-particle-free/bulk-tip/runs'
def read(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding='utf8')).view(np.recarray)
def points(name,step):
    q=np.concatenate([read(p) for p in (runs/name/'output').glob(f'tip-points-{step:04d}-r*.csv')]).view(np.recarray)
    return q[np.lexsort((q.y,q.x))]
result={}
for name,steps in [('mpi02',[4]),('restart01',[6]),('mpi03',[12]),('restart03',[16])]:
    if not (runs/name/'command.json').exists():continue
    for step in steps:
        a,b=points('control04',step),points(name,step)
        sa=read(runs/'control04'/f'output/tip-sections-{step:04d}.csv');sb=read(runs/name/f'output/tip-sections-{step:04d}.csv')
        result[f'{name}:{step}']={'point_max_abs':{k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a.dtype.names if k not in ['cell','q','section']},
            'section_max_abs':{k:float(np.max(np.abs(sa[k].astype(float)-sb[k].astype(float)))) for k in sa.dtype.names}}
result['run_seconds']={p.parent.name:json.loads(p.read_text()).get('seconds',0) for p in runs.glob('*/command.json')}
result['total_simulation_seconds']=sum(result['run_seconds'].values())
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_tip/lifecycle.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
