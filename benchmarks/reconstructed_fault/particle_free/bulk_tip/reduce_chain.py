#!/usr/bin/env python3
"""Reduce an explicit accepted restart chain; never join or modify checkpoints.

Pieces are LABEL:LAST_STEP, in chronological order. Only steps after the previous
piece's endpoint are included. The caller selects the actual accepted ancestry.
"""
import argparse,csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('pieces',nargs='+');p.add_argument('--name',required=True)
p.add_argument('--window',type=float,default=.04);a=p.parse_args()
def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))
rows=[];previous=0;time=0.;pieces=[]
for item in a.pieces:
    label,stop=item.split(':');stop=int(stop)
    run=ROOT/'build-particle-free/bulk-tip/runs'/label
    power={int(r['step']):r for r in read(run/'output/tip-power.csv')}
    selected=[r for r in read(run/'output/bulk-accepted.csv') if previous<int(r['step'])<=stop]
    assert len(selected)==stop-previous,(label,previous,stop)
    for raw in selected:
        r={k:float(v) for k,v in raw.items()};step=int(r['step'])
        assert step==previous+1 and abs(r['time']-time-r['dt'])<1e-12
        r['friction']=float(power[step]['friction']);r['radiation']=float(power[step]['radiation'])
        r['physical_remainder']=r['Wext']-r['energy_change']-r['Db']-r['Dc']-r['friction']-r['radiation']
        r['added_fracture']=r['energy_change']-((r['bulk']+r['cohesive'])-(rows[-1]['bulk']+rows[-1]['cohesive'] if rows else 0.))
        rows.append(r);previous=step;time=r['time']
    pieces.append(dict(label=label,last_step=stop,core_sha256=json.loads((run/'command.json').read_text())['binary_sha256']))
def total(selected):
    terms=('Wext','Db','Dc','friction','radiation','physical_remainder','stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work','ledger_error','added_fracture')
    return {k:dict(signed=sum(r[k] for r in selected),absolute=sum(abs(r[k]) for r in selected)) for k in terms}
result=dict(pieces=pieces,last_time=time,accepted_steps=len(rows),whole=total(rows),window_start=a.window,
            active_window=total([r for r in rows if r['time']>a.window+1e-12]),
            maximum_residuals={k:max(abs(r[k]) for r in rows) for k in ('mechanical_residual','phase_KKT','ledger_error')},
            final_storage={k:rows[-1][k] for k in ('bulk','cohesive','fracture')})
out=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/tip_solver'/(a.name+'.json')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
