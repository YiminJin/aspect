#!/usr/bin/env python3
"""Compare restored native fields and section state with their actual parents."""
from observations import *

records=[]
for label,parent,step in [
 ('overnight-free-turn-a','first-turn-positive-fourcore-end',15),
 ('overnight-free-turn-b-half','overnight-free-turn-b-prefix',15),
 ('overnight-free-turn-c-prefix','overnight-free-turn-c-pilot',14),
 ('overnight-free-turn-c-half','overnight-free-turn-c-prefix',15)]:
 p,q=RUNS/parent/'output',RUNS/label/'output'
 old=native(p,step);restored=gather(q,'restore-phase-r',['cell','i'])
 assert np.array_equal(old.cell,restored.cell) and np.array_equal(old.i,restored.i)
 phase=float(max(abs(old.B-restored.B)))
 solution=float(max(abs(restored.B-restored.solution_B)))
 a,b=read(p/f'tip-sections-{step:04d}.csv'),read(q/'restore-sections.csv')
 section={name:float(max(abs(a[name]-b[name]))) for name in b.dtype.names}
 assert phase==0 and solution==0 and max(section.values())==0,label
 clock=read(q/'restore-clock.csv')[0]
 assert int(clock.accepted_step)==step
 record=dict(label=label,parent=parent,parent_step=step,phase_error=phase,
             solution_phase_error=solution,section_errors=section,
             clock_before_pending_dt_override={name:float(clock[name]) for name in clock.dtype.names})
 if list(p.glob(f'tip-points-{step:04d}-r*')):
  oldpoints=gather(p,f'tip-points-{step:04d}-r',['cell','q'])
  newpoints=gather(q,'restore-points-r',['cell','q'])
  assert np.array_equal(oldpoints.cell,newpoints.cell) and np.array_equal(oldpoints.q,newpoints.q)
  record['point_errors']={name:float(max(abs(oldpoints[name]-newpoints[name]))) for name in newpoints.dtype.names if name!='cell'}
  assert max(record['point_errors'].values())==0
 else:
  record['point_note']='Sparse parent has no full point frame at this endpoint. Full serialized parent copy is byte-checked in completed-preservation.json; no pointwise comparison is claimed.'
 records.append(record)
(OUT/'restore-comparisons.json').write_text(json.dumps(records,indent=2)+'\n')
print('Exact phase/section restores:',[r['label'] for r in records])
