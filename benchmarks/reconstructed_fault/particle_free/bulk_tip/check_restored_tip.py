#!/usr/bin/env python3
"""Compare a read-only tip checkpoint observation with retained endpoint exports."""
from pathlib import Path
import os,json,argparse
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from analyze_material_overnight import read_points
from analyze_time_update import read
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference',type=Path,help='Original run output directory')
parser.add_argument('restored',type=Path,help='Read-only restore output directory')
parser.add_argument('step',type=int)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();p=args.reference;r=args.restored
rows=read(p/'bulk-accepted.csv');row=rows[rows.step==args.step];assert len(row)==1
row=row[0]
def gather(root,pattern,keys):
 a=np.concatenate([read_points(f) for f in sorted(root.glob(pattern))]).view(np.recarray)
 return a[np.lexsort(tuple(a[k] for k in reversed(keys)))]
def compare(a,b,fields):
 assert len(a)==len(b)
 out={}
 for key in fields:
  if a[key].dtype.kind in 'SU':assert np.array_equal(a[key],b[key]);continue
  assert np.array_equal(np.isnan(a[key]),np.isnan(b[key]))
  finite=np.isfinite(a[key]);error=float(max(abs(a[key][finite]-b[key][finite]))) if finite.any() else None
  out[key]=error
  assert error is None or error<=2e-13*max(1.,max(abs(b[key][finite]))),(key,error)
 return out
a=gather(r,'restore-points-r*.csv',['cell','q']);b=gather(p,f'tip-points-{args.step:04d}-r*.csv',['cell','q'])
report={'points':compare(a,b,a.dtype.names)}
a=gather(r,'restore-phase-r*.csv',['cell','i']);b=gather(p,f'tip-balance-{args.step:04d}-r*.csv',['cell','i'])
report['phase']=compare(a,b,['cell','i','B']);report['phase']['solution_history_max']=float(max(abs(a.B-a.solution_B)));assert report['phase']['solution_history_max']==0
report['sections']=compare(read(r/'restore-sections.csv'),read(p/f'tip-sections-{args.step:04d}.csv'),['s','ds','age','slip','V','force'])
c=read(r/'restore-clock.csv');assert c.accepted_step[0]==row.step and abs(c.accepted_time[0]-row.time)<1e-14 and c.publications[0]==row.publications
assert c.pending_step[0]==row.step+1 and abs(c.pending_dt[0]-row.dt)<1e-14
assert abs(c.pending_time[0]-row.time-row.dt)<1e-14
report['clock']={k:float(c[k][0]) for k in c.dtype.names}
report['criterion']='Per-field maximum <= 2e-13*max(1, reference maximum); identical CellId/local ordering; zero native history/solution difference.'
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
