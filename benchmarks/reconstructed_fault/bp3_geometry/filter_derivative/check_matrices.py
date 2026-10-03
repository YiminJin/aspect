#!/usr/bin/env python3
"""Compare the actual assembled filter matrices, not legacy strict-branch diagnostics."""
from pathlib import Path
import csv,json
import numpy as np
r=Path(__file__).resolve().parents[1]
result=[]
def read(path):return list(csv.DictReader(path.open()))
def matrix(rows,diag,edge):
 d=np.array([float(x[diag]) for x in rows]);e=np.array([float(x[edge]) for x in rows[:-1]])
 return np.diag(d)+np.diag(e,1)+np.diag(e,-1)
for observation in [0,1,2]:
 a=read(r/f'output-filter-t0-forward/nodes_{observation}.csv')
 b=read(r/f'output-filter-t0-reverse/nodes_{observation}.csv')
 assert all(abs(float(x['x'])-float(y['x']))<1e-10 and abs(float(x['y'])-float(y['y']))<1e-10 for x,y in zip(a,b[::-1]))
 for name in ['M','K','Mmu']:
  x,y=matrix(a,name+'_diag',name+'_right'),matrix(b,name+'_diag',name+'_right')[::-1,::-1]
  error=float(np.max(abs(x-y)));scale=float(np.max(abs(x)))
  assert error<=5e-10*scale+1e-22
  result.append(dict(observation=observation,matrix=name,max_abs=error,relative=error/scale))
# G is tested with the same nine physical probes used in the prior diagnosis.
a=read(r/'output-filter-t0-forward/G_probes.csv');b=read(r/'output-filter-t0-reverse/G_probes.csv')
x={(int(v['probe']),int(v['row'])):float(v['value']) for v in a}
y={(int(v['probe']),36-int(v['row'])):float(v['value']) for v in b}
for probe in range(9):
 keys=[k for k in x if k[0]==probe];error=max(abs(x[k]-y[k]) for k in keys);scale=max(abs(x[k]) for k in keys)
 assert error<=5e-10*scale+1e-22
 result.append(dict(probe=probe,matrix='G action',max_abs=error,relative=error/scale))
(r/'filter_derivative/results/matrix_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print('Matched filter matrices and all nine G probes pass.')
