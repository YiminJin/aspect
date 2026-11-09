#!/usr/bin/env python3
"""Reproduce the native-basis physical and fixed-parent comparisons."""
import sys,json
from pathlib import Path
import numpy as np
from analyze_efficiency import reduce,compare,frame,read,ROOT
out=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/native_bernstein'
r={};d={}
for name in ('efficiency-localized-coarse01','bernstein-localized01','bernstein-mpi01'):
 r[name],d[name]=reduce(name)
comparisons={a+':'+b:compare(d[a],d[b]) for a,b in [('efficiency-localized-coarse01','bernstein-localized01'),('bernstein-localized01','bernstein-mpi01')]}
for label,c in comparisons.items():
 a,b=label.split(':');x,y=d[a][1],d[b][1]
 for k in ('H','Y'):
  c['point_max'][k]=float(np.max(abs(y[k]-x[k])));c['point_L2'][k]=float(np.sqrt(x.weight@((y[k]-x[k])**2)))
 c['point_scaled_max']={k:v/max(1e-30,float(np.max(abs(x[k])))) for k,v in c['point_max'].items()}
(out/'localized-comparison.json').write_text(json.dumps(dict(cases=r,comparisons=comparisons),indent=2)+'\n')
for k,c in comparisons.items():print(k,json.dumps({q:c[q] for q in ('point_max','section_max','storage_difference')}))
for k,c in r.items(): print(k,'growth',c['damage_increment'],'step_seconds',c['tip-profile'].get('step_seconds'), 'gates',c['maximum_residuals'])
a=read(ROOT/'build-particle-free/bulk-tip/runs/bernstein-parent-export01/output/fixed-parent-action.csv');b=read(ROOT/'build-particle-free/bulk-tip/runs/bernstein-fine01/output/fixed-parent-action.csv')
x={}
for comp in np.unique(a.component):
 m=a.component==comp;x[str(comp)]={}
 for k in ('coordinate','residual','action'):
  diff=float(np.max(abs(a[k][m]-b[k][m])));scale=float(np.max(abs(a[k][m])));x[str(comp)][k]=dict(max_abs=diff,scale=scale,scaled=diff/max(scale,1e-30))
(out/'fixed-parent-components.json').write_text(json.dumps(x,indent=2)+'\n');print('fixed by component',json.dumps(x))

# The difficult comparison imports an explicit parent rather than ordinary
# restart, so select the common endpoint directly without inventing ancestry.
fine=[]
for label in ('efficiency-final01','bernstein-fine01'):
    directory=ROOT/'build-particle-free/bulk-tip/runs'/label/'output'
    rows=read(directory/'bulk-accepted.csv');rows=rows[rows.step==24]
    fine.append((rows,frame(directory,24),read(directory/'tip-sections-0024.csv'),None))
comparison=compare(*fine)
for k in ('H','Y'):
    difference=fine[1][1][k]-fine[0][1][k]
    comparison['point_max'][k]=float(np.max(abs(difference)))
    comparison['point_L2'][k]=float(np.sqrt(fine[0][1].weight@(difference*difference)))
comparison['point_scaled_max']={k:v/max(1e-30,float(np.max(abs(fine[0][1][k])))) for k,v in comparison['point_max'].items()}
comparison['energy_work_difference']={k:float(fine[1][0][k][-1]-fine[0][0][k][-1]) for k in fine[0][0].dtype.names if k not in ('step','time','dt','publications')}
(out/'fine-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')

# Reconstruct cell polynomials from the independent 6x6 FEValues exports.
# Classification below is diagnostic (1e-12), never a solver tolerance change.
def coefficients(points):
    g=(np.polynomial.legendre.leggauss(6)[0]+1)/2
    B=np.array([(1-g)**2,2*g*(1-g),g*g]).T
    E=np.array([[B[q%6,i%3]*B[q//6,i//3] for i in range(9)] for q in range(36)])
    result={}
    for cell in np.unique(points.cell):
        rows=points[points.cell==cell];rows=rows[np.argsort(rows.q)]
        result[str(cell)]=np.linalg.lstsq(E,rows.phi,rcond=None)[0]
    return result
parent={str(cell):a.coordinate[(a.cell==cell)&(a.component==7)][np.argsort(a['shape'][(a.cell==cell)&(a.component==7)])] for cell in np.unique(a.cell)}
sets={}
for label in ('efficiency-final01','bernstein-fine01'):
    coeff=coefficients(frame(ROOT/'build-particle-free/bulk-tip/runs'/label/'output',24))
    result={};counts=dict(lower=0,upper=0,free=0)
    for cell,values in coeff.items():
        for j,value in enumerate(values):
            status='lower' if abs(value-parent[cell][j])<1e-12 else ('upper' if abs(value-1)<1e-12 else 'free')
            result[(cell,j)]=status;counts[status]+=1
    sets[label]=(result,counts)
keys=list(sets);differences=sum(sets[keys[0]][0][k]!=sets[keys[1]][0][k] for k in sets[keys[0]][0])
(out/'active-bounds.json').write_text(json.dumps(dict(classification_tolerance=1e-12,method='least-squares Q2 polynomial from Gauss6 physical exports; diagnostic only',counts={k:v[1] for k,v in sets.items()},changed= differences),indent=2)+'\n')
assert differences==0
