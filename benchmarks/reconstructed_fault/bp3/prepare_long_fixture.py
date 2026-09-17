"""Remove junction-only refinement; preserve fixed physical prestress input."""
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from prepare_wide_fixture import cell

HERE=Path(__file__).resolve().parent
BASE=HERE/'fixtures/modified_bp3'
OUT=HERE/'fixtures/modified_bp3_long_run'
old_mesh=(HERE/'fixtures/modified_bp3_wide/target_cells.txt').read_text().split()
leaves=[]
for entry in (OUT/'ordinary_central_cells.txt').read_text().split():
    path=entry.split(':')[1]
    root=(1,4,3,6)[int(path[0])]
    leaves.append(f'{root}_{len(path)-1}:{path[1:]}')
leaves += [s for s in old_mesh if int(s.split('_')[0]) in (0,2,5,7)]
def positions(entries):
    return {s:cell(int(s.split('_')[0]),s.split(':')[1],True) for s in entries}
old,new=positions(old_mesh),positions(leaves)
assert len(new)==len(leaves) and sum(h*h for x,y,h in new.values())==2e10
# Completion integrals can be retained only where the physical endpoint mesh
# and surface origins are unchanged. The removed junction lies far outside it.
for boundary in (0.,100000.):
    select=lambda mesh:{p for p in mesh.values() if abs(p[1]+p[2]/2-boundary)<2000.}
    assert select(old)==select(new)
fault=np.loadtxt(BASE/'fault.txt')
keep=[0]
for j in range(1,len(fault)):
    if np.linalg.norm(fault[j,:2]-fault[keep[-1],:2])>99. or j==len(fault)-1: keep.append(j)
new_fault=fault[keep]
assert len(fault)-len(new_fault)==80 and len(new_fault)==1156
prestress=np.loadtxt(BASE/'prestress.txt',skiprows=1)
np.testing.assert_array_equal(prestress[:,:2],fault[:,:2])
# The rational prestress coefficients are immutable physical initialization
# data, not the current I_h. Restrict those fields to retained vertices; never
# recalibrate them from a new mechanical result or substitute a new denominator.
coefficients=prestress[keep]
old_completion=np.loadtxt(BASE/'completion.txt',skiprows=1)
nodes=(leggauss(3)[0]+1)/2
completion=[]
nonzero=old_completion[old_completion[:,3]!=0]
for j,(a,b) in enumerate(zip(new_fault[:-1,:2],new_fault[1:,:2])):
    for q,z in enumerate(nodes):
        point=(1-z)*a+z*b
        distances=np.linalg.norm(old_completion[:,1:3]-point,axis=1)
        match=int(np.argmin(distances))
        if distances[match]<1e-9: value=old_completion[match,3]
        else:
            assert 2000.<point[1]<98000.
            value=0.
        completion.append([3*j+q,*point,value])
completion=np.array(completion)
assert np.sum(completion[:,3]!=0)==len(nonzero)
def write(path,text):
    if path.exists(): assert path.read_text()==text, f'Refuse to overwrite different {path}'
    else: path.write_text(text)
write(OUT/'target_cells.txt','\n'.join(sorted(leaves))+'\n')
write(OUT/'fault.txt',''.join(' '.join(f'{v:.17g}' for v in row)+'\n' for row in new_fault))
for name,data in [('prestress.txt',coefficients),('completion.txt',completion)]:
    write(OUT/name,str(len(data))+'\n'+''.join(' '.join(f'{v:.17g}' for v in row)+'\n' for row in data))
manifest={name:{'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest()}
          for name in ('ordinary_central_cells.txt','target_cells.txt','fault.txt','prestress.txt','completion.txt')}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
sn=np.sqrt(3)/2;xt=50000*(1+.5/sn)
crossed=[]
for x,y,h in new.values():
    xc=x+h/2;yc=y+h/2
    if abs((xt-xc)*sn-(100000-yc)*.5)<=h*(sn+.5)/2:
        crossed.append(((xt-xc)*.5+(100000-yc)*sn,h))
report=dict(cells=len(new),removed_cells=len(old)-len(new),vertices=len(new_fault),
            fault_spacing_min_m=float(np.min(np.linalg.norm(np.diff(new_fault[:,:2],axis=0),axis=1))),
            fault_spacing_max_m=float(np.max(np.linalg.norm(np.diff(new_fault[:,:2],axis=0),axis=1))),
            crossed_cell_widths_m=sorted(set(h for xd,h in crossed)),ell_over_h=400/97.65625,
            endpoint_mesh_equal=True,completion_nonzero_rows=len(nonzero),
            prestress='Restriction of immutable rational coefficients at retained physical vertices; no recalibration.')
(OUT/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
