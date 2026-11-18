#!/usr/bin/env python3
"""Evaluate the single settled candidate and freeze its fresh initial geometry."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,shutil,numpy as np
from screen import R,B,O,scan,measures
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase import coefficients,fronts,value,basis,read
out=R/'build-particle-free/bulk-tip/runs/stage4-propagation-candidate-settle/output'
command=json.loads((out.parent/'command.json').read_text());assert command['exit_code']==0
b=coefficients(out,0);initial=np.loadtxt(B/'input/phase.txt',skiprows=1).reshape(128,256,3,3);f=fronts(b,8);f0=fronts(initial,8);a=scan(out);m=measures(a)
# A positive perturbation of a basis nonzero at the extremum advances its
# level set. Report every such native direction and independently recompute
# the contour, without publishing a perturbed field as a physical state.
x,y=f['right_x'],f['right_y'];ix=int(x*32);iy=int(y*32);u=x*32-ix;v=y*32-iy
mask=(a[:,0]==(ix+.5)/32)&(a[:,1]==(iy+.5)/32);rows=a[mask];assert len(rows)==9
candidates=[]
for j in range(9):
 at=float(basis(np.asarray(v))[j//3]*basis(np.asarray(u))[j%3]);net=(rows[j,7]-rows[j,5])/rows[j,4]
 if at>1e-5:
  trial=b.copy();eps=min(1e-4,1-b[iy,ix,j//3,j%3]);trial[iy,ix,j//3,j%3]+=eps
  ff=fronts(trial,8)
  candidates.append(dict(basis=j,value_at_contour=at,net_driving_Pa=float(net),probe_increment=eps,contour_advance_m=ff['right_x']-f['right_x'],note='Read-only infinitesimal-direction contour probe; no finite-energy or physical growth claim.'))
intact=np.max(abs(initial),axis=(2,3))==0
record=dict(prepared_front=f,inherited_front=f0,preparation_right_advance=f['right_x']-f0['right_x'],preparation_left_advance=f0['left_x']-f['left_x'],maximum_coefficient_increment=float(np.max(b-initial)),minimum_coefficient_increment=float(np.min(b-initial)),initially_intact_max_increment=float(np.max((b-initial)[intact])),margins=m,contour_advancing_directions=candidates,static_iterations=read(out/'preparation-iterations.csv').tolist())
(O/'candidate.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
assert record['minimum_coefficient_increment']>=-1e-14
assert f['sampled_components']==1 and f['detached_sample_area']==0
assert m['regions']['all_intact']['max_envelope']<0,'Detached/intact envelope is no longer unfavorable'
assert any(abs(q['net_driving_Pa'])<1e-4 and q['contour_advance_m']>1e-9 for q in candidates),'No checked contour-advancing direction close to energetic favorability'
if '--check-only' in sys.argv:raise SystemExit(0)
target=B/'input-candidate';target.mkdir(exist_ok=False)
for p in (B/'input').iterdir():shutil.copy2(p,target/p.name)
np.savetxt(target/'phase.txt',b.reshape(-1,9),header='256 128',comments='',fmt='%.17g')
planes=np.loadtxt(target/'sections.txt',skiprows=1);occupied=np.max(b,axis=(0,2,3))>0;planes[:,2]=np.maximum(planes[:,2],occupied[np.minimum((planes[:,0]*32).astype(int),255)])
np.savetxt(target/'sections.txt',planes,header=str(len(planes)),comments='',fmt=['%.17g','%.17g','%d'])
print('Candidate geometric input frozen after native preparation and spatial gates')
