#!/usr/bin/env python3
"""Exact x subdivision of the retained .075 native phase, no growth replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'first_free_turn'))
from analyze import polynomial
p=argparse.ArgumentParser();p.add_argument('nx',type=int,choices=[128,256,512]);p.add_argument('--family',choices=['full','targeted'],default='full');a=p.parse_args()
directory=ROOT/f'build-particle-free/stage4-wavelength/input-{a.nx}'
directory.mkdir(exist_ok=False)
old=ROOT/'build-particle-free/bulk-tip/runs/first-turn-positive-fourcore-end/output'
reference=ROOT/'build-particle-free/bulk-tip/runs/stage4-fixed-phase-stiffness-homogeneous/output'
poly=polynomial(old,15,.0625,.03125)
# Bernstein subdivision by exact quadratic value-to-coefficient transformation.
# This preserves each parent's polynomial, including its DG face jumps.
factor=a.nx//128;B=poly.coefficients
refined=[]
for ix in range(a.nx):
    u=(ix%factor+np.array([0.,.5,1.]))/factor
    basis=np.array([(1-u)**2,2*u*(1-u),u*u]).T
    values=B[:,ix//factor]@basis.T
    values[:,:,1]=2*values[:,:,1]-.5*(values[:,:,0]+values[:,:,2])
    refined.append(values)
new=np.stack(refined,axis=1)
np.savetxt(directory/'phase.txt',new.reshape(-1,9),header=f'{a.nx} 128',comments='',fmt='%.17g')
planes=np.genfromtxt(reference/'restoring-sections.csv',delimiter=',',names=True)
formed=np.genfromtxt(old/'tip-planes-0015.csv',delimiter=',',names=True)
assert np.max(abs(planes['angle'][formed['formed']>0]-formed['angle'][formed['formed']>0]))<1e-14
g,w=np.polynomial.legendre.leggauss(6)
x=((np.arange(a.nx)[:,None]+.5+.5*g)*8/a.nx).ravel()
angle=np.interp(x,planes['x'],planes['angle'])
# Formed flag is categorical: nearest original material section. It never
# affects this frozen tangent, because no plane update is invoked.
right=np.clip(np.searchsorted(planes['x'],x),1,len(planes)-1)
nearest=np.where(x-planes['x'][right-1]<planes['x'][right]-x,right-1,right)
np.savetxt(directory/'sections.txt',np.c_[x,angle,planes['formed'][nearest]],header=str(len(x)),comments='',fmt=['%.17g','%.17g','%d'])
z=np.load(HERE.parent/'evidence/stage4_rsf/reduced-operator.npz')
v=np.interp(x,z['x'],z['V'],left=0,right=0)
theta=np.full_like(x,float(z['theta'][0]))
np.savetxt(directory/'operating.csv',np.c_[x,v,theta],delimiter=',',header='x,V,theta',comments='')
patterns=[(n,0) for n in range(1,7)]+[(n,shift) for n in (24,32,48,64) for shift in (0,1)]
if a.nx>=256:patterns += [(96,s) for s in (0,1)]
if a.nx>=512:patterns += [(n,s) for n in (128,192) for s in (0,1)]
if a.family=='targeted':
    assert a.nx==512
    patterns=[(n,0) for n in range(1,7)]+[(n,s) for n in (96,192) for s in (0,1)]
np.savetxt(directory/'patterns.txt',patterns,header=str(len(patterns)),comments='',fmt='%d')
fixture=(HERE.parent/'first_free_turn/positive.prm').read_text()
fixture=fixture.replace('set X repetitions = 128',f'set X repetitions = {a.nx}')
fixture=fixture.replace('set Equilibrate bulk tip seed = true','set Equilibrate bulk tip seed = false')
fixture=fixture.replace('set Initial time step = .01','set Initial time step = .00125')
(directory/'fixture.prm').write_text(fixture)
# Independent checks at interior points, including both traces near DG faces.
rng=np.random.default_rng(421);qx=rng.uniform(0,8,10000);qy=rng.uniform(1.5,2.5,10000)
ix=np.minimum((qx*a.nx/8).astype(int),a.nx-1);iy=np.minimum((qy*32).astype(int),127)
u=qx*a.nx/8-ix;t=qy*32-iy
bx=np.array([(1-u)**2,2*u*(1-u),u*u]).T;by=np.array([(1-t)**2,2*t*(1-t),t*t]).T
value=np.einsum('ki,kij,kj->k',by,new[iy,ix],bx)
error=float(np.max(abs(value-poly.evaluate(qx,qy))));assert error<1e-14
y=np.linspace(1.5,2.5,2001)
np.savetxt(directory/'cross-section.csv',np.c_[y,poly.evaluate(np.full_like(y,4.01),y)],delimiter=',',header='y,phi',comments='')
record=dict(nx=a.nx,ny=128,family=a.family,phase_value_error=error,phase_bounds=[float(new.min()),float(new.max())],
            angle_transfer='piecewise linear in physical x through retained diagnostic sections; all planes frozen',
            rate_transfer='piecewise linear in physical x through original smooth operating point',
            original_formed_angle_difference=0.,theta=float(theta[0]),peak_sampled_V=float(max(v)),
            patterns=patterns,source_phase_step=15,source_phase_time=.075,
            source_files={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [reference/'restoring-sections.csv',old/'tip-planes-0015.csv',HERE.parent/'evidence/stage4_rsf/reduced-operator.npz',*sorted(old.glob('tip-balance-0015-r*.csv.gz'))]})
(directory/'transfer.json').write_text(json.dumps(record,indent=2)+'\n')
print(directory)
