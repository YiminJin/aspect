#!/usr/bin/env python3
"""Quadratic native fixed-phase driving estimates; no numerical parameter sweep."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,numpy as np
from scipy.optimize import brentq
from screen import R,B,O,scan,measures
base=scan(R/'build-particle-free/bulk-tip/runs/stage4-preparation-free-weakening/output')
a=scan(R/'build-particle-free/bulk-tip/runs/stage4-propagation-shear-a/output')
b=scan(R/'build-particle-free/bulk-tip/runs/stage4-propagation-shear-b/output')
assert np.max(abs(base[:,:6]-a[:,:6]))<1e-12 and np.max(abs(base[:,:6]-b[:,:6]))<1e-12
# Parameter z=additional shear/.00025; stress is affine at fixed phase/force.
# On formed planes, driving is quadratic. For intact planes use the conservative
# all-orientation envelope, also quadratic, not fitted maximizing-plane angles.
y=np.stack([base[:,8],a[:,8],b[:,8]])
c0=y[0];c2=(y[2]-2*y[1]+y[0])/2;c1=y[1]-y[0]-c2
intact=np.repeat(np.max(abs(base[:,3].reshape(-1,9)),axis=1)==0,9)
phase=base[:,3].reshape(128,256,9);shift=np.zeros_like(phase);shift[:,1:]=phase[:,:-1];chi=np.maximum(0,shift-phase).ravel()*np.maximum(0,1-((base[:,0]-6.002054379936584)/.5)**2)**2
formed=~intact
# Each direction is restricted to formed phase rows for the polynomial identity;
# the tiny intact part is instead bounded by its envelope in selection.
chi*=formed
cost=chi@base[:,5];mass=chi@base[:,4]
d=np.stack([base[:,7],a[:,7],b[:,7]])@chi
p2=(d[2]-2*d[1]+d[0])/2;p1=d[1]-d[0]-p2
f=lambda z:(d[0]+p1*z+p2*z*z-cost)/mass
intactmax=lambda z:float(np.max((c0+c1*z+c2*z*z-base[:,5])[intact]/base[intact,4]))
record=dict(response_amplitudes=[0,.00025,.0005],native_response=[measures(x) for x in [base,a,b]],profile_zero=None,first_intact_envelope_zero=None)
if f(0)*f(4)<0:record['profile_zero']=brentq(f,0,4)*.00025
if intactmax(0)*intactmax(4)<0:record['first_intact_envelope_zero']=brentq(intactmax,0,4)*.00025
if record['profile_zero'] is not None:
 z=record['profile_zero']/.00025;record['intact_envelope_at_profile_zero']=intactmax(z)
record['note']='Broad profile and intact envelope are reported together; a local cap mode is not equated with a finite connected increment.'
np.savez_compressed(B/'quadratic.npz',base=base,c0=c0,c1=c1,c2=c2)
(O/'calibration.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
