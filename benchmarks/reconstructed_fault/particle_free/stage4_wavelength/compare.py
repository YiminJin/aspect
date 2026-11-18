#!/usr/bin/env python3
"""Transfer checks and continuous-law bounds; no additional mechanical runs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];OUT=HERE.parent/'evidence/stage4_wavelength'
sys.path.insert(0,str(HERE.parent/'stage4_rsf'))
from stability import friction
def read(path):return np.genfromtxt(path,delimiter=',',names=True)
records=[json.loads(f.read_text()) for f in sorted(OUT.glob('spectrum-*.json'))]
result={}
coarse=ROOT/'build-particle-free/bulk-tip/runs/stage4-wavelength-128/output'
s0=read(coarse/'restoring-sections.csv');r0=read(coarse/'restoring-modes.csv');n0=len(s0)
C0=r0['slip_response_per_force_coefficient'].reshape(-1,n0).T
input0=ROOT/'build-particle-free/stage4-wavelength/input-128'
v0=read(input0/'operating.csv')
for record in records:
 nx=record['nx'];run=ROOT/f'build-particle-free/bulk-tip/runs/stage4-wavelength-{nx}/output'
 directory=ROOT/f'build-particle-free/stage4-wavelength/input-{nx}'
 s=read(run/'restoring-sections.csv');raw=read(run/'restoring-modes.csv');C=raw['slip_response_per_force_coefficient'].reshape(-1,len(s)).T
 v=read(directory/'operating.csv');x=s['x'];w=s['ds']
 smooth=[]
 for j in range(6):
  reference=np.interp(x,s0['x'],C0[:,j]);smooth.append(float(np.sqrt(sum(w*(C[:,j]-reference)**2)/sum(w*reference**2))))
 dc=3.2211126126299625e-7
 _,fv,ft=friction(v['V'],v['theta'],dc)
 bound=ft*v['theta']*np.sign(v['V'])/(dc*fv)-abs(v['V'])/dc
 active=s['cohesive_compliance_bound_m_Pa']>0
 # Diagnostic local cohesive-only stiffness limit; NOT a measured native
 # high-wave-number limit, because transverse FE restriction remains fixed.
 k=1/s['cohesive_compliance_bound_m_Pa'][active];q=abs(v['V'][active])/dc
 c=ft[active]*v['theta'][active]*np.sign(v['V'][active])/dc;f=fv[active];r=.01
 linear=f*(r+q)+k-c;constant=q*(f*r+k)-r*c
 roots=(-linear.astype(complex)+np.sqrt(linear**2-4*f*constant+0j))/(2*f)
 result[str(nx)]=dict(smooth_action_relative_errors_vs_coarse_physical_linear_interpolant=smooth,
  represented_V_error=float(max(abs(v['V']-np.interp(x,v0['x'],v0['V'])))),
  theta_error=float(max(abs(v['theta']-v0['theta'][0]))),
  physical_section_measure=float(sum(w)),section_measure_relative_change=float(sum(w)/sum(s0['ds'])-1),
  integrated_operating_rate=float(sum(w*v['V'])),peak_rate=float(max(v['V'])),
  phase_cross_section_error=float(max(abs(read(directory/'cross-section.csv')['phi']-read(input0/'cross-section.csv')['phi']))),
  all_space_growth_upper_bound=float(max(bound)),cohesive_stiffness_lower_bound=float(min(k)),
  cohesive_only_uncoupled_local_max_growth=float(max(roots.real)))
(OUT/'transfer-and-bounds.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
