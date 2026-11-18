#!/usr/bin/env python3
"""Read-only diagnosis of small positive phase and material-plane registration."""
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path);p.add_argument('--report',type=Path,required=True)
a=p.parse_args();result=[]
for f in sorted(a.output.glob('tip-planes-*.csv')):
 step=int(f.stem.split('-')[-1])
 if step<=1:continue
 rows=list(csv.DictReader(f.open()))
 new=[r for r in rows if r['formed']=='1' and r['parent_formed']=='0']
 if not new:continue
 fields=[np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf8') for path in sorted(a.output.glob(f'tip-points-{step:04d}-r*.csv.gz'))]
 if not fields:continue
 fields=np.concatenate(fields)
 balance=np.concatenate([np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf8') for path in sorted(a.output.glob(f'tip-balance-{step:04d}-r*.csv.gz'))])
 mask=np.isin(fields['section'],[int(r['section']) for r in new])
 cells=np.unique(fields['cell'][mask]);b=balance[np.isin(balance['cell'],cells)]
 # Exported residual is the physical force Kf*raw; Kf=2 Pa here.
 prediction=b['stage_B']-b['residual']/(2*b['mass'])
 result.append(dict(step=step,new_sections=len(new),x_min=min(float(r['x']) for r in new),x_max=max(float(r['x']) for r in new),
   new_phase_max=float(max(fields['phi'][mask])),new_A_norm_max=float(max(np.sqrt(fields['Axx'][mask]**2+fields['Ayy'][mask]**2+fields['Azz'][mask]**2+2*fields['Axy'][mask]**2))),
   phase_coefficient_max=float(max(b['B'])),phase_lower_max=float(max(b['lower'])),
   physical_residual_min=float(min(b['residual'])),physical_residual_max=float(max(b['residual'])),
   lower_active_rows=int(np.count_nonzero(prediction<=b['lower'])),rows=len(b),
   angle_min=min(float(r['angle']) for r in new),angle_max=max(float(r['angle']) for r in new),
   slip_max=max(abs(float(r['slip'])) for r in new),parent_age_min=min(float(r['parent_age']) for r in new),
   diagnostic_cutoff_counts={str(t):int(np.count_nonzero(fields['phi'][mask]>t)) for t in [0.,1e-14,1e-12,1e-10,1e-8]}))
a.report.write_text(json.dumps({'run':str(a.output),'registrations':result,'cutoffs_are_read_only_observations':True},indent=2)+'\n')
print(json.dumps(result[:1],indent=2))
