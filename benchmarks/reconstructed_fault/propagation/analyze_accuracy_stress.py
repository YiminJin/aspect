#!/usr/bin/env python3
"""Separate authoritative history accuracy from FE representation and drift."""
import csv,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'accuracy_energy_evidence'
path=ROOT/'build-propagation-integration/output-accuracy_stress_final/bulk-stress-native.csv'
rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(path.open())]
assert len(rows)==354
result={}
for pattern in range(3):
 a=[r for r in rows if r['pattern']==pattern]
 assert max(r['particle_error_Pa'] for r in a)<2e-10 and max(r['FE_additional_drift_Pa'] for r in a)<2e-10
 result[str(pattern)]=dict(description=['constant','affine normals, bilinear shear','Q1 mesh-scale nonconstant'][pattern],max_particle_error_Pa=max(r['particle_error_Pa'] for r in a),max_FE_drift_Pa=max(r['FE_additional_drift_Pa'] for r in a),max_FE_representation_error=max(r['FE_representation_relative'] for r in a),max_energy_representation_change=max(abs(r['representation_energy_change']) for r in a),finals=[r for r in a if r['time']==1])
 for eta in [1e8,1e30]:
  finals=[r for r in a if r['time']==1 and r['eta']==eta]
  assert len(finals)==3
  for key in ['particle_energy','FE_energy','analytic_energy']:
   assert max(r[key] for r in finals)-min(r[key] for r in finals)<2e-12
(OUT/'stress.json').write_text(json.dumps(dict(paths=18,native_updates=336,records=354,patterns=result),indent=2)+'\n')
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-accuracy-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,3.2),constrained_layout=True)
for pattern in range(3):
 a=[r for r in rows if r['pattern']==pattern and r['eta']==1e8 and r['dt']==.03125]
 for ax,key in zip(axes,['particle_error_Pa','FE_additional_drift_Pa','FE_representation_relative']):
  ax.plot([r['time'] for r in a],[r[key] for r in a],label=result[str(pattern)]['description'])
for ax,title,y in zip(axes,['Particle Maxwell error','Additional FE transfer drift','FE representation error'],['Pa','Pa','Relative L2']):
 ax.set(title=title,xlabel='Time (s)',ylabel=y);ax.grid(alpha=.25)
axes[2].legend(fontsize=7)
fig.savefig(OUT/'stress.png',dpi=160);fig.savefig(OUT/'stress.pdf')
for k,v in result.items():print(k,v['max_particle_error_Pa'],v['max_FE_drift_Pa'],v['max_FE_representation_error'])
