#!/usr/bin/env python3
"""Export the bounded PF6a spatial, temporal and work comparisons."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from assess_pf6a import Ref, samples_at

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs', type=Path, default=Path('build-particle-free/runs'))
p.add_argument('--assessment', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
x = json.loads(a.assessment.read_text())
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
y = np.linspace(0, 1, 1601)
for item in [x['runs'][0], x['runs'][2], x['runs'][3]]:
    name = item['label']; step = int(item['metrics'][-1]['step'])
    r = Ref(a.runs/name/'run'/f'state-{step}')
    phase, _, stress = samples_at(r, np.full_like(y, .25), y)
    label = ('64 cells, dt/4' if 'spatial' in name else '32 cells, '+('dt/4' if name.endswith('-4') else 'dt'))
    axes[0, 0].plot(y, phase, label=label)
    axes[0, 1].plot(y, stress[:, 2]/1000, label=label)
axes[0, 0].set(xlabel='Normal coordinate (m)', ylabel='Physical phase', xlim=(.2, .8))
axes[0, 1].set(xlabel='Normal coordinate (m)', ylabel='Physical shear stress (kPa)', xlim=(.2, .8))
axes[0, 0].legend(fontsize=8)
c = x['comparisons']
for field in ['phase', 'T', 'stress']:
    z = [c[k][field+'_L2_relative'] for k in ['temporal_dt_dt2', 'temporal_dt2_dt4']]
    axes[1, 0].loglog([1/32, 1/64], z, 'o-', label=field)
axes[1, 0].set(xlabel='Larger timestep (s)', ylabel='Relative L2 difference')
axes[1, 0].legend(fontsize=8)
work = [sum(r['ledger']['external_remainder'] for r in q['metrics'] if 'ledger' in r) for q in x['runs'][:3]]
axes[1, 1].loglog([1/32, 1/64, 1/128], np.abs(work), 'o-')
axes[1, 1].set(xlabel='Timestep (s)', ylabel='Accumulated work remainder (J/m)')
for ax in axes.ravel(): ax.grid(alpha=.2)
fig.suptitle('PF6a: temporal control passes; normal spatial sensitivity remains')
fig.savefig(a.output, dpi=170)
