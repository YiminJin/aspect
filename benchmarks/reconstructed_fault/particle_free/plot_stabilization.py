#!/usr/bin/env python3
"""Plot the measured prescribed PF4 histories, with no coupled-physics claim."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from assess_transport import Entry, shape
from assess_stabilization import finite_scalar

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--assessment', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
e = Entry(a.run)
result = json.loads(a.assessment.read_text())
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
methods = [('unlimited', 'Unlimited'), ('kxrcf', 'KXRCF + WENO'), ('fu_shu', 'Fu–Shu + WENO'), ('scaling', 'Scaling')]
for panel, case in zip(axes[:2], ['signed_jump', 'finite_width']):
    for method, label in methods:
        data = e.coefficients_for(case+'-'+method)
        x, y = [], []
        for k in np.argsort(e.cells['x0']):
            c = e.cells[k]
            if c['y0'] != np.min(e.cells['y0']):
                continue
            r = np.linspace(0, 1, 80)
            N, _, _ = shape(r, np.full_like(r, .5), e.basis)
            x.extend(c['x0']+c['dx']*r)
            y.extend((N@data[e.indices[k]])[:, 0]/1000)
        panel.plot(x, y, label=label, linewidth=1.4)
    x = np.linspace(0, 1, 3000)
    reference = np.where((x >= .33) & (x < .58), 1., np.where((x >= .58) & (x < .83), -1., 0.)) if case == 'signed_jump' else finite_scalar(x, .2, float(e.constants['width']))
    panel.plot(x, reference, 'k--', label='Analytic', linewidth=1.)
    panel.set(xlabel='x (m)', ylabel='Axx / 1000 Pa')
axes[0].set(title='Sharp pulse, 32 × 64, dt/4', xlim=(.2, .92))
axes[1].set(title='Finite-width pulse: first transition', xlim=(.2, .5), ylim=(0., 1.))
axes[0].legend(fontsize=7)
for method, label in methods:
    values = result['timestep_comparison']['signed_jump-'+method]
    axes[2].semilogy([1, 2], [100*c['relative_difference'] for c in values], 'o-', label=label)
axes[2].set(title='Sharp-pulse timestep sensitivity', ylabel='Full-tensor field difference (%)', xticks=[1, 2], xticklabels=['dt vs dt/2', 'dt/2 vs dt/4'])
axes[2].legend(fontsize=7)
fig.tight_layout()
for suffix in ['.png', '.pdf']:
    fig.savefig(a.output.with_suffix(suffix), dpi=160)
plt.close(fig)
