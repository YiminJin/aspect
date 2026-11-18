#!/usr/bin/env python3
"""Read-only review of accepted pilot geometry and logged constrained-solve cost."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-propagation-pilot-mpl')
from pathlib import Path
import csv
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase import read

ROOT = Path(__file__).resolve().parents[4]
BUILD = ROOT / 'build-particle-free/stage4-propagation-pilot'
RUN = ROOT / 'build-particle-free/bulk-tip/runs/stage4-propagation-pilot-resume51'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot'
log = (RUN/'run.log').read_text()
chunks = re.split(r'\*\*\* Timestep (\d+):', log)
rows = []
for index in range(1, len(chunks), 2):
    step, body = int(chunks[index]), chunks[index+1]
    solves = re.findall(r'Tip bound linear solve: pass=(\d+), iterations=(\d+), residual=([^,]+), seconds=([\d.e+\-]+)', body)
    newton = re.findall(r'Bulk tip coupled Newton (\d+): mechanics=([^,]+), phase KKT=([^,]+)', body)
    rows.append(dict(step=step, accepted='Postprocessing:' in body,
                     logged_linear_minutes=sum(float(v[3]) for v in solves)/60,
                     linear_solves=len(solves),
                     maximum_active_set_pass=max((int(v[0]) for v in solves), default=0),
                     last_newton_iteration=int(newton[-1][0]) if newton else -1,
                     last_newton_KKT=float(newton[-1][2]) if newton else np.nan,
                     shifted_searches=len(re.findall(r'accepted search: shift=(?!0,)', body))))
with (OUT/'cost-review.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
history = read(OUT/'through70-history.csv')
fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True, sharex=True)
axes[0].bar([r['step'] for r in rows], [r['logged_linear_minutes'] for r in rows],
            color=['tab:blue' if r['accepted'] else 'tab:red' for r in rows])
axes[0].set(ylabel='Logged linear solves (min)', title='Pilot cost: accepted steps blue; stopped trial red')
axes[0].text(.01,.96,'Excludes assembly, checkpoints and other overhead',transform=axes[0].transAxes,va='top',fontsize=9)
increments=np.diff(history['right_advance']); steps=history['step'][1:]; mask=steps>=52
axes[1].bar(steps[mask],increments[mask]*1e3,color='tab:orange')
axes[1].set(xlabel='Native timestep number (dt = 80 microseconds)',ylabel='Right contour increment (mm)',title='Accepted connected phi = 0.05 contour; no trial geometry included')
for extension in ['png','pdf']:
    fig.savefig(OUT/('cost-review.'+extension),dpi=180)
record=dict(stopped_trial=rows[-1],accepted_endpoint=.0056,
            ordinary_median_linear_minutes=float(np.median([r['logged_linear_minutes'] for r in rows if r['step']<=66])),
            final_right_advance=float(history['right_advance'][-1]),
            final_right_advance_cells=float(history['right_advance'][-1]/.03125),
            note='Logged cost is a lower bound on wall time. The interrupted trial is not a native solver failure. Alternating contour increments require a timestep comparison before a physical interpretation.')
(OUT/'cost-review.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
