#!/usr/bin/env python3
"""Phase, taper-rate and physical-work corroboration for a single pilot."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-propagation-pilot-mpl')
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase import coefficients, read, value

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot'
parser = argparse.ArgumentParser()
parser.add_argument('label')
parser.add_argument('--output', type=Path, help='Explicit composed accepted-history view')
args = parser.parse_args()
out = ROOT / 'build-particle-free/bulk-tip/runs' / ('stage4-propagation-' + args.label) / 'output'
if args.output:
    out = args.output
history = read(EVIDENCE / (args.label + '-history.csv'))
initial = coefficients(out, 0)
final = coefficients(out, int(history['step'][-1]))
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
x = np.linspace(5.5, 6.3, 1201)
# Both the original axis and the prepared upper tip show the finite-width tail.
for y, style in [(2., '-'), (2.009536049889988, '--')]:
    p0, p = value(initial, x, y), value(final, x, y)
    for data, color, label in [(p0, '0.4', 'initial'), (p, 'tab:blue', 'final'),
                                (p-p0, 'tab:orange', 'increment')]:
        axes[0, 0].plot(x, data, style, color=color, label=f'{label}, y={y:.4f}')
axes[0, 0].axhline(.05, color='k', lw=.7)
axes[0, 0].set(xlabel='x (m)', ylabel='Phase', title='Original right endpoint; absolute level .05')
axes[0, 0].legend(fontsize=7)
increment = np.mean(final-initial, axis=(2, 3))
im = axes[0, 1].imshow(increment, origin='lower', extent=[0, 8, 0, 4],
                       aspect='auto', cmap='magma')
axes[0, 1].set(xlim=(5.3, 6.5), ylim=(1.7, 2.3), xlabel='x (m)', ylabel='y (m)',
               title='Exact cell-mean phase increment')
fig.colorbar(im, ax=axes[0, 1], label='Mean phase increment')
positions = [4.5, 5.5, 5.9, 6.0]
rates = []
for step in history['step'].astype(int):
    sections = read(out / f'pilot-sections-{step:04d}.csv')
    rates.append([np.interp(x, sections['x'], sections['rate_stage']) for x in positions])
for x, series in zip(positions, np.array(rates).T):
    axes[1, 0].plot(history['stage_time']*1e3, series*1e3, label=f'x={x:g} m')
axes[1, 0].set(xlabel='Stage time (ms)', ylabel='Signed slip rate (mm/s)', title='Patch and formed taper')
axes[1, 0].legend()
axes[1, 1].plot(history['time']*1e3, history['added_fracture'], label='Added fracture')
axes[1, 1].plot(history['time']*1e3, np.cumsum(history['work_error']), label='Signed work error')
axes[1, 1].plot(history['time']*1e3, np.cumsum(abs(history['work_error'])), '--', label='Accumulated absolute error')
axes[1, 1].set(xlabel='Endpoint time (ms)', ylabel='J/m', title='Physical work, before ledger corrections', yscale='symlog')
axes[1, 1].set_yscale('symlog', linthresh=1e-12)
axes[1, 1].legend()
for ext in ['png', 'pdf']:
    fig.savefig(EVIDENCE / (args.label + '-response.' + ext), dpi=180)
