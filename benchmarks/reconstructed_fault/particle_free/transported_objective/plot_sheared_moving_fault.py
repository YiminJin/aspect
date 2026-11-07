#!/usr/bin/env python3
"""Plot saved four-run field, geometry, spin, energy and cost diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/sheared_moving_fault'
os.environ['MPLCONFIGDIR'] = '/tmp/sheared-moving-fault-matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    result = json.loads((OUT/'comparison.json').read_text())
    rows = list(result['cases'].values())
    labels = ['32², dt=.1', '32², dt=.05', '32², dt=.025', '64², dt=.025']
    colors = ['#777777', '#e69f00', '#0072b2', '#009e73']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    fields = [('stress_norm', 'Physical deviatoric stress norm', 'Pa'),
              ('shear_traction', 'Shear traction on own curve', 'Pa'),
              ('total_normal_traction', 'Normal traction (tensile positive)', 'Pa'),
              ('slip_rate', 'Endpoint slip rate', 'm/s'),
              ('cumulative_slip', 'Cumulative slip after initialization', 'm'),
              ('normalization', 'Localization normalization', 'm')]
    for ax, (key, title, units) in zip(axes.flat, fields):
        for row, label, color in zip(rows, labels, colors):
            p = row['final_profiles']; ax.plot(p['labels'], p[key], label=label, color=color, lw=1.3)
        ax.axvspan(0, .2, color='#cccccc', alpha=.25)
        ax.axvspan(.8, 1, color='#cccccc', alpha=.25)
        ax.set(title=title, xlabel='Material label coordinate', ylabel=units, xlim=(0, 1))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Moving fault with shear 0.5 s⁻¹ and translation 0.4 m/s — t=0.4 s\nOrdinary DGQ2 / SSPRK3; fixed 18 surface labels; shaded endpoint regions')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'sheared-fields.{ext}', dpi=170)
    plt.close(fig)

    maps = np.load(OUT/'boundary-difference-maps.npz')
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
    titles = ['32²: dt .1 versus .05', '32²: dt .05 versus .025', 'dt .025: 32² versus 64²']
    vmax = max(float(maps[k].max()) for k in maps.files)
    for ax, key, title in zip(axes, ['dt_to_half', 'half_to_quarter', 'space'], titles):
        im = ax.imshow(maps[key].T, origin='lower', extent=(-2, 2, -2, 2), norm=LogNorm(1e-4, vmax), cmap='magma')
        X = np.array(rows[2]['trajectories'][-1]['geometry']); ax.plot(X[:, 0], X[:, 1], color='cyan', lw=1)
        ax.set(title=title, xlabel='x (m)', ylabel='y (m)')
    fig.colorbar(im, ax=axes, label='Cell RMS of tensor difference (Pa)', shrink=.8)
    fig.suptitle('Whole-domain discrepancies concentrate at the upper-left boundary corner')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'sheared-boundary-differences.{ext}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    ax = axes[0, 0]
    X = np.array(rows[0]['trajectories'][0]['geometry']); ax.plot(X[:, 0], X[:, 1], 'k--', label='Initial')
    for row, label, color in zip(rows, labels, colors):
        X = np.array(row['trajectories'][-1]['geometry']); ax.plot(X[:, 0], X[:, 1], '.-', color=color, label=label, ms=3)
    ax.set(title='Curve geometry (domain is [-2,2]²)', xlabel='x (m)', ylabel='y (m)'); ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for row, label, color in zip(rows, labels, colors):
        spin = np.array(row['spin_at_labels']); ax.plot(np.linspace(0, 1, len(spin)), spin*180/np.pi, '.-', label=label, color=color)
    ax.axhline(-.1*180/np.pi, color='k', ls='--', lw=1, label='Nominal affine spin')
    ax.set(title='Accumulated accepted-gradient spin', xlabel='Material label coordinate', ylabel='degrees'); ax.legend(fontsize=8)
    ax = axes[1, 0]
    for row, label, color in zip(rows, labels, colors):
        steps = row['coupled_work']['steps'][1:]; t = [s['time'] for s in steps]
        err = np.array([s['coupled_endpoint_remainder'] for s in steps])
        ax.plot(t, np.cumsum(err), color=color, label=label)
        ax.plot(t, np.cumsum(abs(err)), color=color, ls='--')
    ax.set(title='Coupled endpoint energy remainder\nsolid: signed; dashed: accumulated absolute', xlabel='Time (s)', ylabel='J/m'); ax.legend(fontsize=8)
    ax = axes[1, 1]; x = np.arange(4)
    ax.bar(x, [r['wall_s'] for r in rows], color=colors)
    for i, r in enumerate(rows):
        ax.text(i, r['wall_s']+5, f"{r['linear_solves']} solves", ha='center', fontsize=8)
    ax.set(xticks=x, xticklabels=labels, ylabel='Wall time (s), four MPI ranks', title='Cost including initialization and output')
    ax.set_ylim(top=max(r['wall_s'] for r in rows)*1.15); ax.tick_params(axis='x', labelrotation=15)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'sheared-diagnostics.{ext}', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
