#!/usr/bin/env python3
"""Plot retained midpoint failures, successful time controls and loading screens."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-tip-time-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update'
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    ax = axes[0, 0]
    names = ['tip-time-parent01', 'tip-time-parent-half01', 'tip-time-parent-quarter01']
    end = .060000000000000019
    start = .057500000000000016
    progress = []
    for label in names:
        p = RUNS / label / 'output/bulk-accepted.csv'
        times = np.atleast_1d(np.genfromtxt(p, delimiter=',', names=True))['time'] if p.exists() else []
        progress.append(100 * (max(times, default=start) - start) / (end - start))
    bars = ax.bar(['Δt', 'Δt/2', 'Δt/4'], progress, color='#ad4e4e')
    for bar, value in zip(bars, progress):
        ax.text(bar.get_x()+bar.get_width()/2, value+3, 'Failed', ha='center')
    ax.set(ylim=(0, 110), ylabel='Completed fraction of window (%)',
           title='Common parent .0575 → .06 s; Δt = .0025')
    ax.axhline(100, color='gray', ls='--', lw=1)

    ax = axes[0, 1]
    for label, name in [('tip-time-local01', 'Δt=.005'),
                        ('tip-time-local-half01', 'Δt=.0025')]:
        rows = np.atleast_1d(np.genfromtxt(RUNS/label/'output/bulk-accepted.csv',
                                         delimiter=',', names=True))
        rem = rows['Wext']-rows['energy_change']-rows['Db']-rows['Dc']-rows['Df']
        ax.plot(rows['time'], np.cumsum(rem)*1e6, label=name+' signed')
        ax.plot(rows['time'], np.cumsum(abs(rem))*1e6, ls=':', label=name+' absolute')
    ax.set(xlabel='Time (s)', ylabel='Physical work remainder (μJ/m)',
           title='Old loading; accepted midpoint growth controls')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for filename, label, style in [('screen-width12.json', 'tip-time-screen-seed01', '-'),
                                    ('screen-width20.json', 'tip-time-screen-wide-seed01', '--')]:
        samples = read(filename)['cases'][label]['samples']
        width = '1.2' if style == '-' else '2.0'
        for region, color in [('tip', '#237a46'), ('remote', '#3974aa'), ('boundary', '#bb4d36')]:
            ax.plot([s['time'] for s in samples],
                    [s['regions'][region]['max_stage_Y_over_Kf'] for s in samples],
                    style, color=color, label=f'{region}, width {width}')
    ax.axhline(1, color='gray', lw=1)
    ax.set(yscale='log', xlabel='Time (s)', ylabel='Maximum Y / Kf',
           title='Fixed seed; boundary driving precedes tip driving')
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    pilot = read('compact-pilot.json')['cases']['tip-time-pilot02']
    values = [pilot['energy'][k]['signed'] for k in
              ['stress_loss', 'traction_loss', 'fracture_loss']]+[pilot['added_fracture']]
    ax.bar(['BE stress\nloss', 'BE cohesive\nloss', 'BE fracture\nloss', 'Added\nfracture'],
           values, color=['#bb4d36', '#cc8d4e', '#d5bc6b', '#237a46'])
    ax.set(yscale='log', ylabel='Energy (J/m)',
           title='Compact BE pilot; stopped at boundary damage')
    fig.savefig(OUT/'time-update-summary.png', dpi=170)
    fig.savefig(OUT/'time-update-summary.pdf')


if __name__ == '__main__':
    main()
