"""Plot the accepted native growing-fault stress for the two public-modulus runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def frame(output):
    params = json.loads((output / 'parameters.json').read_text())['Material model']
    modulus = float(params['Phase field fault']['Shear modulus']['value'])
    with (output / 'balance-0001.csv').open() as stream:
        clock = float(next(csv.DictReader(stream))['time'])
    samples = {}
    for path in sorted(output.glob('points-0001-r*.csv')):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = (row['cell'], int(row['q']))
                if key in samples:
                    raise ValueError(f'Duplicate native point {key}')
                samples[key] = tuple(float(row[name]) for name in ('x', 'y', 'tauxy'))
    return modulus, clock, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('public_1', 'public_2', 'figure'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    g1, t1, a = frame(args.public_1)
    g2, t2, b = frame(args.public_2)
    if (g1, g2) != (1e6, 2e6) or t1 != t2 or abs(t1 - .005) > 1e-15:
        raise ValueError('Expected 1 MPa and 2 MPa public moduli at the accepted 5 ms frame')
    if not a or a.keys() != b.keys():
        raise ValueError('Native material identities changed between runs')
    keys = sorted(a)
    x, y, tau1 = np.array([a[k] for k in keys]).T
    x2, y2, tau2 = np.array([b[k] for k in keys]).T
    if not np.array_equal(x, x2) or not np.array_equal(y, y2):
        raise ValueError('Native sample positions changed')
    change = tau2 - tau1
    if not np.isfinite(change).all() or not np.any(change):
        raise ValueError('Expected a finite, nonzero response to the modulus edit')
    for i in (0, np.argmax(np.abs(change))):
        print(f'cell {keys[i][0]} q {keys[i][1]} at ({x[i]:.8g}, {y[i]:.8g}) m: '
              f'{tau1[i]:.12g} -> {tau2[i]:.12g} Pa, change {change[i]:+.12g} Pa')

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), layout='constrained',
                             sharex=True, sharey=True)
    fig.suptitle(r'Accepted native shear stress $\tau_{xy}$ at 5 ms', fontsize=14)
    absolute = Normalize(min(tau1.min(), tau2.min()), max(tau1.max(), tau2.max()))
    for ax, data, title in zip(axes, (tau1, tau2, change),
                               ('G = 1 MPa', 'G = 2 MPa', 'Change from 1 to 2 MPa')):
        diff = ax == axes[2]
        artist = ax.scatter(x, y, c=data, s=4, linewidths=0, marker='s',
                            cmap='cividis' if diff else 'viridis',
                            norm=None if diff else absolute)
        ax.set(xlim=(0, 2), ylim=(0, 1), xlabel='x (m)', title=title, aspect='equal')
        ax.set_xticks([0, .5, 1, 1.5, 2])
        ax.set_yticks([0, .25, .5, .75, 1])
    axes[0].set_ylabel('y (m)')
    fig.colorbar(axes[0].collections[0], ax=axes[:2], orientation='horizontal',
                 shrink=.70, label=r'Accepted $\tau_{xy}$ (Pa)')
    fig.colorbar(artist, ax=axes[2], orientation='horizontal',
                 shrink=.95, label=r'Change in $\tau_{xy}$ (Pa)')
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
