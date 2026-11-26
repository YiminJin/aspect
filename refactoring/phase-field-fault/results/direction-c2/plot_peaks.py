"""Plot ordinary strip-8 endpoint errors against the direct native-point peaks at 5 ms."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    current = [row for row in rows(here / 'accepted-peaks.csv') if float(row['time_s']) == 0.005]
    prior = {row['case']: row for row in rows(here.parent / 'direction-c1' / 'peak-summary.csv')
             if row['method'] == 'production'}
    labels = {'frozen-2': 'Constant, 2 ranks', 'affine-1': 'Affine, 1 rank', 'affine-2': 'Affine, 2 ranks'}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for ax, old_key, error_key, reference_key, absolute, xlabel in (
            (axes[0], 'angle_rad', 'beta_minus_peak_rad', 'peak_rad', 1e-7, 'Absolute angle error (rad)'),
            (axes[1], 'length_m', 'ds_minus_expected_m', 'expected_ds_m', 1e-11, 'Absolute arclength error (m)')):
        for y, row in enumerate(current):
            ref = float(row[reference_key])
            before = abs(float(prior[row['case']][old_key]) - ref)
            after = abs(float(row[error_key]))
            budget = absolute + 1e-7 * abs(ref)
            ax.plot([after, before], [y, y], color='0.8', zorder=1)
            ax.scatter(before, y, color='#994b23', marker='s', label='Before C2' if y == 0 else None)
            ax.scatter(after, y, color='#176879', marker='o', label='Corrected' if y == 0 else None)
            ax.plot(budget, y, marker='|', linestyle='none', color='#282828', markersize=18, markeredgewidth=2,
                    label='Tolerance' if y == 0 else None)
        ax.set(xscale='log', xlabel=xlabel, ylim=(len(current) - 0.6, -0.45))
        ax.grid(axis='x', alpha=0.25)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(current)), [labels[row['case']] for row in current])
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Strip 8 at the first accepted turn (5 ms)')
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    fig.savefig(here / 'peak-errors.png', dpi=180, facecolor='white')
