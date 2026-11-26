"""Plot the direct point-score loss at each selected orientation, including the close pair at the summit."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


CASES = (('frozen-2', 'Constant pressure · 2 MPI ranks'),
         ('affine-1', 'Affine pressure · 1 MPI rank'),
         ('affine-2', 'Affine pressure · 2 MPI ranks'))
MARKERS = {'production': ('#a13f17', 's'), 'candidate': ('#007762', '^'),
           'independent': ('#202a45', '*')}


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def draw(data, destination):
    summary = {(r['case'], r['method']): r for r in read_rows(data/'peak-summary.csv')}
    curves = read_rows(data/'peak-curves.csv')
    fig, axes = plt.subplots(3, 2, figsize=(9.3, 7.9), width_ratios=(1.6, 1))
    for (case, title), row in zip(CASES, axes):
        for ax, window, x_scale, y_scale in zip(row, ('wide', 'inset'), (1e6, 1e9), (1e21, 1e25)):
            samples = sorted((x_scale * float(r['offset_rad']),
                              y_scale * float(r['direct_minus_peak_Pa2_m2']))
                             for r in curves if r['case'] == case and r['window'] == window)
            if len(samples) < 5:
                raise ValueError(f'Missing {window} curve for {case}')
            x, y = zip(*samples)
            ax.plot(x, y, color='#525d70', lw=1.6)
            for method, (color, marker) in MARKERS.items():
                point = summary[case, method]
                dx = x_scale * float(point['angle_minus_peak_rad'])
                if window == 'inset' and not x[0] <= dx <= x[-1]:
                    continue
                ax.plot(dx, y_scale * float(point['direct_minus_peak_Pa2_m2']),
                        color=color, marker=marker, ms=10 if marker == '*' else 7,
                        ls='none', zorder=4, markeredgecolor='white', markeredgewidth=.55)
            ax.set_ylim(top=.09 * max(abs(min(y)), abs(max(y))))
            ax.grid(axis='y', lw=.5, color='#d5d8dc')
            ax.spines[['right', 'top']].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=9)
        row[0].set_title(title, loc='left', size=11, fontweight='bold')
        beta = float(summary[case, 'independent']['angle_rad'])
        row[1].set_title(f'Direct peak at {beta:.12f} rad', loc='left', size=10)
        row[0].set_ylabel('Score difference\n($10^{-21}$ Pa² m²)')
        row[1].set_ylabel('Score difference\n($10^{-25}$ Pa² m²)')
    axes[-1, 0].set_xlabel('Angle relative to direct peak (µrad)')
    axes[-1, 1].set_xlabel('Angle relative to direct peak (nrad)')
    fig.suptitle('Strip 8 at the first 5 ms turn', x=.10, ha='left', fontsize=16)
    names = ('Production', 'Candidate', 'Independent peak')
    handles = [Line2D([], [], ls='none', marker=m, color=c, markersize=8, label=name)
               for (c, m), name in zip(MARKERS.values(), names)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.09, .942), ncol=3, frameon=False)
    fig.subplots_adjust(top=.855, bottom=.073, left=.11, right=.97, hspace=.65, wspace=.38)
    fig.savefig(destination, dpi=195)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='Directory with peak-summary.csv and peak-curves.csv')
    parser.add_argument('figure', type=Path)
    args = parser.parse_args()
    draw(args.data, args.figure)
