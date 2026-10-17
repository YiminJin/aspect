#!/usr/bin/env python3
"""Plot native final pulse polynomials and physical means from the CLS audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_methods import Data
from assess_stabilization import finite_scalar
from assess_limiter import polynomial_matrices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assessment', required=True, type=Path)
    p.add_argument('--runs-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    measured = json.loads(args.assessment.read_text())['transport']
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharex='col')
    colors = {'kxrcf': '#d95f02', 'fu_shu': '#1b9e77', 'unlimited': '#555555'}
    xx = np.linspace(0, 1, 4001)
    for column, case in enumerate(['signed_jump', 'finite_width']):
        z = xx-.08
        analytic = np.where((z >= .25) & (z < .5), 1., np.where((z >= .5) & (z < .75), -1., 0.)) if case == 'signed_jump' else finite_scalar(xx, .2, .15625)
        for ax in axes[:, column]:
            ax.plot(xx, analytic, color='black', linestyle=':', linewidth=1.5, label='analytic')
            ax.grid(alpha=.2)
            ax.set_ylabel('scalar amplitude')
        rows = [r for r in measured if r['case'] == case and r['divisor'] == 4]
        for record in sorted(rows, key=lambda r: (r['detector'] != 'unlimited', r['construction'], r['detector'])):
            d = Data(args.runs_root/record['label'])
            values = d.load('field-'+case+'-'+record['detector'])
            cells = [k for k, c in enumerate(d.cells) if .5 < c['y0']+.5*c['dy'] < .5+c['dy']]
            cells.sort(key=lambda k: d.cells['x0'][k])
            r = np.linspace(0, 1, 17)
            N = d.shape(r, np.full_like(r, .5))
            Q, _, w = polynomial_matrices(d)
            x, a, centers, averages = [], [], [], []
            for k in cells:
                c = d.cells[k]
                x.extend(c['x0']+c['dx']*r)
                a.extend((N@values[k])[:, 0]/1000)
                centers.append(c['x0']+.5*c['dx']); averages.append(float(w@Q@values[k, :, 0]/1000))
            detector = record['detector']
            label = 'unlimited' if detector == 'unlimited' else detector+' '+record['construction']
            style = '-' if record['construction'] == 'cls' or detector == 'unlimited' else '--'
            axes[0, column].plot(x, a, color=colors[detector], linestyle=style, linewidth=1.1, label=label)
            axes[1, column].plot(centers, averages, color=colors[detector], linestyle=style, marker='.', markersize=3, linewidth=1, label=label)
        axes[0, column].set_title(('Sharp pulse' if case == 'signed_jump' else 'Finite-width pulse')+'; native DGQ2, dt/4')
        axes[1, column].set_xlabel('x (m)')
        axes[1, column].set_title('Physical cell means along the middle row')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8)
    fig.tight_layout(rect=[0, .065, 1, 1])
    fig.savefig(args.output, dpi=170)


if __name__ == '__main__':
    main()
