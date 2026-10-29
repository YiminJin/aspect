#!/usr/bin/env python3
"""Equal physical scales for the single aligned-wide mesh comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from assess_spatial import CASES, q1, RADII, ANGLES
from assess_native import M, G, gauss

COLORS = {'coarse': 'tab:blue', 'refined': 'tab:orange'}
LABELS = {'coarse': r'$48\times64$', 'refined': r'$96\times128$'}


def main(output):
    report = json.loads((output/'assessment.json').read_text())
    data = {name: dict(np.load(output/f'{name}-fields.npz')) for name in CASES}
    xx, yy = np.meshgrid(np.linspace(.45, .86, 247), np.linspace(.15, .85, 421))
    points = np.stack([xx, yy], axis=-1); maps = {}; full_qp = {}; full_maps = {}; time = .15
    X, Ygrid = np.meshgrid(np.linspace(0, 1.5, 361), np.linspace(0, 1, 241))
    full_points = np.stack([X, Ygrid], axis=-1); global_peaks = []
    for name, d in data.items():
        L = dict(x=d['x'], y=d['y'], dx=d['x'][1]-d['x'][0],
                 dy=d['y'][1]-d['y'][0], grid_ids=d['grid_ids'])
        step = np.flatnonzero(abs(d['time']-time) < 1e-12)[-1]
        global_peaks.append(float(np.max(d['phase'][step]-d['phase'][0])))
        p0 = q1(L, d['phase'][0], points)[0]
        delta = q1(L, d['phase'][step]-d['phase'][0], points)[0]
        # Initial dense rows and compact q have identical original-QP order.
        dense = np.loadtxt(CASES[name]/'bulk-0000.csv', delimiter=',', skiprows=1, usecols=(1, 2))
        pq = q1(L, d['phase'][step], dense)[0]
        q = np.fromfile(CASES[name]/f'quadrature-q-{step:04d}.bin')
        Y = M*(1+3*pq)/(1-pq)**3*q*q/(2*G)
        # Display each original-QP value in its reference-coordinate midpoint
        # bin. Bin areas are not quadrature weights. No history is interpolated
        # or reevaluated; quantitative averages still use original JxW.
        centers = dense.reshape(-1, 64, 2).mean(axis=1)
        ix = np.rint(centers[:, 0]/L['dx']-.5).astype(int)
        iy = np.rint(centers[:, 1]/L['dy']-.5).astype(int)
        values = np.empty((8*(len(d['y'])-1), 8*(len(d['x'])-1)))
        values[iy[:, None, None]*8+np.arange(8)[None, :, None],
               ix[:, None, None]*8+np.arange(8)[None, None, :]] = Y.reshape(-1, 8, 8)
        z, _ = gauss(8); edges = np.r_[0., (z[1:]+z[:-1])/2, 1.]
        xe = np.r_[(d['x'][:-1, None]+L['dx']*edges[None, :-1]).ravel(), d['x'][-1]]
        ye = np.r_[(d['y'][:-1, None]+L['dy']*edges[None, :-1]).ravel(), d['y'][-1]]
        full_qp[name] = xe, ye, values
        full_maps[name] = (q1(L, d['phase'][0], full_points)[0],
                           q1(L, d['phase'][step]-d['phase'][0], full_points)[0])
        maps[name] = p0, delta
    norms = [Normalize(0, .6), Normalize(0, max(global_peaks)),
             Normalize(0, max(np.max(q[2]) for q in full_qp.values()))]
    fig, axes = plt.subplots(3, 2, figsize=(11, 15), constrained_layout=True)
    for col, (name, d) in enumerate(data.items()):
        segments = [np.array([[x, .15], [x, .85]]) for x in d['x'] if .45 <= x <= .86]
        segments += [np.array([[.45, y], [.86, y]]) for y in d['y'] if .15 <= y <= .85]
        for row in range(3):
            ax = axes[row, col]
            if row < 2:
                artist = ax.pcolormesh(xx, yy, maps[name][row], shading='auto', cmap='viridis', norm=norms[row], rasterized=True)
            else:
                xe, ye, values = full_qp[name]
                artist = ax.pcolormesh(xe, ye, values, shading='flat', cmap='viridis', norm=norms[row], rasterized=True)
            ax.add_collection(LineCollection(segments, colors='.6', linewidths=.25, alpha=.4))
            ax.contour(xx, yy, maps[name][0], levels=[.001, .1, .3, .5], colors='white', linewidths=.7, linestyles='--')
            if row:
                ax.contour(xx, yy, maps[name][0]+maps[name][1], levels=[.001, .1, .3, .5], colors='cyan', linewidths=.7)
                for onset in report['cases'][name]['onset'].values():
                    first = onset['nodes']
                    ax.scatter([p['x'] for p in first], [p['y'] for p in first],
                               facecolors='none', edgecolors='red', s=35, linewidths=.8)
            ax.plot(.5, .5, 'rx', ms=6)
            ax.set(xlim=(.45, .86), ylim=(.15, .85), aspect='equal', xlabel='x [m]', ylabel='y [m]')
            if not col:
                fig.colorbar(artist, ax=axes[row, :], label=['Initial phase [1]', r'Damage increment $\Delta\phi$ [1]', 'Native-QP mechanical driving Y [Pa]'][row], shrink=.8)
        dx, dy = d['x'][1]-d['x'][0], d['y'][1]-d['y'][0]
        axes[0, col].set_title(LABELS[name]+f'\n'+r'$\Delta x$'+f' = {dx:g} m, '+r'$\Delta y$'+f' = {dy:g} m')
    fig.suptitle(r'Same physical specimen and $\Delta t=0.00125$ s; fresh continuum seeds on each mesh'+'\n'
                 +'Growth/driving at 0.15 s. White dashed: initial phase; cyan: evolved. Red circles: first growth; cross: original endpoint.')
    for suffix in ('png', 'pdf'): fig.savefig(output/f'spatial-fields.{suffix}', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for col, (name, d) in enumerate(data.items()):
        segments = [np.array([[x, 0], [x, 1]]) for x in d['x']]
        segments += [np.array([[0, y], [1.5, y]]) for y in d['y']]
        for row in range(3):
            ax = axes[row, col]
            if row < 2:
                artist = ax.pcolormesh(X, Ygrid, full_maps[name][row], shading='auto', cmap='viridis', norm=norms[row], rasterized=True)
            else:
                xe, ye, values = full_qp[name]
                artist = ax.pcolormesh(xe, ye, values, shading='flat', cmap='viridis', norm=norms[row], rasterized=True)
            ax.add_collection(LineCollection(segments, colors='.6', linewidths=.15, alpha=.25))
            ax.contour(X, Ygrid, full_maps[name][0], levels=[.001, .1, .3, .5], colors='white', linewidths=.6, linestyles='--')
            if row:
                ax.contour(X, Ygrid, full_maps[name][0]+full_maps[name][1], levels=[.001, .1, .3, .5], colors='cyan', linewidths=.6)
            ax.add_patch(Rectangle((.45, .15), .41, .7, fill=False, edgecolor='magenta', lw=.9))
            ax.set(xlim=(0, 1.5), ylim=(0, 1), aspect='equal', xlabel='x [m]', ylabel='y [m]')
            if not col:
                fig.colorbar(artist, ax=axes[row, :], label=['Initial phase [1]', r'Damage increment $\Delta\phi$ [1]', 'Native-QP driving Y [Pa]'][row], shrink=.8)
        axes[0, col].set_title(LABELS[name]+r'; $\Delta x=$'+f'{d["x"][1]-d["x"][0]:g} m, '+r'$\Delta y=$'+f'{d["y"][1]-d["y"][0]:g} m')
    fig.suptitle(r'Whole specimen; identical scales and $\Delta t=0.00125$ s'+'\n'
                 +'Damage/driving at 0.15 s. White dashed: initial phase; cyan: evolved; magenta box: cap close-up.')
    for suffix in ('png', 'pdf'): fig.savefig(output/f'spatial-domain.{suffix}', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    rows = report['comparison']['times']; times = [r['time'] for r in rows]
    for col, (key, ylabel) in enumerate([('peak', r'Peak $\Delta\phi$ [1]'),
                                        ('integrated_growth', r'$\int_{cap\ flank}\Delta\phi\,dA$ [m$^2$]'),
                                        ('added_fracture', 'Added cap-flank fracture energy [J/m]')]):
        ax = axes[0, col]
        for name in data:
            for flank, style in [('lower', '-'), ('upper', '--')]:
                ax.plot(times, [r['metrics'][name][flank][key] for r in rows], style,
                        color=COLORS[name], label=LABELS[name]+' '+flank)
        ax.set(xlabel='Physical time [s]', ylabel=ylabel, xlim=(.125, .16)); ax.grid(alpha=.2)
        for t in (.05, .15): ax.axvline(t, ls=':', color='.5', lw=.7)
        if not col: ax.legend(fontsize=8)
    for j, radius in enumerate(RADII):
        ax = axes[1, j]
        for name, d in data.items():
            for t, style in [(.135, ':'), (.15, '-')]:
                curve = d['arc_'+str(t)].reshape(3, len(ANGLES), 3)[j, :, 2]
                ax.plot(ANGLES, curve, style, color=COLORS[name], label=LABELS[name]+f', {t:g} s')
        ax.set(xlabel='Angle from physical +x [degrees]', ylabel='Kernel-averaged Y [Pa]', title=f'Physical radius {radius:.6g} m')
        ax.grid(alpha=.2)
        if not j: ax.legend(fontsize=8)
    fig.suptitle(r"Both flanks, identical $\Delta t$; growth uses each mesh's own initialization"+'\n'
                 +'Angular curves average original-QP Y using the same physical radius-0.0625 m kernel.')
    for suffix in ('png', 'pdf'): fig.savefig(output/f'spatial-response.{suffix}', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
