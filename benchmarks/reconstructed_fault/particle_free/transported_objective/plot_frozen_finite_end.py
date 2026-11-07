#!/usr/bin/env python3
"""Plot the frozen comparison from compact saved inputs and coefficients only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from compare_frozen_finite_end import OUT, GRIDS, METRIC, index_basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/frozen-finite-end-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = json.loads((args.output/'comparison.json').read_text())
    rows = {(c['n'], c['q']): c for c in r['cases']}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    h = 4/np.array(GRIDS)
    for law, name, color in [(0, 'Constant viscosity', 'tab:blue'), (1, 'Layered viscosity', 'tab:orange')]:
        for q, style, method in [('cut', '-', 'best DGQ2'), (8, '--', 'Gauss 8')]:
            ax[0, 0].loglog(h, [100*rows[n, q]['localized_source_relative_L2'][law] for n in GRIDS],
                            marker='o' if q == 'cut' else 'x', linestyle=style, color=color,
                            label=f'{name}, {method}')
    ax[0, 0].set(xlabel='Tensor cell width h (m)', ylabel='L2 error / localized source (%)',
                 title='Finite-end approximation limits accuracy')
    ax[0, 0].legend(fontsize=7)
    for n in [32, 64, 128]:
        ax[0, 1].loglog(r['orders'], [rows[n, q]['quadrature_L2'][0] for q in r['orders']], '.-', label=f'{n}²')
    ax[0, 1].axhline(rows[64, 'cut']['best_L2'][0], linestyle=':', color='black', label='64² best-space error')
    ax[0, 1].set(xlabel='Uncut cell Gauss order', ylabel='Tensor L2 error (Pa m)',
                 title='Quadrature refinement cannot remove that limit')
    ax[0, 1].legend(fontsize=8)
    for q, style, name in [('cut', '-', 'cut reference'), (8, '--', 'Gauss 8')]:
        ax[1, 0].loglog(h, [100*rows[n, q]['point_traction']['all']['relative_L2'][0] for n in GRIDS],
                        'o'+style, color='tab:red', label=f'Point, {name}')
        ax[1, 0].loglog(h, [100*rows[n, q]['work_traction_relative_L2'][0] for n in GRIDS],
                        's'+style, color='tab:green', label=f'Work, {name}')
    ax[1, 0].set(xlabel='Tensor cell width h (m)', ylabel='Relative shear-traction error (%)',
                 title='Point and work traction differ (constant viscosity)')
    ax[1, 0].legend(fontsize=7)

    # Evaluate the line source from the saved Q1 fixture, without native exports.
    f = np.load(args.output/'frozen-input.npz')
    surface = f['surface']
    X = surface[:, 1:3]
    tangent = X[-1]-X[0]
    length = np.linalg.norm(tangent)
    tangent /= length
    normal = np.array([-tangent[1], tangent[0]])
    S = np.array([tangent[0]*normal[0], tangent[1]*normal[1],
                  .5*(tangent[0]*normal[1]+tangent[1]*normal[0])])
    along = length+np.linspace(-.16, .16, 2000)
    x = X[0]+along[:, None]*tangent
    ids, B = index_basis(x, int(f['profile_n']))
    phi = np.einsum('pk,pk->p', B, f['phase_coeff'][ids])
    labels = (X-X[0])@tangent
    j = np.clip(np.searchsorted(labels, along)-1, 0, len(labels)-2)
    u = (along-labels[j])/(labels[j+1]-labels[j])
    I = (1-u)*surface[j, 5]+u*surface[j+1, 5]
    V = (1-u)*f['prescribed_V'][j]+u*f['prescribed_V'][j+1]
    chi = float(f['m'])*phi*(1+phi)/(1-phi)**2/I*(along <= length)
    k = -10*np.expm1(-float(f['G'])*float(f['dt'])/10)
    src = 2*k*(np.array([.03, -.03, .01])-(chi*V)[:, None]*S)
    ax[1, 1].plot(along-length, src@(METRIC@S), color='black', label='Local source, fixed Q1 profile')
    coeff = np.load(args.output/'projection-coefficients.npz')
    ids, B = index_basis(x, 64)
    for tag, style, name in [('cut', '-', '64² best DGQ2'), ('q8', '--', '64² Gauss 8')]:
        tau = np.einsum('pk,pkc->pc', B, coeff[f'n64_{tag}'][ids, :, 0])
        ax[1, 1].plot(along-length, tau@(METRIC@S), style, label=name)
    ax[1, 1].axvline(0, color='gray', linewidth=.8)
    ax[1, 1].set(xlabel='Distance along curve from right end (m)', ylabel='Frozen shear stress (Pa)',
                 title='The physical source terminates inside a tensor cell')
    ax[1, 1].legend(fontsize=7)
    for a in ax.flat:
        a.grid(alpha=.25)
    fig.savefig(args.output/'frozen-finite-end.png', dpi=180)
    fig.savefig(args.output/'frozen-finite-end.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
