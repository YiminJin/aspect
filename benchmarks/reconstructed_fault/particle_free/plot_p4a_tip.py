#!/usr/bin/env python3
"""Actual native quadrature and ray geometry, with physical phase and stress."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from assess_p3_extension import csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    q = csv(a.path/'bulk-quadrature.csv')
    r = csv(a.path/'normalization-rays.csv')
    s = csv(a.path/'old-surface.csv')
    phases = csv(a.path/'old-phase.csv')
    tip = s['x'][-1]
    px, py = np.unique(phases['x']), np.unique(phases['y'])
    phases.sort(order=['y', 'x'])
    P = phases['phi'].reshape(len(py), len(px))
    gx = np.linspace(tip-.13, tip+.33, 701)
    gy = np.linspace(.32, .68, 501)
    X, Y = np.meshgrid(gx, gy)
    ix, iy = np.minimum((X*32).astype(int), 31), np.minimum((Y*64).astype(int), 63)
    u, v = X*32-ix, Y*64-iy
    field = (1-u)*(1-v)*P[iy,ix]+u*(1-v)*P[iy,ix+1]+(1-u)*v*P[iy+1,ix]+u*v*P[iy+1,ix+1]
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 200})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout='constrained')
    im = None
    regions = [(tip-.125, tip+.33, .32, .68), (.49, .54, .47, .53)]
    all_rows = []
    for k, (ax, (left, right, bottom, top)) in enumerate(zip(axes, regions)):
        ax.axvspan(tip, right, color='#f3b8b8', alpha=.12)
        im = ax.pcolormesh(X, Y, field, vmin=0., vmax=.6, cmap='cividis', shading='auto', rasterized=True)
        cs = ax.contour(X, Y, field, levels=[.01,.1,.3,.5,.59], colors='white', linewidths=.9)
        if k == 0:
            ax.clabel(cs, fontsize=8, fmt='%g')
        for x in px:
            if left <= x <= right:
                ax.axvline(x, color='#777777', lw=.55, alpha=.8)
        for y in py:
            if bottom <= y <= top:
                ax.axhline(y, color='#777777', lw=.55, alpha=.8)
        ax.axvline(tip, color='#b11736', ls='--', lw=1.4)
        ax.plot(s['x'], s['y'], color='#b11736', lw=2.5, zorder=6)
        ax.plot(tip, .5, 's', ms=7, color='#f23d5e', mec='black', zorder=8)
        # These are actual normal integration intervals, not Gauss particles.
        selected = np.unique(r['profile'])[::8] if k == 0 else np.unique(r['profile'])[::3]
        lines = [((row['x'], row['lower']), (row['x'], row['upper']))
                 for row in r if row['profile'] in selected and left < row['x'] < right]
        ax.add_collection(LineCollection(lines, colors='#f29d26', linewidths=.65, alpha=.9))
        inside = (q['x'] > left) & (q['x'] < right) & (q['y'] > bottom) & (q['y'] < top)
        rows = q[inside]
        all_rows.append(dict(panel=k, points=len(rows), active_points=int(np.sum(rows['active'])),
                             JxW_sum=float(np.sum(rows['JxW'])),
                             active_JxW_sum=float(np.sum(rows['JxW'][rows['active'] != 0])),
                             minimum_JxW=float(np.min(rows['JxW'])), maximum_JxW=float(np.max(rows['JxW']))))
        # Keep context readable; the close-up shows EVERY native QP in view.
        if k == 1:
            for active, color in [(1, '#18cddd'), (0, '#e83c78')]:
                mask = rows['active'] == active
                ax.scatter(rows['x'][mask], rows['y'][mask], s=3, color=color, alpha=.85, linewidths=0, rasterized=True, zorder=5)
        ax.set(xlim=(left,right), ylim=(bottom,top), xlabel='x [m]', ylabel='y [m]')
        ax.set_aspect('equal')
    axes[0].set_title('Represented phase cap and actual normalization rays')
    axes[0].annotate('Rounded cap ahead of physical endpoint:\nfracture energy + obstacle; no local\nmechanical phase driving at fixed geometry',
                     xy=(tip+.12,.55), xytext=(tip-.11,.665), fontsize=9, va='top',
                     arrowprops={'arrowstyle':'->', 'color':'white'}, color='white')
    weights = r['surface_weight']
    axes[0].text(.02, .02, 'Normal intervals are integrated analytically; rays are not particles.\n'
                 +f'Gauss8 surface weights: {weights.min():.3g}–{weights.max():.3g} m.\n'
                 +'No normalization rays / coupling surface ahead of the endpoint.',
                 transform=axes[0].transAxes, fontsize=8, va='bottom',
                 bbox={'facecolor':'white','alpha':.92,'edgecolor':'none'})
    axes[1].set_title('Every bulk mechanics QP in this window')
    stat = all_rows[1]
    axes[1].text(.02, .02, f"{stat['points']} points; {stat['active_points']} associated\n"
                 +f"Σ plotted JxW = {stat['JxW_sum']:.6g} m²\n"
                 +f"associated ΣJxW = {stat['active_JxW_sum']:.6g} m²\n"
                 +f"JxW range {stat['minimum_JxW']:.2e}–{stat['maximum_JxW']:.2e} m²",
                 transform=axes[1].transAxes, fontsize=8, va='bottom', color='black',
                 bbox={'facecolor':'white','alpha':.92,'edgecolor':'none'})
    fig.colorbar(im, ax=axes, label=r'Physical Q1 phase $\phi$ (dimensionless)', fraction=.023)
    handles = [Line2D([],[],color='#b11736',lw=2,label='Finite Γ / coupling support ends here'),
               Line2D([],[],marker='s',color='#f23d5e',ls='',mec='black',label='Actual reconstructed endpoint'),
               Line2D([],[],color='#f29d26',lw=1,label='Selected actual normalization rays'),
               Line2D([],[],marker='.',color='#18cddd',ls='',label='Associated bulk QP (coupling / local driving)'),
               Line2D([],[],marker='.',color='#e83c78',ls='',label='Unassociated QP (bulk stress; zero local driving)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, fontsize=9)
    fig.suptitle('Finite endpoint INSIDE a bulk cell — native frozen-state diagnostic\n'
                 'Markers have equal size: point counts are not integration weights', fontsize=13)
    for ext in ['png','pdf']:
        fig.savefig(a.output/f'tip_quadrature.{ext}')
    plt.close(fig)
    (a.output/'figure-measures.json').write_text(json.dumps(all_rows, indent=2)+'\n')
    # A separate field view uses actual exported QPs, without smoothing the
    # support discontinuity or confusing auxiliary T with physical stress.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    inside = (q['x'] >= tip-.1) & (q['x'] <= tip+.125) & (np.abs(q['y']-.5) <= .125)
    rows = q[inside]
    for ax, values, label in [(axes[0], rows['tauxy']/1000, r'Physical $\tau_{xy}$ [kPa]'),
                              (axes[1], rows['local_mechanical_derivative'], r'Local $\partial e_{bulk}/\partial\phi$ at fixed $I$ [Pa]')]:
        bound = max(np.max(np.abs(values)), 1e-20)
        im = ax.scatter(rows['x'], rows['y'], c=values, s=3, cmap='RdBu_r',
                        vmin=-bound, vmax=bound, linewidths=0, rasterized=True)
        ax.contour(X, Y, field, levels=[.1,.3,.5,.59], colors='black', linewidths=.4, alpha=.7)
        ax.plot(s['x'], s['y'], color='#333333', lw=2)
        ax.plot(tip,.5,'s',color='#f23d5e',mec='black',ms=6)
        ax.axvline(tip,color='#b11736',lw=1,ls='--')
        ax.set(xlim=(tip-.1,tip+.125),ylim=(.375,.625),xlabel='x [m]',ylabel='y [m]')
        ax.set_aspect('equal')
        fig.colorbar(im,ax=ax,label=label,fraction=.045)
    axes[0].set_title(r'Reconstructed stress, not auxiliary $T$')
    axes[1].set_title('Local mechanical derivative = 0 ahead')
    fig.suptitle('Frozen loaded snapshot — no mechanical equilibrium or phase-evolution claim\n'
                 'Complete nodal derivative also includes the independently verified normalization adjoint',fontsize=11)
    for ext in ['png','pdf']:
        fig.savefig(a.output/f'tip_fields.{ext}')
    plt.close(fig)


if __name__ == '__main__':
    main()
