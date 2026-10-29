#!/usr/bin/env python3
"""Physical-coordinate, fixed-scale views of the one accepted trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import Normalize
from assess_spatial import q1
from assess_continued import RUN, LEVELS
from assess_native import G, ETA, M, gauss, csv


def main(output, run, movie):
    report = json.loads((output/'assessment.json').read_text())
    history = json.loads((output/'history.json').read_text())
    connection = json.loads((output/'contours.json').read_text())
    data = dict(np.load(output/'fields.npz')); times = data['time']; phases = data['phase']
    energy = csv(output/'energy.csv')
    L = dict(x=data['x'], y=data['y'], dx=data['x'][1]-data['x'][0],
             dy=data['y'][1]-data['y'][0], grid_ids=data['grid_ids'])
    # Active-cell ordering is recovered from the actual initial native-QP file.
    xy = np.loadtxt(run/'bulk-0000.csv', delimiter=',', skiprows=1, usecols=(1, 2)).reshape(-1, 64, 2)
    centers = xy.mean(axis=1); ix = np.rint(centers[:, 0]/L['dx']-.5).astype(int)
    iy = np.rint(centers[:, 1]/L['dy']-.5).astype(int)
    rows = iy[:, None, None]*8+np.arange(8)[None, :, None]
    cols = ix[:, None, None]*8+np.arange(8)[None, None, :]
    z, _ = gauss(8); edges = np.r_[0., (z[1:]+z[:-1])/2, 1.]
    xe = np.r_[(L['x'][:-1, None]+L['dx']*edges[None, :-1]).ravel(), L['x'][-1]]
    ye = np.r_[(L['y'][:-1, None]+L['dy']*edges[None, :-1]).ravel(), L['y'][-1]]
    X, Y = np.meshgrid(np.linspace(0, 1.5, 385), np.linspace(0, 1, 257))
    points = np.stack([X, Y], axis=-1); initial = q1(L, phases[0], points)[0]
    pq = q1(L, phases[0], xy)[0]
    # Pure geometric Q1 basis weights can be reused without caching stress.
    # Read actual q/rate at their native points; no local constitutive-state interpolation.
    def rate(step):
        f = run/f'quadrature-rate-{step:04d}.bin'
        if f.exists(): return np.fromfile(f).reshape(-1, 64)
        if not step: return np.zeros(xy.shape[:2])
        a = []
        for n in (step-1, step):
            p = q1(L, phases[n], xy)[0]
            q = np.fromfile(run/f'quadrature-q-{n:04d}.bin').reshape(-1, 64)
            a.append(M*p*(1+p)/(1-p)**2*q/G)
        dt = times[step]-times[step-1]
        return ((1+dt*G/ETA)*a[1]-a[0])/dt
    def rate_grid(step):
        values = np.empty((len(ye)-1, len(xe)-1))
        values[rows, cols] = abs(rate(step)).reshape(-1, 8, 8)
        return values
    vmax = [max(.6, float(phases.max())), max(r['peak_damage'] for r in history),
            max(r['max_rate'] for r in history)]
    norms = [Normalize(0, v if v > 0 else 1.) for v in vmax]
    labels = ['Physical phase phi [1]', 'Damage phi(t) - phi(0) [1]', 'Local slip strain rate |w| [1/s]']
    strain = np.cumsum([r['strain_increment'] for r in history])
    Q = np.array([r['Q'] for r in history]); t = times
    E = np.array([r['bulk']+r['cohesive']+r['fracture']-history[0]['fracture'] for r in history])
    D = np.r_[0., np.cumsum(energy['physical_dissipation'])]
    W = np.r_[0., np.cumsum(energy['trapezoidal_work'])]
    be = np.r_[0., np.cumsum(energy['endpoint_remainder'])]
    tr = np.r_[0., np.cumsum(energy['trapezoidal_remainder'])]
    tv = np.r_[0., np.cumsum(abs(energy['trapezoidal_remainder']))]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), gridspec_kw={'height_ratios': [1., 1.1, 1.]})
    fig.subplots_adjust(left=.055, right=.94, bottom=.105, top=.89, hspace=.40, wspace=.38)
    maps = []; dynamic_contours = []
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            artist = ax.pcolormesh(xe, ye, rate_grid(0), shading='flat', cmap='magma', norm=norms[col], rasterized=True) if col == 2 else ax.pcolormesh(X, Y, initial if col == 0 else np.zeros_like(initial), shading='auto', cmap='viridis', norm=norms[col], rasterized=True)
            maps.append(artist)
            ax.contour(X, Y, initial, levels=LEVELS, colors='white', linewidths=.6, linestyles='--')
            dynamic_contours.append(ax.contour(X, Y, initial, levels=LEVELS, colors='cyan', linewidths=.7))
            ax.plot([0, .5], [.5, .5], color='red', lw=1.)
            ax.plot(.5, .5, 'rx', ms=5)
            ax.set(xlim=(0, 1.5) if row == 0 else (.35, 1.1),
                   ylim=(0, 1) if row == 0 else (.12, .88), aspect='equal', xlabel='x [m]', ylabel='y [m]')
            if not row: ax.set_title(labels[col], fontsize=10)
            if row == 1:
                fig.colorbar(artist, ax=axes[:2, col], location='bottom', pad=.075, shrink=.8, fraction=.035)
    for ax in axes[2]: ax.grid(alpha=.25)
    axes[2, 0].plot(strain, Q, color='.8', lw=2)
    reaction, = axes[2, 0].plot([], [], color='tab:blue', lw=1.6)
    marker, = axes[2, 0].plot([], [], 'ro', ms=5)
    axes[2, 0].set(xlabel='Imposed engineering shear strain [1]', ylabel='Generalized reaction Q [J/m]', title='Measured reaction-displacement path')
    energy_lines = []
    for values, text, color in [(E, 'Storage change', 'tab:blue'), (D, 'Physical dissipation', 'tab:orange'), (W, 'Trapezoidal work', 'black')]:
        axes[2, 1].plot(t, values, color=color, alpha=.17)
        line, = axes[2, 1].plot([], [], color=color, label=text); energy_lines.append((line, values))
    axes[2, 1].set(xlabel='Time [s]', ylabel='Energy [J/m]', title='Physical storage, work and dissipation')
    axes[2, 1].legend(fontsize=7)
    remainder_lines = []
    for values, text, color in [(be, 'Endpoint BE remainder', 'tab:purple'), (tr, 'Trapezoidal signed', 'tab:green'), (tv, 'Trapezoidal sum |step|', 'tab:red')]:
        axes[2, 2].plot(t, values, color=color, alpha=.17)
        line, = axes[2, 2].plot([], [], color=color, label=text); remainder_lines.append((line, values))
    axes[2, 2].set(xlabel='Time [s]', ylabel='Remainder [J/m]', title='BE ledger is separate from physical accuracy')
    axes[2, 2].set_yscale('symlog', linthresh=1e-6)
    axes[2, 2].legend(fontsize=7)
    for ax in axes[2, 1:]:
        ax.axvline(.15, color='.5', ls=':', lw=.8); ax.axvline(.175, color='.5', ls='--', lw=.8)
    title = fig.suptitle('', fontsize=12)
    topology_text = axes[0, 0].text(.98, .97, '', transform=axes[0, 0].transAxes,
        ha='right', va='top', fontsize=7, color='white', bbox=dict(facecolor='black', alpha=.5, edgecolor='none'))
    intact_text = axes[0, 1].text(.98, .97, '', transform=axes[0, 1].transAxes,
        ha='right', va='top', fontsize=7, color='white', bbox=dict(facecolor='black', alpha=.5, edgecolor='none'))
    fig.text(.5, .015, 'Whole domain above; tip close-up below. White dashed: initial contours; cyan: current (.001,.01,.05,.1,.3,.5). Red: original seed axis/end.\n'
             'Fixed colour scales. Rate bins are display areas, not integration weights. Remainder axis: symmetric log, linear within +/-1e-6 J/m.\n'
             'Growth speed and spatial accuracy unqualified.', ha='center', fontsize=8)
    def draw(step):
        nonlocal dynamic_contours
        p = q1(L, phases[step], points)[0]; rg = rate_grid(step)
        for c in dynamic_contours: c.remove()
        dynamic_contours = []
        for row in range(2):
            for col in range(3):
                values = [p, p-initial, rg][col]
                maps[row*3+col].set_array(values.ravel())
                dynamic_contours.append(axes[row, col].contour(X, Y, p, levels=LEVELS, colors='cyan', linewidths=.7))
        reaction.set_data(strain[:step+1], Q[:step+1]); marker.set_data([strain[step]], [Q[step]])
        current_geometry = connection.get(str(step), connection['0'])
        topology_text.set_text(f"Disconnected contours at phi=.001: {current_geometry['0.001']['disconnected_components']}\n"
            f"Seed component max x: {current_geometry['0.001']['rightmost']:.3f} m")
        intact_text.set_text(f"Initially zero-phase nodes activated: {history[step]['newly_active_zero_phase_nodes']}\n"
            f"Disconnected activation patches (node graph): {history[step]['disconnected_native_activation_components']}")
        for line, values in energy_lines+remainder_lines: line.set_data(t[:step+1], values[:step+1])
        stage = 'unloaded' if t[step] <= .05+1e-12 else ('continued loading' if t[step] <= .175+1e-12 else 'fixed-displacement hold')
        title.set_text(f'Exploratory 96 x 128 specimen: t = {t[step]:.5f} s, {stage}\n'
                       r'$\Delta t=0.00125$ s; same seed/law; loading stop extended from 0.150 to 0.175 s')
        return []
    draw(len(t)-1)
    fig.savefig(output/'continued-final.png', dpi=160)
    fig.savefig(output/'continued-final.pdf', dpi=160)
    if movie:
        # Every accepted time in order: no varying sampling interval that could
        # manufacture an apparent acceleration during the growth interval.
        frames = list(range(len(t)))
        animation = FuncAnimation(fig, draw, frames=frames, interval=140, blit=False, repeat=False)
        animation.save(output/'continued-loading.mp4', writer=FFMpegWriter(fps=7, codec='libx264', bitrate=2400,
                       extra_args=['-pix_fmt', 'yuv420p']), dpi=100)
        # Compact inline preview; the MP4 retains the readable complete figure.
        preview = sorted(set(list(range(0, len(t), 4))+[len(t)-1]))
        animation2 = FuncAnimation(fig, draw, frames=preview, interval=220, blit=False, repeat=False)
        animation2.save(output/'continued-loading-preview.gif', writer=PillowWriter(fps=5), dpi=65)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].plot(t, Q); axes[0, 0].set(ylabel='Generalized reaction Q [J/m]', xlabel='Time [s]')
    axes[0, 1].plot(strain, Q); axes[0, 1].set(ylabel='Generalized reaction Q [J/m]', xlabel='Imposed shear strain [1]')
    for key, text in [('bulk', 'Bulk'), ('cohesive', 'Cohesive'), ('fracture', 'Added fracture')]:
        values = np.array([r[key] for r in history]); values -= values[0]
        axes[0, 2].plot(t, values, label=text)
    axes[0, 2].set(ylabel='Storage change [J/m]', xlabel='Time [s]'); axes[0, 2].legend()
    for key in ['Db', 'Dc', 'Df']: axes[1, 0].plot(t[1:], np.cumsum(energy[key]), label=key)
    axes[1, 0].set(ylabel='Physical dissipation [J/m]', xlabel='Time [s]'); axes[1, 0].legend()
    for values, text in [(W, 'Trapezoidal work'), (E+D, 'Storage + physical dissipation')]: axes[1, 1].plot(t, values, label=text)
    axes[1, 1].set(ylabel='Energy [J/m]', xlabel='Time [s]'); axes[1, 1].legend()
    for values, text in [(be, 'Endpoint BE'), (tr, 'Trapezoidal signed'), (tv, 'Trapezoidal absolute')]: axes[1, 2].plot(t, values, label=text)
    axes[1, 2].set(ylabel='Cumulative remainder [J/m]', xlabel='Time [s]'); axes[1, 2].legend()
    axes[1, 2].set_yscale('symlog', linthresh=1e-6)
    axes[1, 2].set_title('Symmetric log; linear within +/-1e-6 J/m', fontsize=9)
    for ax in axes.ravel():
        ax.grid(alpha=.25)
        if ax is not axes[0, 1]:
            for time in [.05, .15, .175]: ax.axvline(time, color='.6', ls=':', lw=.8)
    fig.suptitle('Continued shear to .175 s, hold to .185 s: endpoint work and physical path work remain distinct')
    for suffix in ['png', 'pdf']: fig.savefig(output/f'continued-energy.{suffix}', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    indices = np.array(sorted(map(int, connection))); ct = t[indices]
    for level in LEVELS:
        key = str(level); initial_c = connection['0'][key]
        values = [connection[str(i)][key] for i in indices]
        axes[0, 0].plot(ct, [1000*(v['centerline']-initial_c['centerline']) for v in values], label=f'{level:g}')
        axes[0, 1].plot(ct, [1000*(v['rightmost']-initial_c['rightmost']) for v in values], label=f'{level:g}')
        line, = axes[0, 2].plot(ct, [1000*(v['upper']['ray45_radius']-initial_c['upper']['ray45_radius']) for v in values], label=f'{level:g}')
        axes[0, 2].plot(ct, [1000*(v['lower']['ray45_radius']-initial_c['lower']['ray45_radius']) for v in values], color=line.get_color(), ls='--')
        axes[1, 1].step(ct, [v['disconnected_components'] for v in values], where='post', label=f'{level:g}')
    axes[0, 0].set(ylabel='Centerline contour advance [mm]', title='Seed-connected phase contours')
    axes[0, 1].set(ylabel='Rightmost contour advance [mm]', title='Seed-connected extent (includes joining)')
    axes[0, 2].set(ylabel='45-degree flank contour advance [mm]', title='45-degree cuts: upper solid, lower dashed')
    axes[0, 0].legend(title='phi level', ncol=2, fontsize=8)
    for key, text in [('cap_growth', 'Cap'), ('continuum_intact_growth', 'Initially intact (continuum)'), ('represented_zero_growth', 'Initially zero Q1')]:
        axes[1, 0].plot(t, [r[key] for r in history], label=text)
    axes[1, 0].set(ylabel='Integrated damage increment [m²]'); axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(ylabel='Disconnected superlevel components', title='Disconnected phase-level components')
    axes[1, 2].plot(t, [r['peak_step_damage'] for r in history]); axes[1, 2].axhline(.001, color='red', ls='--', label='Saved-interval diagnostic flag')
    axes[1, 2].set(ylabel='Maximum per-step damage increment [1]'); axes[1, 2].legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xlabel('Time [s]'); ax.grid(alpha=.25)
        for time in [.15, .175]: ax.axvline(time, color='.6', ls=':', lw=.8)
    fig.suptitle('Contour displacement is measured from the initial diffuse cap; it is not inserted fault-surface extension')
    for suffix in ['png', 'pdf']: fig.savefig(output/f'continued-growth.{suffix}', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued'))
    parser.add_argument('--no-movie', action='store_true')
    args = parser.parse_args(); main(args.output, args.run, not args.no_movie)
