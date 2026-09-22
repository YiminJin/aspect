"""Plot copied BP5 profiles directly, without requiring an output index or VTUs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

YEAR = 31557600.
FIELDS = ('V_m_per_s', 'Theta_s', 'slip_m', 'q_weak_Pa', 'sigma_n_weak_Pa')
NOTE = ('Θ is committed post-update state; mechanics used incoming Θ. Tractions are total consistent-Q1 weak projections,\n'
        'not raw stress/history fields; positive normal traction means compression. Saved data only, without smoothing.')


def edges(x):
    return np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profiles', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.profiles.parent / 'fault_evolution'
    paths = sorted(args.profiles.glob('fault_*.csv'), key=lambda p: int(p.stem.split('_')[-1]))
    if len(paths) < 2:
        raise ValueError('At least two complete saved profiles are required')
    geometry = None
    history = {key: [] for key in FIELDS}
    times, steps = [], []
    for path in paths:
        a = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
        a = a[np.argsort(a['xd_m'])]
        if any(not np.isfinite(a[key]).all() for key in a.dtype.names):
            raise ValueError(f'Nonfinite profile: {path}')
        if (np.any(a['fault'] != 0) or len(np.unique(a['node'])) != len(a)
                or np.any(np.diff(a['xd_m']) <= 0)
                or np.any(a['time_s'] != a['time_s'][0])
                or np.any(a['step'] != int(path.stem.split('_')[-1]))):
            raise ValueError(f'Invalid profile geometry/time/identity: {path}')
        g = np.column_stack([a[key] for key in ('node', 'xd_m', 'x_m', 'y_m')])
        if geometry is None:
            geometry = g
        elif not np.array_equal(g, geometry):
            raise ValueError(f'Incomplete profile or changed geometry: {path}')
        times.append(a['time_s'][0] / YEAR)
        steps.append(int(a['step'][0]))
        for key in FIELDS:
            history[key].append(a[key])
    times = np.array(times)
    if np.any(np.diff(times) <= 0):
        raise ValueError('Nonincreasing saved physical times')
    history = {key: np.array(value) for key, value in history.items()}
    xd = geometry[:, 1] / 1000
    selected = sorted({0, len(times)-1} | {int(np.argmin(abs(times-t)))
                      for t in (1, 5, 20, 50, 100, 150) if times[0] <= t <= times[-1]})
    q, sn = history['q_weak_Pa']/1e6, history['sigma_n_weak_Pa']/1e6
    panels = [('Slip rate (m/s)', history['V_m_per_s'], True),
              ('Committed state Θ (s)', history['Theta_s'], True),
              ('Cumulative slip (m)', history['slip_m'], False),
              ('Total weak shear traction (MPa)', q, False),
              ('Total weak normal traction (MPa)', sn, False),
              ('Weak normal change from initial (MPa)', sn-sn[0], False)]
    cmap, norm = plt.get_cmap('viridis'), Normalize(times[0], times[-1])
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.text(.02, .015, NOTE, fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, .91))
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{ext}', dpi=180)
        plt.close(fig)

    for limit, name in ((45., 'profiles_shallow'), (xd[-1], 'profiles_full_fault')):
        mask = xd <= limit
        fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
        for ax, (label, values, logarithmic) in zip(axes.flat, panels):
            for i in selected:
                if logarithmic and np.any(values[i] <= 0):
                    raise ValueError(f'Nonpositive {label}')
                ax.plot(xd[mask], values[i, mask], color=cmap(norm(times[i])), lw=.9,
                        label=f'{times[i]:.2f} yr')
            if logarithmic:
                ax.set_yscale('log')
            ax.set(ylabel=label, xlim=(0, limit))
            for boundary in (30, 33):
                ax.axvline(boundary, color='.5', ls=':', lw=.7)
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8, ncol=2)
        for ax in axes[-1]:
            ax.set_xlabel('Down-dip distance (km)')
        fig.suptitle('BP5 — saved fault-property evolution\nDotted lines: 30–33 km transition')
        save(fig, name)

    maps = panels[:3] + [('Weak shear change from initial (MPa)', q-q[0], False),
                       panels[-1], ('Element slip gradient (m/km)',
                                    np.diff(history['slip_m'], axis=1)/np.diff(xd), False)]
    mask = xd <= 45
    n = np.count_nonzero(mask)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for j, (ax, (label, values, logarithmic)) in enumerate(zip(axes.flat, maps)):
        z = values[:, :n-1].T if j == 5 else values[:, mask].T
        y = xd[:n] if j == 5 else edges(xd[mask])
        scale, color = None, 'viridis'
        if logarithmic:
            if np.any(z <= 0):
                raise ValueError(f'Nonpositive {label}')
            scale = LogNorm(z.min(), z.max())
        elif z.min() < 0 < z.max():
            scale, color = Normalize(-abs(z).max(), abs(z).max()), 'RdBu_r'
        m = ax.pcolormesh(edges(times), y, z, cmap=color, norm=scale, rasterized=True)
        ax.set(title=label, ylabel='Down-dip distance (km)', ylim=(45, 0))
        for boundary in (30, 33):
            ax.axhline(boundary, color='.5', ls=':', lw=.7)
        fig.colorbar(m, ax=ax, shrink=.85)
    for ax in axes[-1]:
        ax.set_xlabel('Physical time (yr)')
    fig.suptitle('BP5 — space–time evolution\nMidpoint bins show saved samples, not resolved intermediate states')
    save(fig, 'space_time_shallow')
    summary = dict(saved_profiles=len(times), vertices=len(xd), first_step=steps[0], last_step=steps[-1],
                   first_year=times[0], last_year=times[-1],
                   selected=[dict(step=steps[i], year=times[i]) for i in selected],
                   definitions=NOTE,
                   coverage='Available CSVs only; no output index or accepted-state log supplied for cross-checking.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Wrote plots to {output}')


if __name__ == '__main__':
    main()
