#!/usr/bin/env python3
"""Compare accepted parity cases, retaining initial representation differences."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / 'true-pressure'
sys.path.insert(0, str(REFERENCE))
from verify_pilot import read, stats

FIELDS = ('sigma', 'p', 'minus_tauN', 'V', 'slip')


def response(base, suffix, k):
    a, b = (read(base / f'{name}{suffix}-surface-{k}.csv') for name in ('pilot', 'homogeneous'))
    assert np.array_equal(a['s'], b['s'])
    values = {name: a[name]-b[name] for name in ('sigma', 'p', 'V', 'slip')}
    values['minus_tauN'] = -a['tauN']+b['tauN']
    assert max(abs(values['sigma']-values['p']-values['minus_tauN'])) < 1e-9
    return a['s'], values


def restricted(x, v, a, b):
    points = np.r_[a, x[(x > a) & (x < b)], b]
    return stats(points, np.interp(points, x, v))


def main():
    records = [json.loads((base / f'{name}{suffix}-verification.json').read_text())
               for base, suffix in ((REFERENCE, '64'), (HERE, ''))
               for name in ('pilot', 'homogeneous')]
    assert len({r['resources']['executable_sha256'] for r in records}) == 1
    assert len({r['resources']['plugin_sha256'] for r in records}) == 1
    assert all([s['time_s'] for s in r['steps']] == [0., .5, 1.] for r in records)
    report = dict(scope='parity, normal spacing, hierarchy and particle/initial representation sensitivity; not pure alignment',
                  reference='provisional 64x256', initialization={}, steps=[])
    for base, suffix, label in ((REFERENCE, '64', 'baseline'), (HERE, '', 'staggered')):
        surface = read(base / f'pilot{suffix}/surface_0.csv')
        phase = read(base / f'pilot{suffix}/phase_0.csv')
        assert np.array_equal(phase, read(base / f'homogeneous{suffix}/phase_0.csv'))
        centers = []
        for xx in np.unique(phase['x']):
            col = np.sort(np.unique(phase[phase['x'] == xx][['y', 'phi']]), order='y')
            centers.append(float(np.interp(0, col['y'], col['phi'])))
        seg = read(base / f'pilot{suffix}/segments_0.csv')
        report['initialization'][label] = dict(
            fields={n: stats(surface['x'], surface[n]) for n in ('Theta', 'C', 'Ih', 'V')},
            interpolated_phi_at_y0_range=[min(centers), max(centers)],
            support_half_width_range=[float(min(seg['half_width_plus'])), float(max(seg['half_width_plus']))])
    fig, axes = plt.subplots(5, 3, figsize=(13, 15))
    for k, t in enumerate((0., .5, 1.)):
        bx, b = response(REFERENCE, '64', k)
        sx, s = response(HERE, '', k)
        x = np.unique(np.r_[bx, sx])
        b = {n: np.interp(x, bx, b[n]) for n in FIELDS}
        s = {n: np.interp(x, sx, s[n]) for n in FIELDS}
        row = dict(time_s=t, fields={})
        columns, headers = [x], ['s']
        for i, name in enumerate(FIELDS):
            e = s[name]-b[name]
            row['fields'][name] = dict(baseline=stats(x, b[name]), staggered=stats(x, s[name]),
                                       change=stats(x, e), cuts=[])
            for cut in (.0078125, .015625, .03125, .046875):
                inside = restricted(x, e, cut, .25-cut)
                signal = restricted(x, b[name], cut, .25-cut)
                left, right = restricted(x, e, 0, cut), restricted(x, e, .25-cut, .25)
                total = stats(x, e)['rms']**2*.25
                row['fields'][name]['cuts'].append(dict(width_each_end_m=cut,
                    interior_change=inside, baseline_interior_signal=signal,
                    left_change=left, right_change=right,
                    endpoint_squared_error_fraction=(left['rms']**2+right['rms']**2)*cut/total if total else None,
                    interior_change_over_baseline_signal=inside['rms']/signal['rms'] if signal['rms'] else None))
            for label, value in (('baseline', b[name]), ('staggered', s[name]), ('change', e)):
                columns.append(value); headers.append(label+'_'+name)
            for label, value, style in (('64x256', b[name], '--'), ('64x255', s[name], '-')):
                axes[i, k].plot(x, value, style, label=label)
            axes[i, k].set(title=f'{name}, t={t:g} s', xlabel='s (m)')
            axes[i, k].grid(alpha=.25); axes[i, k].legend()
        np.savetxt(HERE / f'comparison-{k}.csv', np.column_stack(columns), delimiter=',',
                   comments='', header=','.join(headers))
        report['steps'].append(row)
    fig.tight_layout(); fig.savefig(HERE / 'comparison.png', dpi=150)
    # The shared fault spacing permits a direct endpoint-coordinate view;
    # retain signed amplitudes and show both ends, not a fitted peak collapse.
    bx, b = response(REFERENCE, '64', 1)
    sx, s = response(HERE, '', 1)
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    for i, name in enumerate(('sigma', 'p', 'minus_tauN')):
        for j, side in enumerate(('left', 'right')):
            for label, x, values, style in (('64x256', bx, b[name], '--'), ('64x255', sx, s[name], '-')):
                distance = x if side == 'left' else x[-1]-x[::-1]
                v = values if side == 'left' else values[::-1]
                axes[i, j].plot(distance[:7]/.0078125, v[:7], style+'o', label=label)
            axes[i, j].set(title=f'{side} {name}, t=.5 s', ylabel='Pa',
                           xlabel='distance / h_Gamma')
            axes[i, j].grid(alpha=.25); axes[i, j].legend()
    fig.tight_layout(); fig.savefig(HERE / 'endpoints.png', dpi=150)
    (HERE / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
