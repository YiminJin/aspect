#!/usr/bin/env python3
"""Common-coordinate audit of three domain-rule K2 runs; no fitted tolerances.

Weak traction is the saved M^{-1}Q representation, never a bulk column mean.
The report separates initial projection and subsequent error changes without
claiming that initial-error subtraction removes its physical influence.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_cases import norms
from measure_case import summarize


FIELDS = ('particle_q_Q1', 'C_evaluated_Q1', 'friction_Q1', 'radiation_Q1',
          'F_Q1', 'V', 'Theta', 'C_retained', 'slip')


def samples(breaks, order=8):
    gauss, weight = np.polynomial.legendre.leggauss(order)
    x = ((breaks[:-1, None]+breaks[1:, None])/2+np.diff(breaks)[:, None]*gauss/2).ravel()
    w = (np.diff(breaks)[:, None]*weight/2).ravel()
    return x, w


def differences(coarse, fine, initial_coarse, initial_fine):
    breaks = np.unique(np.r_[coarse['s'], fine['s'], initial_coarse['s'], initial_fine['s'],
                              .0625, .1875])
    x, w = samples(breaks)
    interior = (x > .0625) & (x < .1875)
    row = {}
    for name in FIELDS:
        difference = lambda c, f, points: np.interp(points, c['s'], c[name])-np.interp(points, f['s'], f[name])
        d = difference(coarse, fine, x)
        d0 = difference(initial_coarse, initial_fine, x)
        mean, initial_mean = np.average(d, weights=w), np.average(d0, weights=w)
        vertices = difference(coarse, fine, breaks)
        change = d-d0
        controls = np.array([0., .0625, .125, .1875, .25])
        control_error = difference(coarse, fine, controls)
        control_initial = difference(initial_coarse, initial_fine, controls)
        row[name] = dict(total=norms(d, w), anomaly=norms(d-mean, w),
            total_maximum=float(max(abs(vertices))), anomaly_maximum=float(max(abs(vertices-mean))),
            inside_bump_anomaly=norms((d-mean)[interior], w[interior]),
            outside_bump_anomaly=norms((d-mean)[~interior], w[~interior]),
            change_from_initial_difference=norms(change, w),
            anomaly_change_from_initial_difference=norms(change-(mean-initial_mean), w),
            controls=[dict(s_m=float(s), total=float(e), anomaly=float(e-mean),
                           change_from_initial=float(e-e0),
                           anomaly_change_from_initial=float(e-e0-mean+initial_mean))
                      for s, e, e0 in zip(controls, control_error, control_initial)])
        if name == 'V':
            cv = np.interp(breaks, coarse['s'], coarse[name])
            fv = np.interp(breaks, fine['s'], fine[name])
            if min(cv) <= 0 or min(fv) <= 0:
                raise ValueError('Logarithmic slip-rate comparison requires positive states')
            row[name]['maximum_log10_ratio'] = float(max(abs(np.log10(cv/fv))))
    return row


def load_case(directory):
    measurements = directory.with_name(directory.name+'-measurements')
    report = json.loads((measurements/'report.json').read_text())
    if not report['allowances_pass'] or not report['completed_to_2_seconds']:
        raise ValueError('Incomplete trajectory or failed support/normalization allowance')
    if any(row['surface_rule'] != 'domain integrated' for row in report['steps']):
        raise ValueError('Do not mix point-rule data into the domain convergence sequence')
    log = summarize(directory.with_suffix('.log'))
    if len(log) != len(report['steps']):
        raise ValueError('The accepted exports and solver log have different step counts')
    rows = {}
    for k, step in enumerate(log):
        last = step['nonlinear'][-1]
        # These are the existing fixture criteria, not new convergence targets.
        if not (last['bulk'] < last['bulk target']
                and last['surface'] < 1e-8*last['surface scale']
                and all(r['fresh'] <= r['target'] for r in step['linear'])):
            raise ValueError('A solver criterion failed')
        if any(r['bulk scale'] != last['bulk scale'] or r['surface scale'] != last['surface scale']
               for r in step['nonlinear']):
            raise ValueError('A merit scale changed within a nonlinear solve')
        if step['time'] != report['steps'][k]['time_s']:
            raise ValueError('The actual accepted timestep sequences differ')
        rows[step['time']] = np.atleast_1d(np.genfromtxt(
            measurements/f'surface_balance_{k}.csv', names=True, delimiter=','))
    return report, rows, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('spatial', 'temporal'))
    parser.add_argument('directories', type=Path, nargs=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [load_case(path) for path in args.directories]
    times = sorted(set.intersection(*(set(case[1]) for case in cases)))
    if times[0] != 0 or times[-1] != 2:
        raise ValueError('The compared common-time trajectory must reach 2 s')
    widths = [c[0]['support_half_width_m'] for c in cases]
    if max(widths)-min(widths) > 1e-12:
        raise ValueError('Support width changed between runs')
    result = dict(kind=args.kind, status='measured; interpret resolution trends before proceeding',
                  same_support_only=True, common_times_s=times, cases=[], adjacent=[])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for directory, (measurement, rows, log) in zip(args.directories, cases):
        initial = rows[0.]
        x, w = samples(np.unique(np.r_[initial['s'], .0625, .1875]))
        analytic = 200+10*np.where(abs(x-.125) < .0625, (1+np.cos(np.pi*(x-.125)/.0625))**2/4, 0)
        initialization = dict(Theta_projection_rms_s=norms(np.interp(x, initial['s'], initial['Theta'])-analytic, w)['L2'],
            Theta_min_s=float(min(initial['Theta'])), Theta_max_s=float(max(initial['Theta'])),
            C_mean_Pa=float(np.average(np.interp(x, initial['s'], initial['C_retained']), weights=w)))
        response = []
        for time, data in sorted(rows.items()):
            sx, sw = samples(data['s'])
            fields = {}
            for name in ('V', 'Theta', 'C_retained', 'slip', 'particle_q_Q1'):
                values = np.interp(sx, data['s'], data[name])
                mean = float(np.average(values, weights=sw))
                fields[name] = dict(mean=mean, minimum=float(min(data[name])), maximum=float(max(data[name])),
                    anomaly_rms=norms(values-mean, sw)['L2'],
                    mirror_difference_max=float(max(abs(data[name]-np.interp(.25-data['s'], data['s'], data[name])))))
            response.append(dict(time_s=time, fields=fields))
        result['cases'].append(dict(directory=str(directory), initialization=initialization,
            measurements=measurement, response=response,
            returned_linear_checks=sum(len(s['linear']) for s in log)))
        final = rows[2.]
        axes[0, 0].plot(initial['s'], initial['Theta']-200, '.-', label=directory.name)
        for ax, name in ((axes[0, 1], 'V'), (axes[1, 0], 'particle_q_Q1')):
            ax.plot(final['s'], final[name]-response[-1]['fields'][name]['mean'], '.-', label=directory.name)
    for index in range(2):
        coarse, fine = cases[index][1], cases[index+1][1]
        pair = f'{args.directories[index].name}-{args.directories[index+1].name}'
        for time in times:
            terms = differences(coarse[time], fine[time], coarse[0.], fine[0.])
            result['adjacent'].append(dict(pair=pair, time_s=time, terms=terms))
            if time == 2:
                c, f = coarse[time], fine[time]
                s = np.unique(np.r_[c['s'], f['s']])
                q = np.interp(s, c['s'], c['particle_q_Q1'])-np.interp(s, f['s'], f['particle_q_Q1'])
                axes[1, 1].plot(s, q-terms['particle_q_Q1']['total']['mean'], '.-', label=pair)
    for ax, title, ylabel in zip(axes.flat,
        ('Realized initial state', 'Slip-rate anomaly at 2 s', 'Actual weak traction anomaly at 2 s',
         'Adjacent traction-anomaly differences at 2 s'),
        ('Theta - 200 (s)', 'V - mean(V) (m/s)', 'q - mean(q) (Pa)', 'difference (Pa)')):
        ax.set(title=title, xlabel='fault coordinate (m)', ylabel=ylabel)
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
