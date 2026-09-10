#!/usr/bin/env python3
"""Compare saved point-rule and domain-rule weak traction through one second.

This is a same-support diagnostic, not a new reference solution. Surface
values are M^{-1} times the actual weak loads; raw parent stresses and normal
column averages are not substituted for those loads.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure_case import project


def read(path):
    return np.atleast_1d(np.genfromtxt(path, names=True, delimiter=','))


def q1_mean(x, values):
    return float(np.sum(np.diff(x)*(values[:-1]+values[1:])/2)/(x[-1]-x[0]))


def q1_errors(x, values):
    mean = q1_mean(x, values)
    rms = lambda v: float(np.sqrt(np.sum(np.diff(x)*
        (v[:-1]**2+v[:-1]*v[1:]+v[1:]**2)/3)/(x[-1]-x[0])))
    return dict(mean=mean, rms=rms(values), maximum=float(max(abs(values))),
                anomaly_rms=rms(values-mean), anomaly_maximum=float(max(abs(values-mean))))


def load_case(directory):
    measurements = directory.with_name(directory.name+'-measurements')
    result = {}
    for k in range(3):
        time = float(read(directory/f'time_{k}.csv')['time'][0])
        surface = read(measurements/f'surface_balance_{k}.csv')
        x = surface['s']
        weak_path = measurements/f'weak_balance_{k}.csv'
        if weak_path.exists():
            weak = read(weak_path)
            mass = np.diag(weak['Mdiag']) + np.diag(weak['Moff'][:-1], 1)
            mass += np.diag(weak['Moff'][:-1], -1)
            rhs = np.column_stack([weak[name] for name in ('q', 'C', 'friction', 'damping', 'F')])
        else:
            # Reconstruct the OLD documented point rule from its saved parent
            # samples, without overwriting the valid original measurements.
            particle = read(measurements/f'particle_balance_{k}.csv')
            samples = np.column_stack([particle[name] for name in
                                       ('q', 'C_evaluated', 'friction', 'radiation', 'F')])
            mass, rhs, _ = project(particle['segment'].astype(int), particle['xi'],
                                   particle['volume'], samples, len(x))
        represented = np.linalg.solve(mass, rhs)
        if max(abs(represented[:, 0]-surface['particle_q_Q1'])) > 1e-9:
            raise ValueError('Saved represented traction does not match actual weak load')
        controls = {}
        for label, coordinate in (('left', 0), ('interior_left', .0625),
                                  ('center', .125), ('interior_right', .1875), ('right', .25)):
            i = int(np.argmin(abs(x-coordinate)))
            controls[label] = dict(s_m=float(x[i]), q_Pa=float(represented[i, 0]),
                q_anomaly_Pa=float(represented[i, 0]-q1_mean(x, represented[:, 0])),
                weak_q_Pa_m2=float(rhs[i, 0]), weak_F_Pa_m2=float(rhs[i, -1]),
                row_mass_m2=float(sum(mass[i])), Mdiag_m2=float(mass[i, i]),
                Mleft_m2=float(mass[i, i-1]) if i else 0.,
                Mright_m2=float(mass[i, i+1]) if i+1<len(x) else 0.)
        result[time] = dict(x=x, surface=surface, controls=controls,
                           full_admitted_measure_m2=float(mass.sum()),
                           weak_shear_load_Pa_m2=float(rhs[:, 0].sum()),
                           maximum_weak_balance_Pa_m2=float(max(abs(rhs[:, -1]))))
    if set(result) != {0., .5, 1.}:
        raise ValueError('The bounded replay must contain accepted times 0, 0.5 and 1 s')
    return result


def compare(coarse, fine):
    initial_error = {}
    rows = []
    for time in sorted(coarse):
        c, f = coarse[time], fine[time]
        x = np.unique(np.r_[c['x'], f['x']])
        row = dict(time_s=time, fields={})
        for name in ('particle_q_Q1', 'V', 'Theta', 'C_retained', 'slip'):
            error = np.interp(x, c['x'], c['surface'][name])-np.interp(x, f['x'], f['surface'][name])
            if time == 0:
                initial_error[name] = (x, error)
            initial_x, initial_values = initial_error[name]
            change = error-np.interp(x, initial_x, initial_values)
            row['fields'][name] = dict(total=q1_errors(x, error),
                change_from_initial_difference=q1_errors(x, change),
                left=float(error[0]), right=float(error[-1]),
                left_change_from_initial=float(change[0]), right_change_from_initial=float(change[-1]))
        rows.append(row)
    return rows


def main():
    here = Path(__file__).resolve().parent
    paths = dict(old64=here.parent/'output', old128=here.parent/'refinement/space128',
                 new64=here/'k2_64', new128=here/'k2_128')
    cases = {name: load_case(path) for name, path in paths.items()}
    report = dict(same_support_only=True, controls={name: [dict(time_s=t,
        full_admitted_measure_m2=row['full_admitted_measure_m2'],
        weak_shear_load_Pa_m2=row['weak_shear_load_Pa_m2'],
        maximum_weak_balance_Pa_m2=row['maximum_weak_balance_Pa_m2'], **row['controls'])
        for t, row in sorted(case.items())] for name, case in cases.items()}, comparisons={})
    for label, a, b in (('old_spatial', 'old64', 'old128'), ('new_spatial', 'new64', 'new128'),
                         ('rule_change64', 'new64', 'old64'), ('rule_change128', 'new128', 'old128')):
        report['comparisons'][label] = compare(cases[a], cases[b])
    report['realized_surface'] = {name: [dict(time_s=time, fields={field: dict(
        mean=q1_mean(row['x'], row['surface'][field]),
        minimum=float(min(row['surface'][field])), maximum=float(max(row['surface'][field])))
        for field in ('V', 'Theta', 'C_retained', 'slip')}) for time, row in sorted(case.items())]
        for name, case in cases.items()}
    report['unchanged_initial_inputs'] = {}
    for resolution in (64, 128):
        old, new = paths[f'old{resolution}'], paths[f'new{resolution}']
        po, pn = read(old/'particles_0.csv'), read(new/'particles_0.csv')
        po, pn = np.sort(po, order='id'), np.sort(pn, order='id')
        if not np.array_equal(po['id'], pn['id']):
            raise ValueError('Initial particle identities changed')
        report['unchanged_initial_inputs'][resolution] = dict(
            phase_bitwise_equal=bool(np.array_equal(read(old/'phase_0.csv'), read(new/'phase_0.csv'))),
            particle_max_difference={name: float(max(abs(po[name]-pn[name])))
                for name in ('x', 'y', 'volume', 'H', 'tau_xx', 'tau_yy', 'tau_xy')})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
