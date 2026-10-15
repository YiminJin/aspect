#!/usr/bin/env python3
"""Fixed-state resampling/trace operators and one short controller comparison."""
import csv
import json
from pathlib import Path

import numpy as np

from analyze_section7b import FIELDS, read
from analyze_section7cd import metric
from analyze_managed_particles import check

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'build-bp3-native/managed-event-audit'
EVENT = BASE / 'runs/event'


def operator_pair(before, after):
    a, b = [read(EVENT/(name+'-loads.csv')) for name in (before, after)]
    for key in ['node', 's_m', 'V', 'theta', 'mass_diagonal', 'mass_edge', 'weak_pressure', 'pressure_Pa']:
        assert np.array_equal(a[key], b[key]), (before, key)
    result = {}
    for region, selected in [('whole', np.ones(len(a['s_m']), dtype=bool)),
                             ('lower_200m', a['s_m'] >= a['s_m'][-1]-200)]:
        result[region] = {}
        for key in ['weak_residual', 'weak_shear', 'weak_friction', 'weak_raw_normal',
                    'weak_normal', 'weak_tensor', 'raw_Pa', 'filtered_Pa', 'shear_Pa', 'tensor_Pa']:
            delta = b[key]-a[key]
            index = np.flatnonzero(selected)[np.argmax(np.abs(delta[selected]))]
            result[region][key] = dict(max_abs=float(np.max(np.abs(delta[selected]))),
                                      l2=float(np.linalg.norm(delta[selected])),
                                      signed_sum=float(np.sum(delta[selected])),
                                      absolute_sum=float(np.sum(np.abs(delta[selected]))),
                                      max_at_s_m=float(a['s_m'][index]))
            if key.endswith('_Pa'):
                result[region][key].update(metric(a['s_m'][selected], a[key][selected], b[key][selected], 50e6))
    qp = [read(EVENT/(name+'-qp.csv')) for name in (before, after)]
    for key in ['q', 'x', 'y', 'weight', 'phi', 'chi', 'pressure']:
        assert np.array_equal(qp[0][key], qp[1][key]), key
    w = qp[0]['weight']
    result['history'] = {}
    beta = np.exp(-2e8*32038120320./1e26)
    recurrence = sum(factor*((qp[1][new]-qp[0][new])-beta*(qp[1][old]-qp[0][old]))**2
                     for factor, old, new in zip([1, 1, 2], ['old_xx', 'old_yy', 'old_xy'],
                                                 ['stress_xx', 'stress_yy', 'stress_xy']))
    result['fixed_strain_Maxwell_delta_defect_Pa'] = float(np.sqrt(np.max(recurrence)))
    for representation, fields in [('FE', ['old_xx', 'old_yy', 'old_xy']),
                                   ('direct_LLS', ['lls_xx', 'lls_yy', 'lls_xy'])]:
        norm = sum(factor*(qp[1][name]-qp[0][name])**2 for factor, name in zip([1, 1, 2], fields))
        result['history'][representation] = dict(max_norm_Pa=float(np.sqrt(np.max(norm))),
                                                weighted_rms_Pa=float(np.sqrt(np.sum(w*norm)/np.sum(w))))
    result['FE_minus_direct_LLS'] = []
    for q in qp:
        norm = sum(factor*(q[a]-q[b])**2 for factor, a, b in zip([1, 1, 2],
                    ['old_xx', 'old_yy', 'old_xy'], ['lls_xx', 'lls_yy', 'lls_xy']))
        result['FE_minus_direct_LLS'].append(dict(max_norm_Pa=float(np.sqrt(np.max(norm))),
                                                weighted_rms_Pa=float(np.sqrt(np.sum(w*norm)/np.sum(w)))))
    return result


def main():
    result = {'population': operator_pair('population-44-before', 'population-44-after'),
              'face_operator': operator_pair('face-34-native', 'face-34-host')}
    particles = []
    for label in ['before', 'after']:
        with (EVENT/f'population-44-{label}-particles.csv').open() as stream:
            particles.append({int(row['id']): row for row in csv.DictReader(stream)})
    before, after = particles
    survivors = before.keys() & after.keys()
    assert all(before[i] == after[i] for i in survivors)
    result['population']['particles'] = dict(before=len(before), after=len(after), survivors=len(survivors),
                                             births=len(after.keys()-before.keys()), removals=len(before.keys()-after.keys()),
                                             survivor_change=0.)
    # The before snapshot follows sorting/outflow; count management alone is
    # isolated here. Keep lower-end counts separate from whole-domain totals.
    # The prescribed straight segment is the same input used by native BP3.
    vertices = np.loadtxt(Path(__file__).with_name('fault.txt'), comments='#')
    profile = read(EVENT/'coupled-profile-44.csv')
    x0, y0 = vertices[0, :2]
    tangent = (vertices[-1, :2]-vertices[0, :2])/profile['s_m'][-1]
    down_dip = lambda row: (float(row['x'])-x0)*tangent[0]+(float(row['y'])-y0)*tangent[1]
    lower = lambda row: down_dip(row) >= profile['s_m'][-1]-200
    result['population']['lower_200m_counts'] = dict(
        births=int(sum(lower(after[i]) for i in after.keys()-before.keys())),
        removals=int(sum(lower(before[i]) for i in before.keys()-after.keys())))
    state = [read(EVENT/f'population-44-{stage}-fe.csv') for stage in ['before', 'after']]
    for key in ['block', 'index', 'frozen']:
        assert np.array_equal(state[0][key], state[1][key]), key
    result['population']['frozen_bulk_field_change'] = 0.
    result['population']['FE_coefficient_change'] = float(np.max(np.abs(state[1]['reconstructed']-state[0]['reconstructed'])))
    result['event_lifecycle'] = check(EVENT)
    qp = [read(EVENT/f'population-44-{stage}-qp.csv') for stage in ['before', 'after']]
    change = sum(factor*(qp[1][name]-qp[0][name])**2
                 for factor, name in zip([1, 1, 2], ['old_xx', 'old_yy', 'old_xy']))
    index = int(np.argmax(change))
    result['population']['maximum_history_change_at'] = {
        key: float(qp[0][key][index]) for key in ['x', 'y', 'phi', 'chi', 'weight']}
    sweep = read(EVENT/'face-34-sweep.csv')
    result['face_sweep'] = {k: v.tolist() for k, v in sweep.items()}
    mesh = []
    with (EVENT/'initial-mesh.csv').open() as stream:
        mesh = list(csv.DictReader(stream))
    eligibility = []
    for x, y, count in zip(sweep['x'], sweep['y'], sweep['traces']):
        cells = [row['cell'] for row in mesh
                 if abs(x-float(row['x'])) <= float(row['edge'])*(.5+1e-6)
                 and abs(y-float(row['y'])) <= float(row['edge'])*(.5+1e-6)]
        assert len(cells) == int(count), (x, y, count, cells)
        eligibility.append(cells)
    result['independent_eligible_cells'] = eligibility
    text = (EVENT/'face-34-particle.txt').read_text()
    host = np.array([float(x) for x in next(line for line in text.splitlines() if line.startswith('host '))[5:].split()])
    native = np.array([float(x) for x in next(line for line in text.splitlines() if line.startswith('native '))[7:].split()])
    result['face_stress_difference_Pa'] = float(np.linalg.norm(host-native))
    result['face_identity'] = text
    kappa = lambda dt: -1e26*np.expm1(-dt*32038120320./1e26)
    result['fixed_field_face_stress_scaling'] = {str(dt): result['face_stress_difference_Pa']*kappa(dt)/kappa(2e8)
                                              for dt in [2e8, 4e7, 4e6, 2e6]}
    retained = ROOT/'build-bp3-native/managed-coupled/runs/managed-v3'
    result['replay_max_differences'] = {k: 0. for k in FIELDS}
    for step in range(45):
        a, b = [read(path/f'coupled-profile-{step}.csv') for path in [retained, EVENT]]
        for key in FIELDS:
            result['replay_max_differences'][key] = max(result['replay_max_differences'][key], float(np.max(np.abs(a[key]-b[key]))))
    cases = {name: BASE/f'runs/{name}' for name in ['production-time', 'production-time-tight']}
    loaded = {}
    result['time_checks'] = {}
    for name, path in cases.items():
        for filename in ['initial-mesh.csv', 'initial-phase.csv', 'mechanical-quadrature.csv']:
            assert (path/filename).read_bytes() == (EVENT/filename).read_bytes(), filename
        journal, accepted = read(path/'coupled-checks.csv'), read(path/'accepted_steps.csv')
        assert np.array_equal(journal['step'], accepted['step'])
        assert np.all(journal['publications'] == 1)
        assert journal['time_s'][-1] == 2e7
        assert np.all(accepted['normalized_nonlinear_residual'] < 1e-8)
        loaded[name] = {float(d['time_s'][0]): d for d in (read(p) for p in path.glob('coupled-profile-*.csv'))}
        cap = 4e7 if name == 'production-time' else 2e6
        logarithms = []
        for p in loaded[name].values():
            v, theta = p['V_m_s'], p['theta_s']
            predicted = theta*np.exp(-v*cap/.008)+.008/v*(-np.expm1(-v*cap/.008))
            logarithms.append(float(np.max(np.abs(np.log(predicted/theta)))))
        result['time_checks'][name] = dict(positive_steps=int(journal['step'][-1]),
            times_s=journal['time_s'].tolist(), dt_s=journal['dt_s'].tolist(),
            max_residual=float(np.max(accepted['normalized_nonlinear_residual'])),
            max_stress_defect_Pa=float(np.max(journal['stress_error_Pa'])),
            max_predicted_log_theta_change_at_cap=max(logarithms),
            maximum_step_slip_over_Dc=float(np.max(accepted['max_step_slip_over_Dc'])))
        result['time_checks'][name]['lifecycle'] = check(path)
    common = sorted(loaded['production-time'].keys() & loaded['production-time-tight'].keys())
    result['time_comparison'] = []
    for time in common:
        a, b = [loaded[name][time] for name in cases]
        assert np.array_equal(a['s_m'], b['s_m'])
        assert np.array_equal(a['I_h_m'], b['I_h_m'])
        fields = {}
        for key in FIELDS:
            scale = dict(V_m_s=1e-9, theta_s=8e6, Omega=1., slip_m=max(.008, 1e-9*time)).get(key, 50e6)
            fields[key] = metric(a['s_m'], a[key], b[key], scale)
        result['time_comparison'].append(dict(time_s=time, fields=fields))
    evidence = ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'
    result['runs'] = {p.stem: json.loads(p.read_text()) for p in evidence.glob('bp3-event-*.json')}
    result['simulation_seconds'] = sum(r['seconds'] for r in result['runs'].values())
    (BASE/'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
