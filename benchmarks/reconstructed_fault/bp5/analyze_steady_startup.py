"""Native weak initialization, split lifecycle, timestep and restart checks."""
import argparse
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

from run_steady_startup import OUT
from check_startup_30km import table, raw_qps, load_parts, difference
from check_first_cycle_restart import convergence


def raw(path, step):
    return np.sort(np.concatenate([raw_qps(p) for p in sorted(path.glob(f'work_qp_{step}_rank*.csv'))]),
                   order=['cell', 'qp'])


def check(case):
    path = OUT / case
    assert json.loads((path / 'execution.json').read_text())['passed']
    conv = convergence(path / 'run.log')
    clock = table(path / 'accepted_steps.csv')
    assert clock[-1]['time'] == 900
    assert np.all(clock['fresh_linear_checks_passed'] == 1)
    assert np.all(clock['free'] == 1156) and np.all(clock['lower_active'] == 0)
    assert max(clock['Theta_relative_error']) < 1e-12
    assert not (path / 'weak_initialization.csv').exists()
    previous = table(OUT / 'startup/state_work_2.csv') if case == 'resume' else None
    initialized_path = OUT / 'startup' if case == 'resume' else path
    init = table(initialized_path / 'steady_initialization.csv')
    rows = {}
    for k in conv:
        s = table(path / f'state_work_{k}.csv')
        w = table(path / f'work_weak_{k}.csv')
        assert max(abs(w['bg']-init['background_load'])/w['weight']) < 1e-5
        if k == 0:
            init = table(path / 'steady_initialization.csv')
            np.testing.assert_array_equal(s['Theta_in'], np.full(len(s), 1e8))
            np.testing.assert_array_equal(s['Theta_in'], s['Theta_out'])
            assert max(abs(init['weak_error_Pa'])) < 1e-5
            assert max(abs(init['weight'] / w['weight'] - 1)) < 1e-12
            assert clock[0]['max_committed_stress_Pa'] == 0
        if previous is not None:
            np.testing.assert_array_equal(s['Theta_in'], previous['Theta_out'])
            dt = s['dt'][0]
            x = s['V'] * dt / .1
            theta = s['Theta_in'] * np.exp(-x) - .1 / s['V'] * np.expm1(-x)
            assert max(abs(s['Theta_out'] / theta - 1)) < 1e-12
            slip = np.array([math.fma(float(dt), float(v), float(old)) for v, old in zip(s['V'], previous['slip'])])
            np.testing.assert_array_equal(s['slip'], slip)
        previous = s
        rows[k] = dict(time=float(s['time'][0]), V_over_Vp=[float(min(s['V']) / 1e-9), float(max(s['V']) / 1e-9)],
                       Theta_range=[float(min(s['Theta_out'])), float(max(s['Theta_out']))],
                       max_step_Vdt_Dc=float(max(s['V'] * s['dt'] / .1)))
    if case == 'resume':
        assert set(conv) == {3}
        assert not (path / 'steady_initialization.csv').exists()
        assert not list(path.glob('initial_mesh_*.csv'))
        result = dict(passed=True, convergence=conv, states=rows, initialization_repeated=False)
    else:
        init = table(path / 'steady_initialization.csv')
        particle_count = 0
        # Fixed fixture layout: id,x,y,H,initial theta,composition,Maxwell,...
        for file in path.glob('audit_particles_0_rank*.csv'):
            theta = np.loadtxt(file, delimiter=',', skiprows=1, usecols=4, ndmin=1)
            np.testing.assert_array_equal(theta, np.full(len(theta),1e8))
            particle_count += len(theta)
        assert particle_count > 0
        arrays = {a.attrib['Name']: np.fromstring(a.text, sep=' ') for a in ET.parse(
            path / 'reconstructed_faults/reconstructed_faults-00000.vtu').findall('.//PointData/DataArray')}
        f = np.clip(arrays['composition_strengthening'], 0, 1)
        qp = raw(path, 0)
        active = qp['source_active'] == 1
        q = qp[active]
        j, xi = q['segment'].astype(int), q['xi']
        mixture = np.clip((1-xi) * arrays['composition_strengthening'][j] + xi * arrays['composition_strengthening'][j+1], 0, 1)
        direct = .004 * (1-mixture) + .04 * mixture
        mu = direct * np.arcsinh(1e-9 / (2e-6) * np.exp((.6 + .03*np.log(1e8*1e-6/.1)) / direct))
        target = 50e6 * mu + 4624440e-9
        bg = (1-xi)*init['tau_bg'][j] + xi*init['tau_bg'][j+1]
        # Independently integrate the actual native QP measure in exported windows.
        weights = q['JxW'] * q['chi']
        load, measure = np.zeros(len(init)), np.zeros(len(init))
        for end, basis in ((0, 1-xi), (1, xi)):
            np.add.at(load, j+end, weights*basis*(bg-target))
            np.add.at(measure, j+end, weights*basis)
        covered = np.abs(measure/init['weight']-1) < 1e-11
        assert covered.sum() > 100
        assert max(abs(load[covered]/measure[covered])) < 1e-5
        predictor = table(path / 'state_startup_predictor.csv')
        ratio = .03 / (.004*(1-f) + .04*f)
        for record in predictor:
            s = table(path / f'state_work_{int(record["accepted_step"])}.csv')
            dt = record['proposed_dt']
            x = s['V']*dt/.1
            theta = s['Theta_out']*np.exp(-x)-.1/s['V']*np.expm1(-x)
            measured = max(ratio*abs(np.log(theta/s['Theta_out'])))
            assert record['limit'] == .02 and measured <= .02 + 1e-14
            assert abs(measured-record['measure']) < 1e-13
        result = dict(passed=True, convergence=conv, states=rows,
                      initialized_Theta=1e8, verified_initial_particles=particle_count,
                      background_range_Pa=[float(min(init['tau_bg'])),float(max(init['tau_bg']))],
                      maximum_native_weak_initial_error_Pa=float(max(abs(init['weak_error_Pa']))),
                      independent_QP_weak_error_Pa=float(max(abs(load[covered]/measure[covered]))),
                      independent_QP_point_error_Pa=float(max(abs(bg-target))),
                      checked_rows=int(covered.sum()), maximum_predictor_measure=float(max(predictor['measure'])))
    (path / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def comparisons():
    a, b, r = OUT / 'startup', OUT / 'half', OUT / 'resume'
    temporal = {}
    for k in (0, 1, 2, 3):
        x, y = table(a / f'state_work_{k}.csv'), table(b / f'state_work_{2*k}.csv')
        w = table(a / f'work_weak_{k}.csv')['weight']
        assert x['time'][0] == y['time'][0]
        temporal[k] = {}
        for field in ('V', 'Theta_out', 'slip', 'weak_q', 'weak_sigma', 'weak_friction'):
            u, v = x[field], y[field]
            if field.startswith('weak_'):
                u, v = u/w, v/w
            d = u-v
            temporal[k][field] = dict(max_absolute=float(max(abs(d))),
                rms=float(np.sqrt(np.dot(w,d*d)/sum(w))), relative_max=float(max(abs(d))/max(abs(v))) if max(abs(v)) else 0.)
    restarted = {}
    x, y = table(a / 'state_work_3.csv'), table(r / 'state_work_3.csv')
    for field in ('V', 'Theta_in', 'Theta_out', 'slip', 'weak_q', 'weak_sigma', 'weak_friction'):
        restarted[field] = difference(x[field],y[field])
    # These early increments are tiny compared with 1e8 s. An ordinary relative
    # tolerance alone could miss a reset, so explicitly require identical state.
    assert max(abs(x['Theta_out']-1e8)) > 1e-5
    np.testing.assert_array_equal(x['Theta_in'],y['Theta_in'])
    np.testing.assert_array_equal(x['Theta_out'],y['Theta_out'])
    restarted['evolved_state_bitwise_identical'] = True
    for pattern in ('audit_bulk_3_rank*.csv', 'audit_particles_3_rank*.csv'):
        x,y = load_parts(a,pattern),load_parts(r,pattern)
        np.testing.assert_array_equal(x[:,0],y[:,0])
        if 'bulk' in pattern:
            np.testing.assert_array_equal(x[:,1],y[:,1])
            restarted[pattern] = {str(int(c)):difference(x[x[:,1]==c,2],y[y[:,1]==c,2]) for c in np.unique(x[:,1])}
        else:
            restarted[pattern] = {str(c):difference(x[:,c],y[:,c]) for c in range(1,x.shape[1])}
    x,y = raw(a,3),raw(r,3)
    for key in ('cell','qp','x','y','source_active','segment','xi'):
        np.testing.assert_array_equal(x[key],y[key])
    restarted['constitutive'] = {key:difference(x[key],y[key]) for key in
        ('phi','Ih','chi','V','p','tau_xx','tau_yy','tau_xy','sigma_n','q')}
    x,y = table(a/'work_weak_3.csv'),table(r/'work_weak_3.csv')
    restarted['background'] = difference(x['bg'],y['bg'])
    np.testing.assert_array_equal(x['bg'],y['bg'])
    np.testing.assert_array_equal(table(a/'accepted_steps.csv')['time'],table(r/'accepted_steps.csv')['time'])
    # Resumed predictor appends without a header, as in the existing plugin.
    fields = ('accepted_step','time','maximum_dt','proposed_dt','measure','limit')
    p = np.atleast_1d(np.genfromtxt(r/'state_startup_predictor.csv',delimiter=',',names=fields))
    original = table(a/'state_startup_predictor.csv')[-1:]
    for key in fields:
        restarted['predictor_'+key] = difference(original[key],p[key])
    result = dict(passed=True, temporal=temporal, restart=restarted)
    (OUT/'comparisons.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['startup','half','resume','compare'])
    args = parser.parse_args()
    comparisons() if args.case == 'compare' else check(args.case)
