#!/usr/bin/env python3
"""Independent reductions of the bounded native evidence; no simulation."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BUILD = ROOT / 'build-particle-free/objective-stress'
RUNS = BUILD / 'runs'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/objective_stress'


def table(path):
    if path.name == 'objective-summary.csv' and not path.read_text().startswith('step,'):
        # Restart appends a nonzero-step fragment in its fresh branch directory;
        # use the same declared schema as the parent, preserving the raw bytes.
        names = 'step,time_s,dt_s,error_Pa,relative_error,max_angle_rad,tensor_square_sum_Pa2,publications,retries'
        return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=names))
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))


def tensors(label, step):
    a = np.concatenate([table(p) for p in sorted((RUNS / label / 'output').glob(f'mature_history_{step}_rank*.csv'))])
    a.sort(order='id')
    assert len(np.unique(a['id'])) == len(a)
    value = np.column_stack([a['tau_xx'], a['tau_yy'], a['tau_xy'], -a['tau_xx']-a['tau_yy']])
    return a['id'], value


def metrics(a, b, tensor=False):
    delta = b-a
    weights = np.array([1, 1, 2, 1]) if tensor else 1
    square = delta*delta*weights
    scale = a*a*weights
    return dict(max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.sum(square)/len(a))),
                relative_l2=float(np.sqrt(np.sum(square)/max(1e-300, np.sum(scale)))), exact=bool(np.array_equal(a, b)))


def compare(first, second, step):
    ida, a = tensors(first, step); idb, b = tensors(second, step)
    assert np.array_equal(ida, idb)
    pa = table(RUNS / first / f'output/profiles/fault_{step}.csv')
    pb = table(RUNS / second / f'output/profiles/fault_{step}.csv')
    assert np.array_equal(pa['node'], pb['node'])
    result = dict(particles=len(a), stress=metrics(a, b, tensor=True))
    for field in ['V_m_per_s', 'Theta_s', 'q_weak_Pa', 'sigma_n_weak_Pa', 'slip_m']:
        result[field] = metrics(pa[field], pb[field])
    return result


def main():
    OUT.mkdir(exist_ok=True)
    result = dict(
        legacy_retained_vs_current=compare('legacy-retained', 'legacy-current', 3),
        spin_off_vs_on=compare('legacy-current', 'spin-on-qualified', 3),
        same_rank_restart=compare('spin-four', 'spin-restart', 4),
        repartition_2_to_4=compare('spin-four', 'spin-repartition', 4),
        rejection_vs_matching_schedule=compare('spin-retry-control', 'spin-rejection', 3),
        final_admission_vs_qualified=compare('spin-on-qualified', 'spin-final-admission', 3),
    )
    # These are local lifecycle checks, with the established native nonlinear
    # tolerance 1e-8. MPI ownership changes are not required to be bitwise exact.
    for key in ['same_rank_restart', 'rejection_vs_matching_schedule', 'final_admission_vs_qualified']:
        assert all(v['exact'] for v in result[key].values() if isinstance(v, dict))
    for value in result['repartition_2_to_4'].values():
        if isinstance(value, dict):
            assert value['relative_l2'] < 1e-8
    cases = {}
    for label in ['legacy-retained', 'legacy-current', 'spin-on-qualified', 'spin-four',
                  'spin-restart', 'spin-repartition', 'spin-rejection', 'spin-retry-control', 'spin-final-admission']:
        directory = RUNS / label
        record = json.loads((directory / 'command.json').read_text())
        case = dict(seconds=record['seconds'], command=record['command'], binary_sha256=record['binary_sha256'],
                    plugin_sha256=record['plugin_sha256'], input_sha256=record['input_sha256'])
        summaries = directory / 'output/objective-summary.csv'
        if summaries.exists():
            rows = table(summaries)
            case.update(recurrence_error_Pa=float(np.max(rows['error_Pa'])),
                        recurrence_relative=float(np.max(rows['relative_error'])),
                        maximum_angle_rad=float(np.max(rows['max_angle_rad'])),
                        publications=rows['publications'].astype(int).tolist(), retries=int(max(rows['retries'])))
        energy = [table(p)[0] for p in sorted((directory / 'output').glob('objective-energy-*.csv'))]
        if energy:
            case['energy_max_closure_J_m'] = max(abs(float(a['closure_J_m'])) for a in energy)
            case['energy_max_error_fraction'] = max(abs(float(a['closure_J_m']/a['bound_J_m'])) for a in energy)
            case['energy_last'] = {key: float(energy[-1][key]) for key in energy[-1].dtype.names}
        cases[label] = case
    result['cases'] = cases
    audit = (RUNS / 'spin-on-qualified/run.log').read_text()
    result['residual_audit'] = dict(
        frozen_load_change=max(map(float, re.findall(r'frozen_load_change=([^ ]+)', audit))),
        affine_error_max=max(map(float, re.findall(r'(?<!_)affine_error=([^ ]+)', audit))),
        represented_action_error_max=max(map(float, re.findall(r'represented_action_error=([^ ]+)', audit))))
    _, values = tensors('spin-on-qualified', 3)
    norms = np.sqrt(np.sum(values*values*np.array([1, 1, 2, 1]), axis=1))
    result['branch_scales'] = dict(maximum_tau_over_G=float(max(norms)/32038120320.),
        maximum_spin_times_Maxwell_time=cases['spin-on-qualified']['maximum_angle_rad']/200000.*(1e26/32038120320.),
        summed_maximum_rotation_upper_bound_rad=sum(table(RUNS / 'spin-on-qualified/output/objective-summary.csv')['max_angle_rad']))
    # Independent bounds on the actual endpoint branch-rate norm from the
    # saved tensor recurrence. Congruence is an isometry; no per-particle spin
    # field is reconstructed from a single reported maximum angle.
    lower, upper, integrated_upper = 0., 0., 0.
    beta = np.exp(-200000.*32038120320./1e26)
    eta_ve = -1e26*np.expm1(-200000.*32038120320./1e26)
    for step in [1, 2, 3]:
        ids, a = tensors('spin-on-qualified', step-1); ids_next, b = tensors('spin-on-qualified', step)
        assert np.array_equal(ids, ids_next)
        na = np.sqrt(np.sum(a*a*np.array([1, 1, 2, 1]), axis=1))
        nb = np.sqrt(np.sum(b*b*np.array([1, 1, 2, 1]), axis=1))
        lower = max(lower, float(max(abs(nb-beta*na))/(2*eta_ve)))
        interval_upper = float(max(nb+beta*na)/(2*eta_ve))
        upper = max(upper, interval_upper)
        integrated_upper += 200000.*interval_upper
    result['branch_scales'].update(maximum_branch_rate_norm_bounds_per_s=[lower, upper],
        maximum_branch_rate_times_Maxwell_time_bounds=[lower*1e26/32038120320., upper*1e26/32038120320.],
        integrated_branch_rate_norm_upper_bound=integrated_upper)
    # File hashes retain the exact produced evidence without copying every native field.
    manifest = {}
    for label in cases:
        for p in (RUNS / label).rglob('*'):
            if p.is_file() and (p.suffix in ['.csv', '.prm', '.json', '.log'] or p.name == 'bp3_accepted_state.txt'):
                manifest[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / 'native-results.json').write_text(json.dumps(result, indent=2)+'\n')
    (OUT / 'native-output-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
