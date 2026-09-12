"""Compare saved K3 normal levels without rerunning or resetting reference histories."""
import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = {128: 'spatial0375_n128_f32_periodic',
         256: 'spatial0375_n256_f32_periodic_floor',
         512: 'spatial0375_n512_f32_periodic_floor'}


def profile(case, field, step):
    return np.genfromtxt(HERE / case / f'comparison_{field}_{step}.csv',
                         delimiter=',', names=True)


def comparison_row(step, time, metric, values):
    # Ratios describe errors, not signed feedback or differences of maxima.
    assert all(value >= 0 and np.isfinite(value) for value in values)
    return dict(step=step, time_s=time, metric=metric,
                error128=values[0], error256=values[1], error512=values[2],
                factor128_256=values[0]/values[1] if values[1] else None,
                factor256_512=values[1]/values[2] if values[2] else None,
                monotone=values[0] >= values[1] >= values[2])


def main():
    runs = {n: json.loads((HERE / f'{case}-comparison.json').read_text())
            for n, case in CASES.items()}
    assert all(run['complete_smoke'] and len(run['states']) == 9
               for run in runs.values())
    metrics, signals, checks = {}, {}, {}
    for n, run in runs.items():
        case = CASES[n]
        initial = run['states'][0]
        phi0, H0 = profile(case, 'phase', 0), profile(case, 'H', 0)
        metrics[n], signals[n] = [], []
        for step, state in enumerate(run['states']):
            coarse = runs[128]['states'][step]
            assert (state['time_s'], state['dt_s'], state['reference']) == (
                coarse['time_s'], coarse['dt_s'], coarse['reference'])
            phi, H = profile(case, 'phase', step), profile(case, 'H', step)
            assert np.array_equal(phi['y'], phi0['y'])
            assert np.array_equal(H['initial_y'], H0['initial_y'])
            ih = state['surface_means']['Ih'] - initial['surface_means']['Ih']
            ref_ih = state['reference']['Ih'] - initial['reference']['Ih']
            previous = run['states'][max(step-1, 0)]
            ih_step = state['surface_means']['Ih'] - previous['surface_means']['Ih']
            ref_ih_step = state['reference']['Ih'] - previous['reference']['Ih']
            metrics[n].append(dict(
                total_phi=state['phi_max_absolute_error'],
                total_H_Pa=state['H_profile_max_absolute_error'],
                total_Ih_m=abs(state['surface_means']['Ih']-state['reference']['Ih']),
                raw_stress_rms_Pa=state['raw_stress_error_rms_Pa'],
                raw_stress_max_Pa=state['raw_stress_error_max_Pa'],
                H_increment_Pa=state['H_increment_profile_max_error'],
                phi_increment=state['phi_increment_profile_max_error'],
                Ih_increment_m=abs(ih_step-ref_ih_step),
                cumulative_H_Pa=float(np.max(abs(H['difference']-H0['difference']))),
                cumulative_phi=float(np.max(abs(phi['difference']-phi0['difference']))),
                cumulative_Ih_m=abs(ih-ref_ih)))
            signals[n].append(dict(step=step, time_s=state['time_s'],
                Ih_cumulative_m=ih, reference_Ih_cumulative_m=ref_ih,
                Ih_increment_m=ih_step, reference_Ih_increment_m=ref_ih_step,
                H_cumulative_max_Pa=float(np.max(H['production_mean_H']-H0['production_mean_H'])),
                phi_cumulative_max=float(np.max(abs(phi['production_mean_phi']-phi0['production_mean_phi'])))))
        checks[n] = dict(
            states=len(run['states']), fresh_linear_checks=len(run['fresh_linear_checks']),
            max_fresh_over_target=max(r['fresh']/r['target'] for r in run['fresh_linear_checks']),
            max_omitted_fraction=max(s['guard']['max_omitted_fraction'] for s in run['states']),
            max_supported_normalization_error=max(s['guard']['max_supported_normalization_error'] for s in run['states']),
            max_surface_balance_rms_Pa=max(s['surface_balance_rms_Pa'] for s in run['states']),
            max_theta_update_error_s=max(s.get('theta_update_max_error_s', 0) for s in run['states']),
            active_nodes=[s['active_nodes'] for s in run['states']],
            resources=json.loads((HERE / f'{case}.resources.json').read_text()))
        # Verify the terminal base residual, not merely a successful exit or
        # a fresh linear direction. The fixture's surface target stays 1e-8.
        final_residuals = []
        for block in (HERE / f'{case}.log').read_text().split('*** Timestep ')[1:]:
            matches = re.findall(r'Fault nonlinear residual: bulk=([^,]+), bulk scale=[^,]+, '
                                 r'surface=([^,]+), surface scale=([^,]+), velocity=[^,]+, '
                                 r'scaled continuity=[^,]+, bulk precision=[^,]+, bulk target=([^,]+),', block)
            bulk, surface, scale, target = map(float, matches[-1])
            assert bulk <= target and surface <= 1e-8*scale
            final_residuals.append(dict(bulk=bulk, bulk_target=target,
                                        surface=surface, surface_target=1e-8*scale))
        assert len(final_residuals) == 9
        checks[n]['terminal_nonlinear_residuals'] = final_residuals
    assert checks[256]['resources']['sha256'] == checks[512]['resources']['sha256']
    rows = [comparison_row(step, runs[128]['states'][step]['time_s'], key,
                           [metrics[n][step][key] for n in CASES])
            for step in range(9) for key in metrics[128][step]]
    peaks = [comparison_row('trajectory_max', None, key,
                            [max(m[key] for m in metrics[n]) for n in CASES])
             for key in metrics[128][0]]
    report = dict(cases=CASES, checks=checks, rows=rows, trajectory_max=peaks,
                  signals=signals,
                  nonmonotone_steps={key: [r['step'] for r in rows
                      if r['metric'] == key and not r['monotone']]
                      for key in metrics[128][0]})
    (HERE / 'normal512-comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    with (HERE / 'normal512-errors.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows + peaks)
    print(json.dumps(dict(final=[r for r in rows if r['step'] == 8],
                          trajectory_max=peaks,
                          nonmonotone_steps=report['nonmonotone_steps'],
                          checks=checks), indent=2))


if __name__ == '__main__':
    main()
