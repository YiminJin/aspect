#!/usr/bin/env python3
"""Reduce the selected synthetic uniform-shear preflight, with native phase rows."""
import argparse
import json
import os
import re
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
from analyze_time_update import frame, read, balance

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/synthetic_propagation'


def regions(x, y):
    left = (abs(x - 2) <= .5) & (abs(y - 2) <= .5)
    right = (abs(x - 6) <= .5) & (abs(y - 2) <= .5)
    return {'left_tip': left, 'right_tip': right, 'tip': left | right,
            'ahead': (abs(x - 4) >= 2) & (abs(y - 2) <= .5),
            'core': (abs(x - 4) < 1) & (abs(y - 2) <= .5),
            'remote': abs(x - 4) > 3,
            'boundary': np.minimum.reduce([x, 8 - x, y, 4 - y]) <= .25}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    run = ROOT / 'build-particle-free/bulk-tip/runs' / args.label
    directory = run / 'output'
    command = json.loads((run / 'command.json').read_text())
    input_text = (run / 'input.prm').read_text()
    def parameter(name):
        return float(re.findall(r'^\s*set ' + re.escape(name) + r'\s*=\s*([^\n#]+)', input_text, re.M)[-1])
    assert parameter('X extent') == 8 and parameter('Y extent') == 4, 'Region masks require the selected 8x4 specimen.'
    K = parameter('Critical energy release rates') / ((8 / 3) * parameter('Length scale'))
    G, eta = parameter('Elastic shear moduli'), parameter('Reference viscosities')
    assert '.01*(y-2);0' in input_text, 'Nominal reference requires the selected shear loading.'
    rows = read(directory / 'bulk-accepted.csv')
    initial = frame(directory, 0)
    result = {'label': args.label, 'command': command, 'regions':
              'tip: +/-0.5 m around x=2,6, |y-2|<=.5; core: |x-4|<1, |y-2|<=.5; '
              'ahead: |x-4|>=2, |y-2|<=.5; remote: |x-4|>3; boundary: distance<=.25 to any wall; overlapping masks',
              'K': K, 'initial_fracture': float(read(directory / 'tip-initial.csv').fracture[0]),
              'steps': []}
    cumulative_positive = cumulative_negative = 0.
    for row in rows:
        step = int(row.step)
        points = frame(directory, step)
        native = balance(directory, f'tip-balance-{step:04d}-r*.csv')
        sections = read(directory / f'tip-sections-{step:04d}.csv')
        assert np.array_equal(points.cell, initial.cell) and np.array_equal(points.q, initial.q)
        phase_error = float(max(abs(points.phi - initial.phi)))
        assert phase_error == 0.
        work = row.dt * points.weight * points.w * sections.force[points.section.astype(int)]
        positive, negative = float(sum(np.maximum(work, 0))), float(sum(np.minimum(work, 0)))
        cumulative_positive += positive
        cumulative_negative += negative
        section_work = row.dt * sum(sections.ds * sections.force * sections.V)
        absolute = float(sum(sections.ds * sections.integral_abs_w))
        absolute_net = float(sum(sections.ds * abs(sections.V)))
        opposite = multi_band = 0
        for index in np.unique(points.section):
            part = points[points.section == index]
            part = part[np.argsort(part.y)]
            threshold = max(1e-10, 1e-6 * max(abs(part.w)))
            opposite += int(np.any(part.w > threshold) and np.any(part.w < -threshold))
            active = abs(part.w) > threshold
            multi_band += int(sum(active & ~np.r_[False, active[:-1]]) > 1)
        # Native residual includes SIPG faces. A positive release is an outward
        # phase-growth direction when the artificial fixed-phase upper bound is removed.
        release = (native.driving_load - native.fracture_gradient) / (K * native.mass)
        row_identity_error = float(max(abs(native.residual + native.driving_load - native.fracture_gradient)))
        regional = {}
        native_masks = regions(native.x, native.y)
        for name, mask in regions(points.x, points.y).items():
            p = points[mask]
            n = native_masks[name]
            candidate = n & (native.B < 1 - 1e-12)
            virgin = candidate & (native.B < 1e-12)
            candidate_ids = np.flatnonzero(candidate)
            peak_row = candidate_ids[np.argmax(release[candidate])]
            peak_point = int(np.argmax(p.Y))
            regional[name] = {
                'max_Y_over_K': float(max(p.Y) / K),
                'max_Y_location': [float(p.x[peak_point]), float(p.y[peak_point]), float(p.phi[peak_point])],
                'max_release_row': {'cell': str(native.cell[peak_row]), 'i': int(native.i[peak_row]),
                                    'center': [float(native.x[peak_row]), float(native.y[peak_row])],
                                    'B': float(native.B[peak_row])},
                'max_abs_shear': float(max(abs(p.tauxy))),
                'max_abs_dynamic_normal': float(max(max(abs(p.stage_normal_x)), max(abs(p.stage_normal_y)))),
                'positive_w_integral': float(sum(p.weight * np.maximum(p.w, 0))),
                'negative_w_integral': float(sum(p.weight * np.minimum(p.w, 0))),
                'max_native_release': float(max(release[candidate])) if np.any(candidate) else None,
                'max_virgin_native_release': float(max(release[virgin])) if np.any(virgin) else None,
                'positive_native_rows': int(sum(release[candidate] > 1e-7)),
                'driving_load_sum': float(sum(native.driving_load[n])),
                'fracture_gradient_sum': float(sum(native.fracture_gradient[n])),
            }
        result['steps'].append({
            'step': step, 'time': float(row.time), 'phase_error': phase_error,
            'nominal_tau': float(eta * .01 * -np.expm1(-row.time * G / eta)),
            'regions': regional,
            'positive_w_integral': float(sum(sections.ds * sections.positive_w)),
            'negative_w_integral': float(sum(sections.ds * sections.negative_w)),
            'absolute_w_integral': absolute, 'absolute_V_integral': absolute_net,
            'cancellation_fraction': 1 - absolute_net / absolute if absolute else 0,
            'opposite_sign_sections': opposite, 'multiple_band_sections': multi_band,
            'max_V': float(max(abs(sections.V))), 'max_slip': float(max(abs(sections.slip))),
            'age_range': [float(min(sections.age)), float(max(sections.age))],
            'max_abs_resistance': float(max(abs(sections.force))),
            'positive_local_work': positive, 'negative_local_work': negative,
            'section_work': float(section_work), 'power_identity_error': float(sum(work) - section_work),
            'native_row_identity_error': row_identity_error,
        })
    remainder = rows.Wext - rows.energy_change - rows.Db - rows.Dc - rows.Df
    result['energy'] = {k: float(sum(rows[k])) for k in [
        'Wext', 'energy_change', 'Db', 'Dc', 'Df', 'stress_loss', 'traction_loss', 'phase_work']}
    result['energy'].update(physical_remainder_signed=float(sum(remainder)),
                            physical_remainder_absolute=float(sum(abs(remainder))),
                            positive_local_work=cumulative_positive,
                            negative_local_work=cumulative_negative,
                            fracture_change=float(rows.fracture[-1] - result['initial_fracture']))
    result['gates'] = {k: float(max(abs(rows[k]))) for k in ['mechanical_residual', 'phase_KKT', 'ledger_error']}
    result['cost'] = {'wall_seconds': command['seconds'], 'physical_intervals': len(rows)}
    assert max(abs(s['power_identity_error']) for s in result['steps']) < 1e-10
    assert max(s['native_row_identity_error'] for s in result['steps']) < 1e-10
    (OUT / f'{args.label}-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'final': result['steps'][-1], 'energy': result['energy'],
                      'gates': result['gates'], 'cost': result['cost']}, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    time = [s['time'] for s in result['steps']]
    for name in ['tip', 'core', 'remote', 'boundary']:
        axes[0, 0].plot(time, [s['regions'][name]['max_abs_shear'] for s in result['steps']], marker='.', ls='--' if name == 'boundary' else '-', label=name)
        axes[0, 1].plot(time, [s['regions'][name]['max_Y_over_K'] for s in result['steps']], marker='.', ls='--' if name == 'boundary' else '-', label=name)
        axes[1, 0].plot(time, [s['regions'][name]['max_native_release'] for s in result['steps']], marker='.', ls='--' if name == 'boundary' else '-', label=name)
    axes[0, 0].set(ylabel='max |shear stress| (Pa)', title='Physical stress')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(1, color='k', lw=.8, ls=':')
    axes[0, 1].set(ylabel='max Y/K', title='Local driving (not the phase criterion)')
    axes[1, 0].axhline(0, color='k', lw=.8, ls=':')
    axes[1, 0].set(ylabel='max (driving − fracture)/(K mass)', title='Native row release tendency')
    axes[1, 1].plot(time, [s['positive_w_integral'] for s in result['steps']], label='positive local w')
    axes[1, 1].plot(time, [s['negative_w_integral'] for s in result['steps']], label='negative local w')
    axes[1, 1].plot(time, [s['absolute_V_integral'] for s in result['steps']], ls='--', label='integral |V| ds')
    axes[1, 1].set(ylabel='integrated fault deformation (m²/s)', title='Signed deformation and section slip')
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel('time (s)')
        axis.grid(alpha=.2)
    fig.suptitle('Synthetic uniform-shear specimen — phase held fixed')
    fig.savefig(OUT / 'preflight.png', dpi=170)
    fig.savefig(OUT / 'preflight.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
