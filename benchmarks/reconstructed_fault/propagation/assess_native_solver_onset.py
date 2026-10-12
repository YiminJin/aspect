#!/usr/bin/env python3
"""Check the economical schedule, recovery and separate native work channels."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E = HERE/'native_endpoint_evidence'
S = HERE/'native_solver_evidence'


def assess(case, live=False):
    comparison = json.loads((E/f'{case}-comparison.json').read_text())
    output = Path(comparison['provenance']['output'])
    metrics = np.atleast_1d(np.genfromtxt(output/'metrics.csv', delimiter=',', names=True))
    schedule = np.loadtxt(S/'onset-schedule.txt')
    times = metrics['time'][1:]
    assert np.array_equal(times, schedule[:len(times)])
    assert np.array_equal(metrics['dt'][1:], np.diff(np.r_[0., times]))
    assert comparison['initialization_phase_error'] == 0.
    assert all(row['passes'] for row in comparison['published_KKT'])
    assert comparison['maximum_healing'] == 0.
    phase_previous = None
    minimum_I = np.inf
    increments = []
    history_difference = None
    for row in metrics:
        step = int(row['step'])
        phase = np.genfromtxt(output/f'phase-{step}.csv', delimiter=',', names=True)['phase']
        surface = np.genfromtxt(output/f'surface-{step}.csv', delimiter=',', names=True)
        assert min(phase) >= 0. and max(phase) < 1.
        assert min(surface['I']) > 0.
        minimum_I = min(minimum_I, min(surface['I']))
        if phase_previous is not None:
            assert min(phase-phase_previous) >= 0.
            increments.append(float(max(phase-phase_previous)))
            if step == 54:
                history_difference = increments[-1]
        phase_previous = phase
    recovery = None
    if (output/'complete-step-recovery.txt').exists():
        with (output/'complete-step-properties.csv').open() as file:
            properties = list(csv.DictReader(file))
        assert all(row['same_bytes'] == '1' for row in properties)
        assert history_difference is not None and history_difference > 0.
        assert sum(metrics['step'] == 55) == 1
        recovery = dict(surface_components=len(properties),
                        unchanged_nan_sentinels=sum(row['before'].lower() == 'nan' for row in properties),
                        differences=0, distinct_previous_phase_max_difference=history_difference,
                        retry_accepted=True, scope='Forced complete outer solve before publication; no postpublication or restart claim.')
    if not live:
        assert comparison['completed'] and comparison['resolved_core_growth']
        assert comparison['accepted_interval_targets_pass'] and len(times) == len(schedule)
        assert recovery is not None

    ledger = np.atleast_1d(np.genfromtxt(output/'path-energy.csv', delimiter=',', names=True))
    assert np.array_equal(ledger['time'], times)
    prototype = json.loads((Path(comparison['prototype_directory'])/'energy.json').read_text())
    reference = [row for row in prototype if row['time'] in times]
    assert np.array_equal([row['time'] for row in reference], times)
    pairs = dict(bulk_rate_work='bulk_work', lifting_work='lifting_work',
                 boundary_reaction_work='external_work', bulk_viscous='bulk_viscous',
                 cohesive_viscous='cohesive_viscous', friction='friction',
                 mechanical_residual_work='mechanical_power',
                 friction_residual_work='friction_residual_power', phase_residual_work='phase_power',
                 profile_bulk_work='bulk_profile_power', profile_cohesive_work='normalization_power',
                 absolute_mechanical_work='mechanical_power_absolute',
                 absolute_friction_work='friction_residual_power_absolute',
                 absolute_phase_work='phase_power_absolute', external_remainder='external_remainder')
    energy = {}
    for key, refkey in pairs.items():
        # Native remainder is storage-work+loss; prototype uses the opposite sign.
        values = np.array([row[refkey] for row in reference])
        if key == 'external_remainder':
            values = -values
        difference = ledger[key]-values
        energy[key] = dict(native_signed=float(sum(ledger[key])),
                           native_sum_absolute_step=float(sum(abs(ledger[key]))),
                           native_maximum_step=float(max(abs(ledger[key]))),
                           prototype_signed=float(sum(values)),
                           prototype_sum_absolute_step=float(sum(abs(values))),
                           difference_signed=float(sum(difference)),
                           sum_absolute_step_difference=float(sum(abs(difference))),
                           maximum_step_difference=float(max(abs(difference))))
    result = dict(case=case, completed=not live and comparison['completed'],
                  accepted_steps=len(times), exact_schedule=True, minimum_normalization=float(minimum_I),
                  maximum_phase_increment=max(increments), recovery=recovery, energy=energy,
                  identity_remainder=dict(signed=float(sum(ledger['identity_remainder'])),
                      sum_absolute_step=float(sum(abs(ledger['identity_remainder']))),
                      maximum_step=float(max(abs(ledger['identity_remainder'])))),
                  scope='Exact matching schedule and unaligned fields. Absolute power channels are separate from sums of absolute signed step work. Stress comparisons are limited to exported off-quadrature states; no full trajectory stability or energy-closure claim.')
    (S/f'{case}-qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(case, ledger, reference)
    print(json.dumps(result))


def plot(case, ledger, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for key, refkey, label, color in zip(
            ['mechanical_residual_work', 'friction_residual_work', 'phase_residual_work'],
            ['mechanical_power', 'friction_residual_power', 'phase_power'],
            ['mechanical', 'friction', 'phase'], colors):
        axes[0, 0].plot(ledger['time'], np.cumsum(ledger[key]), color=color, label=label)
        axes[0, 0].plot(ledger['time'], np.cumsum([row[refkey] for row in reference]), '--', color=color)
    for key, label, color in zip(['absolute_mechanical_work', 'absolute_friction_work', 'absolute_phase_work'],
                                ['mechanical', 'friction', 'phase'], colors):
        axes[0, 1].plot(ledger['time'], np.cumsum(ledger[key]), color=color, label=label)
    axes[1, 0].plot(ledger['time'], np.cumsum(ledger['profile_bulk_work']), label='bulk profile work')
    axes[1, 0].plot(ledger['time'], np.cumsum(ledger['profile_cohesive_work']), label='normalization work')
    axes[1, 0].plot(ledger['time'], np.cumsum(ledger['storage_change']), label='total storage change')
    axes[1, 1].plot(ledger['time'], np.cumsum(ledger['external_remainder']), label='external remainder')
    axes[1, 1].plot(ledger['time'], np.cumsum(ledger['identity_remainder']), label='identity remainder')
    for ax, title in zip(axes.flat, ['Signed residual work (dashed: prototype)',
                                     'Integral of absolute residual power',
                                     'Profile work and storage', 'Separate energy remainders']):
        ax.set(title=title, xlabel='t (s)', ylabel='J/m')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(S/f'{case}-energy.png', dpi=150)
    fig.savefig(S/f'{case}-energy.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    assess(args.case, args.live)
