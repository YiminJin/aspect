"""Assess saved bounded candidates, without launching or searching for a run."""
import argparse
import json
from pathlib import Path

import numpy as np
from reference import Reference, read_parameters


def initial_omission(report, directory):
    if 'omitted_fraction' in report['initial']:
        return report['initial']['omitted_fraction']
    # Older valid reports predate this diagnostic field. Integrate their saved
    # Q1 phi0 directly; never substitute the potentially different phi1 tail.
    model = Reference(read_parameters(Path(report['resolved_parameters'])), 8, 8)
    data = np.loadtxt(directory/'phase-0.csv', delimiter=',', skiprows=1)
    nodes, phi = data.T
    xi, weights = np.polynomial.legendre.leggauss(8)
    y = nodes[:-1, None]+np.diff(nodes)[:, None]*(xi+1)/2
    p = phi[:-1, None]+np.diff(phi)[:, None]*(xi+1)/2
    _, h = model.degradation(np.maximum(p, 0))
    integrals = np.diff(nodes)[:, None]*weights*h/2
    return float(np.sum(integrals*(abs(y)>model.support))/np.sum(integrals))


def assess(coarse, fine):
    a = json.loads((coarse/'report.json').read_text())
    b = json.loads((fine/'report.json').read_text())
    assert a['ramp_peak_m_per_s'] == b['ramp_peak_m_per_s']
    # Compare the entire sampled state, not just the possibly unchanged core.
    noise = dict(H=0., phi=0., Ih=0.)
    for k in range(3):
        for name, field in (('history', 'H'), ('phase', 'phi')):
            x = np.loadtxt(coarse/f'{name}-{k}.csv', delimiter=',', skiprows=1)
            y = np.loadtxt(fine/f'{name}-{k}.csv', delimiter=',', skiprows=1)
            noise[field] = max(noise[field], float(np.max(abs(x[:, 1]-np.interp(x[:, 0], y[:, 0], y[:, 1])))))
    states_a = [a['initial'], *a['steps']]
    states_b = [b['initial'], *b['steps']]
    noise['Ih'] = max(abs(x['Ih']-y['Ih']) for x, y in zip(states_a, states_b))
    def signal(report):
        return dict(H=report['steps'][0]['H_change_max'],
                    phi=report['steps'][1]['phi_change_max'],
                    Ih=abs(report['steps'][1]['Ih']-report['steps'][0]['Ih']))
    signal_a, signal_b = signal(a), signal(b)
    for key in noise:
        noise[key] = max(noise[key], abs(signal_a[key]-signal_b[key]))
    ratios = {key: signal_a[key]/max(noise[key], np.finfo(float).tiny) for key in noise}
    accuracy = noise['phi'] <= 1e-6
    for x, y in zip(states_a, states_b):
        for key in ('Ih', 'C', 'H_max'):
            accuracy &= abs(x[key]/y[key]-1) <= 1e-5
    for x, y in zip(a['steps'], b['steps']):
        accuracy &= abs(x['V']/y['V']-1) <= 1e-5
    # Initial upsilon has zero history correction, so its supported deficit is
    # the initial h omission. Reuse its saved profile if an older JSON lacks it.
    initial_a = initial_omission(a, coarse)
    initial_b = initial_omission(b, fine)
    initial_ok = all(r['initial']['phi_min'] >= -1e-4 and r['initial']['phi_max'] < .8 for r in (a, b))
    budget = a['smoke_budget_pass'] and b['smoke_budget_pass'] and max(initial_a, initial_b) <= 1e-4
    return dict(reference_kind=a['reference_kind'], ramp_peak_m_per_s=a['ramp_peak_m_per_s'],
                initial_omitted_fractions=[initial_a, initial_b],
                signal=signal_a, noise=noise, signal_to_noise=ratios,
                reference_accuracy_pass=bool(accuracy), support_budget_pass=bool(budget),
                feedback_pass=all(value >= 10 for value in ratios.values()),
                passed=bool(initial_ok and accuracy and budget and min(ratios.values()) >= 10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coarse', type=Path)
    parser.add_argument('fine', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.coarse, args.fine)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
