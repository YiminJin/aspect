"""Reference-only K3 timestep/support audit; never launches ASPECT.

Width queries integrate frozen profiles. They do not widen the admitted H
update, change C0, or recompute a trajectory with a different support policy.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq

from reference import Reference, read_parameters


class ProfileIntegral:
    """Partial-element Gauss integral of h for an existing P1 profile."""

    def __init__(self, model, phi, order=10):
        self.model, self.phi = model, phi
        self.points, self.weights = np.polynomial.legendre.leggauss(order)
        values = (phi[:-1, None]*(1-self.points)/2
                  + phi[1:, None]*(1+self.points)/2)
        _, h = model.degradation(np.maximum(values, 0))
        cell = model.dy*np.sum(h*self.weights, axis=1)/2
        self.prefix = np.r_[0., np.cumsum(cell)]

    def primitive(self, y):
        model = self.model
        cell = min(np.searchsorted(model.y, y, side='right')-1, len(model.dy)-1)
        length = y-model.y[cell]
        xi = length*(1+self.points)/(2*model.dy[cell])
        phi = (1-xi)*self.phi[cell]+xi*self.phi[cell+1]
        _, h = model.degradation(np.maximum(phi, 0))
        return self.prefix[cell]+length*np.dot(self.weights, h)/2

    def strip(self, width):
        return self.primitive(width)-self.primitive(-width)


def required_width(error, support, target):
    # Signed history need not make error monotone. Find the last sampled
    # failing interval, then its crossing; retain a wider-width envelope check.
    widths = np.linspace(support, .5, 513)
    errors = np.array([error(w) for w in widths])
    failing = np.flatnonzero(errors > target)
    if not len(failing):
        return support
    assert failing[-1] < len(widths)-1, 'Full-domain identity failed'
    i = failing[-1]
    root = brentq(lambda w: error(w)-target, widths[i], widths[i+1], xtol=1e-12)
    assert max(error(w) for w in np.linspace(root+1e-11, .5, 129)) <= target+1e-12
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt', type=float, choices=(.5, .375, .25, .125), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    here = Path(__file__).resolve().parent
    parameters = here/'smoke/parameters.prm'
    model = Reference(read_parameters(parameters), 2048, 6, .00225)
    sequence = [dict(time=k*args.dt, dt=args.dt,
                     U=1e-4+(.00225-1e-4)*min(k*args.dt/2, 1))
                for k in range(1, round(3/args.dt)+1)]
    report, snapshots = model.run(sequence)
    report['time_sequence_source'] = 'prescribed reference-only common timestep cap through 3 s'
    report['first_timestep']['maximum_s'] = args.dt
    # Legacy names in the two-step preflight are not meaningful for this audit.
    report['feedback']['final_phi_minus_phi0_max'] = report['feedback'].pop('phi2_minus_phi0_max')
    report['feedback']['final_phi_residual_with_H0'] = report['feedback'].pop('phi2_residual_with_H0')
    report.pop('smoke_budget_pass')
    report['parameters_sha256'] = hashlib.sha256(parameters.read_bytes()).hexdigest()
    report['reference_sha256'] = hashlib.sha256((here/'reference.py').read_bytes()).hexdigest()
    report['initialization_interval_s'] = 2.
    report['width_interpretation'] = ('Frozen-trajectory mechanical/history integration diagnostic only; '
        'C0 projection, H update admission, full Ih and all histories unchanged. '
        'Widths are restricted to extensions of current support, with a sampled wider-width envelope check.')
    rows = []
    old = report['initial']['retained_histories']
    for k, (phi, H) in enumerate(snapshots):
        previous_phi, previous_H = snapshots[max(0, k-1)]
        response = report['initial']['evaluated'] if k == 0 else report['steps'][k-1]
        dt = 2. if k == 0 else response['dt_s']
        beta = math.exp(-dt*model.G/model.eta)
        kappa = -model.eta*math.expm1(-dt*model.G/model.eta)
        factor = beta*old['C']/kappa
        _, h, I = model.localization(phi)
        _, hp, _ = model.localization(previous_phi)
        V = response['V']
        scale = max(abs(V), model.Vref)
        instantaneous = V*h/I
        history = factor*(h*old['Ih']/I-hp)
        omitted_inst = float(np.sum(model.w*instantaneous*(~model.admitted)))
        omitted_hist = float(np.sum(model.w*history*(~model.admitted)))
        error = abs(float(np.sum(model.w*(instantaneous+history)*model.admitted))-V)/scale
        current_integral = ProfileIntegral(model, phi)
        previous_integral = ProfileIntegral(model, previous_phi)

        def width_error(width):
            hc = current_integral.strip(width)
            ho = previous_integral.strip(width)
            return abs(V*hc/I+factor*(hc*old['Ih']/I-ho)-V)/scale

        # Independent partial-cell quadrature must reproduce the original
        # fixed-support/full-domain integrals before its widths are interpreted.
        integration_difference = abs(width_error(model.support)-error)
        assert integration_difference < 1e-9
        assert width_error(.5) < 1e-9
        if k:
            assert abs(error-response['supported_slip_normalization_error']) < 1e-12
            assert np.min(H-previous_H) >= 0
        row = dict(step=k, time_s=0. if k == 0 else response['time_s'], dt_s=dt,
            omitted_h_fraction=float(np.sum(model.w*h*(~model.admitted))/I),
            omitted_instantaneous_m_per_s=omitted_inst,
            omitted_history_signed_m_per_s=omitted_hist,
            omitted_history_signed_normalized=omitted_hist/scale,
            omitted_total_signed_normalized=(omitted_inst+omitted_hist)/scale,
            supported_instantaneous_m_per_s=float(np.sum(model.w*instantaneous*model.admitted)),
            supported_history_signed_m_per_s=float(np.sum(model.w*history*model.admitted)),
            total_supported_normalization_error=error,
            full_normalization_error=abs(float(np.sum(model.w*(instantaneous+history)))-V)/scale,
            full_history_integral_m_per_s=float(np.sum(model.w*history)),
            H_increment_max_Pa=float(np.max(H-previous_H)),
            H_cumulative_increment_max_Pa=float(np.max(H-snapshots[0][1])),
            phi_increment_max=float(np.max(abs(phi-previous_phi))),
            phi_cumulative_increment_max=float(np.max(abs(phi-snapshots[0][0]))),
            Ih=I, Ih_increment=0. if k == 0 else I-old['Ih'],
            Ih_cumulative_increment=I-report['initial']['Ih'],
            phi_min=float(np.min(phi)), phi_max=float(np.max(phi)),
            V=V, Theta=old['Theta'] if k == 0 else response['Theta'],
            C=old['C'] if k == 0 else response['C'],
            q_evaluated=response['q'], q_retained=old['q'] if k == 0 else response['q'],
            mechanical_residual=response['residual'], F_at_Vmin=response['F_at_Vmin'],
            phase_residual=report['initial']['reference_phase_residual'] if k == 0
                else response['phase_residual_history'][-1],
            admissible=bool(np.min(phi)>=-1e-4 and np.max(phi)<.8 and V>model.Vmin),
            half_width_1e4_m=required_width(width_error, model.support, 1e-4),
            half_width_5e5_m=required_width(width_error, model.support, 5e-5),
            partial_quadrature_error_difference=integration_difference)
        rows.append(row)
        if k:
            old = response
    report['diagnostics'] = rows
    report['fixed_support_all_states_pass'] = all(r['total_supported_normalization_error']<=1e-4
        and r['omitted_h_fraction']<=1e-4 and r['admissible'] for r in rows)
    report['elapsed_seconds'] = time.monotonic()-start
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'steps.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output/'profiles.npz', y=model.y, qy=model.qy,
                        phi=np.array([p for p, _ in snapshots]), H=np.array([h for _, h in snapshots]))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(dt=args.dt, steps=len(sequence), support=model.support,
        all_states_pass=report['fixed_support_all_states_pass'],
        max_error=max(r['total_supported_normalization_error'] for r in rows),
        final_error=rows[-1]['total_supported_normalization_error'],
        required_half_width_1e4=max(r['half_width_1e4_m'] for r in rows),
        required_half_width_5e5=max(r['half_width_5e5_m'] for r in rows),
        elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
