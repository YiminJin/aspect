"""Reintegrate saved K3 reference profiles; no trajectory solves or ASPECT."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from diagnose_timestep_support import ProfileIntegral, required_width
from reference import Reference, read_parameters


here = Path(__file__).resolve().parent
root = here/'timestep-support'
model = Reference(read_parameters(here/'smoke/parameters.prm'), 2048, 6, .00225)
summary = []
all_rows = []
for case in ('dt050', 'dt025', 'dt0125'):
    report = json.loads((root/case/'report.json').read_text())
    profiles = np.load(root/case/'profiles.npz')
    old = report['initial']['retained_histories']
    rows = []
    for k, saved in enumerate(report['diagnostics']):
        phi = profiles['phi'][k]
        previous_phi = profiles['phi'][max(k-1, 0)]
        g, h, I = model.localization(phi)
        _, hp, _ = model.localization(previous_phi)
        beta = math.exp(-saved['dt_s']*model.G/model.eta)
        kappa = -model.eta*math.expm1(-saved['dt_s']*model.G/model.eta)
        factor = beta*old['C']/kappa
        V = saved['V']
        scale = max(abs(V), model.Vref)
        hc, ho = ProfileIntegral(model, phi), ProfileIntegral(model, previous_phi)

        def width_error(width):
            current, previous = hc.strip(width), ho.strip(width)
            return abs(V*current/I+factor*(current*old['Ih']/I-previous)-V)/scale

        # Search the complete physical strip here, including widths smaller
        # than current support. These are required widths, not proposed policy.
        row = dict(case=case, **saved)
        row['required_half_width_1e4_m'] = required_width(width_error, 0., 1e-4)
        row['required_half_width_5e5_m'] = required_width(width_error, 0., 5e-5)
        check_hc, check_ho = ProfileIntegral(model, phi, 20), ProfileIntegral(model, previous_phi, 20)
        differences = []
        for width in (model.support, row['required_half_width_1e4_m'], row['required_half_width_5e5_m'], .5):
            current, previous = check_hc.strip(width), check_ho.strip(width)
            error = abs(V*current/I+factor*(current*old['Ih']/I-previous)-V)/scale
            differences.append(abs(error-width_error(width)))
        row['width_quadrature_10_vs_20_error'] = max(differences)
        assert max(differences) < 1e-9
        row['candidate_minus_H_old_max_Pa'] = 0.
        row['candidate_over_H_old_max'] = 0.
        if k:
            # Audit why the existing maximum retained H; this does not update
            # the saved state or re-evaluate mechanics with committed history.
            a = saved['C']/g
            b = np.divide(beta*hp*old['C'], 1-g, out=np.zeros_like(g), where=g!=1)
            candidate = saved['dt_s']*(a-b)*(a+b)/(2*kappa)
            H_old = profiles['H'][k-1]
            row['candidate_minus_H_old_max_Pa'] = float(np.max((candidate-H_old)[model.admitted]))
            row['candidate_over_H_old_max'] = float(np.max((candidate/H_old)[model.admitted]))
            np.testing.assert_allclose(profiles['H'][k],
                np.where(model.admitted, np.maximum(H_old, candidate), H_old), rtol=0, atol=1e-8)
            old = report['steps'][k-1]
        rows.append(row)
    all_rows.extend(rows)
    summary.append(dict(case=case, dt=rows[1]['dt_s'], real_steps=len(rows)-1,
        max_omitted_h_fraction=max(r['omitted_h_fraction'] for r in rows),
        max_total_normalization_error=max(r['total_supported_normalization_error'] for r in rows),
        final_total_normalization_error=rows[-1]['total_supported_normalization_error'],
        max_required_half_width_1e4_m=max(r['required_half_width_1e4_m'] for r in rows),
        max_required_half_width_5e5_m=max(r['required_half_width_5e5_m'] for r in rows),
        max_history_omission_signed_normalized=max(r['omitted_history_signed_normalized'] for r in rows),
        max_candidate_over_H_old=max(r['candidate_over_H_old_max'] for r in rows[1:]),
        max_candidate_minus_H_old_Pa=max(r['candidate_minus_H_old_max_Pa'] for r in rows[1:]),
        max_partial_quadrature_error_difference=max(r['partial_quadrature_error_difference'] for r in rows),
        max_width_quadrature_10_vs_20_error=max(r['width_quadrature_10_vs_20_error'] for r in rows),
        elapsed_seconds=report['elapsed_seconds'], final=rows[-1]))
with (root/'all-steps.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps([{key: value for key, value in case.items() if key!='final'} for case in summary], indent=2))
