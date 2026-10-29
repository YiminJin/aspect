#!/usr/bin/env python3
"""Compact physical-slip comparisons; no constitutive changes or refitting."""
import csv
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from mature_gradient_strip import WORK, OUT, CrossFault, G, ELL


def read(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))


def column(rows, key):
    return np.array([float(r[key]) for r in rows])


def write(name, rows):
    with (OUT/name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def profile_error(a, b, key):
    ya, yb = column(a, 'y_m'), column(b, 'y_m')
    knots = np.unique(np.r_[ya, yb]); q, w = leggauss(3)
    x = ((knots[:-1, None]+knots[1:, None])/2+(knots[1:]-knots[:-1])[:, None]*q/2).ravel()
    weight = ((knots[1:]-knots[:-1])[:, None]*w/2).ravel()
    va = np.interp(x, ya, column(a, key)); vb = np.interp(x, yb, column(b, key))
    return float(np.sqrt(weight @ ((va-vb)**2)/(weight @ (vb*vb))))


def analyze():
    history, profiles = read('history.csv'), read('profiles.csv')
    cases = list(dict.fromkeys(r['case'] for r in history))
    final, matched, verification = [], [], []
    tau0 = .1*10000/np.sqrt(1+.1**2)
    for case in cases:
        rows = [r for r in history if r['case'] == case]
        last = rows[-1]; end = [r for r in profiles if r['case'] == case and float(r['time_s']) == 2.]
        start = next(i for i, r in enumerate(rows) if float(r['V_m_s']) > 1e-12)
        rate = column(end, 'w_engineering_s1'); mass = column(end, 'weight_m'); y = column(end, 'y_m')
        Q = column(end, 'Qxy_Pa'); V = float(last['V_m_s'])
        core = abs(y) <= ELL/2; support = abs(y) <= ELL
        p = dict(last)
        p.update(onset_lower_s=float(rows[start-1]['time_s']) if start else 0., onset_upper_s=float(rows[start]['time_s']),
                 extra_traction_fraction=float(last['traction_Pa'])/tau0-1.,
                 gradient_resistance_fraction=float(last['Rgradient_Pa'])/tau0,
                 Rcohesive_core_Pa=float(mass[core] @ (Q[core]*rate[core])/V),
                 Rcohesive_shoulders_Pa=float(mass[support & ~core] @ (Q[support & ~core]*rate[support & ~core])/V),
                 Rcohesive_outside_Pa=float(mass[~support] @ (Q[~support]*rate[~support])/V),
                 core_rate_fraction=float(mass[core] @ rate[core]/V))
        for k in ['work_J_m2', 'Db_W_m2', 'Dc_W_m2', 'Df_W_m2', 'BE_stress_J_m2', 'BE_cohesive_J_m2', 'BE_gradient_J_m2']:
            p['cumulative_'+k.replace('_W_m2', '_J_m2')] = float(sum(float(r[k])*(float(r['dt_s']) if '_W_' in k else 1.) for r in rows))
        final.append(p)
        delta = column(rows, 'delta_m')
        for target in [.00025, .0005, .001]:
            if target > max(delta):
                continue
            r = dict(case=case, delta_m=target, interpolation='Linear between accepted monotonic-slip states; no history transfer or constitutive reevaluation.')
            for key in ['time_s', 'traction_Pa', 'Rgradient_Pa', 'Rcohesive_Pa', 'Rfriction_Pa', 'V_m_s',
                        'rate_participation_width_m', 'K_participation_width_m', 'gradient_J_m2']:
                r[key] = float(np.interp(target, delta, column(rows, key)))
            r['gradient_resistance_fraction'] = r['Rgradient_Pa']/tau0
            matched.append(r)
        verification.append(dict(case=case, accepted_steps=len(rows), max_force_error=max(column(rows, 'force_error')),
                                 max_source_error=max(column(rows, 'source_error')), min_contact_Pa=min(column(rows, 'physical_contact_min_Pa')),
                                 max_ledger_error_J_m2=max(abs(column(rows, 'ledger_error_J_m2'))),
                                 max_gradient_BE_error_J_m2=max(abs(column(rows, 'gradient_identity_error_J_m2'))),
                                 max_traction_partition_error_Pa=max(abs(column(rows, 'traction_partition_error_Pa'))),
                                 max_compatibility_error_m=max(column(rows, 'compatibility_error_m'))))
    sensitivities = []
    for case, ref, kind in [('ls6-n129', 'ls6-n513', 'spatial'), ('ls6', 'ls6-n513', 'spatial'), ('ls6', 'ls6-half', 'temporal')]:
        for time in [1., 2.]:
            a = [r for r in profiles if r['case'] == case and float(r['time_s']) == time]
            b = [r for r in profiles if r['case'] == ref and float(r['time_s']) == time]
            ha = next(r for r in history if r['case'] == case and abs(float(r['time_s'])-time) < 1e-12)
            hb = next(r for r in history if r['case'] == ref and abs(float(r['time_s'])-time) < 1e-12)
            sensitivities.append(dict(case=case, reference=ref, kind=kind, time_s=time,
                                      traction_relative_difference=abs(float(ha['traction_Pa'])/float(hb['traction_Pa'])-1.),
                                      rate_profile_relative_L2=profile_error(a, b, 'w_engineering_s1'),
                                      K_profile_relative_L2=profile_error(a, b, 'k_engineering'),
                                      width_relative_difference=abs(float(ha['rate_participation_width_m'])/float(hb['rate_participation_width_m'])-1.)))
    frozen = []
    for length in [.001, .003, .006]:
        mesh = CrossFault(257, length)
        for shape in ['cohesive_chi', 'computed_final_K']:
            if shape == 'cohesive_chi':
                rho = mesh.h/mesh.I
            else:
                rows = [r for r in profiles if r['case'] == f'ls{round(1000*length)}' and float(r['time_s']) == 2.]
                k = column(rows, 'k_engineering'); rho = k/(mesh.mass @ k)
            J = float(rho @ (mesh.L @ rho))
            frozen.append(dict(length_m=length, shape=shape, normalization=float(mesh.mass @ rho),
                               J_per_m3=J, fixed_shape_Hgrad_Pa_per_m=G*length*length*J,
                               slip_at_5percent_fixed_shape_m=.05*tau0/(G*length*length*J),
                               meaning='Frozen-shape algebraic diagnostic, not an extrapolated evolving solution.'))
    write('final.csv', final); write('matched-slip.csv', matched); write('sensitivity.csv', sensitivities); write('frozen-shape.csv', frozen)
    (OUT/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    decision = dict(selected_length_m=.006, selected_gradient_compatibility=False, coupled_phase_task_started=False,
                    reason='Accumulated-gradient hardening in the intended fixed-width mature limit; 3/6 mm also fail the declared finite-slip screen.',
                    scope='1 mm may approximate a short friction window; this does not establish a non-hardening mature/RSF limit or select a new calibrated length.',
                    threshold_gradient_fraction=.05, slip_window_m=[.00025,.001],
                    all_candidate_lengths_uncalibrated=True)
    (OUT/'decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(sensitivities, indent=2), flush=True)


if __name__ == '__main__':
    analyze()
