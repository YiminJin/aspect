"""One frozen-profile mechanical sensitivity check; not a coupled K3 reference.

Inject only the measured cumulative I_h-feedback error, with zero initial
offset. Reuse scalar mechanics and evolve q/C/Theta histories from the retained
reference initial state once; never reset them from subsequent ASPECT output.
No phase or H solve, support correction, or production execution is performed.
"""
import json
import math
from pathlib import Path
import time

from reference import Reference, read_parameters
from compare_normal_levels import CASES

HERE = Path(__file__).resolve().parent


def replay(model, reference, errors):
    old = reference['initial']['retained_histories'].copy()
    slip = 0.
    rows = []
    for expected, error in zip(reference['steps'], errors, strict=True):
        dt = expected['dt_s']
        response, _, _ = model.mechanics(old, dt, expected['U'], expected['Ih']+error)
        decay = math.exp(-response['V']*dt/model.Dc)
        theta = old['Theta']*decay + model.Dc/response['V']*(-math.expm1(-response['V']*dt/model.Dc))
        slip += dt*response['V']
        rows.append(dict(time_s=expected['time_s'], **response, Theta=theta, slip=slip))
        old = dict(q=response['q'], C=response['C'], Theta=theta, Ih=response['Ih'])
    return rows


def main():
    start = time.monotonic()
    reference = json.loads((HERE/'spatial0375_n128_f32-reference/report.json').read_text())
    assert reference['reference_kind'] == 'independent continuum initialization'
    model = Reference(read_parameters(HERE/CASES[128]/'parameters.prm'), 2048, 6, .00225)
    baseline = replay(model, reference, [0.]*8)
    # Verify the isolated replay reproduces the saved reference mechanics.
    for row, expected in zip(baseline, reference['steps'], strict=True):
        for key in ('V', 'q', 'C', 'Theta', 'Ih'):
            assert abs(row[key]-expected[key]) <= 1e-12*max(1., abs(expected[key]))
    results = {}
    for n, case in CASES.items():
        states = json.loads((HERE/f'{case}-comparison.json').read_text())['states']
        initial = states[0]
        errors = []
        for state, ref in zip(states[1:], reference['steps'], strict=True):
            assert state['time_s'] == ref['time_s'] and state['dt_s'] == ref['dt_s']
            errors.append((state['surface_means']['Ih']-initial['surface_means']['Ih'])
                          -(ref['Ih']-reference['initial']['Ih']))
        perturbed = replay(model, reference, errors)
        measures = {}
        for key in ('V', 'q', 'C', 'Theta', 'slip'):
            absolute = [abs(a[key]-b[key]) for a,b in zip(perturbed, baseline)]
            relative = [d/abs(b[key]) for d,b in zip(absolute, baseline)]
            measures[key] = dict(max_absolute=max(absolute), max_relative=max(relative),
                                 final_signed=perturbed[-1][key]-baseline[-1][key])
        results[n] = dict(Ih_feedback_error_m=errors,
                          max_error_over_total_Ih=max(abs(e)/r['Ih'] for e,r in zip(errors, reference['steps'])),
                          mechanical_impact=measures, trajectory=perturbed)
    report = dict(kind='frozen-profile, prescribed-Ih-feedback-error scalar sensitivity; not a new coupled reference',
                  limitation='No H/phase recomputation: this does not bound fully coupled future feedback or support normalization.',
                  baseline_reproduction_pass=True, results=results,
                  elapsed_seconds=time.monotonic()-start)
    (HERE/'ih-feedback-impact.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}, indent=2))
    for n,r in results.items():
        print(n, json.dumps({k:v for k,v in r.items() if k!='trajectory'}, indent=2))


if __name__ == '__main__':
    main()
