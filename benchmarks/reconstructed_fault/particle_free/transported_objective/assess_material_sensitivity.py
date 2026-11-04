#!/usr/bin/env python3
"""Summarize bounded material discretization/cost checks without fitting an order."""
import json
from pathlib import Path
import numpy as np
from assess_material_native import assess, norm3, OUT


def main():
    result = {}
    cases = ['matched-positive', 'matched-half-dt', 'matched-fine-final',
             'heterogeneous-positive', 'heterogeneous-half-dt-final']
    data = {}
    for name in cases:
        report, states = assess(name)
        rows = [r for r in report['steps'] if r['dt_s'] > 1e-10*report['command']['arguments']['dt']]
        data[name] = (report, states[rows[-1]['step']])
        result[name] = {'last_substantive_step': rows[-1], 'wall_s': report['command']['wall_s'],
                        'requested_dt': report['command']['arguments']['dt'],
                        'cells': 4**report['command']['arguments']['level']}
    comparisons = {}
    for a, b in [('matched-positive', 'matched-half-dt'), ('matched-positive', 'matched-fine-final'),
                 ('heterogeneous-positive', 'heterogeneous-half-dt-final')]:
        _, x = data[a]
        _, y = data[b]
        difference = norm3(x.stress(x.x)-y.stress(x.x))
        scale = norm3(y.stress(x.x))
        interior = np.max(np.abs(x.x), axis=1)<1.25
        relative = lambda mask: float(np.sqrt(np.sum(x.weights[mask]*difference[mask]**2)/np.sum(x.weights[mask]*scale[mask]**2)))
        comparisons[a+' vs '+b] = dict(final_physical_stress_relative_L2=relative(np.ones(len(x.x), dtype=bool)),
            interior_final_physical_stress_relative_L2=relative(interior),
            final_age_max_difference_s=float(np.max(np.abs(x.surface[:,3]-y.surface[:,3]))),
            final_memory_max_difference_Pa_m=float(np.max(np.abs(x.surface[:,4]-y.surface[:,4]))))
    output = dict(cases=result, endpoint_comparisons=comparisons,
      interpretation='Two resolutions do not establish asymptotic order. One-step characteristic error and final-state difference are distinct. Interior one-step error is not monotone under time halving. Open-boundary/profile-projection effects remain unresolved; no production accuracy claim.')
    (OUT/'material-sensitivity.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(comparisons))


if __name__ == '__main__':
    main()
