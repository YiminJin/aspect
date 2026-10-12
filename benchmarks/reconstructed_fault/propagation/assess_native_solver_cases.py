#!/usr/bin/env python3
"""Report scoped native entry checks without qualifying fresh evolution."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import native_endpoint as native
from assess_native_tangent_cost import records

E = Path(__file__).resolve().parent/'native_solver_evidence'


def assess(case):
    provenance = json.loads((native.E/f'{case}-run.json').read_text())
    assert provenance['returncode'] == 0
    output = Path(provenance['output'])
    text = (native.E/f'{case}.log').read_text()
    targets = {}
    label = None
    for line in text.splitlines():
        match = re.match(r'Fixed-entry diagnostic: ([^,]+),', line)
        if match:
            label = match.group(1)
        match = re.search(r'Endpoint phase target: initial=([^,]+), target=(\S+)', line)
        if match:
            targets[label] = float(match.group(2))*64.00000000008
    results = []
    with (output/'fixed-entries.csv').open() as file:
        for row in csv.DictReader(file):
            name = row.pop('case')
            row = {k:float(v) for k,v in row.items()}
            fields = np.genfromtxt(output/f'{name}-phase.csv',delimiter=',',names=True)
            assert min(fields['phase']-fields['lower']) >= 0 and max(fields['phase']) < 1
            assert row['healing'] == 0 and row['I_min'] > 0
            assert row['KKT'] <= targets[name]
            row.update(case=name,physical_KKT_target=targets[name],KKT_pass=True,
                       phase_equivalence_error=min(row['phase_error'],row['reflected_phase_error']),
                       global_reflection=row['reflected_phase_error']<row['phase_error'],
                       energy_decrease=row['initial_potential']-row['final_potential'],
                       tangential_phase_variation=float(max(np.ptp(fields['phase'][fields['y']==y]) for y in np.unique(fields['y']))))
            if 'saddle' in name:
                allowance=32*np.finfo(float).eps*max(64.,abs(row['initial_potential']))
                assert row['energy_decrease']>4*allowance
                row['energy_allowance_J_per_m']=allowance
                row['resolved_energy_departure']=True
            results.append(row)
    phase = records(text,'Endpoint cost:')
    assert len(phase)==len(results)
    for row,cost in zip(results,phase):
        row['cost']=cost
    result=dict(case=case,provenance=provenance,entries=results,
                intermediate_accuracy=records(text,'Endpoint intermediate accuracy:'),
                slow_directions=records(text,'Endpoint slow direction:'),
                history_published=False,
                scope='Scoped FE reference and immutable obstacle; numerical solver/escape diagnostic, not checkpoint conversion or fresh physical evolution. Mechanical residual norms are measured independently; production relative convergence remains unchanged.')
    (E/f'{case}-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['intermediate_accuracy','slow_directions']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case')
    assess(parser.parse_args().case)
