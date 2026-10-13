#!/usr/bin/env python3
"""Retained complete-growth / noncommitting-saddle equivalence and costs."""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE / 'direct_mechanical_evidence'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['growth', 'saddle'])
    parser.add_argument('tag')
    args = parser.parse_args()
    record = json.loads((E / (args.tag + '-run.json')).read_text())
    assert record['returncode'] == 0
    baseline = 'core-final-r1' if args.stage == 'growth' else 'saddle-final-r1'
    reference = json.loads((HERE / 'native_mpi_evidence' / (baseline + '-run.json')).read_text())
    out, old = Path(record['output']), Path(reference['output'])
    current_row = list(csv.DictReader((out / 'fixed-entries.csv').open()))[0]
    old_row = list(csv.DictReader((old / 'fixed-entries.csv').open()))[0]
    name = current_row['case']
    differences = {}
    for kind in ('phase', 'surface', 'stress'):
        a = np.genfromtxt(out / (name + '-' + kind + '.csv'), delimiter=',', names=True)
        b = np.genfromtxt(old / (name + '-' + kind + '.csv'), delimiter=',', names=True)
        keys = ['x', 'y'] if 'y' in a.dtype.names else ['x']
        a, b = np.sort(a, order=keys), np.sort(b, order=keys)
        assert len(a) == len(b)
        for key in keys:
            assert np.array_equal(a[key], b[key])
        differences[kind] = {field: float(max(abs(a[field]-b[field]))) for field in a.dtype.names if field not in keys}
    log = (E / (args.tag + '.log')).read_text()
    old_log = (HERE / 'native_mpi_evidence' / (baseline + '.log')).read_text()
    def costs(text):
        line = re.findall(r'Endpoint cost: ([^\n]+)', text)[-1]
        values = {k.strip(): float(v) for k, v in [x.split('=') for x in line.split(', ')]}
        matches = re.findall(r'Fixed-entry (?:accepted-step|whole-step) seconds=([^,\n]+)', text)
        values['entry_seconds'] = float(matches[0])
        published = re.findall(r'committed once; mechanical trials=(\d+)', text)
        logged = re.findall(r'phase iteration [^\n]+mechanical trials=(\d+)', text)
        values['mechanical_trials_final'] = int(published[-1]) if published else None
        values['mechanical_trials_last_KKT_log'] = int(logged[-1])
        values['direct_linear_calls'] = text.count('Endpoint direct solve:')
        values['bulk_setups'] = text.count('Endpoint bulk Schur factor:')
        values['surface_setups'] = text.count('Endpoint surface Schur setup:')
        return values
    cur_cost, ref_cost = costs(log), costs(old_log)
    row = {k: float(v) for k, v in current_row.items() if k != 'case'}
    energies = {k: row[k]-float(old_row[k]) for k in ('final_potential', 'bulk', 'cohesive', 'fracture')}
    curvature = [float(x) for x in re.findall(r'L2 Rayleigh quotient=([^\n]+)', log)]
    scaled_target = float(re.findall(r'Endpoint phase target: initial=[^,]+, target=([^\n]+)', log)[0])
    # Same effective fracture scale as the fixed reference input: Gc/(c0*ell).
    physical_target = scaled_target * (26.6666666667 / ((8./3.)*.15625))
    assert row['healing'] == 0 and row['increment'] > 1e-6 and row['I_min'] > 0
    assert row['KKT'] <= physical_target and min(row['phase_error'], row['reflected_phase_error']) < .001
    assert curvature[-1] > 0
    # Current retained state follows the same branch; no frame-by-frame alignment.
    assert differences['phase']['phase'] < 1e-6
    assert differences['surface']['q'] < 1.
    if args.stage == 'growth':
        assert 'Complete retained-entry publication verified' in log
    else:
        assert curvature[0] < 0 and 'no history published' in log
    result = dict(stage=args.stage, tag=args.tag, **{'pass': True}, row=row, differences=differences,
                  original_KKT_target=physical_target,
                  energy_differences=energies, cost=cur_cost, reference_cost=ref_cost,
                  retained_reference=baseline, curvature=curvature,
                  process_seconds=record['wall_seconds'], reference_process_seconds=reference['wall_seconds'],
                  entry_speedup=ref_cost['entry_seconds']/cur_cost['entry_seconds'],
                  timing_note='FGMRES complete-entry timing is retained and PMPI instrumented; fixed-state paired batch is the current controlled comparison.')
    (E / (args.stage + '-assessment.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
