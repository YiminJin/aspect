"""Reduce a step-68 row audit using only logged trial data and the native row map."""
import argparse
import csv
import json
import math
import re
from pathlib import Path


def read_tables(files):
    for path in sorted(files):
        with path.open() as stream:
            yield from csv.DictReader(stream)


def parse_passes(path):
    passes = {}
    iteration = None
    for line in path.read_text().splitlines():
        match = re.search(r'Newton (\d+):', line)
        if match:
            iteration = int(match[1])
        if 'Row audit pass=' in line:
            fields = line.split('pass=')[1].split()
            p = int(fields[0])
            row = int(fields[1].split('=')[1])
            value, gradient, lower, upper, old, new = map(float, fields[2:8])
            counts = list(map(int, fields[8:]))
            assert len(counts) == 9 and sum(counts) == 294912
            residual = value - max(lower, min(upper, value - gradient))
            passes[iteration, p] = dict(iteration=iteration, linear_pass=p, row=row,
                                        stage_value=value, gradient_per_mass=gradient,
                                        lower=lower, upper=upper, old_status=int(old),
                                        new_status=int(new), projected=residual,
                                        transition_counts=counts)
        if 'Tip linearized VI: pass=' in line:
            metrics = dict(re.findall(r'(\w+)=([\d.eE+-]+)', line))
            entry = passes[iteration, int(metrics['pass'])]
            assert abs(abs(entry['projected']) - float(metrics['phase'])) < 1e-14 * max(1, float(metrics['phase']))
            entry['feasible'] = bool(int(metrics['feasible']))
        if 'Tip bound linear solve: pass=' in line or 'Same mask pass=' in line:
            metrics = dict(re.findall(r'(\w+)=([\d.eE+-]+)', line))
            entry = passes[iteration, int(metrics.pop('pass'))]
            entry.update({k: float(v) for k, v in metrics.items()})
    return list(passes.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('scratch', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    passes = parse_passes(args.log)
    selected = {p['row'] for p in passes}
    locations = {int(r['index']): r for r in
                 read_tables(args.scratch.glob('phase-map-resume-r*.csv'))
                 if int(r['index']) in selected}
    for p in passes:
        r = locations[p['row']]
        p.update(cell=r['cell'], local_index=int(r['i']),
                 cell_center=[float(r['x']), float(r['y'])])
    # Leaders can belong to different ranks. Compare actual projected residuals,
    # including rows already chosen for the incoming or tangent diagnostics.
    leaders = {}
    for r in read_tables((args.scratch/'output').glob('tip-search-rows-r*.csv')):
        key = (int(r['iteration']), float(r['shift']), float(r['alpha']))
        if key not in leaders or abs(float(r['trial_projected'])) > abs(float(leaders[key]['trial_projected'])):
            leaders[key] = r
    iteration = None
    for line in args.log.read_text().splitlines():
        match = re.search(r'Newton (\d+):', line)
        if match:
            iteration = int(match[1])
        if 'Tip search trial:' in line:
            fields = dict(re.findall(r'(\w+)=([\d.eE+-]+)', line))
            key = (iteration, float(fields['shift']), float(fields['alpha']))
            if key in leaders:
                row = leaders[key]
                assert math.isclose(abs(float(row['trial_projected'])),
                                    float(fields['phase']), rel_tol=1e-12, abs_tol=1e-15)
                target = float(row['trial']) - float(row['trial_gradient_over_mass'])
                expected = float(row['trial']) - min(float(row['upper']), max(float(row['lower']), target))
                assert math.isclose(expected, float(row['trial_projected']), abs_tol=1e-14)
                row.update(mechanical_norm=float(fields['mechanics']),
                           merit=float(fields['merit']))
    result = dict(transition_order='row-major previous to new status, each ordered lower/free/upper; pass zero starts with free scratch mask',
                  linear_passes=passes, trial_leaders=[leaders[k] for k in sorted(leaders)])
    args.destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
