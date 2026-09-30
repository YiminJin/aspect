#!/usr/bin/env python3
"""Exact matched-rank comparison against the qualified R2b proposal-3 executable."""
import argparse
import csv
import json
import math
import pathlib
import xml.etree.ElementTree as ET

root = pathlib.Path(__file__).resolve().parent
reference_root = root.with_name('refactoring_r2b_cache')
results = {}


def measure(label, columns, left, right):
    assert left.keys() == right.keys(), (label, 'row identities differ')
    assert left, (label, 'empty comparison')
    assert all(len(row) == len(columns) for row in (*left.values(), *right.values())), label
    fields = {}
    for i, column in enumerate(columns):
        pairs = [(left[k][i], right[k][i]) for k in left]
        assert all(math.isfinite(a) and math.isfinite(b) for a, b in pairs), label
        fields[column] = dict(
            max_abs=max(abs(a-b) for a, b in pairs),
            max_relative_nonzero_reference=max((abs(a-b)/abs(a) for a, b in pairs if a != 0), default=0),
            max_abs_zero_reference=max((abs(b) for a, b in pairs if a == 0), default=0),
            max_abs_reference=max(abs(a) for a, b in pairs))
    results[label] = dict(rows=len(left), fields=fields)


def csv_rows(path):
    with path.open() as stream:
        reader = csv.reader(stream)
        header = next(reader)
        return header, [[float(v) for v in row] for row in reader]


def indexed_csv(path):
    header, rows = csv_rows(path)
    data = {row[0]: row[1:] for row in rows}
    assert len(data) == len(rows), path
    return header[1:], data


def ranked(directory, prefix, step, identities=None):
    data = {}
    paths = list(directory.glob(f'{prefix}_{step}_rank*.csv'))
    assert paths, (directory, prefix, step)
    for path in paths:
        _, rows = csv_rows(path)
        for row in rows:
            key = row[0] if identities is None else identities[row[0]]
            assert key not in data, (path, key)
            data[key] = row[1:]
    return data


def particle_identities(directory):
    return {key: tuple(row[:2]) for key, row in ranked(directory, 'audit_particles', 0).items()}


def fault_arrays(directory):
    tree = ET.parse(directory / 'reconstructed_faults/reconstructed_faults-00006.vtu')
    return {node.attrib['Name']: [float(x) for x in node.text.split()]
            for node in tree.findall('.//PointData/DataArray')}


particle_columns = ['x', 'y', 'H', 'initial_theta', 'strengthening',
                    'tau_xx', 'tau_yy', 'tau_xy', 'integrator_x', 'integrator_y']
for mode in ('legacy-one','legacy-two','automatic-one','automatic-two','automatic-split'):
    baseline = reference_root / f'output-candidate-bp3-{mode}'
    candidate = root / f'output-candidate-bp3-{mode}'
    steps = range(5,7) if mode.endswith("split") else range(7)
    for step in steps:
        columns, left_rows = csv_rows(baseline / f'profiles/fault_{step}.csv')
        _, right_rows = csv_rows(candidate / f'profiles/fault_{step}.csv')
        measure(f'{mode}/profile/{step}', columns[2:],
                {(r[0], r[1]): r[2:] for r in left_rows},
                {(r[0], r[1]): r[2:] for r in right_rows})
        measure(f'{mode}/particles/{step}', particle_columns,
                ranked(baseline, 'audit_particles', step),
                ranked(candidate, 'audit_particles', step))
        bulk_a = ranked(baseline, 'audit_bulk', step)
        bulk_b = ranked(candidate, 'audit_bulk', step)
        assert bulk_a.keys() == bulk_b.keys()
        assert all(bulk_a[k][0] == bulk_b[k][0] for k in bulk_a)
        for component in sorted({v[0] for v in bulk_a.values()}):
            measure(f'{mode}/bulk/{step}/component{int(component)}', ['value'],
                    {k: v[1:] for k, v in bulk_a.items() if v[0] == component},
                    {k: v[1:] for k, v in bulk_b.items() if v[0] == component})
    columns, left = indexed_csv(baseline / 'accepted_steps.csv')
    _, right = indexed_csv(candidate / 'accepted_steps.csv')
    measure(f'{mode}/solver', columns, {k: left[k] for k in steps}, {k: right[k] for k in steps})
    a, b = fault_arrays(baseline), fault_arrays(candidate)
    assert a.keys() == b.keys()
    for field in a:
        measure(f'{mode}/fault/{field}', [field],
                {i: [v] for i, v in enumerate(a[field])}, {i: [v] for i, v in enumerate(b[field])})


# Full CSV equality covers correction integrals, mechanical work and observer counters.
checks = {}
small = ()
import re
for name in (*small, 'bp3-legacy-one','bp3-legacy-two','bp3-automatic-one','bp3-automatic-two','bp3-automatic-split'):
    a, b = reference_root/f'output-candidate-{name}', root/f'output-candidate-{name}'
    for pattern in ('cache_counts_*.csv','ih_*completion_rank*.csv','work_weak_*.csv','common_fe_weak_*.csv','work_qp_*_rank*.csv'):
        left = {p.name:p.read_bytes() for p in a.glob(pattern)}
        right = {p.name:p.read_bytes() for p in b.glob(pattern)}
        checks[f'{name}/{pattern}'] = left == right
        if pattern == 'cache_counts_*.csv': assert left
    if name in small:
        checks[f'{name}/statistics'] = (a/'statistics').read_bytes()==(b/'statistics').read_bytes()
        label = 'Distributed I_h:' if name.startswith('warm') else 'I_h value cache:'
        checks[f'{name}/existing-assertions'] = all(label in (d/'log.txt').read_text() for d in (a,b))
    def work(variant):
        return [re.sub(r', geometry seconds=[^,]+', '', line)
                for line in ((reference_root if variant == 'reference' else root)/f'evidence/candidate-{name}.log').read_text().splitlines()
                if line.startswith('Cell I_h:')]
    left, right = work('reference'),work('candidate')
    checks[f'{name}/cell-work'] = left==right
    if 'cell' in name or name.startswith('bp3'):
        assert left, name
    if name.startswith('warm'):
        checks[f'{name}/traversal-reuse'] = any(int(re.search(r'reused profiles=(\d+)',line)[1])>0 for line in right)
    snapshots = [csv_rows(path)[1][0] for path in b.glob('cache_counts_*.csv')]
    if not name.startswith('warm'):
        # A resumed run is cold at step 5 and first reuses at step 6. Require
        # both paths over the case, not a hit before its first integration.
        assert any(row[1]>0 for row in snapshots) and all(row[2]>0 for row in snapshots), (name,'hit/miss not exercised')
nonexact = [label for label, result in results.items()
            if any(f['max_abs'] != 0 for f in result['fields'].values())]
(root/'evidence/state-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
(root/'evidence/lifecycle-comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
print(f'{len(results)} field groups; nonexact: {nonexact}')
print(f'{len(checks)} lifecycle/work checks; failures:',[k for k,v in checks.items() if not v])
assert not nonexact and all(checks.values()), 'Do not loosen the exact extraction criterion.'
