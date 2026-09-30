#!/usr/bin/env python3
"""Measure automatic completion against the exact qualified move-only baseline."""
import argparse
import csv
import json
import math
import pathlib
import xml.etree.ElementTree as ET

root = pathlib.Path(__file__).resolve().parent
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


reference = root
particle_columns = ['x', 'y', 'H', 'initial_theta', 'strengthening',
                    'tau_xx', 'tau_yy', 'tau_xy', 'integrator_x', 'integrator_y']
for mode, baseline, candidate, steps in (
        ('one-rank', reference / 'output-bp3-one', root / 'output-qualified-bp3-one', range(7)),
        ('two-rank', reference / 'output-bp3-two', root / 'output-qualified-bp3-two', range(7))):
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

nonexact = [label for label, result in results.items()
            if any(f['max_abs'] != 0 for f in result['fields'].values())]
(root / 'evidence/automatic-state-comparison.json').write_text(json.dumps(results, indent=2)+'\n')
print(f'{len(results)} field groups; {len(nonexact)} nonexact (details in automatic-state-comparison.json)')
