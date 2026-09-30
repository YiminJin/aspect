#!/usr/bin/env python3
"""Measure saved R1 fields; require exact same-rank restart, no fitted tolerances."""
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


baseline = root / 'output-bp3-one'
particle_columns = ['x', 'y', 'H', 'initial_theta', 'strengthening',
                    'tau_xx', 'tau_yy', 'tau_xy', 'integrator_x', 'integrator_y']
for mode, candidate, steps in (
        ('restart', root / 'output-bp3-split', (5, 6)),
        ('cross-rank', root / 'output-bp3-two', range(7))):
    for step in steps:
        # Profiles begin with fault,node: use both columns as the identity.
        columns, left_rows = csv_rows(baseline / f'profiles/fault_{step}.csv')
        _, right_rows = csv_rows(candidate / f'profiles/fault_{step}.csv')
        left = {(r[0], r[1]): r[2:] for r in left_rows}
        right = {(r[0], r[1]): r[2:] for r in right_rows}
        measure(f'{mode}/profile/{step}', columns[2:], left, right)
        ids_a = particle_identities(baseline) if mode == 'cross-rank' else None
        ids_b = particle_identities(candidate) if mode == 'cross-rank' else None
        measure(f'{mode}/particles/{step}', particle_columns,
                ranked(baseline, 'audit_particles', step, ids_a),
                ranked(candidate, 'audit_particles', step, ids_b))
        if mode == 'restart':
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

ordinary = root.parents[2] / 'build-refactor-baseline/tests/output-checkpoint_03_particles'
def ordinary_particles(path):
    rows = [[float(x) for x in line.split()] for line in path.read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    return {i: row for i, row in enumerate(rows)}
measure('ordinary/restart-particles', [f'column{i}' for i in range(10)],
        ordinary_particles(ordinary / 'particles-00009.0000.gnuplot1'),
        ordinary_particles(ordinary / 'particles-00009.0000.gnuplot2'))

out = root / 'evidence/state-comparison.json'
out.write_text(json.dumps(results, indent=2)+'\n')
nonexact = [label for label, result in results.items()
            if label.startswith(('restart/', 'ordinary/'))
            and any(f['max_abs'] != 0 for f in result['fields'].values())]
print(f'{len(results)} field groups compared; exact-restart differences: {nonexact}')
print(out)
assert not nonexact, 'Same-rank reference restart is not exact; inspect the recorded field errors.'
