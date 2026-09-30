#!/usr/bin/env python3
"""Compare R2a with qualified R1 fields using the unchanged R1 error measures."""
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


reference = root.with_name('refactoring_r1')
particle_columns = ['x', 'y', 'H', 'initial_theta', 'strengthening',
                    'tau_xx', 'tau_yy', 'tau_xy', 'integrator_x', 'integrator_y']
for mode, baseline, candidate, steps in (
        ('one-rank', reference / 'output-bp3-one', root / 'output-bp3-one', range(7)),
        ('two-rank', reference / 'output-bp3-two', root / 'output-bp3-two', range(7)),
        ('baseline-restart', reference / 'output-bp3-one', root / 'output-bp3-split', (5, 6))):
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

# Compare existing lifecycle outputs and cold cell-cache work, omitting only time.
import re
checks = {}
def cell_work(path):
    return [re.sub(r', geometry seconds=[^,]+', '', line)
            for line in path.read_text().splitlines() if line.startswith('Cell I_h:')]

for name in ('ih-converged', 'ih-converged-two', 'ih-no-composition-converged', 'ih-cell', 'ih-cell-two'):
    a, b = reference / f'output-{name}', root / f'output-{name}'
    checks[f'{name}/statistics'] = (a / 'statistics').read_text() == (b / 'statistics').read_text()
    checks[f'{name}/verified'] = 'Distributed I_h:              verified' in (b / 'log.txt').read_text()
for name in ('ih-cell', 'ih-cell-two', 'bp3-one', 'bp3-two'):
    a = cell_work(reference / 'evidence' / f'{name}.log')
    b = cell_work(root / 'evidence' / f'{name}.log')
    checks[f'{name}/cell-work'] = bool(a) and a == b

def rollback_markers(path):
    return [line for line in path.read_text().splitlines()
            if 'Reconstructed-fault line search accepted after' in line
            or 'Stage-I rollback after an accepted Newton update: verified' in line]

repo = root.parents[2]
for ranks, suffix in ((1, ''), (2, '_mpi')):
    a = rollback_markers(repo / 'build-refactor-baseline/tests' /
                         f'output-phase_field_fault_stage_i_rollback{suffix}/screen-output.tmp')
    b = rollback_markers(root / 'evidence' / f'rollback-original-{ranks}.log')
    checks[f'rollback-original-{ranks}/accepted-and-restored'] = len(a) == 2 + ranks and a == b
for name in ('rollback-open-top', 'rollback-open-top-two'):
    a = rollback_markers(reference / 'evidence' / f'{name}.log')
    checks[f'{name}/accepted-and-restored'] = bool(a) and a == rollback_markers(root / 'evidence' / f'{name}.log')

ordinary = 'tests/output-convection_box_particles/statistics'
checks['ordinary/statistics'] = (repo / 'build-refactor-baseline' / ordinary).read_text() == (repo / 'build-refactor-r2a' / ordinary).read_text()
nonexact = [label for label, result in results.items()
            if any(f['max_abs'] != 0 for f in result['fields'].values())]
(root / 'evidence/state-comparison.json').write_text(json.dumps(results, indent=2)+'\n')
(root / 'evidence/lifecycle-comparison.json').write_text(json.dumps(checks, indent=2)+'\n')
print(f'{len(results)} field groups; nonexact: {nonexact}')
print('Lifecycle/solver/cache failures:', [name for name, passed in checks.items() if not passed])
assert not nonexact and all(checks.values()), 'Inspect comparisons; do not adjust tolerances.'
