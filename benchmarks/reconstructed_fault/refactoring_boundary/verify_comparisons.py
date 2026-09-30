#!/usr/bin/env python3
"""Checks for the behavior extension; the move-only comparison remains exact."""
import csv, json, math, re, runpy
from pathlib import Path
root = Path(__file__).resolve().parent
ns = runpy.run_path(str(root / 'compare_automatic.py'))
measure, ranked, csv_rows = (ns[x] for x in ('measure', 'ranked', 'csv_rows'))
results = ns['results']
initial = dict(results)
results.clear()

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def completion(directory, prefix):
    result = {}
    for path in directory.glob(prefix + '_rank*.csv'):
        for row in rows(path):
            key = int(row['id'])
            assert key not in result, (path, key, 'double correction')
            result[key] = {k: float(row[k]) for k in ('inside', 'outside', 'completed')}
    return result

# The final executable must retain exact default legacy behavior as well.
a = root / 'output-bp3-one'
b = root / 'output-qualified-legacy-one'
for step in range(7):
    assert csv_rows(a/f'profiles/fault_{step}.csv') == csv_rows(b/f'profiles/fault_{step}.csv')
    for prefix in ('audit_particles', 'audit_bulk'):
        assert ranked(a,prefix,step) == ranked(b,prefix,step)
assert csv_rows(a/'accepted_steps.csv') == csv_rows(b/'accepted_steps.csv')
assert ns['fault_arrays'](a) == ns['fault_arrays'](b)

corrections = {}
for mode in ('one', 'two'):
    legacy = completion(root / f'output-bp3-{mode}', 'ih_bottom_completion')
    auto = completion(root / f'output-qualified-bp3-{mode}', 'ih_automatic_completion')
    assert len(legacy) == 744
    delta = max(abs(row['outside'] - auto.get(k, {'outside': 0.})['outside']) for k, row in legacy.items())
    scale = max(row['outside'] for row in legacy.values())
    assert delta < 1e-10 * scale
    assert all(auto[k]['inside'] == legacy[k]['inside'] for k in auto)
    corrections[mode] = dict(profiles=len(legacy), affected=len(auto), max_abs=delta, max_relative_to_largest=delta/scale)
    for step in range(7):
        for prefix in ('work_weak', 'common_fe_weak'):
            h, a = csv_rows(root / f'output-bp3-{mode}/{prefix}_{step}.csv')
            _, b = csv_rows(root / f'output-qualified-bp3-{mode}/{prefix}_{step}.csv')
            measure(f'legacy-{mode}/{prefix}/{step}', h[1:], {r[0]:r[1:] for r in a}, {r[0]:r[1:] for r in b})

for name, left, right, steps in (
    ('bp3-mpi', 'output-qualified-bp3-one', 'output-qualified-bp3-two', range(7)),
    ('bp3-restart', 'output-qualified-bp3-one', 'output-qualified-bp3-split', (5,6))):
    a, b = root / left, root / right
    ids_a = ns['particle_identities'](a)
    # Restart audit only includes resumed steps. Coordinates are immutable in this fixture.
    ids_b = ids_a if name == 'bp3-restart' else ns['particle_identities'](b)
    for step in steps:
        h, x = csv_rows(a / f'profiles/fault_{step}.csv')
        _, y = csv_rows(b / f'profiles/fault_{step}.csv')
        measure(f'{name}/profile/{step}', h[2:], {(r[0],r[1]):r[2:] for r in x}, {(r[0],r[1]):r[2:] for r in y})
        measure(f'{name}/particles/{step}', ns['particle_columns'], ranked(a,'audit_particles',step,ids_a), ranked(b,'audit_particles',step,ids_b))
        x, y = ranked(a,'audit_bulk',step), ranked(b,'audit_bulk',step)
        assert x.keys() == y.keys()
        for component in sorted({v[0] for v in x.values()}):
            measure(f'{name}/bulk/{step}/{int(component)}', ['value'], {k:v[1:] for k,v in x.items() if v[0]==component}, {k:v[1:] for k,v in y.items() if v[0]==component})

small = {}
for label, left, right in (('oblique-mpi','oblique','oblique-two'), ('multiple-mpi','multiple','multiple-two'), ('reverse','oblique','reversed')):
    def nodal(name):
        return {(int(r['fault']), round(float(r['x']),12), round(float(r['y']),12)):float(r['I_h']) for r in rows(root/f'output-{name}/boundary_state.csv')}
    a, b = nodal(left), nodal(right)
    assert a.keys() == b.keys()
    error = max(abs(a[k]-b[k])/a[k] for k in a)
    assert error < 1e-10
    small[label] = error
for name in ('interior','interior_touch','perpendicular'):
    assert not completion(root/f'output-{name}', 'ih_automatic_completion')
contacts = {}
fd = {}
for name in ('interior','interior_touch','perpendicular','oblique','reversed','left_top','multiple','curved','oblique-two','multiple-two'):
    log = (root/f'evidence/qualified-{name}.log').read_text()
    contacts[name] = re.findall(r'Automatic boundary contact: (.*)', log)
    match = re.search(r'Boundary coupling verified: K=(.*), G=(.*), B=(.*)',log)
    assert match
    fd[name] = dict(zip(('K','G','B'), map(float,match.groups())))
    assert max(fd[name].values()) < 1e-8
for label, record in {**initial, **results}.items():
    for field, value in record['fields'].items():
        if label.endswith('/solver'):
            if field in ('time','dt','free','lower_active','newton_updates','krylov_iterations','min_alpha','fresh_linear_checks_passed'):
                assert value['max_abs'] == 0., (label,field)
            # Near-zero residuals are reported in absolute units, not used as a relative comparison.
        else:
            assert value['max_abs'] <= 1e-10 * max(1.,value['max_abs_reference']), (label,field,value)
summary = dict(final_legacy_exact=True, corrections=corrections, small_nodal_relative=small, contact_summaries=contacts, free_endpoint_derivatives=fd)
(root/'evidence/comparison-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'evidence/mpi-restart-mechanics.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(summary,indent=2))
