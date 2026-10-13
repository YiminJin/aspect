#!/usr/bin/env python3
"""Setup-inclusive cost and independent native/prototype action checks."""
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
    parser.add_argument('tag')
    args = parser.parse_args()
    record = json.loads((E / (args.tag + '-run.json')).read_text())
    assert record['returncode'] == 0
    out = Path(record['output'])
    rows = list(csv.DictReader((out / 'surface-schur.csv').open()))
    assert len(rows) == 32
    groups = {}
    for contact in (0, 1):
        group = {}
        for method in ('FGMRES', 'direct'):
            values = [r for r in rows if int(r['contact']) == contact and r['solver'] == method]
            assert len(values) == 8
            group[method] = {
                'batch_seconds_including_first_setup': sum(float(r['seconds']) for r in values),
                'linear_calls': sum(int(r['linear_calls']) for r in values),
                'Krylov': sum(int(r['Krylov']) for r in values),
                'max_bulk_residual': max(float(r['bulk_residual']) for r in values),
                'max_free_contact': max(float(r['free_contact']) for r in values),
                'max_action_relative_difference': max(float(r['action_error']) / float(r['action_scale'])
                    for r in values if int(r['direction']) % 4 != 3),
                'max_velocity_difference': max(float(r['velocity_error']) for r in values),
                'max_solver_pressure_difference': max(float(r['pressure_error']) for r in values),
                'max_rate_difference': max(float(r['rate_error']) for r in values),
                'max_RSS_MiB': max(float(r['rss_kib']) for r in values) / 1024,
                'peak_RSS_MiB': max(float(r['peak_kib']) for r in values) / 1024}
        groups[str(contact)] = group
    a = np.genfromtxt(out / 'surface-schur-actions.csv', delimiter=',', names=True, dtype=None, encoding='utf8')
    ref = dict(np.load(E / 'prototype-actions.npz'))
    prototype = {}
    for mode in (0, 1):
        selected = a[(a['contact'] == 0) & (a['direction'] == mode) & (a['solver'] == 'direct')]
        actual = np.zeros(513)
        for r in selected:
            actual[round(float(r['y'])*512)] += float(r['action'])
        target = ref['actions'][mode]
        err = float(max(abs(actual-target)))
        scale = float(max(abs(target)))
        prototype[str(mode)] = dict(maximum_error=err, scale=scale, relative=err/scale)
        assert err < 1e-6 * max(1., scale)
    log = (E / (args.tag + '.log')).read_text()
    bulk = [float(x) for x in re.findall(r'Endpoint bulk Schur factor: seconds=([^,]+)', log)]
    surface = re.findall(r'Endpoint surface Schur setup: seconds=([^,]+), free=([^,]+), condition=([^,]+)', log)
    velocity = [float(x) for x in re.findall(r'Endpoint collective Amesos velocity factor: seconds=([^,]+)', log)]
    assert len(bulk) == 1 and len(surface) == 2
    # Both current implementations retain the same velocity preconditioner setup.
    # Include it conservatively on both sides, in addition to all direct setup
    # already charged in the first batch call. Native assembly/history setup is
    # common; report total process cost separately.
    common = sum(velocity)
    f = groups['0']['FGMRES']['batch_seconds_including_first_setup'] + common
    d = groups['0']['direct']['batch_seconds_including_first_setup'] + common
    result = dict(tag=args.tag, completed=True, groups=groups, prototype=prototype,
                  bulk_setup_seconds=bulk, surface_setup=surface,
                  shared_velocity_factor_seconds=common,
                  setup_inclusive_reference_seconds=f, setup_inclusive_direct_seconds=d,
                  setup_inclusive_speedup=f/d,
                  process_seconds=record['wall_seconds'],
                  cost_gate_pass=bool(d < .8*f and groups['0']['direct']['max_action_relative_difference'] <= 2e-7))
    first, second = rows[0], rows[1]
    result['memory_increment'] = dict(before_RSS_MiB=float(first['rss_kib'])/1024,
        after_RSS_MiB=float(second['rss_kib'])/1024,
        retained_increment_MiB=(float(second['rss_kib'])-float(first['rss_kib']))/1024,
        pre_direct_peak_MiB=float(first['peak_kib'])/1024,
        post_direct_peak_MiB=float(second['peak_kib'])/1024,
        note='Paired runs share one process; later FGMRES rows retain direct factors. Peak includes initialization.')
    (E / 'batch-assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
