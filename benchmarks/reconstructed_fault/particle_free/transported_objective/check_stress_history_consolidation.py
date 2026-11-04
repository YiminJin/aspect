#!/usr/bin/env python3
"""Bounded source-refactoring equivalence, never physical requalification."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json
import re
from pathlib import Path
import numpy as np
from compare_bounded_profile import load, reduce, compare, tensor, quadrature, dot3, work_traction, ROOT, RUNS

OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/stress_history_consolidation'


def match(before, after):
    ca, ra, ma, a = load(before)
    cb, rb, mb, b = load(after)
    times_a = {float(r['time']): int(r['step']) for r in ra}
    times_b = {float(r['time']): int(r['step']) for r in rb}
    common = sorted(times_a.keys() & times_b.keys())
    assert common, (before, after)
    steps = []
    for time in common:
        x, y = a[times_a[time]], b[times_b[time]]
        bulk = float(abs(x.values-y.values).max())
        surface = float(abs(x.surface-y.surface).max())
        assert bulk < 5e-13 and surface < 5e-12, (before, after, time, bulk, surface)
        points, weights = quadrature(x.n)
        ta, tb = tensor(x, points), tensor(y, points)
        tensor_error = float(np.sqrt(weights@dot3(ta-tb, ta-tb)))
        wa, _ = work_traction(x)
        wb, _ = work_traction(y)
        steps.append(dict(time=time, bulk_Linf=bulk, surface_Linf=surface,
                          tensor_L2=tensor_error, work_traction_Linf=float(abs(wa-wb).max()),
                          exported_values_bitwise=bool(np.array_equal(x.values,y.values) and
                                                       np.array_equal(x.surface,y.surface))))
    accepted_a = {float(r['time']): r for r in ra}
    accepted_b = {float(r['time']): r for r in rb}
    # Continued restart can report counters only for its current process;
    # compare cumulative history/geometry counts at matching accepted states.
    counters = ['history_publications', 'geometry_publications']
    for time in common:
        for key in counters:
            assert accepted_a[time][key] == accepted_b[time][key], (time, key)
    iterations = lambda name: [int(v) for v in re.findall(r'Fault linear solve: iterations=(\d+)', (RUNS/name/'run.log').read_text())]
    return dict(before=before,after=after,steps=steps,
                before_iterations=iterations(before), after_iterations=iterations(after),
                before_wall_s=ca['wall_s'],after_wall_s=cb['wall_s'])


def main():
    target = OUT/'equivalence.json'
    if target.exists():
        raise SystemExit('Refusing to overwrite completed equivalence evidence.')
    results = [match('consolidate-before-'+case, 'consolidate-after-'+case)
               for case in ['dc', 'sc', 'dl', 'sl']]
    # Three runs have exactly matched duration; layered direct extends the
    # same accepted prefix to provide the uninterrupted restart comparator.
    for row in results:
        if row['before'].endswith(('dc','sc','sl')):
            assert row['before_iterations'] == row['after_iterations'], row['before']
    lifecycle = [match('consolidate-after-dl', 'consolidate-restart4'),
                 match('true-reject', 'consolidate-reject'),
                 match('true-cutback-control', 'consolidate-reject')]
    reductions = {}
    for name in ['consolidate-after-'+case for case in ['dc','sc','dl','sl']]+['consolidate-reject']:
        reduction, _ = reduce(name)
        reductions[name] = {k:v for k,v in reduction.items() if k not in ['input_sha256']}
    # Direct/separated physical differences are retained, not expected to vanish.
    paired = {}
    for law in ['c','l']:
        paired[law] = {}
        for stage in ['before','after']:
            dc, dr, dm, ds = load(f'consolidate-{stage}-d{law}')
            sc, sr, sm, ss = load(f'consolidate-{stage}-s{law}')
            step = next(int(r['step']) for r in sr if abs(float(r['time'])-.1)<1e-14)
            paired[law][stage] = compare(ds[step], ss[step])
    target.write_text(json.dumps(dict(representative=results,lifecycle=lifecycle,
                                     physical_pairs=paired,reductions=reductions),indent=2)+'\n')
    print(json.dumps(dict(representative_max_bulk=max(s['bulk_Linf'] for r in results for s in r['steps']),
                          representative_max_surface=max(s['surface_Linf'] for r in results for s in r['steps']),
                          lifecycle_max_bulk=max(s['bulk_Linf'] for r in lifecycle for s in r['steps']),
                          lifecycle_max_surface=max(s['surface_Linf'] for r in lifecycle for s in r['steps']))))


if __name__ == '__main__':
    main()
