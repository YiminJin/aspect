#!/usr/bin/env python3
"""Compare every velocity quadrature association against the immutable reference."""
from pathlib import Path
import csv, json, math, sys
r = Path(__file__).resolve().parents[1]
def rows(case):
    return {(x['cell'], x['qp']): x for rank in range(2)
            for x in csv.DictReader((r/f'output-{case}/all_sources_rank{rank}.csv').open())}
filtered='--filter-derivative' in sys.argv
reference='endpoint' if filtered else 'qualified-endpoint'
current='filter' if filtered else 'endpoint'
summary = {}
for suffix in ['', '-reverse']:
    old, new = rows(reference+'-dump'+suffix), rows(current+'-dump'+suffix)
    assert old.keys() == new.keys()
    changed = [dict(new[k]) for k in old if old[k]['active'] != new[k]['active']]
    for x in changed:
        x['distance_to_nearest_endpoint_plane'] = min(
            abs((50000-float(x['x'])+float(x['y']))/math.sqrt(2)),
            abs((float(x['x'])-float(x['y'])+50000)/math.sqrt(2)))
    summary['reverse' if suffix else 'forward'] = dict(
        samples=len(old),
        previously_active_preserved_exactly=all(old[k] == new[k] for k in old if old[k]['active'] == '1'),
        new_associations=len(changed),
        positive_phase_new=sum(float(x['phi']) > 0 for x in changed), new=changed)
a, b = rows(current+'-dump'), rows(current+'-dump-reverse')
assert a.keys() == b.keys()
summary['corrected_order_admission_mismatches'] = sum(a[k]['active'] != b[k]['active'] for k in a)
# The retained three-anchor fixture has 36 segments in either native order.
summary['max_physical_shape_coordinate_difference'] = max(
    abs(float(a[k]['segment'])+float(a[k]['xi'])-(36-float(b[k]['segment'])-float(b[k]['xi'])))
    for k in a if a[k]['active'] == '1')
summary['max_reoriented_tangent_difference'] = max(
    abs(float(a[k][d])+float(b[k][d])) for k in a if a[k]['active'] == '1' for d in ['tx', 'ty'])
(r/('filter_derivative/results/full_association_comparison.json' if filtered else 'endpoint/results/full_association_comparison.json')).write_text(json.dumps(summary, indent=2)+'\n')
assert summary['corrected_order_admission_mismatches'] == 0
assert all(summary[s]['previously_active_preserved_exactly'] for s in ['forward', 'reverse'])
assert all(x['distance_to_nearest_endpoint_plane'] < 3e-12
           for s in ['forward', 'reverse'] for x in summary[s]['new'])
print('Full quadrature association comparison passed: 16875 points, both input orders.')
