#!/usr/bin/env python3
"""Compare first registered material planes at a common endpoint, without cutoffs."""
from observations import OUT, RUNS, read, np, json
import argparse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--coarse', default='coarse')
p.add_argument('--fine', default='fine')
p.add_argument('--time', type=float, required=True)
a = p.parse_args()
coarse, fine = [
    {r['section']: r for r in json.loads((OUT / (name + '-registrations.json')).read_text())}
    for name in (a.coarse, a.fine)
]
both = sorted(coarse.keys() & fine.keys())
rows = []
for k in both:
    c, f = coarse[k], fine[k]
    assert abs(c['x'] - f['x']) < 1e-12
    rows.append(dict(section=k, x=c['x'], coarse=c, fine=f,
                     angle_difference=f['angle'] - c['angle'],
                     slip_weight=.5 * (abs(c['final_slip']) * c['final_ds']
                                         + abs(f['final_slip']) * f['final_ds'])))
new = [r for r in rows if not (r['coarse']['initial_seed'] and r['fine']['initial_seed'])]
changed_time = [r for r in new if abs(r['coarse']['time']-r['fine']['time'])>1e-12]

def norms(selected):
    total = sum(r['slip_weight'] for r in selected)
    return dict(count=len(selected), integrated_absolute_slip_weight=total,
                maximum_angle_difference=max((abs(r['angle_difference']) for r in selected), default=0.),
                slip_weighted_angle_RMS=(float(np.sqrt(sum(r['slip_weight'] * r['angle_difference']**2
                                                         for r in selected) / total)) if total else None))

result = dict(time=a.time, convention='Angles in radians. No phase/rate/slip cutoff. '
              'RMS weights average |slip| ds on both grids; unpaired registrations remain explicit.',
              coarse_count=len(coarse), fine_count=len(fine), common=norms(rows),
              common_new=norms(new),
              common_new_right=norms([r for r in new if r['x']>6]),
              coarse_only=[coarse[k] for k in sorted(coarse.keys() - fine.keys())],
              fine_only=[fine[k] for k in sorted(fine.keys() - coarse.keys())],
              largest_new_angle_differences=sorted(new, key=lambda r: abs(r['angle_difference']), reverse=True)[:10],
              largest_new_slip_weights=sorted(new, key=lambda r: r['slip_weight'], reverse=True)[:10],
              largest_new_slip_weights_right=sorted([r for r in new if r['x']>6], key=lambda r: r['slip_weight'], reverse=True)[:10],
              all_common=rows)
changed=norms(changed_time)
changed['fraction_of_all_integrated_absolute_slip']=changed['integrated_absolute_slip_weight']/result['common']['integrated_absolute_slip_weight'] if result['common']['integrated_absolute_slip_weight'] else None
changed['fraction_of_new_integrated_absolute_slip']=changed['integrated_absolute_slip_weight']/result['common_new']['integrated_absolute_slip_weight'] if result['common_new']['integrated_absolute_slip_weight'] else None
changed['sections']=sorted(changed_time,key=lambda r:r['slip_weight'],reverse=True)
result['first_registration_time_changed']=changed
# Follow those same material labels in the already completed coarse history.
# This is explicitly separate if the fine history stops before .080; no fine
# value, common-time error, or extrapolation is inferred from this follow-up.
later=read(RUNS/'recover-free-turn-positive-half/output/tip-sections-0017.csv')
later_rows=[]
for r in changed_time:
    k=r['section'];assert abs(float(later.s[k])-r['x'])<1e-12
    later_rows.append(dict(section=k,x=r['x'],coarse_slip=float(later.slip[k]),
                           coarse_stage_rate=float(later.V[k]),
                           integrated_absolute_slip=float(abs(later.slip[k])*later.ds[k])))
total_later=float(sum(abs(later.slip)*later.ds))
result['retained_coarse_followup_of_changed_registration_labels']=dict(
    endpoint_time=.08,stage_time=.07875,matched_to_fine_endpoint=abs(a.time-.08)<1e-12,
    note='Existing coarse follow-up only; a partial fine history is not extrapolated.',
    fraction_of_coarse_integrated_absolute_slip=sum(r['integrated_absolute_slip'] for r in later_rows)/total_later if total_later else None,
    sections=sorted(later_rows,key=lambda r:r['integrated_absolute_slip'],reverse=True))
(OUT / 'C-registration-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print({k: result[k] for k in ['time', 'coarse_count', 'fine_count', 'common', 'common_new']})
