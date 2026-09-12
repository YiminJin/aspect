"""Offline persistent qualification of saved K4 width differences.

The width budget is the explicitly proposed K4.1b budget, not a replacement
for the original all-time K4.1 criterion. No interpolation between times,
reference evolution, or production simulation occurs here.
"""
import json

from reference_check import compare, dump
from temporal_width import DATA, OUT, load

MAIN=('V','q','C','Theta','slip','velocity_max','velocity_L2')


def persistent_start(times,ratios):
    """Find the first all-passing suffix, not the first isolated pass."""
    last_failure=max((i for i,value in enumerate(ratios) if value>1),default=-1)
    index=last_failure+1
    return None if index==len(times) else times[index]


def main():
    report=json.loads((OUT/'comparison.json').read_text())
    pair=report['new_pair']
    rows=pair['rows']
    times=[r['time_s'] for r in rows]
    qualification={}
    for key in rows[0]['metrics']:
        if key=='Ih':
            qualification[key]=dict(t_qual_s=0.,reason='Frozen profile integral; exactly zero timestep change, no new tolerance assigned.')
            continue
        ratios=[r['metrics'][key]['proposed_ratio'] for r in rows]
        start=persistent_start(times,ratios)
        failures=[r for r in rows if r['metrics'][key]['proposed_ratio']>1]
        qualification[key]=dict(t_qual_s=start,
            last_failure_time_s=failures[-1]['time_s'] if failures else None,
            max_ratio_after_start=max(r['metrics'][key]['proposed_ratio'] for r in rows if r['time_s']>=start) if start is not None else None)
    common=max(qualification[k]['t_qual_s'] for k in MAIN)
    suffix=[r for r in rows if r['time_s']>=common]
    intervals={}
    for key in rows[0]['metrics']:
        m=[r['metrics'][key] for r in suffix]
        intervals[key]=dict(signal_min=min(v['fine_width_difference'] for v in m),
            signal_max=max(v['fine_width_difference'] for v in m),
            max_temporal_change=max(v['matched_temporal_change'] for v in m),
            max_individual_ell0_change=max(v['ell0_temporal_change'] for v in m),
            max_individual_half_change=max(v['half_temporal_change'] for v in m),
            max_common_mode_change=max(v['common_mode_change'] for v in m),
            max_width_budget_ratio=max((v['proposed_ratio'] for v in m if v['proposed_ratio'] is not None),default=None))
    individual={}
    for name in ('ell0','half'):
        a,ua=load(name,.125); b,ub=load(name,.0625)
        comparison=compare(a,b,ua,ub)
        individual[name]=dict(all_time_max_ratio=comparison['max_ratio'],
            interval_max_ratio=max(v['ratio'] for r in comparison['rows'] if r['time_s']>=common for v in r['metrics'].values()))
    old=report['saved_pair']['rows']
    old_starts={key:persistent_start([r['time_s'] for r in old],[r['metrics'][key]['proposed_ratio'] for r in old]) for key in MAIN}
    result=dict(budget='Proposed K4.1b width-specific budget; coefficients and physical scales unchanged.',
        sampled_time_spacing_s=.125,qualification=qualification,
        common_interval_s=[common,6.],samples_in_common_interval=len(suffix),
        interval_signals=intervals,individual_trajectory_criterion=individual,
        older_pair_qualification=old_starts,
        frozen_input_qualification='H, phi, support, localization moments and Ih: t=0; no temporal evolution.',
        original_K41_all_time_pass=False,
        recommendation='Proceed only after review with explicitly post-transient K4.2, initialized at t=0 and compared on [4,6] s; do not certify the early transient.')
    dump(OUT/'post-transient.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
