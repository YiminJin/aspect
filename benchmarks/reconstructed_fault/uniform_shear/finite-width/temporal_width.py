"""Bounded K4.1b: matched width differences, not replacement K4.1 acceptance.

Analyse saved .25/.125 first. The --add-level option runs exactly the approved
.0625 scalar pair, reusing the independent initialization, with no phase solve.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from reference_check import HERE, SCALES, Y, compare, dump, trajectory

DATA=HERE/'k41'
OUT=HERE/'k41b'
NAMES=('ell0','half')
KEYS=('V','q','C','Theta','slip','crack_integral','supported_integral','history_integral','Ih')


def load(name,dt):
    path=DATA/f'{name}-4096'
    return json.loads((path/f'dt{dt:g}.json').read_text()),np.load(path/f'dt{dt:g}.npz')['velocity']


def norm(field,kind):
    if kind=='velocity_L2': return float(np.sqrt(np.trapezoid(np.asarray(field)**2,Y)))
    return float(np.max(abs(field)))


def pair(coarse_dt,fine_dt):
    coarse={n:load(n,coarse_dt) for n in NAMES}
    fine={n:load(n,fine_dt) for n in NAMES}
    assert all([r['time_s'] for r in coarse[n][0]]==[r['time_s'] for r in coarse['ell0'][0]] for n in NAMES)
    lookup={r['time_s']:j for j,r in enumerate(fine['ell0'][0])}
    rows=[]
    profiles=[]
    for i,state in enumerate(coarse['ell0'][0]):
        t=state['time_s']; j=lookup[t]
        metrics={}
        for key in (*KEYS,'velocity_max','velocity_L2'):
            c=[coarse[n][1][i] if key.startswith('velocity') else coarse[n][0][i][key] for n in NAMES]
            f=[fine[n][1][j] if key.startswith('velocity') else fine[n][0][j][key] for n in NAMES]
            diff_c=c[1]-c[0]; diff_f=f[1]-f[0]
            error=[c[k]-f[k] for k in range(2)]
            matched=norm(diff_c-diff_f,key)
            individual=[norm(e,key) for e in error]
            common=norm(.5*(error[0]+error[1]),key)
            if key.startswith('velocity'):
                scale=SCALES['velocity']
            elif key in ('supported_integral','history_integral'):
                scale=SCALES['crack_integral']
            else: scale=SCALES.get(key)
            # PROPOSED only: apply the existing relative coefficient/physical
            # scale to the width signal. This never changes K4.1's readiness.
            proposed=None if scale is None else .25*(.002*norm(diff_f,key)+1e-5*scale)
            metrics[key]=dict(coarse_width_difference=norm(diff_c,key) if key.startswith('velocity') else diff_c,
                fine_width_difference=norm(diff_f,key) if key.startswith('velocity') else diff_f,
                matched_temporal_change=matched,ell0_temporal_change=individual[0],
                half_temporal_change=individual[1],common_mode_change=common,
                individual_change_sum=sum(individual),
                cancellation_ratio=matched/sum(individual) if sum(individual) else None,
                proposed_quarter_width_allowance=proposed,
                proposed_ratio=matched/proposed if proposed else None)
        profiles.append(fine['half'][1][j]-fine['ell0'][1][j])
        rows.append(dict(time_s=t,metrics=metrics))
    summary={key:dict(max_matched=max(r['metrics'][key]['matched_temporal_change'] for r in rows),
                     max_ell0=max(r['metrics'][key]['ell0_temporal_change'] for r in rows),
                     max_half=max(r['metrics'][key]['half_temporal_change'] for r in rows),
                     max_common_mode=max(r['metrics'][key]['common_mode_change'] for r in rows),
                     max_signal=max(abs(r['metrics'][key]['fine_width_difference']) for r in rows),
                     max_proposed_ratio=max((r['metrics'][key]['proposed_ratio'] for r in rows
                                             if r['metrics'][key]['proposed_ratio'] is not None),default=None))
             for key in rows[0]['metrics']}
    for s in summary.values():
        denom=s['max_ell0']+s['max_half']
        s['maxnorm_cancellation_ratio']=s['max_matched']/denom if denom else None
    np.savez(OUT/f'width-profiles-dt{fine_dt:g}.npz',y=Y,times=[r['time_s'] for r in rows],width_difference=profiles)
    return dict(coarse_dt=coarse_dt,fine_dt=fine_dt,rows=rows,summary=summary,
                original_criterion={n:compare(coarse[n][0],fine[n][0],coarse[n][1],fine[n][1])['max_ratio'] for n in NAMES})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--add-level',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    old=pair(.25,.125)
    dump(OUT/'saved-pair.json',old)
    if not args.add_level:
        print(json.dumps(old['summary'],indent=2)); return
    # This is a run-eligibility check, not a scientific acceptance threshold.
    assert all(old['summary'][k]['maxnorm_cancellation_ratio']<.25
               for k in ('V','q','C','Theta','slip','velocity_max','velocity_L2'))
    start=time.monotonic()
    for name in NAMES:
        path=DATA/f'{name}-4096'
        assert not (path/'dt0.0625.json').exists(), 'Never overwrite an earlier trajectory.'
        trajectory(path,.0625)
    seconds=time.monotonic()-start
    new=pair(.125,.0625)
    lookup={r['time_s']:r for r in new['rows']}
    contraction={}
    for key in old['summary']:
        previous=old['summary'][key]['max_matched']
        current=max(lookup[r['time_s']]['metrics'][key]['matched_temporal_change'] for r in old['rows'])
        contraction[key]=dict(previous_max=previous,new_max_same_times=current,
                              factor=previous/current if current else None,
                              pointwise=[dict(time_s=r['time_s'],
                                  old_change=r['metrics'][key]['matched_temporal_change'],
                                  new_change=lookup[r['time_s']]['metrics'][key]['matched_temporal_change']) for r in old['rows']])
    report=dict(saved_pair=old,new_pair=new,contraction_same_times=contraction,
                scalar_pair_seconds=seconds,original_K41_ready=False,
                proposed_criterion_only=True,
                frozen_quantities='H0, phi, Ih, support and localization moments do not evolve; temporal changes are exactly zero.')
    dump(OUT/'comparison.json',report)
    with (OUT/'all-times.csv').open('w') as stream:
        fields=['coarse_dt','fine_dt','time_s','observable',*next(iter(old['rows'][0]['metrics'].values()))]
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for p in (old,new):
            for row in p['rows']:
                for key,metrics in row['metrics'].items():
                    writer.writerow(dict(coarse_dt=p['coarse_dt'],fine_dt=p['fine_dt'],time_s=row['time_s'],observable=key,**metrics))
    print(json.dumps(dict(seconds=seconds,new=new['summary'],
                          contraction={k:{a:b for a,b in v.items() if a!='pointwise'} for k,v in contraction.items()},
                          original_criterion=new['original_criterion']),indent=2))


if __name__=='__main__': main()
