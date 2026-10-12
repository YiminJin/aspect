#!/usr/bin/env python3
"""Summarize measured native work; never infer a physical stall from cost."""
import argparse
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
E=HERE/'native_endpoint_evidence'


def records(text, prefix):
    result=[]
    for line in text.splitlines():
        if prefix not in line:
            continue
        fields=dict(re.findall(r'([A-Za-z_ ]+)\s*=\s*([-+0-9.eE]+)',line.split(prefix,1)[1]))
        result.append({key.strip():float(value) for key,value in fields.items()})
    return result


def assess(case, output):
    text=(E/f'{case}.log').read_text()
    tangent=records(text,'Endpoint linear tangent:')
    profiles=records(text,'Fault linear profile:')
    phase=records(text,'Endpoint cost:')
    factors=records(text,'Endpoint sparse velocity factor:')
    timers={name.strip():dict(calls=int(calls),seconds=float(seconds))
            for name,calls,seconds in re.findall(
                r'\|\s*([^|\n]+)\|\s*(\d+)\s*\|\s*([0-9.eE+-]+)s\s*\|',text)}
    # The one-second observer gives independent, nonoverlapping physical-step
    # entry-to-commit times. A controlled precommit failure is included in the
    # same step's elapsed time; counters printed only on commit omit that trial.
    progress=E/f'{case}-progress.json'
    step_times=[]
    if progress.exists():
        events=json.loads(progress.read_text())['events'];entry=None;step=None
        for event in events:
            match=re.match(r'\*\*\* Timestep (\d+):',event['line'])
            if match:step=int(match.group(1));entry=event['seconds']
            elif entry is not None and 'Endpoint complete state committed' in event['line']:
                step_times.append(dict(step=step,entry_to_commit_seconds=event['seconds']-entry))
    result=dict(case=case,provenance=json.loads((E/f'{case}-run.json').read_text()),
                tangent_actions=tangent,phase_solves=phase,velocity_factors=factors,
                linear_profile_totals={key:sum(row.get(key,0.) for row in profiles)
                                       for key in set().union(*(row.keys() for row in profiles))
                                       if key not in ['step','newton']},
                tangent_calls=len(tangent),
                tangent_seconds=sum(row['seconds'] for row in tangent),
                tangent_linear_calls=sum(row['linear calls'] for row in tangent),
                reported_phase_costs=len(phase),
                mechanical_newton_iterations=len(re.findall('Relative nonlinear residuals',text)),
                native_timers=timers,
                step_entry_to_commit=step_times,
                scope='Exclusive linear-profile times are nested inside tangent/phase wall times; do not add them twice. Failed precommit attempts have no commit-cost line.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items()
                      if key not in ['tangent_actions','phase_solves','velocity_factors']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assess(args.case,args.output)
