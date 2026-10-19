#!/usr/bin/env python3
"""Reduce bounded stationary-DG runs; no simulation or server submission."""
import argparse,csv,json,re
from pathlib import Path
import numpy as np
from analyze_pf6_mature import read,stress_compare
from analyze_pf6_mature_pilot import merged,physical_states,replicated
from analyze_pf6_mature_resolution import surface


def trajectory(folder):
    record=json.loads((folder/'record.json').read_text())
    assert record['exit_code']==0,folder
    text=(folder/'stdout.log').read_text()
    timers={}
    for name,calls,value,unit in re.findall(r'\| ([^|]+?)\s*\|\s*(\d+)\s*\|\s*([\deE.+-]+)(ms|s)\s*\|',text):
        timers[name.strip()]={'calls':int(calls),'seconds':float(value)*(.001 if unit=='ms' else 1.)}
    profiles=[]
    for line in text.splitlines():
        if line.startswith('Fault linear profile:'):
            profiles.append({k:float(v) for k,v in re.findall(r'(\w+)=([-+\deE.]+)',line)})
    linear={k:sum(p.get(k,0.) for p in profiles) for k in set().union(*(p.keys() for p in profiles))-{'step','newton'}}
    counts=[]
    for line in text.splitlines():
        if line.startswith('Fault sampling counters:'):
            counts.append({k:float(v) for k,v in re.findall(r'(\w+)=([-+\deE.]+)',line)})
    linear['B_application_s']=linear.get('B_s',0.)+linear.get('B_sparse_s',0.)
    linear['G_application_s']=linear.get('G_s',0.)+linear.get('G_sparse_s',0.)
    run=folder/'run';accepted=read(run/'accepted_steps.csv')
    checks=[read(f) for f in (list(run.glob('rank-*-coupled-checks.csv')) or [run/'coupled-checks.csv'])]
    assert checks and all(np.all(c['publications']==1) for c in checks)
    assert all(len(np.unique(c['step']))==len(c['step']) for c in checks)
    assert np.all(accepted['fresh_linear_checks_passed']==1)
    assert all(np.all(c['error_fraction_of_allowance']<=1.) for c in checks)
    rss=[json.loads(f.read_text())['max_rss_KiB'] for f in folder.glob('resource-rank-*.json')]
    step_events=[e for e in record.get('timing_events',[]) if e['line'].startswith('*** Timestep')]
    terminal=next((e for e in record.get('timing_events',[]) if e['line'].startswith('Termination requested')),None)
    intervals=[]
    for i,event in enumerate(step_events):
        end=step_events[i+1] if i+1<len(step_events) else terminal
        if end:intervals.append(dict(step=int(re.search(r'Timestep (\d+)',event['line'])[1]),seconds=end['seconds']-event['seconds']))
    return dict(seconds=record['seconds'],timers=timers,linear_profile=linear,rank0_cumulative_counts=counts[-1] if counts else {},
        peak_RSS_MiB_per_rank=[v/1024 for v in rss],native_header_to_next_header_or_termination=intervals,
        accepted_step_sequence=[list(v) for v in zip(accepted['step'],accepted['time'],accepted['dt'])],
        krylov_iterations=accepted['krylov_iterations'].tolist(),newton_updates=accepted['newton_updates'].tolist(),
        max_normalized_nonlinear_residual=float(max(accepted['normalized_nonlinear_residual'])),
        max_surface_RMS_Pa=float(max(accepted['surface_RMS_Pa'])),
        max_publication_allowance_fraction=float(max(max(c['error_fraction_of_allowance']) for c in checks)),
        once_only_publication=True)


def comparison(first,second,step_first,step_second):
    result=dict(persistent=physical_states(first,step_first,second,step_second))
    a=merged(first,f'authoritative-stress-{step_first}.csv',('cell','qp'))
    b=merged(second,f'authoritative-stress-{step_second}.csv',('cell','qp'))
    result['stress']=stress_compare(a,b)
    result['surface']=surface(read(first/f'coupled-profile-{step_first}.csv'),read(second/f'coupled-profile-{step_second}.csv'),
        replicated(first,f'surface-weak-{step_first}.csv'),replicated(second,f'surface-weak-{step_second}.csv'))
    result['stepwise_surface']={}
    for file in first.glob('coupled-profile-*.csv'):
        other=second/file.name
        if not other.is_file():continue
        pa,pb=read(file),read(other)
        if not np.array_equal(pa['time_s'],pb['time_s']):continue
        step=file.stem.rsplit('-',1)[1]
        result['stepwise_surface'][step]=surface(pa,pb,replicated(first,f'surface-weak-{step}.csv'),replicated(second,f'surface-weak-{step}.csv'))
    # Phase/normalization are already checked above; keep all full tensor and
    # pressure diagnostics and report differences rather than accuracy errors.
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('work',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();runs=args.work/'runs'
    def folder(label):
        for prefix in ('qualified-','final-',''):
            candidate=runs/(prefix+label)
            if (candidate/'record.json').is_file():return candidate
        return runs/label
    result=dict(base='b2dfdebe43437d6028b06d4f9f862cc80a00cc71',strict_reference_match_passed=False,runs={},comparisons={})
    for label in ('reference-small','optimized-small','reference-larger','optimized-larger','ownership-r1','ownership-r4','restart-r4','candidate-rejection','outer-rejection','nonlinear-rejection','cutback-control','adaptive-control'):
        if (folder(label)/'record.json').is_file():result['runs'][label]=trajectory(folder(label))
    pairs=[('reference-small','baseline-small',2,2),('optimized-small','reference-small',2,2),('optimized-larger','reference-larger',2,2),
           ('ownership-r1','optimized-small',2,2),('ownership-r4','optimized-small',2,2),
           ('restart-r4','ownership-r4',2,2),('restart-r4','ownership-r1',2,2),('candidate-rejection','optimized-small',2,2),
           ('outer-rejection','cutback-control',3,3),('nonlinear-rejection','cutback-control',3,3)]
    for a,b,sa,sb in pairs:
        if (folder(a)/'record.json').is_file() and (folder(b)/'run'/f'coupled-profile-{sb}.csv').is_file():
            result['comparisons'][a+'__'+b]=comparison(folder(a)/'run',folder(b)/'run',sa,sb)
    for size in ('small','larger'):
        a=result['runs'].get('reference-'+size);b=result['runs'].get('optimized-'+size)
        if a and b:
            assert a['accepted_step_sequence']==b['accepted_step_sequence']
            result.setdefault('speedups',{})[size]=dict(total=a['seconds']/b['seconds'],
                linear_profile={k:a['linear_profile'][k]/v for k,v in b['linear_profile'].items() if k.endswith('_s') and v>0 and a['linear_profile'][k]>0},
                timers={k:a['timers'][k]['seconds']/v['seconds'] for k,v in b['timers'].items() if k in a['timers'] and v['seconds']>0})
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result.get('speedups',{}),indent=2))

if __name__=='__main__':main()
