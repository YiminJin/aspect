"""Exclusive condensed timing, fresh-check and accepted-trajectory evidence."""
import argparse
import json
from pathlib import Path
import re
from compare import compare


def summarize(path):
    log=path.with_suffix('.log').read_text()
    records=[]
    for line in re.findall(r'Fault linear profile: (.*)',log):
        row={key:float(value) for key,value in (part.split('=') for part in line.split(', '))}
        assert abs(sum(value for key,value in row.items() if key.endswith('_s'))-row['elapsed'])<1e-8
        records.append(row)
    assert records, 'No timed coupled linearizations'
    totals={key:sum(row[key] for row in records) for key in records[0] if key not in ('step','newton')}
    attempts=[tuple(map(float,row)) for row in re.findall(
        r'Fault linear solve: iterations=([\d.eE+\-]+), estimated=([\d.eE+\-]+), fresh=([\d.eE+\-]+), target=([\d.eE+\-]+)',log)]
    # A failed fresh check is an attempted direction, not an accepted return.
    # Iteration counts are cumulative across residual-replacement restarts.
    # Preserve every failed attempt and require a subsequent genuine return.
    checks=[]
    rejected=[]
    pending=False
    previous_iteration=0
    previous_target=None
    for row in attempts:
        if pending:
            assert row[0]>previous_iteration, 'Fresh failure was not continued within its solve'
            assert row[3]==previous_target, 'Fresh restart changed its requested tolerance'
        if row[2]<=row[3]:
            checks.append(row)
            pending=False
        else:
            rejected.append(row)
            pending=True
        previous_iteration=row[0]
        previous_target=row[3]
    assert checks and not pending, 'No genuinely converged final linear direction'
    resources=json.loads(path.with_suffix('.resources.json').read_text())
    assert resources['status']==0, 'Trajectory failed'
    return dict(resources=resources,linearizations=records,totals=totals,
                fresh_checks=len(checks),linear_iterations=sum(row[0] for row in checks),
                accepted_linear_checks=checks, rejected_fresh_attempts=rejected,
                fresh_attempt_count=len(attempts),
                max_fresh_over_target=max(row[2]/row[3] for row in checks),
                timed_fraction_of_wall=totals['elapsed']/resources['wall_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result={'runs':{str(p):summarize(p) for p in args.directories}}
    if len(args.directories)==2:
        result['comparison']=compare(*args.directories)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({p:{k:v for k,v in r.items() if k!='linearizations'} for p,r in result['runs'].items()},indent=2))
