#!/usr/bin/env python3
from pathlib import Path
import json,re,csv
r=Path(__file__).resolve().parent;e=r/'evidence';checks={}
logs=[(e/f'{v}-frozen-replay.log').read_text(errors='replace') for v in ('reference','candidate')]
records=[json.loads((e/f'{v}-frozen-replay.json').read_text()) for v in ('reference','candidate')]
checks['same-preexisting-hierarchy-failure']=all(j['exit_code']==1 for j in records) and all('construct_multigrid_hierarchy' in s and 'is_multilevel_hierarchy_constructed()' in s for s in logs)
checks['no-false-full-probe-pass']=all('FROZEN AMG/GMG COMPARISON PASSED' not in s for s in logs)
markers=('Fault linear solve:', 'Relative nonlinear residuals', 'line search accepted after', '*** Timestep')
summaries=[sorted(line.strip() for line in s.splitlines() if any(m in line for m in markers)) for s in logs]
checks['pre-observer-decisions-identical']=bool(summaries[0]) and summaries[0]==summaries[1]
counts=[[re.findall(r'(?:step|newton|\w+_calls)=\d+',line) for line in s.splitlines() if line.startswith('Fault linear profile:')] for s in logs]
checks['pre-observer-work-counts-identical']=bool(counts[0]) and counts[0]==counts[1]
rows=[]
for v in ('reference','candidate'):
 with (r/f'output-{v}-frozen-replay/frozen_gmg.csv').open() as f: rows.append(list(csv.DictReader(f)))
checks['only-AMG-row-reached']=all(len(x)==1 and x[0]['backend']=='AMG' for x in rows)
keys=['backend','step','newton','rhs_norm','tolerance','iterations','fresh','estimated','direction_difference_relative','preconditioner_calls','operator_calls']
checks['frozen-AMG-numerics-and-counters-identical']=all(rows[0][0][k]==rows[1][0][k] for k in keys)
checks['frozen-AMG-direction-and-residual-pass']=all(float(x[0]['fresh'])<=float(x[0]['tolerance']) and float(x[0]['direction_difference_relative'])==0 for x in rows)
record=dict(status='blocked by pre-existing frozen GMG fixture hierarchy setup',checks=checks,matched_AMG={k:rows[0][0][k] for k in keys})
(e/'frozen-comparison.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));assert all(checks.values())
