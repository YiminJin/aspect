#!/usr/bin/env python3
"""Compare original fixture outcomes, solver decisions, markers and statistics."""
from pathlib import Path
import json,re
root=Path(__file__).resolve().parent;e=root/'evidence';checks={}
cases=[f'{name}-{rank}' for name in ('unit','stage-j','temperature','frozen-stress','rollback-original','rollback-open-top') for rank in (1,2)]
cases+=['stage-j-open-top-2','cohesive-resume-direct-2']
markers=('line search accepted after','Stage-I rollback after an accepted Newton update: verified','Stage-J controlled transverse-temperature evaluation:', 'Frozen Maxwell', 'Relative nonlinear residuals (bulk, fault)', 'Fault linear solve:', 'Stage-J history resolution:', 'Stage-J history feedback', 'Timestep ', 'All tests passed')
for name in cases:
 a,b=[json.loads((e/f'{v}-{name}.json').read_text()) for v in ('reference','candidate')]
 checks[name+'/exit']=a['exit_code']==b['exit_code']
 logs=[(e/f'{v}-{name}.log').read_text(errors='replace') for v in ('reference','candidate')]
 # MPI output can interleave per-rank unit summaries and restoration markers.
 summaries=[sorted(line.strip() for line in log.splitlines() if any(m in line for m in markers)) for log in logs]
 checks[name+'/decisions-and-markers']=summaries[0]==summaries[1]
 if name.startswith('rollback'):
  checks[name+'/accepted-update-and-restoration']=all('line search accepted after' in s and 'Stage-I rollback after an accepted Newton update: verified' in s for s in logs)
 if name=='cohesive-resume-direct-2':
  checks[name+'/restored-history']=all('Stage-J checkpoint histories, V, geometry, and bulk: verified' in s for s in logs)
 if name.startswith('stage-j'):
  checks[name+'/failure-or-feedback']=all(('significant pressure' in s and 'incompatibility' in s) or 'Stage-J history resolution:' in s for s in logs)
 if not name.startswith('unit'):
  paths=[root/f'output-{v}-{name}/statistics' for v in ('reference','candidate')]
  checks[name+'/statistics']=paths[0].exists()==paths[1].exists() and (not paths[0].exists() or paths[0].read_bytes()==paths[1].read_bytes())
(e/'history-lifecycle-comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
print(len(checks),'lifecycle checks; failed:',[k for k,v in checks.items() if not v])
assert all(checks.values())
