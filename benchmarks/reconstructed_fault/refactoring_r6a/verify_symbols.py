#!/usr/bin/env python3
from pathlib import Path
import collections,json,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence';checks={}
methods={'open_stress_cycle','open_source_history','selects_stress_cycle','source_history_is_open','record_stress_cycle','record_source_history'}
for label,path in [('independent',e/'history_diagnostics-independent.o'),('linked',repo/'build-refactor-r6a/aspect-release')]:
 s=subprocess.check_output(['nm','-C','--defined-only',str(path)],text=True);found=collections.Counter();lines=[]
 for line in s.splitlines():
  m=re.search(r'^[0-9a-f]+ [TW] aspect::internal::FaultHistoryDiagnostics<([23])>::(\w+)\(',line)
  if m and m[2] in methods:found[m.groups()]+=1;lines.append(line)
 checks[label]=set(found)=={(d,m) for d in ('2','3') for m in methods} and all(v==1 for v in found.values())
 (e/(label+'-symbols.txt')).write_text('\n'.join(lines)+'\n')
(e/'symbol-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(checks.values())
