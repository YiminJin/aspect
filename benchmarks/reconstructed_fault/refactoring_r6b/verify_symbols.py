#!/usr/bin/env python3
from pathlib import Path
import collections,json,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence';checks={}
helper={'open_fault_bound_diagnostic','write_fault_bound_diagnostic','report_fault_bound_diagnostic'}
driver={'solve_reconstructed_fault_stokes','evaluate_reconstructed_fault_coupled_residual','solve_reconstructed_fault_condensed_system'}
for label,paths in [('independent',[e/'reconstructed_fault_stokes-independent.o',e/'reconstructed_fault_bound_diagnostics-independent.o']),('linked',[repo/'build-refactor-r6b/aspect-release'])]:
 found=collections.Counter();lines=[]
 for path in paths:
  s=subprocess.check_output(['nm','-C','--defined-only',str(path)],text=True)
  for line in s.splitlines():
   m=re.search(r'^[0-9a-f]+ [TW] aspect::internal::(\w+)\(',line)
   if m and m[1] in helper:found[('helper',m[1])]+=1;lines.append(line)
   m=re.search(r'^[0-9a-f]+ [TW] aspect::Simulator<([23])>::(\w+)\(',line)
   if m and m[2] in driver and "::{lambda" not in line:found[m.groups()]+=1;lines.append(line)
 expected={('helper',x) for x in helper}|{(d,x) for d in ('2','3') for x in driver}
 checks[label]=set(found)==expected and all(n==1 for n in found.values())
 (e/(label+'-symbols.txt')).write_text('\n'.join(lines)+'\n')
(e/'symbol-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(checks.values())
