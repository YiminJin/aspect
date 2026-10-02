#!/usr/bin/env python3
"""Check unique 2D/3D helper definitions independently and in the linked binary."""
from pathlib import Path
import collections,json,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence';checks={}
for label,path in (('independent',e/'surface_system-independent.o'),('linked',repo/'build-refactor-r5b2/aspect-release')):
 text=subprocess.check_output(['nm','-C','--defined-only',str(path)],text=True)
 counts=collections.Counter();lines=[]
 for line in text.splitlines():
  m=re.search(r'^[0-9a-f]+ [TW] aspect::ReconstructedFaultSurfaceSystem<([23])>::(prepare_surface_linearization|linearize_surface_system)\(',line)
  if m and '::{lambda' not in line:counts[m.groups()]+=1;lines.append(line)
 checks[label]=set(counts)=={(d,n) for d in ('2','3') for n in ('prepare_surface_linearization','linearize_surface_system')} and all(n==1 for n in counts.values())
 (e/(label+'-symbols.txt')).write_text('\n'.join(lines)+'\n')
(e/'symbol-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(checks.values())
