#!/usr/bin/env python3
from pathlib import Path
import collections,json,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
names=json.loads((e/'moved-methods.json').read_text());expected={(dim,name) for dim in ('2','3') for name in names}
checks={}
for label,path in [('remaining',e/'manager-independent.o'),('moved',e/'manager_particle_projection-independent.o'),('linked',repo/'build-refactor-r5a2/aspect-release')]:
 text=subprocess.check_output(['nm','-C','--defined-only',str(path)],text=True)
 found=[];lines=[]
 for line in text.splitlines():
  match=re.search(r'^[0-9a-f]+ [TW] aspect::ReconstructedFaultManager<([23])>::(\w+)\(',line)
  if match and match.group(2) in names:
   found.append(match.groups());lines.append(line)
 (e/(label+'-symbols.txt')).write_text('\n'.join(lines)+'\n')
 counts=collections.Counter(found)
 checks[label]=(not counts) if label=='remaining' else (set(counts)==expected and all(n==1 for n in counts.values()))
(e/'symbol-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(checks.values())
