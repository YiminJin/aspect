#!/usr/bin/env python3
"""Unique independent 2D/3D definitions and lifecycle symbol preservation."""
from pathlib import Path
import collections,json,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
particle='assemble_particle_system';bulk='assemble_bulk_work_system';dispatch='assemble_surface_system';checks={}
for label,path,allowed in (
 ('lifecycle',e/'surface_system-independent.o',{dispatch}),
 ('particle',e/'surface_system_particle-independent.o',{particle}),
 ('bulk',e/'surface_system_bulk_work-independent.o',{bulk}),
 ('linked',repo/'build-refactor-r5b1/aspect-release',{dispatch,particle,bulk})):
 text=subprocess.check_output(['nm','-C','--defined-only',str(path)],text=True)
 counts=collections.Counter();lines=[]
 for line in text.splitlines():
  match=re.search(r'^[0-9a-f]+ [TW] aspect::ReconstructedFaultSurfaceSystem<([23])>::(\w+)\(',line)
  if match and match[2] in {dispatch,particle,bulk}:counts[match.groups()]+=1;lines.append(line)
 expected={(dim,name) for dim in ('2','3') for name in allowed}
 checks[label]=set(counts)==expected and all(n==1 for n in counts.values())
 (e/(label+'-symbols.txt')).write_text('\n'.join(lines)+'\n')
(e/'symbol-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks);assert all(checks.values())
