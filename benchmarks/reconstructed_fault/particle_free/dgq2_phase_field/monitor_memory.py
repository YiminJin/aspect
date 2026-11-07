#!/usr/bin/env python3
"""Low-frequency /proc RSS samples for the selected candidate's MPI ranks.

This observer changes no simulation state. VmHWM includes the time before the
observer started. Sum of per-process high-water marks need not be concurrent;
report it separately from the maximum concurrent sampled RSS. Shared pages may
be counted in more than one rank. The monitor exits when the campaign wrapper
is no longer present, after seeing that wrapper at least once.
"""
import json
import time
from pathlib import Path
root=Path(__file__).resolve().parents[4]
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled/runtime-memory.jsonl'
assert not out.exists(),out
seen=False
with out.open('x') as f:
 while True:
  ranks=[];wrapper=False
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    if 'dgq2_phase_field/run_coupled.py plus plus-half minus plus-fine' in cmd and 'python' in cmd:wrapper=True
    if '/dgq2-phase/candidate/aspect-release ' not in cmd:continue
    status=(p/'status').read_text()
    vals={line.split(':')[0]:line.split(':',1)[1].strip() for line in status.splitlines() if ':' in line}
    if vals.get('Name') not in ('aspect-release','aspect.exe.rele'):continue
    ranks.append(dict(pid=int(p.name),command=cmd,rss_kib=int(vals['VmRSS'].split()[0]),hwm_kib=int(vals['VmHWM'].split()[0])))
   except (OSError,KeyError,ValueError):continue
  f.write(json.dumps(dict(utc_unix=time.time(),ranks=ranks))+'\n');f.flush()
  seen|=wrapper
  if seen and not wrapper:break
  time.sleep(30)
