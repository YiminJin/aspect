#!/usr/bin/env python3
"""Stop only this task's queue owner; leave its current child run/reduction intact."""
import json,os,signal,sys
from pathlib import Path
root=Path(__file__).resolve().parents[4]
expected=str(Path(__file__).with_name('run_coupled.py'))
records=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().split(b'\0')
  if len(argv)<2 or str(Path(argv[1].decode()).resolve())!=expected:continue
  pid=int(p.name)
  if sys.argv[1]=='hold':os.kill(pid,signal.SIGSTOP)
  elif sys.argv[1]=='finish':
   os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
  else:raise ValueError(sys.argv[1])
  records.append(dict(pid=pid,action=sys.argv[1],command=[a.decode() for a in argv if a]))
 except (OSError,UnicodeDecodeError):continue
path=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled'/('queue-'+sys.argv[1]+'.json')
path.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records),flush=True)
