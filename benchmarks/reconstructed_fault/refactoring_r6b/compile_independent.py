#!/usr/bin/env python3
"""Compile the driver and dimension-independent formatter without unity/PCH."""
from pathlib import Path
import json, shlex, subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
commands=json.loads((repo/'build-refactor-r6b/compile_commands.json').read_text());used=[]
for name in ('reconstructed_fault_stokes','reconstructed_fault_bound_diagnostics'):
 entry=next(x for x in commands if x['file'].endswith('/solver/'+name+'.cc'))
 cmd=shlex.split(entry['command']);assert '-include' not in cmd
 cmd[cmd.index('-o')+1]=str(e/(name+'-independent.o'));used.append(cmd)
 subprocess.run(cmd,cwd=entry['directory'],check=True)
(e/'independent-commands.json').write_text(json.dumps(used,indent=2)+'\n')
