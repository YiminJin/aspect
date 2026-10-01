#!/usr/bin/env python3
"""Compile both manager translation units independently, without unity or PCH."""
import json,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent;repo=root.parents[2];build=repo/'build-refactor-r5a1';e=root/'evidence'
entry=next(c for c in json.loads((build/'compile_commands.json').read_text()) if c['file'].endswith('/solver/reconstructed_fault_stokes.cc'))
base=shlex.split(entry['command']);assert '-include' not in base
commands=[]
for name in ('manager','manager_slip_rate'):
 command=base.copy();command[command.index('-o')+1]=str(e/(name+'-independent.o'))
 command[-1]=str(repo/f'source/reconstructed_fault/{name}.cc');commands.append(command)
 result=subprocess.run(command,cwd=entry['directory'])
 if result.returncode:sys.exit(result.returncode)
(e/'independent-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
