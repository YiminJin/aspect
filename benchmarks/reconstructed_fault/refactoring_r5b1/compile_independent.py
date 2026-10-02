#!/usr/bin/env python3
"""Independently compile lifecycle and both backends, without PCH or unity."""
from pathlib import Path
import json,shlex,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence';build=repo/'build-refactor-r5b1'
entry=next(c for c in json.loads((build/'compile_commands.json').read_text()) if c['file'].endswith('/surface_system_particle.cc'))
base=shlex.split(entry['command']);assert '-include' not in base;commands=[]
for name in ('surface_system','surface_system_particle','surface_system_bulk_work'):
 cmd=base.copy();cmd[cmd.index('-o')+1]=str(e/(name+'-independent.o'));cmd[-1]=str(repo/f'source/reconstructed_fault/{name}.cc')
 commands.append(cmd);subprocess.run(cmd,cwd=entry['directory'],check=True)
(e/'independent-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
