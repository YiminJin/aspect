#!/usr/bin/env python3
"""Compile general solver and internal header without unity or precompiled headers."""
import json,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[2]
build=repo/'build-refactor-r4c'
e=root/'evidence'
entry=next(c for c in json.loads((build/'compile_commands.json').read_text()) if c['file'].endswith('/solver/reconstructed_fault_stokes.cc'))
base=shlex.split(entry['command'])
assert '-include' not in base
commands=[]
for name,source in [('solver',repo/'source/simulator/solver.cc'),('driver',repo/'source/simulator/solver/reconstructed_fault_stokes.cc'),('header',e/'header-check.cc')]:
    if name=='header':source.write_text('#include "'+str(repo/'source/simulator/solver/stokes_operators.h')+'"\n')
    command=base.copy()
    command[command.index('-o')+1]=str(e/(name+'-independent.o'))
    command[-1]=str(source)
    commands.append(command)
    result=subprocess.run(command,cwd=entry['directory'])
    if result.returncode:sys.exit(result.returncode)
(e/'independent-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
