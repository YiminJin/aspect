#!/usr/bin/env python3
"""Compile history independently and verify 2D/3D lifecycle definitions."""
from pathlib import Path
import json
import shlex
import subprocess
root = Path(__file__).resolve().parent
repo = root.parents[2]
entries = json.loads((repo/'build-refactor-r3b/compile_commands.json').read_text())
entry = next(x for x in entries if x['file'].endswith('/phase_field_fault/history.cc'))
args = shlex.split(entry['command'])
obj = repo/'build-refactor-r3b/history-separate.o'
args[args.index('-o')+1] = str(obj)
assert '-include' not in args
(root/'evidence/separate-command.json').write_text(json.dumps(args,indent=2)+'\n')
subprocess.run(['python3',str(root/'run_logged.py'),'separate-history','300',*args],check=True)
symbols = subprocess.check_output(['nm','-C',str(obj)]).decode().splitlines()
names = ['sample_accepted_history','compute_history_candidates','validate_history_candidates',
         'publish_history_candidates','commit_reconstructed_fault_mechanical_history']
checks = {f'{dim}/{name}':any(' U ' not in line and f'PhaseFieldFault<{dim}>::{name}(' in line for line in symbols)
          for dim in (2,3) for name in names}
assert all(checks.values()),checks
(root/'evidence/instantiations.json').write_text(json.dumps(checks,indent=2)+'\n')
