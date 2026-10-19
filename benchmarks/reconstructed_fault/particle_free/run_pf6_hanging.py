#!/usr/bin/env python3
"""Prepare one bounded serial fixed-local-mesh cohesive endpoint control.

Launch command.json with run_logged.py; this prepares inputs, not a workflow
controller. Physical seed/materials, Gauss12/source3 and unlimited DG are inherited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('action', choices=['prepare', 'finalize'])
p.add_argument('label')
p.add_argument('--binary', type=Path)
p.add_argument('--plugin', type=Path)
p.add_argument('--restart', type=Path)
p.add_argument('--end', type=float, choices=[.125, .25], default=.25)
p.add_argument('--schedule', choices=['constant', 'half', 'cutback', 'matched'], default='constant')
p.add_argument('--operators', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
runner = Path(__file__).with_name('run_pf6_domain_free.py')
args = [sys.executable, str(runner), a.action, a.label]
if a.action == 'finalize':
    subprocess.run(args, check=True)
    sys.exit()
args += ['--no-deep', '--end', str(a.end)]
for option in ['binary', 'plugin', 'restart']:
    if getattr(a, option):
        args += ['--'+option, str(getattr(a, option))]
subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
out = root/'build-particle-free/runs'/a.label
text = (out/'input.prm').read_text()
expression = dict(constant='0.125', half='0.0625',
                  cutback='if(time<0.25,0.125,0.0625)',
                  matched='if(time<0.125,0.125,0.0625)')[a.schedule]
cap = .0625 if a.schedule == 'half' else .125
text += f'''
set Maximum first time step = {cap}
set Maximum time step = {cap}
subsection Time stepping
 set List of model names = function{', repeat on cutback' if a.schedule == 'cutback' else ''}
 subsection Function
  set Function expression = {expression}
 end
 subsection Repeat on cutback
  set Relative repeat threshold = 0.75
  set Cut back amount = 0.5
 end
end
subsection Boundary velocity model
 set Prescribed velocity boundary indicators = left y:PF6a interval ramp, right y:PF6a interval ramp, bottom:PF6a interval ramp, top:PF6a interval ramp
end
subsection Postprocess
 subsection PF6 transfer checks
  set Fixed local phase evolution = true
  set Constrained operator checks = {'true' if a.operators else 'false'}
 end
end
'''
(out/'input.prm').write_text(text)
r = json.loads((out/'record.json').read_text())
r['input_sha256'] = hashlib.sha256((out/'input.prm').read_bytes()).hexdigest()
r['budget_seconds'] = 300 if a.operators else 180
r['hanging_case'] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
r['scope'] = 'Serial 67-cell fixed locally refined cohesive DG endpoint; fixed surface; no physical remesh or transport'
(out/'record.json').write_text(json.dumps(r, indent=2)+'\n')
print((out/'command.json').read_text())
