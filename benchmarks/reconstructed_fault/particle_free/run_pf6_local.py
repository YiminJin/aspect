#!/usr/bin/env python3
"""Prepare bounded G12 and fixed-surface local cases using the transfer runner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('action', choices=['prepare', 'finalize'])
p.add_argument('label')
p.add_argument('--ranks', type=int, choices=[1, 2, 4], default=1)
p.add_argument('--quadrature', type=int, choices=[6, 12, 24, 32], default=12)
p.add_argument('--end', type=float, choices=[.125, .25], default=.25)
p.add_argument('--restart', type=Path)
p.add_argument('--snapshot', action='store_true')
p.add_argument('--local', action='store_true')
p.add_argument('--deep-uniform', action='store_true', help='Retained 64-to-256 particle-support negative control.')
p.add_argument('--no-deep-uniform', dest='deep_uniform', action='store_false')
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
runner = Path(__file__).with_name('run_pf6_transfer.py')
args = [sys.executable, str(runner), a.action, a.label]
if a.action == 'finalize':
    subprocess.run(args, check=True)
    sys.exit()
args += ['--ranks', str(a.ranks), '--end', str(a.end), '--checkpoint']
if a.restart:
    args += ['--restart', str(a.restart)]
if a.snapshot:
    assert a.restart and a.quadrature == 6 and not a.local
subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
out = root/'build-particle-free/runs'/a.label
text = (out/'input.prm').read_text()
text += f'''\nsubsection Material model
 subsection Phase field fault
  set FE bulk normal quadrature points = {a.quadrature}
 end
end
subsection Postprocess
 subsection PF6 transfer checks
  set Snapshot quadrature control = {'true' if a.snapshot else 'false'}
  set Deep uniform roundtrip = {'true' if a.deep_uniform else 'false'}
 end
end
'''
if a.local:
    text += '''\nsubsection Postprocess
 subsection PF6 transfer checks
  set Local transfer checks = true
 end
end
'''
(out/'input.prm').write_text(text)
r = json.loads((out/'record.json').read_text())
r['input_sha256'] = hashlib.sha256((out/'input.prm').read_bytes()).hexdigest()
r['local_case'] = vars(a) | {'restart': str(a.restart) if a.restart else None}
r['scope'] = 'G12 physical quadrature and conditional frozen fixed-surface local remap'
(out/'record.json').write_text(json.dumps(r, indent=2)+'\n')
print((out/'command.json').read_text())
