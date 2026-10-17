#!/usr/bin/env python3
"""Prepare/finalize the bounded homogeneous mesh-data DG experiment."""
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
p.add_argument('--end', type=float, choices=[.125, .25], default=.25)
p.add_argument('--restart', type=Path)
p.add_argument('--legacy', action='store_true', help='Retained domain-backed homogeneous control.')
p.add_argument('--no-deep', action='store_true')
p.add_argument('--negative', choices=['domains', 'heterogeneous', 'missing-seed', 'particle-fields', 'fit-geometry'])
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
runner = Path(__file__).with_name('run_pf6_local.py')
args = [sys.executable, str(runner), a.action, a.label]
if a.action == 'finalize':
    subprocess.run(args, check=True)
    sys.exit()
args += ['--ranks', str(a.ranks), '--end', str(a.end)]
if not a.no_deep:
    args += ['--deep-uniform']
if a.restart:
    args += ['--restart', str(a.restart)]
subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
out = root/'build-particle-free/runs'/a.label
text = (out/'input.prm').read_text()
if not a.legacy:
    text += '''
subsection Material model
 subsection Phase field fault
  set Use homogeneous mesh data = true
 end
end
subsection Compositional fields
 set Compositional field methods = static, static, static, static
 set Mapped particle properties =
end
subsection Particles
 set Generate particle domains = false
 set Generate CPDI data for particle domains = false
 set Generate face data for particle domains = false
 set List of particle properties = crack driving force, maxwell stress
end
'''
negative = {
    'domains': 'subsection Particles\n set Generate particle domains = true\nend\n',
    'heterogeneous': 'subsection Material model\n subsection Phase field fault\n  set Elastic shear moduli = 1e6, 2e6\n end\nend\n',
    'missing-seed': 'subsection Material model\n subsection Phase field fault\n  set FE initial phase profile file =\n end\nend\n',
    'particle-fields': 'subsection Compositional fields\n set Compositional field methods = particles, particles, particles, particles\nend\n',
    'fit-geometry': 'subsection Fault reconstruction\n set Fit prescribed geometry to phase field = true\nend\n',
}
if a.negative:
    assert not a.legacy
    text += negative[a.negative]
(out/'input.prm').write_text(text)
r = json.loads((out/'record.json').read_text())
r['input_sha256'] = hashlib.sha256((out/'input.prm').read_bytes()).hexdigest()
r['domain_free_case'] = vars(a) | {'restart': str(a.restart) if a.restart else None}
r['scope'] = 'Homogeneous mesh-data DG endpoint; prescribed phase seed; fixed diagnostic points; restored deep uniform remap'
(out/'record.json').write_text(json.dumps(r, indent=2)+'\n')
print((out/'command.json').read_text())
