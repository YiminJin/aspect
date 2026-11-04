#!/usr/bin/env python3
"""Reproduce corrected full-support comparison from completed local runs."""
import json,subprocess,fcntl
from pathlib import Path
root=Path(__file__).resolve().parents[4];b=root/'benchmarks/reconstructed_fault/particle_free/transported_objective';runs=root/'build-particle-free/transported-objective/stage-c/material-runs'
cases=[]
for c in ['c','l']:
 for d in ['p','n']:
  for m in ['q','m']:
   for r in (['base','space','time','joint'] if d=='p' else ['base','joint']):
    name=f'bpf-{c}-{d}-{m}-{r}';cases.append(name+'-live' if name=='bpf-c-p-m-base' else name)
  cases.append(f'bpf-{c}-{d}-m-ref')
for n in cases:assert json.loads((runs/n/'command.json').read_text())['exit_code']==0,n
lock=(runs.parent/'simulation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
args=['python3',str(b/'compare_bounded_profile.py'),*cases]
for c,e in [('c','constant'),('l','layered')]:
 for d,direction in [('p','positive'),('n','negative')]:args.extend(['--reference',f'{e}:{direction}:bpf-{c}-{d}-m-ref'])
subprocess.run(args,check=True)
energy=[n for n in cases if '-p-' in n and any(n.endswith('-'+r) for r in ['space','joint','ref'])]
subprocess.run(['python3',str(b/'diagnose_bounded_energy.py'),*energy],check=True)
subprocess.run(['python3',str(b/'plot_bounded_profile.py')],check=True)
subprocess.run(['python3',str(b/'check_bounded_lifecycle.py'),'--full-support'],check=True)
subprocess.run(['python3',str(b/'check_bounded_surface.py')],check=True)
subprocess.run(['python3',str(b/'summarize_bounded_profile.py')],check=True)
