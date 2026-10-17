#!/usr/bin/env python3
"""Selected serial gates/comparisons for the bounded stationary mature task."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage',choices=('gates','trajectories','sensitivity','spatial'))
p.add_argument('--gate-suffix',default='v2')
a=p.parse_args()
root=Path(__file__).resolve().parents[3]
runs=root/'build-particle-free/mature-dg/runs'
runner=root/'benchmarks/reconstructed_fault/particle_free/run_pf6_mature.py'

def run(label,*options,cap=700):
    spent=sum(json.loads(f.read_text())['seconds'] for f in runs.glob('*/record.json'))
    if spent+cap>3600:
        raise SystemExit(f'Remaining native budget below {cap} s cap; used {spent:.3f} s')
    command=[sys.executable,str(runner),label,'--seconds',str(cap),*map(str,options)]
    print(json.dumps(dict(command=command,spent_seconds=spent)),flush=True)
    subprocess.run(command,cwd=root,check=True)

if a.stage=='gates':
    for backend in ('dg','fe','particle'):
        run('gate-'+backend+'-'+a.gate_suffix,'--backend',backend,'--steps',1,cap=400)
else:
    for backend in ('dg','fe','particle'):
        record=json.loads((runs/('gate-'+backend+'-v2')/'record.json').read_text())
        if record['exit_code']:
            raise SystemExit('Each matched backend must pass its one-step gate first')
    if a.stage=='trajectories':
        for backend in ('dg','fe','particle'):
            run(backend+'-10','--backend',backend)
    elif a.stage=='sensitivity':
        run('dg-q6','--quadrature',6,'--steps',3)
        run('fe-q6','--backend','fe','--quadrature',6,'--steps',3)
        run('dg-q8','--quadrature',8,'--steps',3)
        run('dg-half','--quadrature',6,'--dt',100000,'--steps',6)
    elif a.stage=='spatial':
        run('dg-fine','--quadrature',6,'--steps',3,'--refinement',7)
