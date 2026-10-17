#!/usr/bin/env python3
"""Run only the selected Gauss6 serial short-horizon resolution matrix."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BUILD=Path('build-particle-free/mature-resolution')
CASES=[('dg-fc-final','dg',7,True),('dg-cc','dg',6,False),('dg-ff','dg',7,False),
       ('particle-cc','particle',6,False),('particle-fc','particle',7,True),('particle-ff','particle',7,False)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=[x[0] for x in CASES]);a=p.parse_args()
    runs=ROOT/BUILD/'runs'
    spent=sum(json.loads(f.read_text())['seconds'] for f in runs.glob('*/record.json'))
    for label,backend,refinement,prolong in CASES:
        if a.case and a.case!=label: continue
        seconds=min(700.,3600.-spent)
        if seconds<30: raise SystemExit('Selected native budget exhausted.')
        command=[sys.executable,str(Path(__file__).with_name('run_pf6_mature.py')),
                 label,'--backend',backend,'--build-directory',str(BUILD),'--steps','3',
                 '--quadrature','6','--refinement',str(refinement),'--seconds',str(seconds),
                 '--common-points',str(BUILD/'inputs-lf'/('coarse-points.csv' if refinement==6 else 'fine-points.csv'))]
        if prolong: command+=['--phase-reference',str(BUILD/'inputs-lf/coarse-phase.csv')]
        print(label,flush=True)
        code=subprocess.run(command,cwd=ROOT,check=False).returncode
        record=json.loads((runs/label/'record.json').read_text());spent+=record['seconds']
        print(json.dumps(dict(label=label,exit_code=code,native_seconds=spent)),flush=True)
        if code: raise SystemExit(code)
if __name__=='__main__': main()
