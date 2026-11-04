#!/usr/bin/env python3
"""Sequential bounded local comparison; at most four MPI ranks, no server jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUNS=ROOT/'build-particle-free/transported-objective/stage-c/material-runs'
PHASE='.35*max(0,1-(x/1.4)^2)^4*max(0,1-((y-.3*x)/.6)^2)^4'

def case(coefficient,direction,method,resolution,extra=()):
    label=f'bp-{coefficient}-{direction}-{method}-{resolution}'
    level,dt={'base':(4,.1),'space':(5,.1),'time':(4,.05),'joint':(5,.05),'ref':(6,.025)}.get(resolution,(4,.1))
    args=[label,'--level',str(level),'--dt',str(dt),'--end','.39999999999999997',
          '--profile-method','bounded MacCormack' if method=='m' else 'Q1 remap',
          '--inflow','accepted trace','--shear','0','--phase',PHASE,
          '--speed','.4' if direction=='p' else '-.4',
          '--eta','1' if coefficient=='c' else '1+.5*tanh(4*x)',
          '--ranks','4' if resolution=='ref' else '2','--seconds','600',*extra]
    return args

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group',choices=['base','refinement','reference','lifecycle','energy','identity']);a=p.parse_args()
    cases=[]
    if a.group=='base':
        cases=[case(c,d,m,'base') for c in ['c','l'] for d in ['p','n'] for m in ['q','m']]
        for args in cases:
            if args[0]=='bp-c-p-m-base':args[0]+='-live'
    elif a.group=='refinement':
        cases=[case(c,d,m,r) for c in ['c','l'] for d in ['p','n'] for m in ['q','m'] for r in (['space','time','joint'] if d=='p' else ['joint'])]
    elif a.group=='reference':
        cases=[case(c,d,'m','ref') for c in ['c','l'] for d in ['p','n']]
    elif a.group=='energy':
        for c in ['c','l']:
            for m in ['q','m']:
                for r in ['space','joint']:
                    args=case(c,'p',m,r,['--energy']);args[0]+='-energy';cases.append(args)
    elif a.group=='identity':
        for selector,value,label in [('--profile-method','Q1 remap','profile-mismatch'),('--inflow','zero','inflow-mismatch')]:
            cases.append(case('c','p','m',label,['--restart',str(RUNS/'bp-c-p-m-checkpoint'),selector,value]))
    else:
        for c in ['c','l']:
            cases.extend([case(c,'p','m','reject',['--reject','--end','.2']),
                          case(c,'p','m','schedule',['--schedule','if(time<.099999999,.1,.05)','--end','.2']),
                          case(c,'p','m','checkpoint',['--end','.2']),
                          case(c,'p','m','restart',['--restart',str(RUNS/f'bp-{c}-p-m-checkpoint'),'--ranks','4'])])
    for args in cases:
        target=RUNS/args[0]
        if target.exists():
            record=json.loads((target/'command.json').read_text())
            if a.group=='identity':
                assert record['exit_code']!=0
                assert 'PF5 checkpoint version/material/quadrature mismatch' in (target/'run.log').read_text()
                print('Retained expected rejection '+args[0],flush=True);continue
            if record['exit_code']!=0:raise SystemExit('Retained failed case needs explicit new label: '+args[0])
            print('Retained completed '+args[0],flush=True);continue
        completed=subprocess.run([sys.executable,str(HERE/'run_material_native.py'),*args],cwd=ROOT)
        if a.group=='identity':
            assert completed.returncode!=0,'Incompatible checkpoint was silently accepted.'
            log=(target/'run.log').read_text()
            assert 'PF5 checkpoint version/material/quadrature mismatch' in log,log[-2000:]
        else:completed.check_returncode()

if __name__=='__main__':main()
