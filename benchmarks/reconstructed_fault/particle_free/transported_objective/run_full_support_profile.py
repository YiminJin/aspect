#!/usr/bin/env python3
"""Corrected normal-support comparison; short 0.2 s local material experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_bounded_profile_matrix import HERE,ROOT,RUNS,PHASE

def case(c,d,m,resolution,extra=()):
    level,dt={'base':(4,.1),'space':(5,.1),'time':(4,.05),'joint':(5,.05),'ref':(6,.025)}.get(resolution,(4,.1))
    label=f'bpf-{c}-{d}-{m}-{resolution}'
    if label=='bpf-c-p-m-base':label+='-live'
    return [label,'--level',str(level),'--dt',str(dt),'--end','.2',
            '--profile-method','bounded MacCormack' if m=='m' else 'Q1 remap',
            '--inflow','accepted trace','--support','full normal cross-section','--energy','--shear','0','--phase',PHASE,
            '--speed','.4' if d=='p' else '-.4','--eta','1' if c=='c' else '1+.5*tanh(4*x)',
            '--ranks','4' if resolution=='ref' else '2','--seconds','900',*extra]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group',choices=['base','refinement','reference','lifecycle','identity']);a=p.parse_args()
    if a.group=='base':cases=[case(c,d,m,'base') for c in ['c','l'] for d in ['p','n'] for m in ['q','m']]
    elif a.group=='refinement':
        cases=[case(c,d,m,r) for c in ['c','l'] for d in ['p','n'] for m in ['q','m'] for r in (['space','time','joint'] if d=='p' else ['joint'])]
    elif a.group=='reference':cases=[case(c,d,'m','ref') for c in ['c','l'] for d in ['p','n']]
    elif a.group=='identity':
        cases=[case('c','p','m','old-support-mismatch',['--restart',str(RUNS/'bp-c-p-m-checkpoint')]),
               case('c','p','m','profile-mismatch',['--restart',str(RUNS/'bpf-c-p-m-checkpoint'),'--profile-method','Q1 remap']),
               case('c','p','m','coverage-rejection',['--support','stationary profile'])]
    else:
        cases=[]
        for c in ['c','l']:
            cases.extend([case(c,'p','m','reject',['--reject']),
                          case(c,'p','m','schedule',['--schedule','if(time<.099999999,.1,.05)']),
                          case(c,'p','m','checkpoint',['--end','.1']),
                          case(c,'p','m','restart',['--restart',str(RUNS/f'bpf-{c}-p-m-checkpoint'),'--ranks','4'])])
    for args in cases:
        target=RUNS/args[0]
        if target.exists():
            r=json.loads((target/'command.json').read_text());assert r['exit_code']==(1 if a.group=='identity' else 0),(args[0],r['exit_code'])
            print('Retained '+args[0],flush=True);continue
        run=subprocess.run([sys.executable,str(HERE/'run_material_native.py'),*args],cwd=ROOT)
        assert run.returncode==(1 if a.group=='identity' else 0),(args[0],run.returncode)
        if a.group=='identity':
            log=(target/'run.log').read_text()
            expected='requires Association support' if 'coverage-rejection' in args[0] else 'version/material/quadrature mismatch'
            assert expected in log,(args[0],expected)

if __name__=='__main__':main()
