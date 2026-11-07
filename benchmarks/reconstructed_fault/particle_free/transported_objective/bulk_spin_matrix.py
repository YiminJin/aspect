#!/usr/bin/env python3
"""Bounded 16-square-mesh constitutive comparison; sequential, <=4 MPI ranks."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[4]
RUNNER=Path(__file__).with_name('run_bulk_spin.py')
BASE=['--profile-method','bounded MacCormack','--support','full normal cross-section','--inflow','accepted trace','--phase','.35*max(0,1-(x/1.4)^2)^4*max(0,1-((y-.3*x)/.6)^2)^4','--shear','.5','--end','.1','--level','4','--energy','--stage-energy','--seconds','200']
CASES={}
for law in ['bulk','total']:
 for eta in ['c','l']:
  for rep in ['d','s']:
   name=f'{law}-{eta}{rep}4'
   CASES[name]=['--constitutive-spin','bulk slip subtracted' if law=='bulk' else 'total material','--eta','1' if eta=='c' else '1+.5*tanh(4*x)','--representation','direct physical stress' if rep=='d' else 'separated','--dt','.025','--ranks','1']
for eta in ['c','l']:
 CASES[f'bulk-{eta}d-half']=CASES[f'bulk-{eta}d4']+['--dt','.0125']
CASES['bulk-cd-mpi']=CASES['bulk-cd4']+['--ranks','4']
CASES['bulk-cd-reject']=CASES['bulk-cd4']+['--reject','--end','.0999999999999']
CASES['bulk-cd-retry-control']=CASES['bulk-cd4']+['--schedule','if(time<.025,.025,if(time<.0375,.0125,.025))','--end','.0999999999999']
if __name__=='__main__':
 names=sys.argv[1:] or list(CASES)
 for name in names:
  if name not in CASES:raise SystemExit('unknown bounded case '+name)
  command=[sys.executable,str(RUNNER),name,*BASE,*CASES[name]]
  print(json.dumps(command),flush=True)
  subprocess.run(command,cwd=ROOT,check=True)
