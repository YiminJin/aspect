#!/usr/bin/env python3
"""The four selected DG phase comparisons, sequential and fresh; no production job."""
import argparse,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[4]
here=Path(__file__).resolve().parents[1]/'transported_objective'
candidate=root/'build-particle-free/dgq2-phase/candidate'
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/coupled'
cases={'plus':{},'plus-half':{'dt':.00625},'minus':{'translation':-2},
       'plus-fine':{'dt':.00625,'level':4,'depth_repetitions':3,'seconds':18000}}
p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',nargs='+',choices=list(cases));p.add_argument('--reduce-only',action='store_true');a=p.parse_args()
out.mkdir(exist_ok=True);env=os.environ.copy();env['LAYERED_EVIDENCE']=str(out)
for case in a.cases:
 settings=dict(candidate=str(candidate),spin='bulk slip subtracted',phase_discretization='DGQ2',
               background_shear=3.7,translation=2,halfwidth=6,level=5,dt=.0125,end=.8,ranks=4,seconds=10800)
 settings.update(cases[case]);label='dgphase-'+case
 command=[sys.executable,str(here/'run_layered_campaign.py'),label]
 for key,value in settings.items():command.extend(['--'+key.replace('_','-'),str(value)])
 print(json.dumps(dict(case=case,command=command)),flush=True)
 if not a.reduce_only:subprocess.run(command,cwd=root,env=env,check=True)
 for script in ('check_layered_reference.py','assess_layered_campaign.py'):
  subprocess.run([sys.executable,str(here/script),'layered-'+label],cwd=root,env=env,check=True)
