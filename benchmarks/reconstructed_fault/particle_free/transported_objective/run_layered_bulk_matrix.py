#!/usr/bin/env python3
"""Frozen corrected campaign inputs; sequential MPI runs, followed by serial reduction."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
CANDIDATE=ROOT/'build-particle-free/transported-objective/stage-c/layered-bulk'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign'
CASES={
 'plus':dict(translation=2), 'zero':dict(translation=0), 'minus':dict(translation=-2),
 'plus-half':dict(translation=2,dt=.00625),
 'total-plus':dict(translation=2,spin='total material'),
 'total-plus-half':dict(translation=2,dt=.00625,spin='total material'),
 'plus-fine':dict(translation=2,dt=.00625,level=4,depth_repetitions=3,seconds=14400),
 'plus-fine-cartesian':dict(translation=2,dt=.00625,level=4,depth_repetitions=3,seconds=14400,
                           candidate=str(CANDIDATE.parent/'layered-bulk-cartesian')),
 'total-plus-fine-cartesian':dict(translation=2,dt=.00625,level=4,depth_repetitions=3,seconds=14400,
                                 spin='total material',candidate=str(CANDIDATE.parent/'layered-bulk-cartesian')),
 'plus-wide':dict(translation=2,halfwidth=8,candidate=str(CANDIDATE.parent/'layered-bulk-cartesian')),
}
def main():
 p=argparse.ArgumentParser();p.add_argument('cases',nargs='+',choices=list(CASES));p.add_argument('--reduce-only',action='store_true');a=p.parse_args()
 env=os.environ.copy();env['LAYERED_EVIDENCE']=str(OUT)
 for name in a.cases:
  label='bulk-'+name;d=dict(translation=2,dt=.0125,end=.8,level=5,halfwidth=6,seconds=7200,ranks=4,spin='bulk slip subtracted',background_shear=3.7,candidate=str(CANDIDATE));d.update(CASES[name])
  cmd=[sys.executable,str(HERE/'run_layered_campaign.py'),label]
  for k,v in d.items():cmd += ['--'+k.replace('_','-'),str(v)]
  print(json.dumps(dict(case=name,command=cmd)),flush=True)
  if not a.reduce_only:subprocess.run(cmd,cwd=ROOT,env=env,check=True)
  for script in ['check_layered_reference.py','assess_layered_campaign.py']:
   subprocess.run([sys.executable,str(HERE/script),'layered-'+label],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
