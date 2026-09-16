"""Fresh all-QP observer control or bottom-source candidate; same fixed clock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE/'bottom-completion-50-local4'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=['control','continued','supported','complete-wedge'])
    parser.add_argument('--prepare-only',action='store_true',help='Write a runnable configuration without starting ASPECT.')
    args=parser.parse_args()
    out=HERE/('bottom-source-'+args.case+'-50-local4')
    out.mkdir() # Never replace a failed or accepted run.
    (out/'clock.csv').write_bytes((BASE/'clock.csv').read_bytes())
    (out/'run.prm').write_text(f'include {BASE}/run.prm\nset Output directory = {out}\n')
    if args.case in ['supported','complete-wedge']:
        with (out/'run.prm').open('a') as prm:
            prm.write(f'subsection Postprocess\n  subsection BP3\n    set Bottom normalization completion file = {BASE}/completion.txt\n  end\nend\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(json.loads((BASE/'provenance.json').read_text())['environment'])
    env.update(ASPECT_BP3_TIMESTEP_SEQUENCE=str(out/'clock.csv'),
               ASPECT_BP3_ALL_SOURCE_QPS='1',ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC='1',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    if args.case=='continued':env['ASPECT_BP3_BOTTOM_SOURCE_CONTINUATION']='1'
    if args.case in ['supported','complete-wedge']:env.pop('ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC',None)
    command=['timeout','--signal=TERM','--kill-after=15','600','mpirun','-np','4',
             '--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(out/'run.prm')]
    paths=[HERE/'bp3.cc',HERE/'uniform_sliding.h',Path(__file__),HERE/'build/libbp3.release.so',
           REPO/'build-pf-cpdi/aspect-release',out/'run.prm',out/'clock.csv',BASE/'completion.txt',
           REPO/'source/material_model/phase_field_fault.cc',REPO/'source/reconstructed_fault/manager.cc']
    (out/'provenance.json').write_text(json.dumps(dict(command=command,
        environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    if args.prepare_only:
        print(f'Prepared only: {out}/run.prm',flush=True)
        return
    start=time.monotonic()
    with (out/'run.log').open('x') as log:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
    record=dict(status=result.returncode,seconds=time.monotonic()-start)
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    assert result.returncode==0,'Preserve failure; no automatic retry.'

if __name__=='__main__':main()
