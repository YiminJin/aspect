"""Fresh 300/150-s startup and one capped replay of the preserved bound failure."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from startup_30km import STUDY, HERE, BP3, ROOT, BIN, PLUGIN, FIXTURE, parameters, render, digest

OUT=STUDY/'small-startup'


def prepare(case):
    path=OUT/case;path.mkdir(parents=True)
    original=STUDY/'startup'
    values=parameters((original/'run.prm').read_text())
    record=json.loads((original/'launch.json').read_text())
    # Preserve the already-failed solve's actual plugin, not a freshly relinked
    # implementation. Its diagnostic flag only requests noncommitting exports.
    if case=='bounds':
        plugin=HERE/'server-30km/qualified-local-plugin.release.so'
        assert digest(plugin)==record['hashes'][str(PLUGIN)]
        values['Additional shared libraries',]=values['Additional shared libraries',].replace(str(PLUGIN),str(plugin))
        values['Checkpointing','Steps between checkpoint']='2' # accepted step 1
        record['environment']['ASPECT_FAULT_NONLINEAR_DIAGNOSTIC']='1'
        cap=1800
    else:
        plugin=PLUGIN;dt=int(case)
        values['Maximum time step',]=str(dt)
        values['End time',]='900'
        values['Postprocess','BP3','Last accepted step']=str(900//dt)
        values['Postprocess','BP3','Graceful wall seconds']='1100'
        values['Time stepping','List of model names']+=', BP5 state startup'
        values['Time stepping','BP5 state startup','Maximum logarithmic state change']='0.1'
        values['Checkpointing','Steps between checkpoint']='0'
        values['Termination criteria','Checkpoint on termination']='false'
        values['Postprocess','BP3','Audit full state every step']='false'
        cap=1200
    values['Output directory',]=str(path)
    (path/'run.prm').write_text(render(values))
    inputs=[BIN,plugin,Path(__file__).resolve(),path/'run.prm']+list(FIXTURE.glob('*.txt'))
    if case!='bounds':inputs+=[HERE/'startup_time_step.cc',HERE/'CMakeLists.txt']
    # Retain the independent snapshot/probe library hash from the original run.
    inputs+=list(map(Path,values['Additional shared libraries',].split(', ')))
    record.update(command=['mpirun','-np','4','--bind-to','core','--map-by','core',str(BIN),str(path/'run.prm')],
                  cap_seconds=cap,hashes={str(p):digest(p) for p in inputs})
    (path/'launch.json').write_text(json.dumps(record,indent=2)+'\n')


def run(case):
    path=OUT/case;record=json.loads((path/'launch.json').read_text())
    for p,sha in record['hashes'].items():assert digest(Path(p))==sha,p
    used=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('*/execution.json'))
    assert used+record['cap_seconds']<=4200
    assert shutil.disk_usage(OUT).free>4*1024**3
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment'])
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',LD_BIND_NOW='1')
    start=time.monotonic()
    with (path/'run.log').open('x') as log:
        result=subprocess.run(['timeout','--kill-after=15',str(record['cap_seconds'])]+record['command'],
            cwd=BP3,env=env,stdout=log,stderr=subprocess.STDOUT)
    rows=list(csv.DictReader((path/'accepted_steps.csv').open()))
    if case=='bounds':
        passed=(result.returncode!=0 and 'ExcNonlinearSolverNoConvergence' in (path/'run.log').read_text()
                and [int(r['step']) for r in rows]==[0,1])
    else:passed=result.returncode==0 and float(rows[-1]['time'])==900.
    info=dict(seconds=time.monotonic()-start,status=result.returncode,passed=passed,
              scope='expected failure replay' if case=='bounds' else 'short startup',
              accepted_steps=[int(r['step']) for r in rows],peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (path/'execution.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
    assert passed,'Preserved unexpected outcome; no automatic retry'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run']);parser.add_argument('case',choices=['300','150','bounds'])
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.case)
    else:run(args.case)
