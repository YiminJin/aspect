"""Two fresh four-rank runs with saved local meshes and an admissible shared clock.

No physical fields are exchanged, no old nodal/particle state is imported,
and no failed run is retried. Existing evidence is never overwritten.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--tag',default='');args=p.parse_args()
here=Path(__file__).resolve().parent;repo=here.parents[2]
pair=here/('junction-matched'+('-'+args.tag if args.tag else '')+'-local4')
if args.mode=='prepare':
    pair.mkdir();(pair/'clock').mkdir()
    cells=[]
    for path in sorted((here/'normal-stress-complete-local4').glob('initial_mesh_*.csv')):
        with path.open() as f:cells+=list(csv.DictReader(f))
    assert len(cells)==36106 and len({r['cell'] for r in cells})==36106
    for label in ['coarse','refined']:
        out=pair/label;out.mkdir()
        leaves=[];split=0
        for r in cells:
            x=float(r['x']);y=float(r['y']);st=3**.5/2
            xd=(.5e5*(1+1/(3**.5))-x)*.5+(1e5-y)*st
            refine=label=='refined' and 36000<xd<44000 and float(r['distance'])<1500
            if refine:
                root=r['cell'].split('_')[0];path=r['cell'].split(':')[1]
                leaves += [f'{root}_{len(path)+1}:{path}{i}' for i in range(4)]
                split+=1
            else:leaves.append(r['cell'])
        if label=='refined' and args.tag=='qualified':
            # Preserve the observed ASPECT grading closure. This changes neither
            # the support width nor the standard mesh-smoothing flags.
            target=set(leaves);extra=[];parents=set()
            for path in (here/'junction-matched-local4/mesh-check').glob('mesh_guard_rank*.csv'):
                with path.open() as f:extra+=list(csv.DictReader(f))
            assert len(extra)==316 and all(r['descendant']=='1' for r in extra)
            for r in extra:
                cell=r['cell'];root=cell.split('_')[0];children=cell.split(':')[1]
                parent=f'{root}_{len(children)-1}:{children[:-1]}'
                assert parent in target
                parents.add(parent)
            assert len(parents)==79
            leaves=sorted((target-parents)|{r['cell'] for r in extra})
            assert len(leaves)==42880
        (out/'target_cells.txt').write_text('\n'.join(sorted(leaves))+'\n')
        (out/'run.prm').write_text(f'''# Matched physical-time local-resolution test, not a first-event run.
include {here}/normal_stress_fresh_local4.prm
set Output directory = {out}
set End time = 2232176379.2516127
subsection Mesh refinement
  set Initial adaptive refinement = {6 if label=='coarse' else 7}
  set Strategy = BP3 saved mesh
end
subsection Time stepping
  set List of model names = BP3 shared clock
end
subsection Termination criteria
  set Termination criteria = end time, end step
  set End step = 60
end
''')
        (out/'mesh_plan.json').write_text(json.dumps(dict(split_coarse_cells=split,target_leaves=len(leaves)),indent=2)+'\n')
    print(pair)
else:
    binary=repo/'build-pf-cpdi/aspect-release';plugin=here/'build/libbp3.release.so'
    running={};start=time.monotonic()
    for label in ['coarse','refined']:
        out=pair/label
        env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
        env.update(ASPECT_SOURCE_DIR=str(repo),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
            ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
            ASPECT_FAULT_HISTORY_AUDIT='1',ASPECT_BP3_SHARED_CLOCK=str(pair/'clock'),
            ASPECT_BP3_PAIR_LABEL=label,ASPECT_BP3_TARGET_MESH=str(out/'target_cells.txt'),
            ASPECT_BP3_EXACT_TARGET='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
        cmd=['mpirun','-np','4',str(binary),str(out/'run.prm')]
        files=[binary,plugin,out/'run.prm',out/'target_cells.txt',here/'bp3.cc',here/'matched_resolution.h',
               here/'normal_stress_fresh_local4.prm',here/'first_cycle_coarse/original.prm']
        record={'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith('ASPECT_')},
                'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        with (out/'provenance.json').open('x') as f:json.dump(record,f,indent=2)
        log=(out/'run.log').open('x')
        process=subprocess.Popen(cmd,cwd=here,env=env,stdout=log,stderr=subprocess.STDOUT)
        running[label]=(process,log)
    completed={}
    while len(completed)<2:
        for label,(process,log) in running.items():
            if label in completed:continue
            status=process.poll()
            if status is not None:
                log.close();completed[label]={'status':status,'seconds':time.monotonic()-start}
                (pair/label/'execution.json').write_text(json.dumps(completed[label],indent=2)+'\n')
                if status:(pair/'clock/abort').touch()
                print(label,completed[label],flush=True)
        time.sleep(1)
    (pair/'execution.json').write_text(json.dumps(completed,indent=2)+'\n')
    assert all(r['status']==0 for r in completed.values()),'Preserve failed paired evidence; no retry.'
