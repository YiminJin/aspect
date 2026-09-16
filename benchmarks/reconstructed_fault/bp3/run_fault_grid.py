"""One fresh locally bisected fault trajectory on the qualified fixed bulk mesh.

The recorded 100-m trajectory is reused only while its steps pass the actual
controller guard. Preserve any failure; do not rerun a trajectory implicitly.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

here=Path(__file__).resolve().parent;repo=here.parents[2]
base=here/'junction-matched-qualified-local4/refined'
out=here/'fault-grid-50-local4'
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if args.mode=='prepare':
    out.mkdir()
    with (base/'fault_0.csv').open() as f:old=list(csv.DictReader(f))
    assert len(old)==1156
    points=[];common=[];splits=[]
    for i,r in enumerate(old):
        common.append(len(points));points.append((float(r['x']),float(r['y'])))
        if i+1<len(old):
            b=old[i+1];lo=float(b['xd']);hi=float(r['xd'])
            # Split complete existing elements in the requested window. The
            # uppermost retained old node is 43998.413 m, not exactly 44 km.
            if lo>=36000-1e-7 and hi<=44000+1e-7:
                points.append(tuple((float(r[k])+float(b[k]))*.5 for k in ['x','y']))
                splits.append([i,lo,hi])
    assert len(splits)==80 and len(points)==1236
    assert all(points[j]==(float(r['x']),float(r['y'])) for r,j in zip(old,common))
    assert abs(float(old[755]['xd'])-40000)<1e-7
    # Fixed-geometry reconstruction subdivides each supplied edge using ceil.
    # A 101-m cap prevents re-splitting roundoff-long 100-m edges. The source
    # uses this parameter only for resampling; every desired node is explicit.
    assert max(math.dist(a,b) for a,b in zip(points,points[1:]))<101
    (out/'fault.txt').write_text(''.join(f'{x:.17g} {y:.17g} 0.6\n' for x,y in points))
    plan=dict(old_vertices=len(old),new_vertices=len(points),split_elements=splits,common_node_map=common,
        refined_interval=[min(r[1] for r in splits),max(r[2] for r in splits)],
        old_junction_node=755,new_junction_node=common[755],structural_spacing_cap=101,
        baseline=str(base),baseline_fault_sha256=digest(base/'fault_0.csv'),
        baseline_clock_sha256=digest(base/'accepted_steps.csv'))
    (out/'grid_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'run.prm').write_text(f'''# Fault-grid-only comparison; no late-history interpolation.
include {base}/run.prm
set Output directory = {out}
set Resume computation = false
subsection Fault reconstruction
  set Structural point spacing = 101
  set Prescribed faults file = {out}/fault.txt
end
subsection Time stepping
  set List of model names = convection time step, reconstructed fault time step, BP3 replay cap
end
subsection Termination criteria
  set Termination criteria = end time, end step
  set End step = 13
end
''')
    print(json.dumps({k:v for k,v in plan.items() if k not in ['common_node_map','split_elements']},indent=2))
else:
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(repo),ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
        ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
        ASPECT_FAULT_HISTORY_AUDIT='1',ASPECT_BP3_TIMESTEP_SEQUENCE=str(base/'accepted_steps.csv'),
        ASPECT_BP3_TARGET_MESH=str(base/'target_cells.txt'),ASPECT_BP3_EXACT_TARGET='1',
        ASPECT_BP3_EXPECTED_FAULT=str(out/'fault.txt'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    binary=repo/'build-pf-cpdi/aspect-release'
    cmd=['mpirun','-np','4','--bind-to','core','--map-by','core',str(binary),str(out/'run.prm')]
    files=[binary,here/'build/libbp3.release.so',here/'bp3.cc',here/'matched_resolution.h',
           here/'replay_time_step.h',out/'run.prm',out/'fault.txt',base/'run.prm',base/'target_cells.txt',
           base/'accepted_steps.csv',here/'first_cycle_coarse/original.prm']
    record=dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
                sha256={str(p):digest(p) for p in files})
    with (out/'provenance.json').open('x') as f:json.dump(record,f,indent=2)
    start=time.monotonic()
    with (out/'run.log').open('x') as log:
        result=subprocess.run(cmd,cwd=here,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(status=result.returncode,seconds=time.monotonic()-start)
    (out/'execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert result['status']==0,'Preserve failure; review the admissible-clock fallback, no automatic retry.'
