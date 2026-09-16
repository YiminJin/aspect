"""One causal uniform-sliding experiment: extend bulk refinement to 48 km.

Fault grid, 40-km physical junction, phase law, completed boundary columns,
histories, material and solver settings are retained. New FE phase and I_h are
initialized normally on the actual new mesh; no old volume field is imported.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE/'top-source-paired-50-local4'
OUT=HERE/'deep-mesh-shift48-uniform-local4'


def prepare():
    OUT.mkdir()
    leaves=[];split=0
    for path in sorted(BASE.glob('initial_mesh_*.csv')):
        for row in csv.DictReader(path.open()):
            def visit(cell,x,y,h):
                nonlocal split
                sn=3**.5/2;xt=50000*(1+.5/sn)
                s=(xt-x)*.5+(100000-y)*sn
                r=abs((xt-x)*sn-(100000-y)*.5)
                if 43500<s<48000 and r<1500 and h>48.828125+1e-7:
                    split+=1;root,children=cell.split(':');root=root.split('_')[0]
                    for i in range(4):
                        visit(f'{root}_{len(children)+1}:{children}{i}',
                              x+(1 if i%2 else -1)*h/4,
                              y+(1 if i//2 else -1)*h/4,h/2)
                else:leaves.append(cell)
            visit(row['cell'],float(row['x']),float(row['y']),float(row['h']))
    (OUT/'target_cells.txt').write_text('\n'.join(sorted(leaves))+'\n')
    (OUT/'clock.csv').write_bytes((BASE/'clock.csv').read_bytes())
    # Only boundary profiles receive outside-box completion. Their geometry,
    # physical cells, phase profile and fault QPs are unchanged by this local
    # interior refinement. Reuse that independently checked immutable input.
    (OUT/'completion.txt').write_bytes((BASE/'completion.txt').read_bytes())
    (OUT/'run.prm').write_text(f'''include {BASE}/run.prm
set Output directory = {OUT}
subsection Postprocess
  subsection BP3
    set Bottom normalization completion file = {OUT}/completion.txt
  end
end
subsection Termination criteria
  set Termination criteria = end time, end step, BP3 replay complete
  set End step = 2
end
''')
    (OUT/'preflight.json').write_text(json.dumps(dict(question=
        'Does the fixed uniform-sliding stress/profile feature follow the bulk refinement edge from 44 to 48 km while the fault grid and physical 40-km junction stay fixed?',
        requested_leaves=len(leaves),split_cells=split,mesh_grading='Existing deal.II closure; verify realized descendants offline.',
        expected_seconds='150–240',hard_cap_seconds=600,expected_memory='approximately 1.5 GiB per rank',
        mesh_dependent_inputs='Fresh FE distance profile, volume sampling and projected I_h; endpoint completion unchanged only because endpoint cells/fault QPs are identical.'),indent=2)+'\n')


def run():
    assert (OUT/'run.prm').exists() and not (OUT/'run.log').exists()
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(json.loads((BASE/'provenance.json').read_text())['environment'])
    env.pop('ASPECT_BP3_EXACT_TARGET',None)
    env.update(ASPECT_BP3_TARGET_MESH=str(OUT/'target_cells.txt'),
               ASPECT_BP3_TIMESTEP_SEQUENCE=str(OUT/'clock.csv'),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    command=['timeout','--signal=TERM','--kill-after=15','600','mpirun','-np','4',
             '--bind-to','core','--map-by','core',str(REPO/'build-pf-cpdi/aspect-release'),str(OUT/'run.prm')]
    paths=[Path(__file__),OUT/'run.prm',OUT/'clock.csv',OUT/'target_cells.txt',OUT/'completion.txt',
           HERE/'bp3.cc',HERE/'replay_stop.h',HERE/'build/libbp3.release.so',
           REPO/'build-pf-cpdi/aspect-release',HERE/'fault-grid-50-local4/fault.txt',
           HERE/'mature-fault-50-local4/prestress.txt']
    (OUT/'provenance.json').write_text(json.dumps(dict(command=command,
        environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    start=time.monotonic()
    with (OUT/'run.log').open('x') as log:
        result=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
    record=dict(status=result.returncode,seconds=time.monotonic()-start,
                child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (OUT/'execution.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    assert result.returncode==0,'Preserve failed evidence; no automatic retry.'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run'])
    if parser.parse_args().action=='prepare':prepare()
    else:run()
