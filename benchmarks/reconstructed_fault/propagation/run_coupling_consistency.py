#!/usr/bin/env python3
"""Bounded fixed-phase serial/two-rank check of the selected work pair."""
from pathlib import Path
import os,subprocess,json,hashlib
root=Path(__file__).resolve().parents[3];build=root/'build-propagation-integration';e=Path(__file__).parent/'coupling_evidence'
env=os.environ.copy();env.update(ASPECT_SOURCE_DIR=str(root),ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1');env['PATH']='/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:'+env['PATH']
for n in [1,2]:
 label=f'coupling_consistency_v3_{n}';out=build/('output-'+label);assert not out.exists();prm=build/(label+'.prm');log=e/(label+'.log')
 prm.write_text(f'''include {root}/benchmarks/reconstructed_fault/propagation/recoverable.prm
set Additional shared libraries = {build}/plugin/libcoupling_consistency.release.so
set Output directory = {out}
set End time = 0.5
set Maximum time step = 0.125
subsection Material model
 subsection Phase field fault
  set Evolve phase field = false
  set Cohesive history representation = surface elastic slip
  set Fault mechanical work measure = bulk conjugate
 end
end
subsection Postprocess
 set List of postprocessors = particles, P2 coupling consistency
end
''')
 cmd=([str(build/'aspect')] if n==1 else ['/opt/openmpi/5.0.6/bin/mpirun','-n','2',str(build/'aspect')])+[str(prm)]
 hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [prm,build/'aspect-release',build/'plugin/libcoupling_consistency.release.so']}
 with log.open('w') as f:code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
 log.with_suffix('.json').write_text(json.dumps(dict(command=cmd,code=code,hashes=hashes),indent=2)+'\n');print(n,code,flush=True)
 if code:raise SystemExit(code)

import csv
left=list(csv.DictReader((build/'output-coupling_consistency_v3_1/consistency.csv').open()))
right=list(csv.DictReader((build/'output-coupling_consistency_v3_2/consistency.csv').open()))
assert len(left)==len(right)==55
for a,b in zip(left,right):
 for k in a:
  tolerance=0 if k in ['step','time','node'] else 1e-11*max(1.,abs(float(a[k])))
  assert abs(float(a[k])-float(b[k]))<=tolerance,(k,a[k],b[k])
