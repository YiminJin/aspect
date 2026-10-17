#!/usr/bin/env python3
"""One bounded serial common-field/frozen-mechanics spatial diagnostic."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('label');p.add_argument('--normal-cells',type=int,choices=[32,64,128],default=32)
p.add_argument('--backend',choices=['DG','FE'],default='DG');p.add_argument('--quadrature',type=int,choices=[12,24],default=12)
p.add_argument('--event',action='store_true');p.add_argument('--divisor',type=int,choices=[128,256],default=128)
a=p.parse_args();assert not a.event or a.backend=='DG';root=Path(__file__).resolve().parents[3]
input_path=root/'build-particle-free/pf6a-spatial/inputs'/(a.label+'.prm');input_path.parent.mkdir(parents=True,exist_ok=True)
assert not input_path.exists()
text=f'''include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_free/pf6a-common.prm
set Start time = {.53125 if a.event else 0}
set End time = {.5625 if a.event else 0}
set Maximum time step = {1/a.divisor}
set Maximum first time step = {1/a.divisor}
subsection Geometry model
 subsection Box
  set Y repetitions = {a.normal_cells//2}
 end
end
subsection Material model
 subsection Phase field fault
  set Bulk history representation = {'DG trial stress' if a.backend=='DG' else 'accepted FE strain'}
  set Enable prescribed DG accepted history = {'true' if a.backend=='DG' else 'false'}
  set FE bulk normal quadrature points = {a.quadrature}
 end
end
subsection Boundary velocity model
 subsection Function
  set Function expression = 0.256*max(0,t-0.00390625)*(y-0.5); 0
 end
end
subsection Postprocess
 subsection PF6a spatial
  set Evolving imported entry = {'true' if a.event else 'false'}
  set Common fields file = $ASPECT_SOURCE_DIR/build-particle-free/pf6a-spatial/common/common.txt
 end
end
'''
if a.event:
 text+='''subsection Boundary velocity model
 set Prescribed velocity boundary indicators = left y:PF6a spatial interval ramp, right y:PF6a spatial interval ramp, bottom:PF6a spatial interval ramp, top:PF6a spatial interval ramp
end
'''
input_path.write_text(text)
cmd=[sys.executable,str(root/'benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py'),a.label,'--input',str(input_path),'--plugin',str(root/'build-particle-free/plugin/libpf6a_spatial_checks.release.so'),'--postprocessor','PF6a spatial','--seconds','180']
observer=(Path(__file__).with_name('pf6a_spatial_checks.cc')).read_bytes()
code=subprocess.run(cmd,cwd=root,env=dict(os.environ,ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')).returncode
record=root/'build-particle-free/runs'/a.label/'record.json'
if record.exists():
 r=json.loads(record.read_text());r['spatial_case']=vars(a);r['environment']={'ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION':'1'};r['observer_source_sha256']=hashlib.sha256(observer).hexdigest();(record.parent/'observer-source.cc').write_bytes(observer);record.write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(code)
