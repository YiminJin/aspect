#!/usr/bin/env python3
"""One fresh, bounded native confirmation of the measured 1D selection."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('label');p.add_argument('selection',type=Path);p.add_argument('--batch-snapshots',action='store_true',help='Batch diagnostic stress output; verify the original evaluator at startup/onset/end.');a=p.parse_args()
root=Path(__file__).resolve().parents[3];s=json.loads(a.selection.read_text());assert s['selected_normal_cells']==512 and s['divisor']==128
path=root/'build-particle-free/pf6a-reference/inputs'/(a.label+'.prm');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
path.write_text('''include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_free/pf6a-common.prm
set End time = 0.5625
set Maximum time step = 0.0078125
set Maximum first time step = 0.0078125
subsection Geometry model
 subsection Box
  set Y repetitions = 256
 end
end
subsection Time stepping
 subsection Function
  set Function expression = 0.0078125
 end
end
subsection Material model
 subsection Phase field fault
  set FE bulk normal quadrature points = 12
 end
end
subsection Boundary velocity model
 subsection Function
  set Function expression = 0.256*max(0,t-0.00390625)*(y-0.5); 0
 end
end
subsection Postprocess
 subsection PF6a checks
  set Operator and rejection checks = false
 end
end
''')
if a.batch_snapshots:
 with path.open('a') as stream:stream.write('''
subsection Postprocess
 subsection PF6a checks
  set Batch diagnostic stress evaluation = true
  set Verify batched diagnostic stress = true
 end
end
''')
source=Path(__file__).with_name('pf6a_checks.cc').read_bytes()
cmd=[sys.executable,str(root/'benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py'),a.label,'--input',str(path),'--plugin',str(root/'build-particle-free/plugin/libpf6a_checks.release.so'),'--postprocessor','PF6a cohesive endpoint','--seconds','900']
code=subprocess.run(cmd,cwd=root,env=dict(os.environ,ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')).returncode
record=root/'build-particle-free/runs'/a.label/'record.json'
if record.exists():
 r=json.loads(record.read_text());r['spatial_reference_case']=dict(normal_cells=512,divisor=128,quadrature=12,initialization='same physical seed table; fresh unloaded histories',selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest());r['observer_source_sha256']=hashlib.sha256(source).hexdigest();(record.parent/'observer-source.cc').write_bytes(source);r['environment']={'ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION':'1'};r['batch_diagnostic_snapshots']=a.batch_snapshots;record.write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(code)
