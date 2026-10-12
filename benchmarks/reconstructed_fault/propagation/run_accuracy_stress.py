#!/usr/bin/env python3
"""One native initialization, then recoverable local zero-strain probes."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];B=ROOT/'build-propagation-integration';E=HERE/'accuracy_energy_evidence';E.mkdir(exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('label');label=parser.parse_args().label;out=B/('output-'+label);log=E/(label+'.log');assert not out.exists() and not log.exists()
exe=B/'irreversibility-final-aspect';plugin=B/'plugin/libaccuracy_stress.release.so';prm=B/(label+'.prm')
prm.write_text(f'''include {HERE}/recoverable.prm
set Additional shared libraries = {plugin}
set Output directory = {out}
set End time = 0
subsection Boundary velocity model
 subsection Function
  set Function expression = 0; 0
 end
end
subsection Postprocess
 set List of postprocessors = particles, P2 accuracy stress
end
subsection Material model
 subsection Phase field fault
  set Cohesive history representation = surface elastic slip
  set Fault mechanical work measure = bulk conjugate
 end
end
subsection Solver parameters
 subsection Phase field solver parameters
  set Enforce phase irreversibility = true
 end
end
''')
env=os.environ.copy();env.update(ASPECT_SOURCE_DIR=str(ROOT),OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1');env['PATH']='/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:'+env['PATH']
files=[exe,plugin,prm,HERE/'accuracy_stress.cc',HERE/'recoverable.prm',HERE/'material_feedback.prm',HERE/'coupled.prm',ROOT/'tests/phase_field_fault_stage_i.prm']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};start=time.monotonic()
with log.open('w') as f:
 try:code=subprocess.run([str(exe),str(prm)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
 except subprocess.TimeoutExpired:code=124
r=dict(code=code,seconds=time.monotonic()-start,hashes=hashes);(E/(label+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(code)
