#!/usr/bin/env python3
"""Bounded fresh material comparison; no restart or mechanically prescribed H."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--case',choices=['reference','primary','optional'],required=True);p.add_argument('--amplitude',type=float,choices=[.004,.008,.016],required=True);p.add_argument('--dt',type=float,choices=[.125,.0625],default=.125);a=p.parse_args()
repo=Path(__file__).resolve().parents[3];here=Path(__file__).resolve().parent
build=repo/'build-propagation-integration';e=here/'material_evidence';e.mkdir(exist_ok=True)
output=build/('output-'+a.label);log=e/(a.label+'.log');assert not output.exists() and not log.exists()
Gc={'reference':26.6666666667,'primary':106.6666666667,'optional':266.6666666667}[a.case]
plugin=build/'plugin/libmaterial_feedback.release.so';exe=build/'aspect';prm=build/(a.label+'.prm')
prm.write_text(f'''include {here}/material_feedback.prm
set Maximum time step = {a.dt:.17g}
set Additional shared libraries = {plugin}
set Output directory = {output}
subsection Material model
  subsection Phase field fault
    set Critical energy release rates = {Gc:.17g}, {Gc:.17g}
  end
end
subsection Boundary velocity model
  subsection Function
    set Function expression = {a.amplitude:.17g}*t*(y-0.5); 0
  end
end
''')
env=os.environ.copy();env.update(ASPECT_SOURCE_DIR=str(repo),OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',ASPECT_FAULT_NONLINEAR_DIAGNOSTIC='1');env['PATH']='/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:'+env['PATH']
command=[str(exe),str(prm)]
files=[exe.resolve(),plugin,prm,here/'material_feedback.cc',here/'coupled_feedback.cc',here/'material_feedback.prm',here/'coupled.prm',here/'straight_core06.txt',repo/'tests/phase_field_fault_stage_i.prm',repo/'tests/phase_field_test_access.h',repo/'tests/phase_field_fault_test_access.h']
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
start=time.monotonic()
with log.open('w') as f:
 try:code=subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
 except subprocess.TimeoutExpired:code=124
r=dict(command=command,code=code,seconds=time.monotonic()-start,rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,case=a.case,Gc=Gc,amplitude=a.amplitude,dt=a.dt,hashes=hashes,output_bytes=sum(f.stat().st_size for f in output.rglob('*') if f.is_file()))
log.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='hashes'},indent=2));raise SystemExit(code)
