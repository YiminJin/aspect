#!/usr/bin/env python3
"""One fresh bounded P2 material run; never reuse output or another case's history."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--gc',type=float,required=True);p.add_argument('--amplitude',type=float,required=True);p.add_argument('--dt',type=float,default=.125);a=p.parse_args()
assert a.gc in (26.6666666667,106.6666666667,266.6666666667)
assert a.amplitude in (.004,.008,.016) and a.dt in (.125,.0625)
here=Path(__file__).resolve().parent;repo=here.parents[3];output=here/('output-'+a.label);log=here/'evidence'/(a.label+'.log');prm=here/'inputs'/(a.label+'.prm')
assert not output.exists() and not log.exists() and not prm.exists(), 'Refusing evidence reuse'
plugin=here/'build/libp2_material_feedback.release.so';exe=repo/'build-propagation-integration/aspect-release';fixture=here.parent/'coupled.prm'
prm.write_text(f'''include {fixture}
set Maximum time step = {a.dt:.17g}
set Additional shared libraries = {plugin}
set Output directory = {output}
subsection Material model
  subsection Phase field fault
    set Critical energy release rates = {a.gc:.17g}, {a.gc:.17g}
    set I h projection = consistent
  end
end
subsection Fault reconstruction
  set Prescribed faults file = {here/'inputs/straight06.txt'}
end
subsection Boundary velocity model
  subsection Function
    set Function expression = {a.amplitude:.17g}*t*(y-0.5); 0
  end
end
''')
env=os.environ.copy();env.update(ASPECT_SOURCE_DIR=str(repo),OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',ASPECT_FAULT_NONLINEAR_DIAGNOSTIC='1');env['PATH']='/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:'+env['PATH']
files=[exe,plugin,prm,fixture,here/'inputs/straight06.txt',here/'feedback.cc',here/'diagnostics.h',repo/'tests/phase_field_fault_stage_i.prm',repo/'tests/phase_field_fault_test_access.h',repo/'tests/phase_field_test_access.h']
record=dict(label=a.label,gc=a.gc,amplitude=a.amplitude,dt=a.dt,command=[str(exe),str(prm)],cwd=str(repo),ranks=1,threads=1,environment={k:env[k] for k in ['ASPECT_SOURCE_DIR','OMP_NUM_THREADS','DEAL_II_NUM_THREADS','OPENBLAS_NUM_THREADS','PATH','ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION','ASPECT_FAULT_NONLINEAR_DIAGNOSTIC']},hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
start=time.monotonic()
with log.open('x') as f:
 try:code=subprocess.run(record['command'],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
 except subprocess.TimeoutExpired:code=124
record.update(exit_code=code,seconds=time.monotonic()-start,rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,output_bytes=sum(f.stat().st_size for f in output.rglob('*') if f.is_file()))
log.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('hashes','environment')},indent=2));raise SystemExit(code)
