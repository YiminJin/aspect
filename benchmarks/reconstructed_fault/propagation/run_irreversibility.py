#!/usr/bin/env python3
"""Fresh runs only; all H updates remain native. Preserve failed outputs."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--mode',choices=['zero','fixed','coupled'],default='coupled');p.add_argument('--dt',type=float,choices=[.125,.0625,.03125],default=.125);p.add_argument('--work',action='store_true');p.add_argument('--replay',action='store_true');p.add_argument('--amplitude',type=float,default=.016);a=p.parse_args()
here=Path(__file__).resolve().parent;repo=here.parents[2];b=repo/'build-propagation-integration';e=here/'irreversibility_evidence';e.mkdir(exist_ok=True)
out=b/('output-'+a.label);log=e/(a.label+'.log');assert not out.exists() and not log.exists()
plugin=b/'plugin/libirreversibility_feedback.release.so';exe=b/'aspect';prm=b/(a.label+'.prm')
amplitude=0 if a.mode=='zero' else a.amplitude
prm.write_text(f'''include {here}/recoverable.prm
set Additional shared libraries = {plugin}
set Output directory = {out}
set Maximum time step = {a.dt:.17g}
set End time = {.25 if a.mode=='zero' else 1}
subsection Boundary velocity model
 subsection Function
  set Function expression = {amplitude:.17g}*t*(y-0.5); 0
 end
end
'''+'''subsection Postprocess
 set List of postprocessors = particles, P2 irreversibility feedback
end
subsection Material model
 subsection Phase field fault
  set Cohesive history representation = surface elastic slip
 end
end
'''+('''subsection Material model
 subsection Phase field fault
  set Fault mechanical work measure = bulk conjugate
 end
end
''' if a.work else '')+('''subsection Material model
 subsection Phase field fault
  set Evolve phase field = false
 end
end
''' if a.mode=='fixed' else '')+f'''subsection Solver parameters
 subsection Phase field solver parameters
  set Enforce phase irreversibility = {'false' if a.replay else 'true'}
 end
end
''')
env=os.environ.copy();env.update(ASPECT_SOURCE_DIR=str(repo),OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',ASPECT_FAULT_NONLINEAR_DIAGNOSTIC='1');env['PATH']='/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:'+env['PATH']
files=[exe.resolve(),plugin,prm,*[here/f for f in ['recoverable_feedback.cc','coupling_feedback.cc','irreversibility_feedback.cc','recoverable.prm','material_feedback.cc','coupled_feedback.cc']]]
hashes={str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};start=time.monotonic()
with log.open('w') as f:
 try:code=subprocess.run([str(exe),str(prm)],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240).returncode
 except subprocess.TimeoutExpired:code=124
r=dict(label=a.label,replay=a.replay,mode=a.mode,work=a.work,amplitude=amplitude,dt=a.dt,code=code,seconds=time.monotonic()-start,hashes=hashes)
log.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='hashes'}));raise SystemExit(code)
