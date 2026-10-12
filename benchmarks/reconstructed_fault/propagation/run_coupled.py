#!/usr/bin/env python3
"""Bounded native mechanical-feedback run, with the existing frozen RK2 control."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--dt',type=float,required=True);p.add_argument('--subcritical',action='store_true');a=p.parse_args()
repo=Path(__file__).resolve().parents[3];here=Path(__file__).resolve().parent
build=repo/'build-propagation-integration';evidence=here/'coupled_evidence';evidence.mkdir(exist_ok=True)
output=build/('output-'+a.label);log=evidence/(a.label+'.log');assert not output.exists() and not log.exists()
fixture=here/('coupled_subcritical.prm' if a.subcritical else 'coupled.prm')
plugin=build/'plugin/libcoupled_feedback.release.so';exe=build/'aspect';prm=build/(a.label+'.prm')
prm.write_text(f'include {fixture}\nset Maximum time step = {a.dt:.17g}\nset Additional shared libraries = {plugin}\nset Output directory = {output}\n')
env=os.environ.copy();env.update(ASPECT_SOURCE_DIR=str(repo),OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',ASPECT_FAULT_NONLINEAR_DIAGNOSTIC='1')
env['PATH']='/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:'+env['PATH']
command=[str(exe),str(prm)];files=[exe.resolve(),plugin,prm,here/'coupled_feedback.cc',fixture,repo/'tests/phase_field_fault_test_access.h',repo/'source/material_model/phase_field_fault/history.cc',repo/'source/particle/integrator/rk_2.cc']
input_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
start=time.monotonic()
with log.open('w') as f:
 try:code=subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
 except subprocess.TimeoutExpired:code=124
r=dict(command=command,code=code,seconds=time.monotonic()-start,rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,dt=a.dt,freeze_control='ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1',hashes=input_hashes,output_bytes=sum(f.stat().st_size for f in output.rglob('*') if f.is_file()))
log.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='hashes'},indent=2));raise SystemExit(code)
