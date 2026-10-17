#!/usr/bin/env python3
"""Explicit bounded pilot launch; preparation never invokes this program."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ranks',type=int,choices=(1,2,4),required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent;m=json.loads((root/'manifest.json').read_text())
for name,digest in (m['artifacts']|m['inputs']).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert m['qualification_status']=='bounded fixed-geometry controls passed'
assert a.resume or not (root/'output').exists(),'Fresh pilot refuses existing output; resume explicitly.'
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS'):env[key]='1'
env['OMP_PROC_BIND']='false';env['OMPI_MCA_mpi_yield_when_idle']='1'
# Resolve the packaged BP3 SONAME locally. Dependency libraries still require
# the site's one compatible MPI/deal.II environment, recorded below.
env['LD_LIBRARY_PATH']=str(root)+':'+env.get('LD_LIBRARY_PATH','')
command=['mpirun','--map-by','core','--bind-to','core','-np',str(a.ranks),str(root/m['executable']),str(root/('resume.prm' if a.resume else 'parameters.prm'))]
index=0
while (root/f'launch-{index}.json').exists():index+=1
start=time.monotonic();dependencies={}
for name in m['artifacts']:
    dependencies[name]=subprocess.check_output(['ldd',str(root/name)],text=True,env=env)
with (root/f'launch-{index}.log').open('x') as log:
    process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=process.wait(timeout=m['hard_supervisor_seconds'])
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        code=124
record=dict(command=command,ranks=a.ranks,resume=a.resume,seconds=time.monotonic()-start,exit_code=code,dependencies=dependencies,
            environment={key:value for key,value in env.items() if key.startswith(('OMP_','OMPI_','DEAL_II_','ASPECT_','OPENBLAS_')) or key=='LD_LIBRARY_PATH'})
(root/f'launch-{index}.json').write_text(json.dumps(record,indent=2)+'\n');raise SystemExit(code)
