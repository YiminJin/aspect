#!/usr/bin/env python3
"""Fresh native horizontal-domain controls. C++ advances all physical states."""
import hashlib,json,os,re,shutil,subprocess,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'build-particle-free/cohesive-interface-domain'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_domain'
HOST=ROOT/'build-particle-free/coulomb-native/aspect-release'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name,length,nx,shift,timeout=1800,resume=False):
 folder=WORK/'runs'/name;output=folder/'output';folder.mkdir(parents=True,exist_ok=True)
 assert resume or not output.exists(),f'Preserve prior output {output}'
 label=name+('-resume' if resume else '')
 first=.5/length;last=1-first;seedfirst=(2.5+shift)/length;seedlast=(5.5+shift)/length
 template=(ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.prm').read_text()
 text=template.replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/plugin/libcohesive_interface_native.release.so',str(WORK/'build/libcohesive_interface_native.release.so')).replace('set X extent = 1',f'set X extent = {length}').replace('set Y extent = 0.125','set Y extent = 1').replace('set X repetitions = 4',f'set X repetitions = {nx}').replace('set Y repetitions = 40','set Y repetitions = 80').replace('set End time = 0.004','set End time = 2.47').replace('set Maximum time step = 0.002','set Maximum time step = 0.0025').replace('set Maximum first time step = 0.002','set Maximum first time step = 0.0025').replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/smoke/output',str(output)).replace('set Steps between checkpoint = 1','set Steps between checkpoint = 20')
 text+='''\nsubsection Time stepping
 set List of model names = function
 subsection Function
  set Variable names = time
  set Function expression = if(time < 2.3-1e-10, 0.0025, 0.00125)
 end
end
'''
 if resume:text+='\nset Resume computation = true\n'
 prm=WORK/(label+'.prm');prm.write_text(text)
 env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),CI_EXACT_JUMP='1',CI_FINITE='1',CI_NX=str(nx),CI_LENGTH=str(length),CI_HEIGHT='1',CI_WIDTH='.2',CI_QUADRATURE='3',CI_MODE='front',CI_FIRST=format(first,'.17g'),CI_LAST=format(last,'.17g'),CI_INSERT_TIME='-1',CI_SEED_FIRST=format(seedfirst,'.17g'),CI_SEED_LAST=format(seedlast,'.17g'),CI_LOAD_END='2.45',CI_BUFFER_M='.25',CI_SIGNED_SOLVER='1',CI_REMOTE_GUARD='1',CI_DETACHED_GUARD='1',CI_SNAPSHOT_TIMES='2.30,2.45,2.47',CI_EVOLUTION_FAILURE=str(output/'rejected-signed-trial.txt'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 if resume:env['CI_DERIVATIVE_ROUNDOFF_REFERENCE']='1'
 subprocess.run([str(HOST),'--validate',str(prm)],env=env,stdout=(WORK/(label+'-validate.log')).open('w'),stderr=subprocess.STDOUT,timeout=30,check=True)
 state=dict(last_host_time_s=0.,preserved_checkpoint=False,reader_errors=[]);start=time.perf_counter();observed=0.
 log=(WORK/(label+'-aspect.log')).open('w')
 process=subprocess.Popen([str(HOST),str(prm)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 def reader():
  try:
   for line in process.stdout:
    log.write(line);log.flush();match=re.search(r'Timestep\s+\d+:\s+t=\s*([0-9.eE+-]+)',line)
    if match:state['last_host_time_s']=float(match.group(1))
    if '*** Snapshot' in line and not state['preserved_checkpoint'] and 2.29<state['last_host_time_s']<2.3:
     target=WORK/'preserved'/name;target.mkdir(parents=True,exist_ok=True);shutil.copytree(output/'restart',target/'restart');shutil.copy2(prm,target/'parameters.prm')
     state['preserved_checkpoint']=True;state['checkpoint_host_last_s']=state['last_host_time_s']
  except Exception as e:state['reader_errors'].append(repr(e))
 thread=threading.Thread(target=reader);thread.start();stop=None
 while process.poll() is None:
  try:
   status=Path(f'/proc/{process.pid}/status').read_text();match=re.search(r'VmHWM:\s+(\d+)',status)
   if match:observed=max(observed,int(match.group(1))/1024.)
  except OSError:pass
  elapsed=time.perf_counter()-start
  progress=dict(name=name,elapsed_s=elapsed,observed_peak_RSS_MiB=observed,**state)
  (WORK/(label+'-progress.json')).write_text(json.dumps(progress,indent=2)+'\n')
  if elapsed>timeout or observed>6144:
   stop='wall budget' if elapsed>timeout else '6 GiB memory budget';process.terminate()
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill();process.wait()
   break
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:pass
 thread.join();log.close()
 result=dict(name=name,length_m=length,nx=nx,ny=80,physical_x_shift_m=shift,seed_native_m=[2.5+shift,5.5+shift],seed_physical_m=[2.5,5.5],Gamma_physical_m=[.5-shift,length-.5-shift],dx_m=length/nx,dy_m=.0125,seconds=time.perf_counter()-start,observed_peak_RSS_MiB=observed,exit_code=process.returncode,stop=stop,timeout_s=timeout,plugin_sha256=sha(WORK/'build/libcohesive_interface_native.release.so'),parameter_sha256=sha(prm),**state)
 result['resume']=resume
 (OUT/(label+'-run.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 link=OUT/name
 if not link.exists():link.symlink_to(output,target_is_directory=True)
 assert process.returncode==0,(name,stop,process.returncode)
 assert state['preserved_checkpoint'] and not state['reader_errors'],state
 return output

def main():
 import sys
 OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
 stage=sys.argv[1] if len(sys.argv)>1 else 'reference'
 if stage=='reference':run('period-8',8.,128,0.,900)
 elif stage=='enlarged':
  assert json.loads((OUT/'reference-identity.json').read_text())['passed']
  run('period-16',16.,256,4.,1800)
 elif stage=='enlarged-resume':
  failed=json.loads((OUT/'period-16-run.json').read_text())
  assert failed['exit_code']==1 and (WORK/'failed-first-attempt/derivative_checks.csv').exists()
  run('period-16',16.,256,4.,1200,resume=True)
 else:raise ValueError(stage)
if __name__=='__main__':main()
