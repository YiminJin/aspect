#!/usr/bin/env python3
"""Bounded serial native solves; physical states are computed in C++."""
import json,os,re,shutil,subprocess,threading,time,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'build-particle-free/cohesive-interface-boundary'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_boundary'
HOST=ROOT/'build-particle-free/coulomb-native/aspect-release'

def native(name,L,nx,end,load_end,formula,*,checkpoint=None,fraction=None,extend=False,timeout=1500,snapshots='',expected_failure=None):
 folder=WORK/'runs'/name;output=folder/'output';assert not output.exists(),output
 folder.mkdir(parents=True,exist_ok=True)
 if checkpoint:
  output.mkdir();shutil.copytree(checkpoint,output/'restart')
 template=(ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.prm').read_text()
 replacements={'$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/plugin/libcohesive_interface_native.release.so':str(WORK/'build/libcohesive_interface_native.release.so'),'set X extent = 1':f'set X extent = {L}','set Y extent = 0.125':'set Y extent = 1','set X repetitions = 4':f'set X repetitions = {nx}','set Y repetitions = 40':'set Y repetitions = 80','set End time = 0.004':f'set End time = {end}','set Maximum time step = 0.002':'set Maximum time step = 0.0025','set Maximum first time step = 0.002':'set Maximum first time step = 0.0025','$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/smoke/output':str(output),'set Steps between checkpoint = 1':'set Steps between checkpoint = 20'}
 for a,b in replacements.items():template=template.replace(a,b)
 template+='\nsubsection Time stepping\n set List of model names = function\n subsection Function\n  set Variable names = time\n  set Function expression = '+formula+'\n end\nend\n'
 if checkpoint:template+='\nset Resume computation = true\n'
 prm=WORK/(name+'.prm');prm.write_text(template)
 shift=L/2-4;env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),CI_EXACT_JUMP='1',CI_FINITE='1',CI_NX=str(nx),CI_LENGTH=str(float(L)),CI_HEIGHT='1',CI_WIDTH='.2',CI_QUADRATURE='3',CI_MODE='front',CI_FIRST=format(.5/L,'.17g'),CI_LAST=format(1-.5/L,'.17g'),CI_INSERT_TIME='-1',CI_SEED_FIRST=format((2.5+shift)/L,'.17g'),CI_SEED_LAST=format((5.5+shift)/L,'.17g'),CI_LOAD_END=str(load_end),CI_BUFFER_M='.25',CI_SIGNED_SOLVER='1',CI_REMOTE_GUARD='1',CI_DETACHED_GUARD='1',CI_SNAPSHOT_TIMES=snapshots,CI_DERIVATIVE_ROUNDOFF_REFERENCE='1',CI_EVOLUTION_FAILURE=str(output/'rejected-signed-trial.txt'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 if fraction is not None:env['CI_FAULT_FRACTION']=str(fraction)
 if extend:env['CI_EXTEND_LOAD_END']='1'
 (WORK/(name+'-environment.json')).write_text(json.dumps({k:v for k,v in env.items() if k.startswith('CI_') or k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},indent=2)+'\n')
 with (WORK/(name+'-validate.log')).open('w') as f:subprocess.run([str(HOST),'--validate',str(prm)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
 start=time.perf_counter();peak=0.;state=dict(last_host_time_s=0.,preserved={},reader_errors=[]);log=(WORK/(name+'-aspect.log')).open('w')
 child=subprocess.Popen([str(HOST),str(prm)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 def consume():
  try:
   for line in child.stdout:
    log.write(line);log.flush();match=re.search(r'Timestep\s+\d+:\s+t=\s*([0-9.eE+-]+)',line)
    if match:state['last_host_time_s']=float(match.group(1))
    if '*** Snapshot' in line:
     for label,lo,hi in [('pre-event',2.40,2.43),('pre-hold',2.53,2.55),('hold',2.56,2.58)]:
      if label not in state['preserved'] and lo<state['last_host_time_s']<hi:
       target=WORK/'preserved'/name/label;target.mkdir(parents=True);shutil.copytree(output/'restart',target/'restart');shutil.copy2(prm,target/'parameters.prm');state['preserved'][label]=state['last_host_time_s']
  except Exception as e:state['reader_errors'].append(repr(e))
 thread=threading.Thread(target=consume);thread.start();stop=None
 while child.poll() is None:
  try:
   match=re.search(r'VmHWM:\s+(\d+)',Path(f'/proc/{child.pid}/status').read_text())
   if match:peak=max(peak,int(match.group(1))/1024.)
  except OSError:pass
  elapsed=time.perf_counter()-start
  (WORK/(name+'-progress.json')).write_text(json.dumps(dict(elapsed_s=elapsed,observed_peak_RSS_MiB=peak,**state),indent=2)+'\n')
  if elapsed>timeout or peak>6144:
   stop='wall budget' if elapsed>timeout else '6 GiB observed RSS budget';child.terminate()
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   break
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:pass
 thread.join();log.close()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 result=dict(name=name,seconds=time.perf_counter()-start,exit_code=child.returncode,stop=stop,timeout_s=timeout,observed_peak_RSS_MiB=peak,physical_x_shift_m=shift,fault_y_m=.4 if fraction is None else fraction,load_end_s=load_end,end_s=end,plugin_sha256=sha(WORK/'build/libcohesive_interface_native.release.so'),parameter_sha256=sha(prm),**state)
 if expected_failure:
  result['expected_failure_passed']=child.returncode!=0 and expected_failure in (WORK/(name+'-aspect.log')).read_text() and not (output/'history-resume.csv').exists()
 (OUT/(name+'-run.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 (OUT/name).symlink_to(output,target_is_directory=True)
 assert result.get('expected_failure_passed',child.returncode==0),(name,result)
 assert not state['reader_errors'],state

def main():
 stage=sys.argv[1]
 if stage=='reflection':native('reflected',8,128,2.47,2.45,'if(time < 2.3-1e-10,0.0025,0.00125)',fraction=.6,timeout=900,snapshots='2.30,2.45,2.47')
 elif stage=='longer':
  assert json.loads((OUT/'reflection-verification.json').read_text())['passed']
  checkpoint=ROOT/'build-particle-free/cohesive-interface-domain/preserved/period-16/restart'
  native('longer',16,256,2.58,2.55,'if(time < 2.3-1e-10,0.0025,if(time < 2.45-1e-10,0.00125,0.000625))',checkpoint=checkpoint,extend=True,snapshots='2.45,2.50,2.55,2.58')
 elif stage=='replay':
  assert json.loads((OUT/'longer-completion-run.json').read_text())['exit_code']==0
  checkpoint=WORK/'preserved/longer/pre-event/restart'
  native('event-half',16,256,2.58,2.55,'if(time < 2.45-1e-10,0.000625,0.0003125)',checkpoint=checkpoint,timeout=2100,snapshots='2.55,2.58')
 elif stage=='complete':
  assert json.loads((OUT/'longer-run.json').read_text())['stop']=='wall budget'
  checkpoint=WORK/'runs/longer/output/restart'
  native('longer-completion',16,256,2.58,2.55,'if(time < 2.3-1e-10,0.0025,if(time < 2.45-1e-10,0.00125,0.000625))',checkpoint=checkpoint,timeout=1100,snapshots='2.58')
 elif stage=='reject-past-loading':
  checkpoint=WORK/'preserved/longer/hold/restart'
  native('post-hold-extension-rejected',16,256,2.62,2.60,'0.000625',checkpoint=checkpoint,extend=True,timeout=30,expected_failure='Loading extension would reinterpret accepted boundary history')
 else:raise ValueError(stage)
if __name__=='__main__':main()
