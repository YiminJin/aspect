#!/usr/bin/env python3
"""Sequential native signed-evolution runs; no trajectory supplied to C++."""
import os,sys,json,time,subprocess,shutil,hashlib
import numpy as np
from run_cohesive_interface_evolution import ROOT,WORK,OUT,HOST

def run(name,nx=64,ny=40,dt=.005,end=2.42,first=.125,last=.875,hold=2.7,resume=False,folder=None,signed=True,extra=None,expected_guard=False):
 destination=WORK/'runs'/(folder or name)/'output';destination.parent.mkdir(exist_ok=True,parents=True)
 assert resume or not destination.exists(),f'Preserve {destination}'
 template=(ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.prm').read_text()
 text=template.replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/plugin/libcohesive_interface_native.release.so',str(WORK/'build/libcohesive_interface_native.release.so')).replace('set X extent = 1','set X extent = 8').replace('set Y extent = 0.125','set Y extent = 1').replace('set X repetitions = 4',f'set X repetitions = {nx}').replace('set Y repetitions = 40',f'set Y repetitions = {ny}').replace('set End time = 0.004',f'set End time = {end}').replace('set Maximum time step = 0.002',f'set Maximum time step = {dt}').replace('set Maximum first time step = 0.002',f'set Maximum first time step = {dt}').replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/smoke/output',str(destination)).replace('set Steps between checkpoint = 1','set Steps between checkpoint = 20')
 if resume:text+='\nset Resume computation = true\n'
 prm=WORK/(name+'.prm');prm.write_text(text)
 env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),CI_EXACT_JUMP='1',CI_FINITE='1',CI_NX=str(nx),CI_LENGTH='8',CI_HEIGHT='1',CI_WIDTH='.2',CI_QUADRATURE='3',CI_MODE='front',CI_FIRST=str(first),CI_LAST=str(last),CI_INSERT_TIME='-1',CI_SEED_FIRST='.3125',CI_SEED_LAST='.6875',CI_LOAD_END=str(hold),CI_BUFFER_M='.25',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 env['CI_EVOLUTION_FAILURE']=str(destination/'rejected-signed-trial.txt')
 if signed:env['CI_SIGNED_SOLVER']='1';env['CI_REMOTE_GUARD']='1'
 env.update(extra or {})
 plugin_sha=hashlib.sha256((WORK/'build/libcohesive_interface_native.release.so').read_bytes()).hexdigest()
 start=time.perf_counter()
 with (WORK/(name+'-aspect.log')).open('w') as log:
  result=subprocess.run([str(HOST),str(prm)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 seconds=time.perf_counter()-start
 record=dict(name=name,nx=nx,ny=ny,dt_s=dt,end_s=end,hold_s=hold,first_m=first*8,last_m=last*8,seconds=seconds,exit_code=result.returncode,timeout_s=900,signed=signed,plugin_sha256=plugin_sha,prm_sha256=hashlib.sha256(prm.read_bytes()).hexdigest())
 target=OUT/name;target.mkdir(exist_ok=True,parents=True)
 for pattern in ['*.csv','*.archive','*trial*.txt']:
  for p in destination.glob(pattern):shutil.copy2(p,target/p.name)
 (OUT/(name+'-run.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
 if expected_guard:
  assert result.returncode!=0 and 'Cohesive zone reached declared finite candidate buffer' in (WORK/(name+'-aspect.log')).read_text()
 else:assert result.returncode==0,f'{name}: preserve checkpoint/trial and diagnose.'
 return target

def main():
 stage=sys.argv[1] if len(sys.argv)>1 else 'pilot'
 if stage=='pilot':
  p=run('straight-pilot-final');old=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_insertion/growth-64'
  a=np.genfromtxt(p/'history.csv',names=True,delimiter=',');b=np.genfromtxt(old/'history.csv',names=True,delimiter=',');b=b[b['time_s']<=2.42+1e-12]
  differences={}
  for field in ['bulk_J_m','bond_J_m','fracture_increment_J_m','friction_J_m','Maxwell_J_m']:
   av=a[field] if field in ['bulk_J_m','bond_J_m'] else np.cumsum(a[field]);bv=b[field] if field in ['bulk_J_m','bond_J_m'] else np.cumsum(b[field]);y=np.interp(a['time_s'],b['time_s'],bv);differences[field]=float(max(abs(av-y)))
  (OUT/'pilot-reference.json').write_text(json.dumps(differences,indent=2)+'\n');print(differences)
 elif stage=='long-coarse':run('straight-long-coarse',nx=64,ny=40,dt=.005,end=2.55,first=.0625,last=.9375,hold=2.5)
 elif stage=='hold-coarse':run('straight-hold-coarse',nx=64,ny=40,dt=.0025,end=2.5,first=.0625,last=.9375,hold=2.45)
 elif stage=='long':
  split=0.
  for _ in range(920):split+=.0025
  run('straight-fine-prefix',nx=128,ny=80,dt=.0025,end=split,first=.0625,last=.9375,hold=2.45)
  preserved=WORK/'preserved/fine-prefix';preserved.parent.mkdir(parents=True,exist_ok=True);assert not preserved.exists();shutil.copytree(WORK/'runs/straight-fine-prefix/output',preserved)
  run('straight-long',nx=128,ny=80,dt=.0025,end=2.47,first=.0625,last=.9375,hold=2.45,resume=True,folder='straight-fine-prefix')
 elif stage=='half-window':
  folder=WORK/'runs/straight-half-window/output';assert not folder.exists();folder.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(WORK/'preserved/fine-prefix',folder)
  run('straight-half-window',nx=128,ny=80,dt=.00125,end=2.47,first=.0625,last=.9375,hold=2.45,resume=True,folder='straight-half-window')
 elif stage=='guard-replay':run('guard-replay',nx=64,ny=40,dt=.005,end=2.55,first=.0625,last=.9375,hold=2.5,resume=True,folder='straight-long-coarse',expected_guard=True)
 elif stage=='final-restart':
  folder=WORK/'runs/final-restart/output';folder.parent.mkdir(parents=True,exist_ok=True);assert not folder.exists();shutil.copytree(WORK/'preserved/fine-prefix',folder)
  p=run('final-restart',nx=128,ny=80,dt=.0025,end=2.47,first=.0625,last=.9375,hold=2.45,resume=True,folder='final-restart',extra={'CI_STATE_AUDIT':'1'})
  old=OUT/'straight-long'
  for name in ['states-resume.csv','nodal_history-resume.csv']:
   assert (p/name).read_bytes()==(old/name).read_bytes(),name
  a=np.genfromtxt(p/'history-resume.csv',names=True,delimiter=',');b=np.genfromtxt(old/'history-resume.csv',names=True,delimiter=',');names=[n for n in a.dtype.names if n!='step_seconds'];assert all(np.array_equal(a[n],b[n]) for n in names)
  (OUT/'final-restart.json').write_text(json.dumps(dict(final_source_reproduces_window_bitwise=True,all_Gauss_histories_bitwise=True,nodal_and_bulk_history_bitwise=True,all_physical_history_columns_bitwise=True,steps=len(a)),indent=2)+'\n')
 elif stage=='fields':
  from analyze_cohesive_interface_evolution import case,at
  c=case('straight-long',128);inp=WORK/'physical-field-input.txt'
  with inp.open('w') as f:
   for t in [2.3,2.45,2.47]:
    i=at(c,t);f.write(' '.join(format(float(v),'.17g') for v in [c['h']['time_s'][i],c['h']['wall_memory_m'][i],*c['n']['relaxed_jump_m'][i]])+'\n')
  run('physical-fields',nx=128,ny=80,dt=.0025,end=0,first=.0625,last=.9375,hold=2.45,extra={'CI_FIELD_INPUT':str(inp),'CI_ALL_FIELDS':'1'})
 elif stage=='lifecycle':
  end=0.
  for _ in range(481):end+=.005
  for name,dt,extra in [('growth-restart-control',.0025,{}),('growth-rejection-retry',.005,{'CI_TEST_CUTBACK_TIME':format(end,'.17g')})]:
   folder=WORK/'runs'/name/'output';folder.parent.mkdir(parents=True,exist_ok=True);assert not folder.exists();shutil.copytree(WORK/'runs/straight-long-coarse/output',folder)
   run(name,nx=64,ny=40,dt=dt,end=end,first=.0625,last=.9375,hold=2.5,resume=True,folder=name,extra=dict(extra,CI_STATE_AUDIT='1'))
  a=OUT/'growth-restart-control/accepted-native.archive';b=OUT/'growth-rejection-retry/accepted-native.archive'
  assert a.read_bytes()==b.read_bytes(),'Physical accepted state differs at matching cutback timesteps'
  data=np.loadtxt(OUT/'growth-rejection-retry/native-cutbacks.csv',delimiter=',',skiprows=1,ndmin=2);assert len(data)==1 and data[0,-1]==1
  (OUT/'growth-lifecycle.json').write_text(json.dumps(dict(actual_native_cutback=True,unchanged_before_retry=True,same_accepted_timestep_sequence=True,accepted_archive_bitwise=True,forced_rejections=1),indent=2)+'\n')
 elif stage=='legacy':
  legacy_end=0.
  for _ in range(20):legacy_end+=.005
  p=run('legacy-default-matched',nx=64,ny=40,dt=.005,end=legacy_end,first=.125,last=.875,hold=2.7,signed=False)
  a=np.genfromtxt(p/'history.csv',delimiter=',',names=True);b=np.genfromtxt(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_insertion/growth-64/history.csv',delimiter=',',names=True)[:len(a)]
  quantities=['time_s','wall_memory_m','bulk_J_m','bond_J_m','fracture_increment_J_m','friction_J_m','radiation_J_m','Maxwell_J_m','force_error_Pa']
  assert all(np.array_equal(a[k],b[k]) for k in quantities)
  (OUT/'legacy-default.json').write_text(json.dumps(dict(physical_columns_bitwise=True,accepted_steps=len(a)),indent=2)+'\n')
 else:raise ValueError(stage)
if __name__=='__main__':main()
