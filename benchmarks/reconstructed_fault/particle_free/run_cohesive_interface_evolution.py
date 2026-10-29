#!/usr/bin/env python3
"""Run bounded native evolution; Python only prepares inputs/compares outputs."""
import os,json,time,subprocess,shutil,copy,csv
from pathlib import Path
import numpy as np
import cohesive_interface_admission as reference
ROOT=reference.ROOT
WORK=ROOT/'build-particle-free/cohesive-interface-evolution'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_evolution'
HOST=ROOT/'build-particle-free/coulomb-native/aspect-release'

def emit(f,x):f.write(' '.join(format(float(v),'.17g') for v in np.array(x).ravel())+'\n')
def prepare():
 k=reference.Kernel('native-16-axial');s=reference.initial(k);c=np.r_[s['wall_memory'],s['memory']];response=np.loadtxt(k.path/'response.csv',delimiter=',');bulk=response@c
 history=np.column_stack([s['histories'][key] for key in ['jump','damage','theta','reference','relaxed_jump','absolute_slip']])
 lam,err,peak=reference.bounded_multiplier(k,k.K@c if False else (k.K@c)[1:],np.ones(len(k.s),bool))
 with (WORK/'initial.txt').open('w') as f:
  f.write(f'{k.meta["nx"]} {len(k.nodes)} {len(bulk)} {len(k.s)}\n')
  for x in [c,bulk,s['delta'],s['memory'],history,k.K,k.Q,k.w,k.Nm,k.Np,lam[k.nq]]:emit(f,x)
 with (WORK/'response.txt').open('w') as f:emit(f,response)
 return k

def native(name,end,dt=.0025,resume=False,folder=None):
 dest=WORK/'runs'/(folder or name)/'output';dest.parent.mkdir(exist_ok=True,parents=True)
 assert resume or not dest.exists(),f'Preserve existing run {dest}'
 template=(ROOT/'benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.prm').read_text()
 text=template.replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/plugin/libcohesive_interface_native.release.so',str(WORK/'build/libcohesive_interface_kink_owner.release.so')).replace('cohesive interface native checks','cohesive kink owner checks').replace('set Y repetitions = 40','set Y repetitions = 4').replace('set Maximum time step = 0.002',f'set Maximum time step = {dt}').replace('set Maximum first time step = 0.002',f'set Maximum first time step = {dt}').replace('set End time = 0.004',f'set End time = {end}').replace('$ASPECT_SOURCE_DIR/build-particle-free/cohesive-interface-native/smoke/output',str(dest)).replace('set Steps between checkpoint = 1','set Steps between checkpoint = 20')
 if resume:text+='\nset Resume computation = true\n'
 prm=WORK/(name+'.prm');prm.write_text(text)
 env=dict(os.environ,CI_KINK_INPUT=str(WORK/'initial.txt'),CI_EVOLUTION_RESPONSE=str(WORK/'response.txt'),CI_NATIVE_EVOLUTION='1',ASPECT_SOURCE_DIR=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 start=time.perf_counter()
 with (WORK/(name+'.log')).open('w') as log:r=subprocess.run([str(HOST),str(prm)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 result=dict(name=name,seconds=time.perf_counter()-start,exit_code=r.returncode)
 print(json.dumps(result),flush=True)
 out=OUT/name;out.mkdir(exist_ok=True)
 for p in dest.glob('evolution-*'):shutil.copy2(p,out/p.name)
 (OUT/(name+'-run.json')).write_text(json.dumps(result,indent=2)+'\n')
 assert r.returncode==0,name
 return out

def compare(k,path):
 rows=np.genfromtxt(path/'evolution-history.csv',delimiter=',',names=True)
 hist=np.genfromtxt(path/'evolution-states.csv',delimiter=',',names=True)
 solver=reference.Admission(k);s=reference.initial(k);diff=[]
 for row in np.atleast_1d(rows):
  t=row['time_s']
  if s['generation']==0 and s['clock_s']>=.25-1e-12:s,_=solver.insert(s)
  p=solver.solve(s,float(t),verify=True);s=solver.publish(s,p)
  h=hist[abs(hist['time_s']-t)<1e-10]
  for key,col in [('jump','jump_m'),('damage','damage'),('theta','theta'),('reference','reference_m'),('relaxed_jump','memory_m'),('absolute_slip','absolute_slip_m')]:
   error=max(abs(np.array(s['histories'][key])-h[col]));diff.append(dict(time_s=float(t),quantity=key,maximum_absolute_error=float(error)))
 with (path/'reference-differences.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(diff[0]));w.writeheader();w.writerows(diff)
 error=max(r['maximum_absolute_error'] for r in diff)
 assert error<1e-10,error
 return dict(maximum_history_error=error,maximum_weak_Pa=float(max(rows['weak_error_Pa'])),maximum_ledger_J_m=float(max(abs(rows['ledger_error_J_m']))))
def main():
 import sys
 WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);k=prepare()
 if len(sys.argv)>1 and sys.argv[1]=='smoke':native('smoke',.01);return
 control=native('native-control',.46);comparison=compare(k,control)
 split_time=float(np.genfromtxt(control/'evolution-history.csv',delimiter=',',names=True)['time_s'][159]-.2)
 native('native-split',split_time);resume=native('native-resume',.46,resume=True,folder='native-split')
 assert (control/'evolution-final.archive').read_bytes()==(resume/'evolution-final.archive').read_bytes()
 a=np.genfromtxt(control/'evolution-history.csv',delimiter=',',names=True);b=np.genfromtxt(resume/'evolution-history.csv',delimiter=',',names=True);assert np.array_equal(a,b)
 (OUT/'native-qualification.json').write_text(json.dumps(dict(comparison=comparison,restart_bitwise=True,own_solutions=True),indent=2)+'\n')
 print(comparison)
if __name__=='__main__':main()
