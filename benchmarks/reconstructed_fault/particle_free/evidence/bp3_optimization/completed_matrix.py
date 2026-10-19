import json,subprocess,sys,time
from pathlib import Path
root=Path.cwd();work=root/'build-particle-free/bp3-optimization'
runner='benchmarks/reconstructed_fault/particle_free/run_bp3_optimization.py'
logger='benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py'
cases=[
 ('qualified-reference-small',['--mode','reference'],240),
 ('qualified-optimized-small',[],240),
 ('qualified-reference-larger',['--mode','reference','--mesh','larger','--seconds','320'],340),
 ('qualified-optimized-larger',['--mesh','larger','--seconds','320'],340),
 ('operators-larger',['--operators','--mesh','larger','--steps','1','--seconds','240'],260),
 ('ownership-r1',['--ranks','1'],240),
 ('ownership-r4',['--ranks','4'],240),
 ('restart-split-r1',['--ranks','1','--steps','1'],240),
 ('restart-r4',['--ranks','4','--restart',str(work/'runs/restart-split-r1/run/restart')],240),
 ('candidate-rejection',['--candidate-rejection'],240),
 ('cutback-control',['--steps','3','--schedule','if(time<200000,200000,if(time<300000,100000,200000))'],240),
 ('outer-rejection',['--steps','3','--outer-rejection'],240),
 ('nonlinear-rejection',['--ranks','4','--steps','3','--nonlinear-rejection'],240),
 ('adaptive-control',['--adaptive'],240)]
state=dict(started=time.time(),completed=[],running=None)
progress=work/'matrix-progress.json'
def run(label,cmd,seconds):
 state['running']=label;progress.write_text(json.dumps(state,indent=2)+'\n');print('START '+label,flush=True)
 code=subprocess.run([sys.executable,logger,'bp3-opt-'+label,str(seconds),*cmd]).returncode
 state['completed'].append(dict(label=label,exit_code=code));state['running']=None;progress.write_text(json.dumps(state,indent=2)+'\n')
 if code:raise SystemExit(code)
for label,args,seconds in cases:run(label,[sys.executable,runner,label,*args],seconds)
regression='benchmarks/reconstructed_fault/bp3_native/tests/collective_stop/regression.py'
for case in ['control','wall','wall-resume','scheduled-control','scheduled-split','scheduled-resume']:
 run('wall-'+case,[sys.executable,regression,'run','--work',str(work/'wall'),'--case',case],260)
run('wall-check',[sys.executable,regression,'check','--work',str(work/'wall')],60)
state['finished']=time.time();progress.write_text(json.dumps(state,indent=2)+'\n')
