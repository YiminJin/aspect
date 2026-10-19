import subprocess,sys,json
from pathlib import Path
w=Path('build-particle-free/bp3-homogeneous-coefficients');logger='benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py';runner='benchmarks/reconstructed_fault/particle_free/run_bp3_optimization.py'
cases=[
 ('coefficients-optimized',['--operators']),
 ('coefficients-reference',['--mode','reference']),
 ('coefficients-split',['--ranks','1','--steps','1']),
 ('coefficients-resume',['--ranks','4','--restart',str(Path('build-particle-free/bp3-optimization/runs/coefficients-split/run/restart').resolve())]),
 ('coefficients-rejection',['--candidate-rejection'])]
results=[]
for label,args in cases:
 command=[sys.executable,logger,'bp3-'+label,'240',sys.executable,runner,label,'--adaptive','--vary-bulk-composition',*args]
 code=subprocess.run(command).returncode;results.append(dict(label=label,command=command,exit_code=code))
 (w/'matrix-progress.json').write_text(json.dumps(results,indent=2)+'\n')
 if code:raise SystemExit(code)
