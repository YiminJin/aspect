#!/usr/bin/env python3
"""Run one bounded R1 check; preserve its command, environment, log and outcome."""
import json, os, pathlib, subprocess, sys, time
root = pathlib.Path(__file__).resolve().parent
label, seconds, *command = sys.argv[1:]
evidence = root / 'evidence'
evidence.mkdir(exist_ok=True)
log = evidence / (label + '.log')
if log.exists():
    raise SystemExit('Refusing to overwrite ' + str(log))
env = os.environ.copy()
env['PATH'] = '/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:' + env['PATH']
for key in ('OMP_NUM_THREADS', 'DEAL_II_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    env[key] = '1'
start = time.monotonic()
with log.open('w') as stream:
    try:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=float(seconds), check=False)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
record = dict(command=command, cwd=os.getcwd(), exit_code=code,
              seconds=time.monotonic()-start,
              environment={k:v for k,v in env.items() if k.startswith(('ASPECT_', 'OMP_', 'OPENBLAS_', 'DEAL_II_', 'OMPI_'))})
(evidence / (label + '.json')).write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({'label':label, 'exit_code':code, 'seconds':record['seconds'], 'log':str(log)}), flush=True)
raise SystemExit(code)
