#!/usr/bin/env python3
"""Run one bounded particle replenishment check; preserve its command, environment, log and outcome."""
import json, os, pathlib, subprocess, sys, time, resource, signal
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
peak_rss=0
limit_reason=None
with log.open('w') as stream:
    process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    while process.poll() is None:
        rows=[list(map(int,line.split())) for line in subprocess.check_output(['ps','-e','-o','pid=,ppid=,rss='],text=True).splitlines()]
        family={process.pid}
        for _ in range(12):
            expanded=family|{pid for pid,parent,rss in rows if parent in family}
            if expanded==family:break
            family=expanded
        rss=sum(rss for pid,parent,rss in rows if pid in family)
        peak_rss=max(peak_rss,rss)
        if time.monotonic()-start>float(seconds) or rss>20*1024**2:
            limit_reason='time' if time.monotonic()-start>float(seconds) else '20 GiB aggregate RSS'
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            break
        time.sleep(.5)
    code=124 if limit_reason else process.returncode
record = dict(peak_aggregate_rss_kib=peak_rss, limit_reason=limit_reason, command=command, cwd=os.getcwd(), exit_code=code,
              seconds=time.monotonic()-start,
              child_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              environment={k:v for k,v in env.items() if k.startswith(('ASPECT_', 'OMP_', 'OPENBLAS_', 'DEAL_II_', 'OMPI_'))})
(evidence / (label + '.json')).write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({'label':label, 'exit_code':code, 'seconds':record['seconds'], 'log':str(log)}), flush=True)
raise SystemExit(code)
