#!/usr/bin/env python3
"""Bounded selection/provider/checkpoint guards, including one short restart step."""
from pathlib import Path
import os,subprocess,shutil,json,hashlib,time,signal
ROOT=Path(__file__).resolve().parents[4]
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
BASE=BUILD/'material-runs/exterior-zero025'
OUT=BUILD/'exterior-stress/guard-runs'

def main():
    OUT.mkdir(exist_ok=False);results={}
    base=(BASE/'input.prm').read_text();identity=next(l.strip() for l in base.splitlines() if 'set Material exterior stress identity =' in l)
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('ASPECT_TRUE_STRESS_ENERGY',None)
    for name,marker in [('empty','requires a nonempty provider identity'),('separated','separated exterior histories are not specified'),
                        ('mismatch','matching identity'),('missing','Prescribed exterior stress provider is missing'),
                        ('restart_match',None),('restart_changed','PF5 checkpoint version/material/quadrature mismatch')]:
        run=OUT/name;run.mkdir();text=base.replace(str(BASE),str(run))
        text=text.replace('set End time = 0.4','set End time = 0.025' if name=='missing' else ('set End time = 0.425' if name=='restart_match' else 'set End time = 0.4'))
        if name=='empty':text=text.replace(identity,'set Material exterior stress identity =')
        if name=='separated':text=text.replace('set Material stress representation = direct physical stress','set Material stress representation = separated')
        if name=='mismatch':text=text.replace(identity,identity+'-wrong')
        if name in ('missing','restart_changed'):text=text.replace(', homogeneous shear exterior','')
        if name.startswith('restart_'):
            text=text.replace('set Resume computation = false','set Resume computation = true')
            (run/'output').mkdir();shutil.copytree(BASE/'output/restart',run/'output/restart')
        if name=='restart_changed':
            text=text.replace('set Material tensor inflow = prescribed exterior stress','set Material tensor inflow = accepted trace').replace(identity,'set Material exterior stress identity =')
        (run/'input.prm').write_text(text);(run/'fault.txt').write_bytes((BASE/'fault.txt').read_bytes())
        command=['/opt/openmpi/5.0.6/bin/mpirun','-n','2',str(BUILD/'exterior-stress/aspect-release'),str(run/'input.prm')]
        start=time.perf_counter()
        with (run/'run.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10);code=124
        output=(run/'run.log').read_text();passed=(code==0) if marker is None else (code!=0 and code!=124 and marker in output)
        results[name]=dict(command=command,exit_code=code,expected_marker=marker,passed=passed,seconds=time.perf_counter()-start)
        (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,results[name],flush=True)
        assert passed,name
if __name__=='__main__':main()
