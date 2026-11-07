#!/usr/bin/env python3
"""Replay a saved non-affine velocity in benchmark scratch histories only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[4]
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
RUNS=BUILD/'material-runs'
CASES={'replay32':(5,.025,600)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=CASES)
    p.add_argument('--suffix',default='',help='Distinct suffix for a retained failed-attempt repair')
    a=p.parse_args();level,dt,seconds=CASES[a.case]
    end=0.  # Initialization-only host; all transport states remain scratch.
    assert all(c.isalnum() or c in '-_' for c in a.suffix)
    lock=(BUILD/'simulation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    label='corner-'+a.case+a.suffix;out=RUNS/label;out.mkdir(exist_ok=False)
    base=RUNS/'shear32-dt025';previous=json.loads((base/'command.json').read_text())
    plugin=BUILD/'corner-replay/libcorner_replay.release.so'
    core=ROOT/'build-particle-free/native/aspect-release'
    checker=BUILD/'checks/libtransported_material_energy.release.so'
    text=(base/'input.prm').read_text().replace(str(base),str(out))
    text=text.replace(f'set Additional shared libraries = {checker}',f'set Additional shared libraries = {checker}, {plugin}')
    text=text.replace('set Initial global refinement = 5',f'set Initial global refinement = {level}')
    text=text.replace('set Maximum time step = 0.025',f'set Maximum time step = {dt}')
    text=text.replace('set Maximum first time step = 0.025',f'set Maximum first time step = {dt}')
    text=text.replace('set Function expression = 0.025',f'set Function expression = {dt}')
    text=text.replace('set End time = 0.4',f'set End time = {end}')
    text=text.replace('set List of postprocessors = particles, material transaction checks',
                      'set List of postprocessors = particles, material transaction checks, corner velocity replay')
    (out/'input.prm').write_text(text);(out/'fault.txt').write_bytes((base/'fault.txt').read_bytes())
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',
                                   OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='false',OMPI_MCA_mpi_yield_when_idle='1')
    for key in ['ASPECT_TRUE_STRESS_CHECKS','ASPECT_K1_FLOOR_AUDIT','ASPECT_TRUE_STRESS_ENERGY']:env.pop(key,None)
    env['ASPECT_CORNER_REPLAY_VELOCITY']=str(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_replay/velocity.txt')
    command=['/opt/openmpi/5.0.6/bin/mpirun','-n','4',str(core),str(out/'input.prm')]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    args=previous['arguments']|dict(label=label,level=level,dt=dt,end=end,
                                   requested_end=0,seconds=seconds,stage_energy=False)
    record=dict(arguments=args,command=command,core_sha256=sha(core),plugin_sha256=sha(checker),
                replay_plugin_sha256=sha(plugin),velocity_sha256=sha(Path(env['ASPECT_CORNER_REPLAY_VELOCITY'])),input_sha256=sha(out/'input.prm'),runner_sha256=sha(out/'runner.py'),
                head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                control='frozen shear32-dt025 step16 velocity, manufactured tensor transport',
                base_input_sha256=sha(base/'input.prm'),runtime_environment={k:env[k] for k in env if k.startswith('ASPECT_') or k in ['LD_LIBRARY_PATH','OMP_NUM_THREADS','DEAL_II_NUM_THREADS','OPENBLAS_NUM_THREADS']})
    start=time.perf_counter()
    with (out/'run.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    record.update(exit_code=code,wall_s=time.perf_counter()-start)
    (out/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(case=label,exit_code=code,wall_s=record['wall_s'])),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
