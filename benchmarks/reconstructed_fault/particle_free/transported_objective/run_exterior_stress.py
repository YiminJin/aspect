#!/usr/bin/env python3
"""Five selected coupled shear runs; preserve retained controls and all attempts."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
EXTERIOR=BUILD/'exterior-stress'
RUNS=BUILD/'material-runs'
CASES={'zero025':(5,.025,False,600),'compatible025':(5,.025,True,600),
       'compatible050':(5,.05,True,600),'compatible0125':(5,.0125,True,900),
       'fine0125':(6,.0125,True,1800)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=CASES);p.add_argument('--suffix',default='');p.add_argument('--candidate',type=Path);p.add_argument('--spin',choices=['bulk slip subtracted','total material']);a=p.parse_args()
    assert all(c.isalnum() or c in '-_' for c in a.suffix)
    level,dt,compatible,seconds=CASES[a.case];clock=0.
    for _ in range(round(.4/dt)):clock+=dt
    end=min(.4,clock)
    lock=(BUILD/'simulation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    label='exterior-'+a.case+a.suffix;out=RUNS/label;out.mkdir(exist_ok=False)
    base=RUNS/'shear32-dt025';previous=json.loads((base/'command.json').read_text())
    candidate=a.candidate.resolve() if a.candidate else EXTERIOR
    checker=candidate/'checks/libtransported_material_energy.release.so'
    plugin=candidate/'provider/libexterior_stress.release.so';core=candidate/'aspect-release'
    original_checker=BUILD/'checks/libtransported_material_energy.release.so'
    identity=f'homogeneous-shear-Maxwell-v1:G=10:eta=10:shear=.5:init=.1:p=0:dt={dt:.17g}'
    text=(base/'input.prm').read_text().replace(str(base),str(out)).replace(str(original_checker),str(checker)+', '+str(plugin))
    for old,new in [('set Initial global refinement = 5',f'set Initial global refinement = {level}'),
                    ('set Maximum time step = 0.025',f'set Maximum time step = {dt}'),
                    ('set Maximum first time step = 0.025',f'set Maximum first time step = {dt}'),
                    ('set Function expression = 0.025',f'set Function expression = {dt}'),
                    ('set End time = 0.4',f'set End time = {end}'),
                    ('set Material tensor inflow = accepted trace',f'set Material tensor inflow = prescribed exterior stress\n  set Material exterior stress identity = {identity}'),
                    ('set List of postprocessors = particles, material transaction checks','set List of postprocessors = particles, material transaction checks, homogeneous shear exterior')]:
        assert old in text,old;text=text.replace(old,new)
    if a.spin:text=text.replace('set Material stress representation = direct physical stress','set Material stress representation = direct physical stress\n  set Constitutive spin = '+a.spin)
    if compatible:text+='\nsubsection Boundary traction model\n set Prescribed traction boundary indicators = top:prescribed shear background\nend\n'
    (out/'input.prm').write_text(text);(out/'fault.txt').write_bytes((base/'fault.txt').read_bytes());(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='false',OMPI_MCA_mpi_yield_when_idle='1',ASPECT_TRUE_STRESS_ENERGY='1')
    for key in ['ASPECT_TRUE_STRESS_CHECKS','ASPECT_K1_FLOOR_AUDIT','ASPECT_CORNER_REPLAY_VELOCITY']:env.pop(key,None)
    command=['/opt/openmpi/5.0.6/bin/mpirun','-n','4',str(core),str(out/'input.prm')]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    args=previous['arguments']|dict(label=label,level=level,dt=dt,end=end,requested_end=.4,seconds=seconds,stage_energy=True,
                                  constitutive_spin=a.spin or 'total material',tensor_inflow='prescribed exterior stress',top_loading='compatible background' if compatible else 'zero')
    record=dict(arguments=args,command=command,core_sha256=sha(core),plugin_sha256=sha(checker),exterior_plugin_sha256=sha(plugin),
                input_sha256=sha(out/'input.prm'),runner_sha256=sha(out/'runner.py'),exterior_identity=identity,
                head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
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
    record.update(exit_code=code,wall_s=time.perf_counter()-start);(out/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(case=label,exit_code=code,wall_s=record['wall_s'])),flush=True)
    raise SystemExit(code)
if __name__=='__main__':main()
