#!/usr/bin/env python3
"""Sequential, bounded layered mature-fault runs; unique output per invocation."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
RUNS=BUILD/'material-runs';CANDIDATE=BUILD/'exterior-stress'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('label');p.add_argument('--translation',type=float,default=2.5)
    p.add_argument('--phase-discretization',choices=['Q1','DGQ2'],default='Q1',help='Native phase space; old inputs retain Q1')
    p.add_argument('--dt',type=float,default=.0125);p.add_argument('--end',type=float,default=.4)
    p.add_argument('--depth-repetitions',type=int,default=1,choices=[1,3])
    p.add_argument('--level',type=int,default=5);p.add_argument('--halfwidth',type=float,default=4.)
    p.add_argument('--seconds',type=int,default=3600);p.add_argument('--ranks',type=int,default=4)
    p.add_argument('--max-log-state-change',type=float,default=.01)
    p.add_argument('--adaptive',action='store_true');p.add_argument('--schedule',action='store_true')
    p.add_argument('--function',help='Explicit accepted-schedule control using the existing Function plugin')
    p.add_argument('--checkpoint-steps',type=int,default=16,help='Native checkpoint interval for bounded lifecycle checks')
    p.add_argument('--reject',action='store_true');p.add_argument('--resume-from')
    p.add_argument('--checkpoint-slot',choices=['oldest','01','02','03'])
    p.add_argument('--checkpoint-directory',help='Preserved restart directory from the named parent, before its rotating slots were overwritten')
    p.add_argument('--candidate',type=Path,help='Coherent candidate with core, checks and layered-provider')
    p.add_argument('--spin',choices=['bulk slip subtracted','total material'])
    p.add_argument('--background-shear',type=float,default=2.7)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    assert (a.candidate is None)==(a.spin is None), 'Select a coherent candidate and its constitutive spin together'
    assert not a.checkpoint_directory or a.resume_from, 'A preserved checkpoint requires its parent fixture metadata'
    assert 1<=a.ranks<=4 and a.halfwidth in (4.,6.,8.) and all(c.isalnum() or c in '-_' for c in a.label)
    lock=(BUILD/'simulation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    label='layered-'+a.label;out=RUNS/label;out.mkdir(exist_ok=False)
    base=RUNS/'exterior-compatible025'
    template=Path(__file__).resolve().parent/'layered_background/base.prm'
    text=template.read_text().replace('@ROOT@',str(ROOT)).replace('@RUN@',str(out)) if a.candidate else (base/'input.prm').read_text().replace(str(base),str(out))
    plugin=BUILD/'layered-campaign/provider/liblayered_background.release.so';core=CANDIDATE/'aspect-release'
    oldplugin=CANDIDATE/'provider/libexterior_stress.release.so'
    candidate=a.candidate.resolve() if a.candidate else CANDIDATE
    if a.candidate:
        plugin=candidate/'layered-provider/liblayered_background.release.so';core=candidate/'aspect-release'
        text=text.replace(str(CANDIDATE/'checks/libtransported_material_energy.release.so'),str(candidate/'checks/libtransported_material_energy.release.so'))
    if a.spin:
        text=text.replace('set Material stress representation = direct physical stress',
                          'set Material stress representation = direct physical stress\n  set Constitutive spin = '+a.spin)
    if a.candidate:
        text+=f'\nsubsection Postprocess\n subsection Material transaction fixture\n  set Initial background shear = {a.background_shear}\n end\nend\n'
    else:assert a.background_shear==2.7
    text=text.replace(str(oldplugin),str(plugin))
    dt=a.dt;clock=0.
    for _ in range(round(a.end/dt)):clock+=dt
    end=min(a.end,clock) if not (a.adaptive or a.schedule or a.reject or a.function) else a.end-1e-13
    identity=f'layered-Maxwell-v1:G=10:eta=1..10:width=.35:du=2:init=.1:pref=0:panels=64:U={a.translation:.17g}'
    replacements=[('set Initial global refinement = 5',f'set Initial global refinement = {a.level}'),
      ('set X extent = 4',f'set X extent = {2*a.halfwidth}\n  set X repetitions = {int(a.halfwidth/2)*a.depth_repetitions}'),
      ('set Box origin X coordinate = -2',f'set Box origin X coordinate = {-a.halfwidth}'),
      ('set Maximum time step = 0.025',f'set Maximum time step = {dt}'),
      ('set Maximum first time step = 0.025',f'set Maximum first time step = {dt}'),
      ('set Function expression = 0.025',f'set Function expression = {dt}'),
      ('set End time = 0.4',f'set End time = {end:.17g}'),
      ('left:function, right:function, bottom:function','left:layered background, right:layered background, bottom:layered background'),
      ('homogeneous shear exterior','layered background'),('top:prescribed shear background','top:layered background'),
      ('set Function expression = 1\n','set Function expression = .55+.45*tanh(y/.35)\n'),
      ('((y-.3*x)/.6)','((y-.6*x)/.6)'),('set Steps between checkpoint = 2',f'set Steps between checkpoint = {a.checkpoint_steps}')]
    if a.depth_repetitions!=1:replacements.append(('set Y extent = 4',f'set Y extent = 4\n  set Y repetitions = {a.depth_repetitions}'))
    for old,new in replacements:
        assert old in text,old;text=text.replace(old,new)
    text='\n'.join('  set Material exterior stress identity = '+identity if 'set Material exterior stress identity =' in l else l for l in text.splitlines())+'\n'
    text+=f'\nsubsection Postprocess\n subsection Layered background\n  set Translation = {a.translation}\n end\nend\n'
    modes=['function']
    if a.adaptive:modes+=['reconstructed fault time step']
    if a.reject:modes+=['material transaction rejection']
    text=text.replace('set List of model names = function','set List of model names = '+', '.join(modes))
    if a.schedule:
        # Existing Function controller; explicit handling check, not adaptivity.
        text=text.replace(f'set Function expression = {dt}\n',f'set Function expression = if(time<0.1-1e-10,{dt},if(time<0.2-1e-10,{dt/2},{dt}))\n')
    if a.function:
        assert not a.schedule
        text=text.replace(f'set Function expression = {dt}\n',f'set Function expression = {a.function}\n')
    if a.adaptive:
        text+=f'\nset CFL number = 0.1\nsubsection Time stepping\n subsection Reconstructed fault time step\n  set Maximum logarithmic state change = {a.max_log_state_change}\n end\nend\n'
    if a.resume_from:
        parent=RUNS/a.resume_from
        parent_args=json.loads((parent/'command.json').read_text())['arguments']
        assert parent_args.get('phase_discretization','Q1')==a.phase_discretization, 'Restart phase-space mismatch'
        for key in ['translation','background_shear','spin','level','depth_repetitions','halfwidth','adaptive','schedule','function']+(['max_log_state_change'] if a.adaptive else []):
            assert parent_args.get(key,1 if key=='depth_repetitions' else None)==getattr(a,key),('Restart fixture mismatch',key,parent_args.get(key),getattr(a,key))
        text=text.replace('set Resume computation = false','set Resume computation = true')
        restart_source=Path(a.checkpoint_directory).resolve() if a.checkpoint_directory else parent/'output/restart'
        (out/'output').mkdir();shutil.copytree(restart_source,out/'output/restart')
        if a.checkpoint_slot:
            slots=[p for p in restart_source.iterdir() if p.is_dir() and (p/'resume.z').exists()]
            slot=min(slots,key=lambda p:(p/'resume.z').stat().st_mtime).name if a.checkpoint_slot=='oldest' else a.checkpoint_slot
            (out/'output/restart/last_good_checkpoint.txt').write_text(str(int(slot)))
    if a.phase_discretization=='DGQ2':
        text+='\nsubsection Discretization\n set Use discontinuous phase discretization = true\n set Phase polynomial degree = 2\nend\n'
    (out/'input.prm').write_text(text);(out/'fault.txt').write_text('-.8 -.48 .4\n.8 .48 .4\n')
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib',OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='false',OMPI_MCA_mpi_yield_when_idle='1',ASPECT_TRUE_STRESS_ENERGY='1')
    for key in ['ASPECT_TRUE_STRESS_CHECKS','ASPECT_K1_FLOOR_AUDIT','ASPECT_CORNER_REPLAY_VELOCITY']:env.pop(key,None)
    command=['/opt/openmpi/5.0.6/bin/mpirun','-n',str(a.ranks),str(core),str(out/'input.prm')]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record=dict(arguments=(vars(a)|dict(candidate=str(candidate)))|dict(label=label,end=end,eta='.55+.45*tanh(y/.35)',shear=.5),command=command,
      core_sha256=sha(core),provider_sha256=sha(plugin),checker_sha256=sha(candidate/'checks/libtransported_material_energy.release.so'),
      input_sha256=sha(out/'input.prm'),runner_sha256=sha(out/'runner.py'),identity=identity,
      head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    if a.prepare_only:return
    start=time.perf_counter()
    with (out/'run.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    record.update(exit_code=code,wall_s=time.perf_counter()-start);(out/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(case=label,exit_code=code,wall_s=record['wall_s'])),flush=True)
    raise SystemExit(code)
if __name__=='__main__':main()
