#!/usr/bin/env python3
"""One fresh bounded native bulk-tip run; retains input, identity, log and exit status."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--executable',type=Path,default=root/'build-particle-free/bulk-tip/native/aspect-release')
p.add_argument('--checker',type=Path,default=root/'build-particle-free/bulk-tip/points/libbulk_tip_checks.release.so')
p.add_argument('--fixture',type=Path,default=Path(__file__).with_name('finite_tip.prm'));p.add_argument('--oracle',action='store_true');
p.add_argument('label');p.add_argument('--dt',type=float,default=.005);p.add_argument('--end',type=float,default=.08)
p.add_argument('--ranks',type=int,choices=[1,2,4],default=1);p.add_argument('--refinement',type=int,default=0)
p.add_argument('--seconds',type=int,default=600);p.add_argument('--unloaded',action='store_true')
p.add_argument('--remote-margin',action='store_true')
p.add_argument('--phase-basis',choices=['nodal','Bernstein'])
p.add_argument('--export-parent',type=Path)
p.add_argument('--import-parent',type=Path)
p.add_argument('--restart-dt',type=float,help='Reduce only the pending restart interval using the existing simulator signal')
p.add_argument('--linear-oracle',action='store_true');p.add_argument('--tangent',action='store_true');p.add_argument('--intact',action='store_true');p.add_argument('--rejection',action='store_true')
p.add_argument('--linear-oracle-step',type=int,default=1)
p.add_argument('--tangent-step',type=int,default=8);p.add_argument('--rejection-step',type=int,default=2)
p.add_argument('--snapshot-id',type=int,choices=[1,2,3]);p.add_argument('--resume-from',type=Path);p.add_argument('--no-checker',action='store_true');a=p.parse_args()
assert a.label and all(c.isalnum() or c in '-_' for c in a.label)
run=root/'build-particle-free/bulk-tip/runs'/a.label;run.mkdir(parents=True,exist_ok=False)
text=a.fixture.read_text()
if a.phase_basis:text+=f'\nsubsection Discretization\n set Phase basis = {a.phase_basis}\nend\n'
# The integrated adapter reuses ASPECT's assembled block-AMG Stokes inverse.
if not a.oracle:text+='\nsubsection Solver parameters\n subsection Stokes solver parameters\n  set Stokes solver type = block AMG\n end\nend\n'
if a.remote_margin:text+='\nsubsection Termination criteria\n set Termination criteria = end time, tip remote margin\nend\n'
if a.oracle:text+='\nsubsection Material model\n subsection Phase field fault\n  set Bulk tip solver = benchmark direct\n end\nend\n'
if a.unloaded:
    text=text.replace('.02*(y-.5);0','0;0').replace('.01*(y-1);0','0;0')
if a.intact:text=text.replace('.2*max(0,1-((x-1)/.5)^2)*max(0,1-((y-.5)/.25)^2)','0')
text+=f'\nset Maximum time step = {a.dt:.17g}\nset Maximum first time step = {a.dt:.17g}\nset End time = {a.end:.17g}\nset Output directory = {run}/output\n'
text+=f'\nsubsection Mesh refinement\n set Initial global refinement = {a.refinement}\nend\n'
exe=a.executable.resolve();plugin=a.checker.resolve()
if not a.no_checker:text+=f'\nset Additional shared libraries = {plugin}\nsubsection Postprocess\n set List of postprocessors = bulk tip checks\nend\n'
if a.resume_from:
    # The restored run writes to a fresh directory. Its checkpoint files remain untouched.
    import shutil
    (run/'output').mkdir()
    for file in a.resume_from.glob('restart*'):
        if file.is_dir():shutil.copytree(file,run/'output'/file.name)
        else:shutil.copy2(file,run/'output'/file.name)
    if a.snapshot_id is not None:
        selected=run/'output/restart'/f'{a.snapshot_id:02d}'
        assert selected.is_dir(),f'Missing requested snapshot {selected}'
        (run/'output/restart/last_good_checkpoint.txt').write_text(str(a.snapshot_id))
    text+='\nset Resume computation = true\n'
(run/'input.prm').write_text(text)
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS'):env[key]='1'
if a.remote_margin:env['ASPECT_TIP_REMOTE_MARGIN']='1'
if a.export_parent:env['ASPECT_TIP_EXPORT_PARENT']=str(a.export_parent.resolve())
if a.import_parent:env['ASPECT_TIP_IMPORT_PARENT']=str(a.import_parent.resolve())
if a.restart_dt:env['ASPECT_TIP_RESTART_DT']=str(a.restart_dt)
if a.rejection:
    env['ASPECT_TIP_RECOVERY']='1';env['ASPECT_TIP_RECOVERY_STEP']=str(a.rejection_step)
if a.tangent:
    env['ASPECT_BULK_CHECK_TANGENT']='1';env['ASPECT_TIP_TANGENT_STEP']=str(a.tangent_step)
if a.linear_oracle:
    env['ASPECT_TIP_LINEAR_ORACLE']='1';env['ASPECT_TIP_LINEAR_ORACLE_STEP']=str(a.linear_oracle_step)
command=['/opt/openmpi/5.0.6/bin/mpirun','-np',str(a.ranks),str(exe),str(run/'input.prm')]
sha=lambda file:hashlib.sha256(file.read_bytes()).hexdigest()
files=subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines()
files+=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--','include','source','unit_tests'],cwd=root,text=True).splitlines()
record=dict(source_identity_note='source_sha256 describes the current worktree; executable identity is independent, especially with --executable',command=command,arguments={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()},binary_sha256=sha(exe),input_sha256=sha(run/'input.prm'),
            source_sha256={f:sha(root/f) for f in files if (root/f).is_file()},revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
if not a.no_checker:record['checker_sha256']=sha(plugin)
if a.resume_from:record['restart_source_sha256']={str(f.relative_to(a.resume_from)):sha(f) for f in sorted(a.resume_from.glob('restart*/**/*')) if f.is_file()}
(run/'command.json').write_text(json.dumps(record,indent=2)+'\n')
started=time.monotonic()
with (run/'run.log').open('w') as log:
    job=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=job.wait(timeout=a.seconds)
    except subprocess.TimeoutExpired:
        os.killpg(job.pid,signal.SIGTERM)
        try:job.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);job.wait()
        code=124
record.update(exit_code=code,seconds=time.monotonic()-started,environment={k:v for k,v in env.items() if k.startswith(('ASPECT_','OMP_','OPENBLAS_','DEAL_II_'))})
(run/'command.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(label=a.label,exit_code=code,seconds=record['seconds'],run=str(run))),flush=True)
raise SystemExit(code)
