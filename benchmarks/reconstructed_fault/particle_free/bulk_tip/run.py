#!/usr/bin/env python3
"""One fresh bounded native bulk-tip run; retains input, identity, log and exit status."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('label');p.add_argument('--dt',type=float,default=.005);p.add_argument('--end',type=float,default=.08)
p.add_argument('--ranks',type=int,choices=[1,2,4],default=1);p.add_argument('--refinement',type=int,default=0)
p.add_argument('--seconds',type=int,default=600);p.add_argument('--unloaded',action='store_true')
p.add_argument('--tangent',action='store_true');p.add_argument('--intact',action='store_true');p.add_argument('--rejection',action='store_true')
p.add_argument('--snapshot-id',type=int,choices=[1,2,3]);p.add_argument('--resume-from',type=Path);p.add_argument('--no-checker',action='store_true');a=p.parse_args()
assert a.label and all(c.isalnum() or c in '-_' for c in a.label)
run=root/'build-particle-free/bulk-tip/runs'/a.label;run.mkdir(parents=True,exist_ok=False)
text=Path(__file__).with_name('finite_tip.prm').read_text()
if a.unloaded:text=text.replace('.02*(y-.5);0','0;0')
if a.intact:text=text.replace('.2*max(0,1-((x-1)/.5)^2)*max(0,1-((y-.5)/.25)^2)','0')
text+=f'\nset Maximum time step = {a.dt:.17g}\nset Maximum first time step = {a.dt:.17g}\nset End time = {a.end:.17g}\nset Output directory = {run}/output\n'
text+=f'\nsubsection Mesh refinement\n set Initial global refinement = {a.refinement}\nend\n'
exe=root/'build-particle-free/bulk-tip/native/aspect-release';plugin=root/'build-particle-free/bulk-tip/points/libbulk_tip_checks.release.so'
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
if a.rejection:env['ASPECT_TIP_RECOVERY']='1'
if a.tangent:env['ASPECT_BULK_CHECK_TANGENT']='1'
command=['/opt/openmpi/5.0.6/bin/mpirun','-np',str(a.ranks),str(exe),str(run/'input.prm')]
sha=lambda file:hashlib.sha256(file.read_bytes()).hexdigest()
files=subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines()
files+=[str(f.relative_to(root)) for base in ('include','source','unit_tests') for f in (root/base).rglob('*bulk_tip*') if f.is_file()]
record=dict(command=command,arguments=vars(a)|{'resume_from':str(a.resume_from)},binary_sha256=sha(exe),input_sha256=sha(run/'input.prm'),
            source_sha256={f:sha(root/f) for f in files if (root/f).is_file()},revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
if not a.no_checker:record['checker_sha256']=sha(plugin)
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
