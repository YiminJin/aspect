"""Stage and run bounded filter branches. No checkpoint rewriting or retries.

prepare copies the SAME checkpoint and inputs into three independent branches.
run R writes its ACTUAL constitutive clock; F1/F2 consume it as a safety cap.
Use the server compiler/deal.II/Boost environment compatible with the checkpoint.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(a):
    root=a.destination.resolve()
    if root.exists():
        raise ValueError('Preserve existing evidence; use a new staging directory')
    cp=a.checkpoint.resolve()
    step,t=(cp/'bp3_accepted_state.txt').read_text().split()
    if int(step)!=5612 or float(t)!=5310111071.5634108:
        raise ValueError('Wrong source checkpoint')
    for name in ('resume.z','mesh','mesh.info','mesh_fixed.data','mesh_variable.data'):
        if not (cp/name).is_file():
            raise ValueError('Incomplete checkpoint: '+name)
    root.mkdir(parents=True)
    shutil.copytree(a.fixture,root/'fixture')
    for name in ('libbp5_steady_initialization.release.so','libbp5_normal_stress_diagnostic.release.so'):
        shutil.copy2(a.libraries/name,root/name)
    cp_hash={str(p.relative_to(cp)):sha(p) for p in cp.rglob('*') if p.is_file()}
    for branch in ('R','F1','F2'):
        dest=root/branch
        dest.mkdir()
        shutil.copy2(Path(__file__).with_name(branch+'.prm'),dest/'run.prm')
        if branch=='F2' and getattr(a,'secondary_length',200)==50:
            text=(dest/'run.prm').read_text().replace('set Normal filter length = 200','set Normal filter length = 50')
            (dest/'run.prm').write_text(text)
        shutil.copytree(cp,dest/'output/restart/01')
        (dest/'output/restart/last_good_checkpoint.txt').write_text('1\n')
        for p in (cp/'bp3_output_metadata').iterdir():
            if p.is_file():
                shutil.copy2(p,dest/'output'/p.name)
    manifest={'source_checkpoint':str(cp),'checkpoint_sha256':cp_hash,
              'checkpoint_step':int(step),'checkpoint_time_text':t,
              'selected_lengths_m':[100,getattr(a,'secondary_length',200)],
              'inputs_sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*')
                              if p.is_file() and 'output' not in p.parts}}
    (root/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared; no simulation launched:',root)


def run(a):
    root=a.directory.resolve();dest=root/a.branch
    if (dest/'run.log').exists():
        raise ValueError('Existing execution preserved: no automatic retry')
    manifest=json.loads((root/'provenance.json').read_text())
    for name,digest in manifest['inputs_sha256'].items():
        if sha(root/name)!=digest:
            raise ValueError('Staged input/library changed: '+name)
    for name,digest in manifest['checkpoint_sha256'].items():
        if sha(dest/'output/restart/01'/name)!=digest:
            raise ValueError('Source checkpoint changed: '+name)
    if a.branch!='R':
        ref=json.loads((root/'R/execution.json').read_text())
        if not ref['passed'] or ref['binary_sha256']!=sha(a.binary):
            raise ValueError('R must complete with the identical binary')
        if ref['launcher']!=a.launcher:
            raise ValueError('Use identical MPI ranks and launcher')
        if not (root/'offline.json').exists():
            raise ValueError('Run the offline assessment and inspect length sensitivity before filtering')
        if ref['clock_sha256']!=sha(root/'actual_intervals.txt'):
            raise ValueError('The recorded actual clock changed')
    env={k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_FAULT_EXPLICIT_B='1',ASPECT_FAULT_EXPLICIT_G='1',
               ASPECT_FAULT_SURFACE_SOLVER='tridiagonal',ASPECT_FAULT_PERFORMANCE='1',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    # No freeze-particle/native-history selectors. G uses the exact nonlocal
    # action when filtering is selected, not the old local sparse matrix.
    command=['timeout','--kill-after=15','7200']+a.launcher+[str(a.binary.resolve()),'run.prm']
    start=time.monotonic()
    with (dest/'run.log').open('x') as log:
        status=subprocess.run(command,cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    summary=dest/'output/normal_summary.csv'
    rows=list(csv.DictReader(summary.open())) if summary.exists() else []
    passed=status==0 and len(rows)==10 and [int(r['step']) for r in rows]==list(range(5613,5623))
    accepted_path=dest/'output/accepted_steps.csv'
    accepted=list(csv.DictReader(accepted_path.open())) if accepted_path.exists() else []
    accepted=[r for r in accepted if 5613<=int(r['step'])<=5622]
    passed=passed and len(accepted)==10 and all(r['fresh_linear_checks_passed']=='1' for r in accepted)
    intervals=[float(r['dt']) for r in rows]
    if passed and a.branch=='R':
        (root/'actual_intervals.txt').write_text(''.join(format(x,'.17g')+'\n' for x in intervals))
    if passed and a.branch!='R':
        passed=intervals==ref['actual_intervals']
    info={'returncode':status,'passed':passed,'seconds':time.monotonic()-start,
          'binary_sha256':sha(a.binary),'launcher':a.launcher,'environment':env,
          'actual_intervals':intervals,'cumulative_elapsed':sum(intervals)}
    info['accepted_solver_records']=accepted
    if passed:
        info['clock_sha256']=sha(root/'actual_intervals.txt')
    (dest/'execution.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps({k:v for k,v in info.items() if k!='environment'},indent=2))
    if not passed:
        raise SystemExit('Stopped. Failure and accepted prefix preserved; no automatic retry.')


def half_retry(a):
    original=a.directory.resolve()
    prior=json.loads((original/'provenance.json').read_text())
    if 'retry_of' in prior or (original/'halved_retry.json').exists():
        raise ValueError('Only one common half-clock retry is allowed')
    ref=json.loads((original/'R/execution.json').read_text())
    if not ref['passed'] or len(ref['actual_intervals'])!=10:
        raise ValueError('The original complete R clock is required')
    # Ten half-sized intervals, not twenty replacement steps: at most twenty
    # accepted states per branch across the two attempts. Keep their shorter
    # common elapsed interval explicit in the report.
    failed=False
    for branch in ('R','F1','F2'):
        p=original/branch/'execution.json'
        if p.exists():
            record=json.loads(p.read_text())
            assert len(record['actual_intervals'])<=10
            failed |= not record['passed']
    if not failed:
        raise ValueError('A common retry is only for a recorded failed attempt')
    prepare(SimpleNamespace(destination=a.destination,checkpoint=Path(prior['source_checkpoint']),
                            fixture=original/'fixture',libraries=original,
                            secondary_length=prior['selected_lengths_m'][1]))
    root=a.destination.resolve()
    (root/'actual_intervals.txt').write_text(''.join(format(dt/2,'.17g')+'\n' for dt in ref['actual_intervals']))
    with (root/'R/run.prm').open('a') as f:
        f.write('\nsubsection Time stepping\n set List of model names = convection time step, reconstructed fault time step, BP5 state startup, BP5 filter clock\n subsection BP5 filter clock\n set Actual intervals file = ../actual_intervals.txt\n end\nend\n')
    m=json.loads((root/'provenance.json').read_text());m['retry_of']=str(original)
    m['inputs_sha256']['R/run.prm']=sha(root/'R/run.prm')
    m['inputs_sha256']['actual_intervals.txt']=sha(root/'actual_intervals.txt')
    (root/'provenance.json').write_text(json.dumps(m,indent=2)+'\n')
    (original/'halved_retry.json').write_text(json.dumps({'retry':str(root)},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('prepare')
    for name in ('checkpoint','fixture','libraries','destination'):
        s.add_argument('--'+name,type=Path,required=True)
    s.add_argument('--secondary-length',type=int,choices=[200,50],default=200,
                   help='Use 50 only if the offline 200-m trial plainly oversmooths broad features')
    s=sub.add_parser('run');s.add_argument('directory',type=Path)
    s.add_argument('branch',choices=['R','F1','F2']);s.add_argument('binary',type=Path)
    s.add_argument('launcher',nargs=argparse.REMAINDER)
    s=sub.add_parser('half-retry');s.add_argument('directory',type=Path)
    s.add_argument('destination',type=Path)
    a=p.parse_args()
    if a.action=='run' and not a.launcher:
        p.error('Supply an explicit MPI launcher, e.g. mpirun -np 4')
    {'prepare':prepare,'run':run,'half-retry':half_retry}[a.action](a)
