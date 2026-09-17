"""Four-rank research replay, filesystem restart qualification, or bounded continuation.

Continuation must be explicitly bounded and omits the saved-clock model. No
automatic retries; a wall-limit kill leaves the last ordinary checkpoint intact.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
REFERENCE=HERE/'reference_200km'
REPO = HERE.parents[2]
INPUTS = HERE/'fixtures/modified_bp3'
OUT = HERE/'fully-frictional-cleanup-local4'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'wide-research-local4')
    parser.add_argument('--configuration', choices=('frictional','constrained','wide'), default='wide')
    parser.add_argument('--velocity-preconditioner', choices=('amg','gmg'), default='gmg',
                        help='Velocity-block preconditioner (default: gmg); amg retains the reference path')
    parser.add_argument('--mode', choices=('replay','restart-qualification','continuation'), default='replay')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--mesh-only', action='store_true', help='Verify/export the prepared mesh and stop before mechanics')
    parser.add_argument('--resume', action='store_true', help='Continue the checkpoint in --output; continuation only')
    parser.add_argument('--end-time', type=float, help='Absolute physical time in seconds; continuation requires this')
    parser.add_argument('--end-step', type=int, help='Last accepted timestep number; continuation requires this')
    parser.add_argument('--wall-seconds', type=int, help='Hard process cap; continuation requires this')
    args=parser.parse_args()
    plugin=HERE/'build/libbp3_research.release.so'
    if not plugin.is_file():
        parser.error('The preserved research fixture requires BP3_BUILD_RESEARCH_REFERENCE=ON and target bp3_research. The maintained bp3 target is the clean long-run model.')
    if args.mesh_only and args.mode!='replay':
        parser.error('--mesh-only belongs to a fresh replay configuration.')
    if args.mode=='continuation':
        if not (args.end_time and math.isfinite(args.end_time) and args.end_time>0
                and args.end_step and args.end_step>0 and args.wall_seconds and args.wall_seconds>=120):
            parser.error('Continuation requires finite positive --end-time, positive --end-step and --wall-seconds >= 120.')
        if not args.prepare_only:
            parser.error('Continuation remains prepare-only: same-binary restart qualification hit the frozen-Ih invariant. Resolve and qualify that restart before enabling execution.')
    else:
        if args.resume or any(v is not None for v in (args.end_time,args.end_step,args.wall_seconds)):
            parser.error('Resume and explicit bounds belong only to continuation; replay uses its fixed saved clock.')
    global OUT
    OUT=args.output.resolve()
    fixture=HERE/{'frictional':'bp3_modified_fully_frictional.prm',
                  'constrained':'bp3_constrained_reference.prm',
                  'wide':'bp3_modified_wide.prm'}[args.configuration]
    mesh_inputs=HERE/'fixtures/modified_bp3_wide' if args.configuration=='wide' else INPUTS
    if args.resume:
        assert (OUT/'restart/last_good_checkpoint.txt').is_file()
    else:
        OUT.mkdir()  # Preserve failed evidence; never overwrite or retry.
    manifest=json.loads((INPUTS/'manifest.json').read_text())
    for name,entry in manifest.items():
        assert hashlib.sha256((INPUTS/name).read_bytes()).hexdigest()==entry['sha256'],name
    if mesh_inputs!=INPUTS:
        for name,entry in json.loads((mesh_inputs/'manifest.json').read_text()).items():
            assert hashlib.sha256((mesh_inputs/name).read_bytes()).hexdigest()==entry['sha256'],name
    with (INPUTS/'seven_step_clock.csv').open() as stream:
        clock = list(csv.DictReader(stream))
    with (OUT/'clock.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'time', 'dt'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(clock)
    def parameters(resume, end_step):
        adaptive=args.mode=='continuation'
        # End-step is checked after advance_time: end_step=4 checkpoints the
        # accepted state 4 with the already chosen step-5 time/dt, not dt=0.
        text=f'''include {fixture}
set Additional shared libraries = {plugin}
set Output directory = {OUT}
set Resume computation = {str(resume).lower()}
set End time = {args.end_time if adaptive else clock[-1]['time']}
subsection Termination criteria
  set Termination criteria = end time, end step, wall time{'' if adaptive else ', BP3 replay complete'}
  set End step = {end_step}
  set Wall time = {((args.wall_seconds or 2400)-60)/3600:.17g}
  set Checkpoint on termination = true
end
'''
        if adaptive:
            text+='''subsection Time stepping
  set List of model names = convection time step, reconstructed fault time step
end
'''
        return text
    first_name='continuation' if args.mode=='continuation' else 'run'
    (OUT/f'{first_name}.prm').open('x').write(parameters(args.resume, args.end_step or (4 if args.mode=='restart-qualification' else 7)))
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(REPO), ASPECT_FAULT_EXPLICIT_B='1', ASPECT_FAULT_EXPLICIT_G='1',
               ASPECT_FAULT_SURFACE_SOLVER='tridiagonal', ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
               ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC='1',
               ASPECT_BP3_EXACT_TARGET='1',
               ASPECT_BP3_TARGET_MESH=str(mesh_inputs/'target_cells.txt'),
               ASPECT_BP3_EXPECTED_FAULT=str(INPUTS/'fault.txt'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
    if args.mode!='continuation': env['ASPECT_BP3_TIMESTEP_SEQUENCE']=str(OUT/'clock.csv')
    if args.velocity_preconditioner=='gmg':
        env.update(ASPECT_FAULT_VELOCITY_GMG='1', ASPECT_FAULT_GMG_HIERARCHY='1')
    if args.mesh_only: env['ASPECT_BP3_MESH_ONLY']='1'
    command = ['timeout', '--signal=TERM', '--kill-after=15', str(180 if args.mesh_only else args.wall_seconds or 2400), 'mpirun', '-np', '4',
               '--bind-to', 'core', '--map-by', 'core', str(REPO/'build-pf-cpdi/aspect-release')]
    paths = [Path(__file__), OUT/f'{first_name}.prm', OUT/'clock.csv', fixture, HERE/'bp3_modified_fully_frictional.prm',
             *[INPUTS/name for name in manifest], mesh_inputs/'target_cells.txt',
             REFERENCE/'bp3.cc', REFERENCE/'bp3_model.h', REFERENCE/'work_checks.h', REFERENCE/'work_replay.h',
             plugin, REPO/'build-pf-cpdi/aspect-release']
    record = dict(command=command+[str(OUT/f'{first_name}.prm')], environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
                  sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/f'{first_name}_provenance.json').open('x').write(json.dumps(record, indent=2)+'\n')
    if args.mode=='replay':
        (OUT/'provenance.json').open('x').write(json.dumps(record,indent=2)+'\n')
    (OUT/f'{first_name}_source.patch').open('xb').write(subprocess.check_output(['git', 'diff'], cwd=REPO))
    if args.prepare_only:
        print('Prepared only:', OUT/f'{first_name}.prm')
        return
    def execute(name):
        start = time.monotonic()
        with (OUT/f'{name}.log').open('x') as stream:
            result = subprocess.run(command+[str(OUT/f'{name}.prm')], cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
        log = (OUT/f'{name}.log').read_text()
        linear = re.findall(r'Fault linear solve: iterations=(\d+),(?: estimated=[^,\n]+,)? fresh=([^,\n]+), target=([^,\n]+)', log)
        record = dict(status=result.returncode, seconds=time.monotonic()-start,
                  peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  fresh_linear_checks=len(linear), fresh_linear_passed=bool(linear) and all(float(a)<=float(b) for _,a,b in linear),
                  krylov=sum(int(i) for i,_,_ in linear), first_update_passed='BP3 WORK REPLAY FIRST UPDATE PASSED' in log)
        (OUT/f'{name}_execution.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2), flush=True)
        if args.mesh_only:
            assert result.returncode not in (0,124,137)
            assert 'Intentional mesh-only stop before mechanics.' in log
            assert 'BP3 exact fault-grid guard:' in log
            if args.configuration=='wide':
                subprocess.run(['python3',str(HERE/'prepare_wide_fixture.py'),
                                '--check-export',str(OUT)],check=True)
            print('Mesh-only preflight passed; no mechanical trajectory was run.')
            return record
        assert record['status']==0 and record['fresh_linear_passed']
        return record
    first=execute(first_name)
    if args.mesh_only: return
    if args.mode=='continuation': return
    if args.mode=='replay': (OUT/'execution.json').write_text(json.dumps(first,indent=2)+'\n')
    assert first['first_update_passed']
    if args.mode=='restart-qualification':
        with (OUT/'accepted_steps.csv').open() as stream:
            assert [int(r['step']) for r in csv.DictReader(stream)]==list(range(5))
        # Preserve the ordinary accepted-step-4 checkpoint before rotation.
        shutil.copytree(OUT/'restart',OUT/'step4_checkpoint')
        (OUT/'resume.prm').open('x').write(parameters(True,7))
        execute('resume')
    with (OUT/'accepted_steps.csv').open() as stream:
        accepted = list(csv.DictReader(stream))
    assert [int(r['step']) for r in accepted]==list(range(8))
    if args.configuration in ('frictional','wide'):
        for r in accepted:
            assert int(r['free'])==1236 and int(r['lower_active'])==0
    for a,b in zip(accepted, clock):
        for key in ('time','dt'):
            # Initial dt output is the artificial interval, not elapsed time.
            if key=='dt' and int(a['step'])==0: continue
            assert abs(float(a[key])-float(b[key]))<=1e-12*max(1.,float(b[key]))


if __name__=='__main__':
    main()
