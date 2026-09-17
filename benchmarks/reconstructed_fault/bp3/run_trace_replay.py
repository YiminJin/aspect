"""Fresh independent V/Theta/slip traces, original clock, seven real steps only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = HERE/'work-replay-50-local4'
OUT = HERE/'trace-replay-seven-local4'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fully-frictional',action='store_true',
                        help='Release the whole deep segment; retain continuous Q1 without split traces.')
    args=parser.parse_args()
    global OUT
    if args.fully_frictional:
        OUT=HERE/'fully-frictional-seven-local4'
    OUT.mkdir()  # Preserve failed evidence; never overwrite or retry.
    with (BASE/'accepted_steps.csv').open() as stream:
        clock = list(csv.DictReader(stream))[:8]
    with (OUT/'clock.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'time', 'dt'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(clock)
    (OUT/'run.prm').write_text(f'''include {BASE}/run.prm
set Output directory = {OUT}
set Resume computation = false
set End time = {clock[-1]['time']}
subsection Termination criteria
  set Termination criteria = end time, BP3 replay complete
end
''')
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(REPO), ASPECT_FAULT_EXPLICIT_B='1', ASPECT_FAULT_EXPLICIT_G='1',
               ASPECT_FAULT_SURFACE_SOLVER='tridiagonal', ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
               ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC='1', ASPECT_BP3_SPLIT_TRACE_REPLAY='795',
               ASPECT_BP3_TIMESTEP_SEQUENCE=str(OUT/'clock.csv'), ASPECT_BP3_EXACT_TARGET='1',
               ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
               ASPECT_BP3_EXPECTED_FAULT=str(HERE/'fault-grid-50-local4/fault.txt'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
    if args.fully_frictional:
        env.pop('ASPECT_BP3_SPLIT_TRACE_REPLAY')
        env['ASPECT_BP3_FULLY_FRICTIONAL_REPLAY']='1'
    command = ['timeout', '--signal=TERM', '--kill-after=15', '2400', 'mpirun', '-np', '4',
               '--bind-to', 'core', '--map-by', 'core', str(REPO/'build-pf-cpdi/aspect-release'), str(OUT/'run.prm')]
    paths = [Path(__file__), OUT/'run.prm', OUT/'clock.csv', BASE/'run.prm', BASE/'accepted_steps.csv',
             HERE/'mature-fault-50-local4/prestress.txt', HERE/'top-source-paired-50-local4/completion.txt',
             HERE/'bp3.cc', HERE/'bp3_model.h', HERE/'cohesion_diagnostic.h', HERE/'work_replay.h',
             HERE/'build/libbp3.release.so', REPO/'build-pf-cpdi/aspect-release']
    record = dict(command=command, environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
                  sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    (OUT/'source.patch').write_bytes(subprocess.check_output(['git', 'diff'], cwd=REPO))
    start = time.monotonic()
    with (OUT/'run.log').open('x') as stream:
        result = subprocess.run(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
    log = (OUT/'run.log').read_text()
    linear = re.findall(r'Fault linear solve: iterations=(\d+), estimated=[^,]+, fresh=([^,]+), target=([^,]+)', log)
    record = dict(status=result.returncode, seconds=time.monotonic()-start,
                  peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  fresh_linear_checks=len(linear), fresh_linear_passed=bool(linear) and all(float(a)<=float(b) for _,a,b in linear),
                  krylov=sum(int(i) for i,_,_ in linear), first_update_passed='BP3 WORK REPLAY FIRST UPDATE PASSED' in log)
    (OUT/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    assert record['status']==0 and record['fresh_linear_passed'] and record['first_update_passed']
    with (OUT/'accepted_steps.csv').open() as stream:
        accepted = list(csv.DictReader(stream))
    assert [int(r['step']) for r in accepted]==list(range(8))
    for a,b in zip(accepted, clock):
        for key in ('time','dt'):
            # Initial dt output is the artificial interval, not elapsed time.
            if key=='dt' and int(a['step'])==0: continue
            assert abs(float(a[key])-float(b[key]))<=1e-12*max(1.,float(b[key]))


if __name__=='__main__':
    main()
