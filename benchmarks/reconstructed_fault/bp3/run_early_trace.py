"""Reconstruct only accepted step 1, then one disposable step-2 free trace."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = HERE/'work-replay-50-local4'
ROOT = HERE/'early-free-trace'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prefix', 'probe'])
    parser.add_argument('--directory', default='early-free-trace-matched')
    args = parser.parse_args()
    global ROOT
    ROOT = HERE/args.directory
    out = ROOT/args.action
    out.mkdir(parents=True)
    with (BASE/'accepted_steps.csv').open() as stream:
        clock = list(csv.DictReader(stream))
    fields = ['step', 'time', 'dt']
    with (out/'clock.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        # Include the next step even when stopping after step 1: ASPECT
        # checkpoints the already selected next dt, then restores it verbatim.
        writer.writeheader(); writer.writerows(clock[:3])
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(ASPECT_SOURCE_DIR=str(REPO), ASPECT_FAULT_EXPLICIT_B='1', ASPECT_FAULT_EXPLICIT_G='1',
               ASPECT_FAULT_SURFACE_SOLVER='tridiagonal', ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC='1',
               ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC='1', ASPECT_BP3_TIMESTEP_SEQUENCE=str(out/'clock.csv'),
               ASPECT_BP3_TARGET_MESH=str(HERE/'junction-matched-qualified-local4/refined/target_cells.txt'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
    if args.action == 'prefix':
        env.update(ASPECT_BP3_EXACT_TARGET='1', ASPECT_BP3_EXPECTED_FAULT=str(HERE/'fault-grid-50-local4/fault.txt'))
        text = f'''include {BASE}/run.prm
set Output directory = {out}
set Resume computation = false
set End time = {clock[1]['time']}
'''
    else:
        assert json.loads((ROOT/'prefix/execution.json').read_text())['status'] == 0
        assert json.loads((ROOT/'prefix/comparison.json').read_text())['passed']
        checkpoint_id = (ROOT/'prefix/restart/last_good_checkpoint.txt').read_text().strip()
        checkpoint = ROOT/'prefix/restart'/f'{int(checkpoint_id):02d}'
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir() if p.is_file()}
        shutil.copytree(checkpoint, out/'restart/01')
        (out/'restart/last_good_checkpoint.txt').write_text('1\n')
        env.update(ASPECT_BP3_WITHIN_STEP_DIAGNOSTIC=str(ROOT/'prefix'), ASPECT_BP3_EARLY_TRACE_STEP='2',
                   ASPECT_FAULT_FREE_TRACE_DIAGNOSTIC='795', ASPECT_FAULT_NONCOMMITTING_DIAGNOSTIC='1',
                   ASPECT_FAULT_COMPARE_COUPLING='1', ASPECT_FAULT_COMPARE_SURFACE_INVERSE='1')
        # The target clock/geometry is the original accepted step 2; its
        # incoming state is the separately verified fresh prefix, never fitted.
        shutil.copy2(BASE/'fault_2.csv', ROOT/'prefix/fault_2.csv')
        text = f'''include {BASE}/run.prm
set Output directory = {out}
set Resume computation = true
subsection Postprocess
  subsection BP3
    set Committing work-measure replay = false
  end
end
'''
    (out/'run.prm').write_text(text)
    command = ['timeout', '--signal=TERM', '--kill-after=15', '600', 'mpirun', '-np', '4',
               '--bind-to', 'core', '--map-by', 'core', str(REPO/'build-pf-cpdi/aspect-release'), str(out/'run.prm')]
    record = dict(command=command, environment={k:v for k,v in env.items() if k.startswith('ASPECT_')},
                  executable_sha256=hashlib.sha256((REPO/'build-pf-cpdi/aspect-release').read_bytes()).hexdigest(),
                  plugin_sha256=hashlib.sha256((HERE/'build/libbp3.release.so').read_bytes()).hexdigest())
    if args.action == 'probe': record.update(checkpoint=str(checkpoint), checkpoint_sha256=hashes)
    (out/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    start = time.monotonic()
    with (out/'run.log').open('x') as stream:
        result = subprocess.run(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
    log = (out/'run.log').read_text()
    linear = re.findall(r'Fault linear solve: iterations=(\d+), estimated=[^,]+, fresh=([^,]+), target=([^,]+)', log)
    record.update(status=result.returncode, seconds=time.monotonic()-start,
                  peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  fresh_linear_checks=len(linear), fresh_linear_passed=bool(linear) and all(float(a)<=float(b) for _,a,b in linear),
                  krylov=sum(int(i) for i,_,_ in linear))
    if args.action == 'probe':
        record.update(converged='Noncommitting fault diagnostic converged:' in log,
                      rollback='BP3 noncommitting rollback verified:' in log,
                      trace_checks='Independent trace checks passed:' in log,
                      accepted_output_written=(out/'accepted_steps.csv').exists(),
                      original_unchanged=all(hashlib.sha256((checkpoint/n).read_bytes()).hexdigest()==v for n,v in hashes.items()),
                      copy_unchanged=all(hashlib.sha256((out/'restart/01'/n).read_bytes()).hexdigest()==v for n,v in hashes.items()))
    (out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    assert record['fresh_linear_passed']
    if args.action == 'prefix':
        assert result.returncode == 0 and 'BP3 WORK REPLAY FIRST UPDATE PASSED' in log
        with (out/'accepted_steps.csv').open() as stream:
            accepted = list(csv.DictReader(stream))
        assert [int(r['step']) for r in accepted] == [0,1]
    else:
        assert all(record[k] for k in ('converged', 'rollback', 'trace_checks', 'original_unchanged', 'copy_unchanged'))
        assert not record['accepted_output_written']


if __name__ == '__main__':
    main()
