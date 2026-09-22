"""Bounded steady-state initialization: 300/150 s startup and step-2 restart.

Uses the existing physical mesh/profile fixture and keeps the 0.02 predictor.
Each case is immutable, four-rank, capped at 1200 s; no automatic retries.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

from startup_30km import HERE, BP3, ROOT, BIN, FIXTURE, STUDY, parameters, render, digest, restore_prefix

OUT = STUDY / 'steady-startup'
PLUGIN = HERE / 'build/libbp5_steady_initialization.release.so'
BASE = STUDY / 'startup-followup/adaptive'


def prepare(case):
    path = OUT / case
    path.mkdir(parents=True)
    values = parameters((BASE / 'run.prm').read_text())
    values['Additional shared libraries',] = str(PLUGIN) + ', ' + values['Additional shared libraries',].split(', ', 1)[1]
    values['Output directory',] = str(path)
    values['Postprocess', 'BP3', 'Mature prestress file'] = ''
    values['Maximum time step',] = '150' if case == 'half' else '300'
    values['End time',] = '900'
    values['Postprocess', 'BP3', 'Last accepted step'] = '6' if case == 'half' else '3'
    values['Postprocess', 'BP3', 'Graceful wall seconds'] = '1100'
    values['Checkpointing', 'Steps between checkpoint'] = '3' if case == 'startup' else '0'
    values['Termination criteria', 'Checkpoint on termination'] = 'false'
    values['Resume computation',] = 'false'
    if case == 'resume':
        source = OUT / 'startup'
        assert json.loads((source / 'execution.json').read_text())['passed']
        checkpoints = [p for p in (source / 'restart').glob('[0-9][0-9]')
                       if int((p / 'bp3_accepted_state.txt').read_text().split()[0]) == 2]
        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        shutil.copytree(checkpoint, path / 'restart/01')
        (path / 'restart/last_good_checkpoint.txt').write_text('1\n')
        for p in (checkpoint / 'bp3_output_metadata').iterdir():
            shutil.copy2(p, path / p.name)
        shutil.copy2(source / 'cumulative_slip.csv', path / 'cumulative_slip.csv')
        restore_prefix(path / 'cumulative_slip.csv', 2)
        values['Resume computation',] = 'true'
        (path / 'checkpoint_source.json').write_text(json.dumps(dict(source=str(checkpoint),
            hashes={str(p.relative_to(checkpoint)): digest(p) for p in checkpoint.rglob('*') if p.is_file()}), indent=2) + '\n')
    (path / 'run.prm').write_text(render(values))
    record = json.loads((BASE / 'launch.json').read_text())
    sources = [HERE / 'steady_initialization.h', HERE / 'startup_time_step.cc', HERE / 'CMakeLists.txt',
               BP3 / 'bp3.cc', BP3 / 'bp3_model.h', BP3 / 'work_replay.h', BP3 / 'mature_fault.h',
               Path(__file__).resolve()]
    snapshot = path / 'source-tested'
    for p in sources:
        dest = snapshot / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    (snapshot / 'working-tree.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    record['source_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    inputs = [BIN, path / 'run.prm', *sources, *map(Path, values['Additional shared libraries',].split(', '))]
    inputs += [FIXTURE / name for name in ('fault.txt', 'target_cells.txt', 'completion.txt')]
    record.update(command=['mpirun', '-np', '4', '--bind-to', 'core', '--map-by', 'core', str(BIN), str(path / 'run.prm')],
                  cap_seconds=1200, hashes={str(p): digest(p) for p in inputs},
                  initialization='uniform Dc/Vinit state; native weak projected variable background')
    (path / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')


def run(case):
    path = OUT / case
    record = json.loads((path / 'launch.json').read_text())
    for name, sha in record['hashes'].items():
        assert digest(Path(name)) == sha, name
    used = sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('*/execution.json'))
    assert used + record['cap_seconds'] <= 3600
    assert shutil.disk_usage(OUT).free > 4 * 1024**3
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment'])
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1', LD_BIND_NOW='1')
    start = time.monotonic()
    with (path / 'run.log').open('x') as stream:
        result = subprocess.run(['timeout', '--kill-after=15', str(record['cap_seconds'])] + record['command'],
                                cwd=BP3, env=env, stdout=stream, stderr=subprocess.STDOUT)
    clock = list(csv.DictReader((path / 'accepted_steps.csv').open())) if (path / 'accepted_steps.csv').exists() else []
    passed = result.returncode == 0 and bool(clock) and float(clock[-1]['time']) == 900.
    info = dict(seconds=time.monotonic() - start, status=result.returncode, passed=passed,
                accepted_steps=[int(r['step']) for r in clock],
                peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (path / 'execution.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))
    assert passed, 'Unexpected result preserved; do not automatically retry'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('case', choices=['startup', 'half', 'resume'])
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.case)
