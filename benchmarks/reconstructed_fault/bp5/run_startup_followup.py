"""One 75-s comparison, genuine predictor-limited startup, and same-rank restart."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from startup_30km import STUDY, HERE, BP3, BIN, PLUGIN, FIXTURE, parameters, render, digest, restore_prefix

OUT = STUDY/'startup-followup'


def prepare(case):
    path = OUT/case
    path.mkdir(parents=True)
    source = STUDY/'small-startup/300'
    values = parameters((source/'run.prm').read_text())
    record = json.loads((source/'launch.json').read_text())
    # Keep the exact previously verified executable and plugin. All changes
    # below are clock/termination/checkpoint/output settings, not mechanics.
    for p in [BIN, PLUGIN]:
        assert digest(p) == record['hashes'][str(p)], p
    values['Output directory',] = str(path)
    values['Postprocess','BP3','Last accepted step'] = '4'
    values['Postprocess','BP3','Graceful wall seconds'] = '1100'
    if case == '75':
        values['Maximum time step',] = '75'
        values['End time',] = '300'
    else:
        values['Maximum time step',] = '4e6'
        values['End time',] = '16000000'  # Step guard stops at four, without clipping dt.
        values['Time stepping','BP5 state startup','Maximum logarithmic state change'] = '0.02'
        values['Postprocess','BP3','Audit full state every step'] = 'true'
        values['Checkpointing','Steps between checkpoint'] = '3' if case == 'adaptive' else '0'
    if case == 'resume':
        reference = OUT/'adaptive'
        assert json.loads((reference/'execution.json').read_text())['passed']
        checkpoints = [p for p in (reference/'restart').glob('[0-9][0-9]')
                       if int((p/'bp3_accepted_state.txt').read_text().split()[0]) == 2]
        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        shutil.copytree(checkpoint, path/'restart/01')
        (path/'restart/last_good_checkpoint.txt').write_text('1\n')
        for p in (checkpoint/'bp3_output_metadata').iterdir():
            shutil.copy2(p, path/p.name)
        shutil.copy2(reference/'cumulative_slip.csv', path/'cumulative_slip.csv')
        restore_prefix(path/'cumulative_slip.csv', 2)
        values['Resume computation',] = 'true'
        (path/'checkpoint_source.json').write_text(json.dumps(dict(source=str(checkpoint),
            hashes={str(p.relative_to(checkpoint)):digest(p) for p in checkpoint.rglob('*') if p.is_file()}),indent=2)+'\n')
    (path/'run.prm').write_text(render(values))
    inputs = [BIN, PLUGIN, Path(__file__).resolve(), path/'run.prm', HERE/'startup_time_step.cc']
    inputs += list(FIXTURE.glob('*.txt'))
    inputs += list(map(Path, values['Additional shared libraries',].split(', ')))
    record.update(command=['mpirun','-np','4','--bind-to','core','--map-by','core',str(BIN),str(path/'run.prm')],
                  cap_seconds=1200, hashes={str(p):digest(p) for p in inputs})
    (path/'launch.json').write_text(json.dumps(record,indent=2)+'\n')


def run(case):
    path = OUT/case
    record = json.loads((path/'launch.json').read_text())
    for p, sha in record['hashes'].items():
        assert digest(Path(p)) == sha, p
    used = sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('*/execution.json'))
    assert used + record['cap_seconds'] <= 3600
    assert shutil.disk_usage(OUT).free > 4*1024**3
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
    env.update(record['environment'])
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1', LD_BIND_NOW='1')
    start = time.monotonic()
    with (path/'run.log').open('x') as log:
        result = subprocess.run(['timeout','--kill-after=15',str(record['cap_seconds'])]+record['command'],
                                cwd=BP3,env=env,stdout=log,stderr=subprocess.STDOUT)
    rows = list(csv.DictReader((path/'accepted_steps.csv').open())) if (path/'accepted_steps.csv').exists() else []
    passed = result.returncode == 0 and bool(rows) and int(rows[-1]['step']) == 4
    if case == '75':
        passed = passed and float(rows[-1]['time']) == 300.
    info = dict(seconds=time.monotonic()-start, status=result.returncode, passed=passed,
                accepted_steps=[int(r['step']) for r in rows],
                peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (path/'execution.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))
    assert passed, 'Unexpected outcome preserved; no automatic retry'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('case',choices=['75','adaptive','resume'])
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.case)
