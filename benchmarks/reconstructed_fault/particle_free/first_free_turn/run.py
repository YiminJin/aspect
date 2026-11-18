#!/usr/bin/env python3
"""One explicitly selected native control or free-turn window; no hidden campaign."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('case', choices=['smoke-baseline', 'smoke-native', 'smoke-strips', 'strip-append',
                                    'positive-prefix', 'reflected-prefix', 'positive-end', 'reflected-end'])
parser.add_argument('--label', required=True)
parser.add_argument('--resume-from', type=Path)
parser.add_argument('--rejection', action='store_true')
parser.add_argument('--rejection-step', type=int)
parser.add_argument('--snapshot-id', type=int, choices=[1, 2, 3])
parser.add_argument('--seconds', type=int, default=7200)
parser.add_argument('--ranks', type=int, choices=[2,4], default=2)
parser.add_argument('--end-step', type=int)
parser.add_argument('--point-stride', type=int)
args = parser.parse_args()
evidence = root/'benchmarks/reconstructed_fault/particle_free/evidence/first_free_turn'
run = root/'build-particle-free/bulk-tip/runs'/args.label
assert not run.exists()
smoke = args.case.startswith('smoke')
if smoke:
    fixture = Path(__file__).with_name('strip-smoke.prm' if args.case == 'smoke-strips' else 'straight-smoke.prm')
else:
    fixture = Path(__file__).with_name('positive.prm' if args.case.startswith('positive') else 'reflected.prm')
if args.case == 'strip-append':
    fixture = evidence/(args.label+'.prm')
    fixture.write_text(Path(__file__).with_name('positive.prm').read_text()
                       .replace('set X repetitions = 128','set X repetitions = 32')
                       .replace('set Y repetitions = 128','set Y repetitions = 16'))
tested = root/'build-particle-free/first-free-turn/tested'
if args.case == 'smoke-baseline':
    tested = root/'build-particle-free/tip-step89/tested'
env = os.environ.copy()
for key in list(env):
    if key.startswith('ASPECT_TIP_'):
        env.pop(key)
env['ASPECT_TIP_POSITIVE_METRIC_FIRST'] = '1'
env['ASPECT_TIP_POINT_EXPORT_STRIDE'] = str(args.point_stride or (1 if smoke or args.case == 'strip-append' else 4))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'DEAL_II_NUM_THREADS'):
    env[key] = '1'
command = ['python3', str(root/'benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py'), args.label,
           '--fixture', str(fixture), '--executable', str(tested/'aspect-release'),
           '--checker', str(tested/'libbulk_tip_checks.release.so'), '--ranks', str(args.ranks),
           '--dt', '.005', '--end', '.01' if smoke else '.08', '--time-update', 'midpoint',
           '--phase-balance', '--checkpoint-steps', '1', '--seconds', str(args.seconds)]
if args.ranks==4:
    command += ['--bind-cores']
if not smoke:
    command += ['--integrity-stop']
if args.end_step or args.case.endswith('prefix'):
    command += ['--end-step', str(args.end_step or 12)]
if args.resume_from:
    env['ASPECT_TIP_RESTORE_AUDIT'] = '1'
    command += ['--resume-from', str(args.resume_from.resolve()), '--copy-selected-snapshot-only']
if args.snapshot_id is not None:
    assert args.resume_from
    command += ['--snapshot-id', str(args.snapshot_id)]
if args.rejection:
    command += ['--rejection', '--rejection-step', str(args.rejection_step if args.rejection_step is not None else (2 if smoke else 13))]
(evidence/(args.label+'-launch.json')).write_text(json.dumps(dict(command=command,
    environment={k:v for k,v in env.items() if k.startswith(('ASPECT_TIP_', 'OMP_', 'OPENBLAS_', 'DEAL_II_'))}), indent=2)+'\n')
records = []


def compress_closed(last, restore_closed=False):
    paths=list((run/'output').glob('tip-*.csv'))
    if restore_closed:
        paths+=list((run/'output').glob('restore-points-r*.csv'))
        paths+=list((run/'output').glob('restore-phase-r*.csv'))
    for p in sorted(paths):
        match = re.match(r'tip-(?:points|balance)-(\d+)-r\d+\.csv$', p.name)
        step = int(match[1]) if match else 0 if p.name.startswith(('tip-seed-fracture-r','restore-')) else None
        if step is None or step > last:
            continue
        target = p.with_suffix('.csv.gz')
        assert not target.exists()
        digest = hashlib.sha256()
        with p.open('rb') as source, gzip.open(target, 'wb', compresslevel=6) as output:
            for block in iter(lambda: source.read(1048576), b''):
                digest.update(block)
                output.write(block)
        with gzip.open(target, 'rb') as source:
            assert hashlib.file_digest(source, 'sha256').hexdigest() == digest.hexdigest()
        records.append(dict(file=str(target.relative_to(root)), raw_sha256=digest.hexdigest(),
                            raw_bytes=p.stat().st_size, gzip_bytes=target.stat().st_size))
        (evidence/(args.label+'-compression.json')).write_text(json.dumps(records, indent=2)+'\n')
        p.unlink()


job = subprocess.Popen(command, env=env)
while job.poll() is None:
    log = run/'run.log'
    if log.exists() and args.ranks<4:
        content=log.read_text()
        steps = re.findall(r'\*\*\* Timestep (\d+):', content)
        if steps:
            compress_closed(int(steps[-1])-1, 'Bulk tip accepted parent:' in content)
    time.sleep(3)
compress_closed(10**9, (run/'run.log').exists() and 'Bulk tip accepted parent:' in (run/'run.log').read_text())
raise SystemExit(job.returncode)
