"""Peak-.60 screen or selected continuation, with lossless export compression."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('mesh', choices=['coarse', 'fine', 'axial'], help='fine refines y; axial refines x only.')
p.add_argument('--growth', action='store_true')
p.add_argument('--dt', type=float, default=.01)
p.add_argument('--end', type=float, default=.08)
p.add_argument('--seconds', type=int, default=1800)
p.add_argument('--checkpoint-steps', type=int, default=8)
p.add_argument('--wall-hours', type=float, help='Graceful existing ASPECT wall-time limit.')
p.add_argument('--label', help='Distinct fresh output label for a selected continuation.')
p.add_argument('--evidence-directory', type=Path)
p.add_argument('--resume-from', type=Path, help='Accepted output directory, copied by run.py.')
p.add_argument('--snapshot-id', type=int, choices=[1, 2, 3])
p.add_argument('--end-step', type=int)
p.add_argument('--checker', type=Path, help='Explicit benchmark checker; protected default is retained.')
p.add_argument('--extend-terminal-restart', action='store_true', help='Opt-in checker rescheduling of a roundoff-sized terminal interval.')
p.add_argument('--ranks', type=int, choices=[2, 4], default=2)
p.add_argument('--defer-compression', action='store_true', help='Compress after MPI exits; required with four ranks.')
a = p.parse_args()
assert a.ranks == 2 or a.defer_compression, 'Respect four aggregate numerical workers.'
root = Path(__file__).resolve().parents[4]
evidence = root / 'benchmarks/reconstructed_fault/particle_free/evidence/resolution_matched_tip'
if a.evidence_directory:
    evidence = a.evidence_directory.resolve()
evidence.mkdir(parents=True, exist_ok=True)
label = a.label or ('resolution60-' + a.mesh + ('-growth' if a.growth else '-screen'))
assert label and all(c.isalnum() or c in '-_' for c in label)
run = root / 'build-particle-free/bulk-tip/runs' / label
assert not run.exists(), run
fixture = evidence / (label + '.prm')
text = Path(__file__).with_name('resolution_matched_seed.prm').read_text()
if a.mesh == 'fine':
    text = text.replace('set Y repetitions = 128', 'set Y repetitions = 256')
elif a.mesh == 'axial':
    text = text.replace('set X repetitions = 128', 'set X repetitions = 256')
fixture.write_text(text)
env = os.environ.copy()
for key in list(env):
    if key.startswith('ASPECT_TIP_'):
        env.pop(key)
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'DEAL_II_NUM_THREADS'):
    env[key] = '1'
if a.growth:
    env['ASPECT_TIP_POSITIVE_METRIC_FIRST'] = '1'
if a.resume_from:
    env['ASPECT_TIP_RESTORE_AUDIT'] = '1'
if a.extend_terminal_restart:
    assert a.resume_from and a.checker
    env['ASPECT_TIP_EXTEND_TERMINAL_RESTART_DT'] = str(a.dt)
command = ['python3', str(Path(__file__).with_name('run.py')), label,
           '--executable', str(root / 'build-particle-free/tip-step89/tested/aspect-release'),
           '--checker', str(a.checker.resolve() if a.checker else root / 'build-particle-free/tip-step89/tested/libbulk_tip_checks.release.so'),
           '--fixture', str(fixture), '--time-update', 'midpoint', '--ranks', str(a.ranks),
           '--dt', str(a.dt), '--end', str(a.end), '--phase-balance', '--integrity-stop',
           '--checkpoint-steps', str(a.checkpoint_steps), '--seconds', str(a.seconds)]
if a.wall_hours:
    command += ['--wall-hours', str(a.wall_hours)]
if a.resume_from:
    command += ['--resume-from', str(a.resume_from.resolve()), '--restart-dt', str(a.dt)]
if a.snapshot_id is not None:
    assert a.resume_from
    command += ['--snapshot-id', str(a.snapshot_id)]
if a.end_step is not None:
    command += ['--end-step', str(a.end_step)]
if not a.growth:
    command.append('--fixed-phase')
if a.defer_compression:
    # Measured coarse exports ~180 MB/frame; fine doubles cells. Leave room for checkpoints.
    estimate = (round(a.end/a.dt)+1)*(400_000_000 if a.mesh != 'coarse' else 200_000_000)+500_000_000
    assert shutil.disk_usage(root).free > estimate, 'Insufficient space for deferred lossless compression.'
processes = subprocess.check_output(['ps', '-eo', 'comm,args'], text=True).splitlines()
assert not [s for s in processes if s.split()[0] in ('aspect-release', 'mpirun') and 'input.prm' in s], 'An existing numerical job needs preserving.'
archive = []


def compress_closed(last_closed):
    """Only this fresh run's closed exports; checkpoints are never touched."""
    for path in sorted((run / 'output').glob('tip-*.csv')):
        m = re.match(r'tip-(?:points|balance)-(\d+)-r\d+\.csv$', path.name)
        step = int(m[1]) if m else 0 if path.name.startswith('tip-seed-fracture-r') else None
        if step is None or step > last_closed:
            continue
        target = path.with_suffix('.csv.gz')
        assert not target.exists()
        digest = hashlib.sha256()
        with path.open('rb') as source, gzip.open(target, 'wb', compresslevel=1) as destination:
            for block in iter(lambda: source.read(1048576), b''):
                digest.update(block)
                destination.write(block)
        check = hashlib.sha256()
        with gzip.open(target, 'rb') as source:
            for block in iter(lambda: source.read(1048576), b''):
                check.update(block)
        assert check.digest() == digest.digest()
        archive.append(dict(file=str(target.relative_to(root)), raw_sha256=digest.hexdigest(),
                            raw_bytes=path.stat().st_size, compressed_bytes=target.stat().st_size))
        path.unlink()  # Newly generated, closed file; all bytes verified in .gz.
        (evidence / (label + '-compression.json')).write_text(json.dumps(archive, indent=2) + '\n')


job = subprocess.Popen(command, env=env)
while job.poll() is None:
    log = run / 'run.log'
    if log.exists() and not a.defer_compression:
        steps = re.findall(r'\*\*\* Timestep (\d+):', log.read_text())
        if steps:
            compress_closed(int(steps[-1]) - 1)
    time.sleep(3)
compress_closed(10**9)
raise SystemExit(job.returncode)
