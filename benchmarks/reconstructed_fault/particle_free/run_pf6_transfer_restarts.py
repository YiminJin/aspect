#!/usr/bin/env python3
"""Bounded same-build 1/2/4-rank checkpoint/restart matrix, sequential jobs.

Use through refactoring_boundary/run_logged.py on hosts requiring MPI sockets.
Each native child has a 180-second process-group timeout. Existing outputs are
never overwritten. The final cross-rank case redistributes a serial checkpoint.
"""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--prefix', default='pf6-transfer-checked')
p.add_argument('--resume', action='store_true', help='Reuse only matching completed cases; never overwrite a native run.')
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
runner = Path(__file__).with_name('run_pf6_transfer.py')
logger = root/'benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py'
assessor = Path(__file__).with_name('assess_pf6_transfer.py')
evidence = root/'build-particle-free/transfer-qualification'


def run_case(label, ranks, end=.25, restart=None):
    out = root/'build-particle-free/runs'/label
    if a.resume and out.exists():
        r = json.loads((out/'record.json').read_text())
        assert r.get('exit_code') == 0 and r['ranks'] == ranks
        assert r['case']['end'] == end and r['case']['restart'] == (str(restart) if restart else None)
        assert r['executable_sha256'] == hashlib.sha256((root/'build-particle-free/native/aspect-release').read_bytes()).hexdigest()
        assert r['plugin_sha256'] == hashlib.sha256((root/'build-particle-free/plugin/libpf6_transfer_checks.release.so').read_bytes()).hexdigest()
        print('Reusing completed matching case '+label, flush=True)
        return out
    args = [sys.executable, str(runner), 'prepare', label, '--ranks', str(ranks),
            '--end', str(end), '--checkpoint']
    if restart:
        args += ['--restart', str(restart)]
    subprocess.run(args, cwd=root, check=True, stdout=subprocess.DEVNULL)
    out = root/'build-particle-free/runs'/label
    command = json.loads((out/'command.json').read_text())
    subprocess.run([sys.executable, str(logger), label, '190', 'timeout', '--signal=TERM',
                    '--kill-after=5s', '180s']+command, cwd=root, check=True)
    subprocess.run([sys.executable, str(runner), 'finalize', label], cwd=root, check=True)
    return out


for ranks in [1, 2, 4]:
    baseline = root/'build-particle-free/runs'/f'{a.prefix}-r{ranks}'
    if not (baseline/'run.log').exists():
        subprocess.run([sys.executable, str(runner), 'finalize', baseline.name], cwd=root, check=True)
    assert json.loads((baseline/'record.json').read_text()).get('exit_code') == 0

for ranks in [1, 2, 4]:
    half = run_case(f'{a.prefix}-half-r{ranks}', ranks, .125)
    resumed = run_case(f'{a.prefix}-restart-r{ranks}', ranks, restart=half/'run/restart')
    subprocess.run([sys.executable, str(assessor),
                    str(root/'build-particle-free/runs'/f'{a.prefix}-r{ranks}'),
                    '--compare', str(resumed), '--output', str(evidence/f'checked-restart-r{ranks}.json')],
                   cwd=root, check=True)
serial = root/'build-particle-free/runs'/f'{a.prefix}-half-r1'
cross = run_case(f'{a.prefix}-cross-1-to-4', 4, restart=serial/'run/restart')
subprocess.run([sys.executable, str(assessor),
                str(root/'build-particle-free/runs'/f'{a.prefix}-r4'),
                '--compare', str(cross), '--output', str(evidence/'checked-cross-1-to-4.json')],
               cwd=root, check=True)
print('Same-rank 1/2/4 and cross-rank 1->4 restart comparisons passed.', flush=True)
