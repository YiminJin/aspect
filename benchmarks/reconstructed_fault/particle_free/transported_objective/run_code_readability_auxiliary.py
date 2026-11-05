#!/usr/bin/env python3
"""Run retained stationary, serial-support and non-fault controls sequentially."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/code_readability'
BUILD = ROOT/'build-particle-free/code-readability'


def main():
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/opt/openmpi/5.0.6/lib'
    for key in ('OMP_NUM_THREADS', 'DEAL_II_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'
    commands = {'stationary': [sys.executable, str(Path(__file__).with_name('run_stage_c_native.py')),
                               'readability-stationary', '--ranks', '2', '--steps', '1',
                               '--refinement', '3', '--seconds', '120']}
    for case in ('limiter', 'stabilization', 'nonfault-after'):
        commands[case] = ['/opt/openmpi/5.0.6/bin/mpirun', '-n', '1',
                          str(ROOT/'build-particle-free/native/aspect-release'),
                          str(BUILD/case/'input.prm')]
    for name, command in commands.items():
        log = OUT/('aux-'+name+'.log')
        assert not log.exists(), 'Preserve completed evidence.'
        start = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=180)
        record = dict(command=command, exit_code=result.returncode,
                      seconds=time.monotonic()-start,
                      core_sha256=hashlib.sha256((ROOT/'build-particle-free/native/aspect-release').read_bytes()).hexdigest())
        (OUT/('aux-'+name+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
