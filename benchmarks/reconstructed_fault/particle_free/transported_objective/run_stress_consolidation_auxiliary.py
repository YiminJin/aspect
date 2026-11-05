#!/usr/bin/env python3
"""Short stationary and ordinary non-fault dispatch checks; fresh outputs only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/stress_consolidation_guide'
BUILD = ROOT/'build-particle-free/stress-consolidation-guide'


def main():
    nonfault = BUILD/'nonfault'
    nonfault.mkdir(exist_ok=False)
    old = ROOT/'build-particle-free/code-readability/nonfault-after'
    # Explicit retained comparator dependency for this regression, not a guide
    # example. The material guide's complete input generator is independent.
    text = (old/'input.prm').read_text().replace(str(old), str(nonfault))
    (nonfault/'input.prm').write_text(text)
    commands = {
        'stationary': [sys.executable, str(Path(__file__).with_name('run_stage_c_native.py')),
                       'consolidation-stationary', '--ranks', '2', '--steps', '1',
                       '--refinement', '3', '--seconds', '120'],
        'nonfault': ['/opt/openmpi/5.0.6/bin/mpirun', '-n', '1',
                     str(ROOT/'build-particle-free/native/aspect-release'), str(nonfault/'input.prm')]}
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib', OMP_NUM_THREADS='1',
               DEAL_II_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for name, command in commands.items():
        start = time.monotonic()
        with (OUT/('aux-'+name+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=180)
        (OUT/('aux-'+name+'.json')).write_text(json.dumps(dict(command=command,
            exit_code=result.returncode, seconds=time.monotonic()-start), indent=2)+'\n')
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
