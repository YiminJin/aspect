"""Bounded installed pf-rsf probes for a surface-reaching, right-dipping thrust."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path, help='Assignment D installation with benchmark plugins')
    parser.add_argument('destination', type=Path, help='new local directory')
    args = parser.parse_args()
    root, prefix = args.destination.resolve(), args.prefix.resolve()
    root.mkdir(parents=True, exist_ok=False)
    aspect = prefix/'bin/aspect-release'
    examples = prefix/'benchmarks/fixed_fault'
    ridge, probe = examples/'phase_ridge', examples/'transfer_work'
    env = dict(os.environ, ASPECT_SOURCE_DIR=str(prefix), DEAL_II_NUM_THREADS='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/'fresh.log').open('w') as log:
        subprocess.run([sys.executable, '-B', ridge/'run.py', aspect, ridge/'libphase_ridge.release.so',
                        root/'free-32', 'free', '32'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (root/'reference.log').open('w') as log:
        subprocess.run([sys.executable, '-BO', '-c',
                        'from pathlib import Path; from check import check_case; check_case(Path.cwd(),"free",32)'],
                       cwd=root, env=dict(env, PYTHONPATH=str(ridge)), stdout=log, stderr=subprocess.STDOUT, check=True)

    controls = {
        'surface': ('subsection Material model\n subsection Phase field fault\n'
                    '  set Fault geometry = fixed interior\n  set Fit trace to phase = false\n end\nend',
                    'A seed vertex is outside its supported box region'),
        'friction': ('subsection Material model\n subsection Phase field fault\n'
                     '  set Direct effect = 0.010, 0.025\n end\nend', 'does not match the given pattern')
    }
    for name, (settings, diagnostic) in controls.items():
        case = root/name
        case.mkdir()
        prm = case/'input.prm'
        prm.write_text(f"include {root/'free-32/input.prm'}\nset Output directory = {case/'output'}\n{settings}\n")
        log = case/'log'
        with log.open('w') as stream:
            result = subprocess.run([aspect, prm], cwd=case, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode == 0 or diagnostic not in ' '.join(log.read_text().split()):
            raise ValueError(f'{name}: expected {diagnostic!r}, see {log}')
        print(f'{name}: rejected ({diagnostic})', flush=True)

    # The physical-sense test also continues a newly checkpointed, right-dipping
    # boundary-spanning trace. Its pressure and work controls use a frozen parent.
    subprocess.run([sys.executable, '-B', probe/'check_sense.py', aspect,
                    probe/'libtransfer_work.release.so', root/'slip-sense', '--checkpoint'],
                   cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
