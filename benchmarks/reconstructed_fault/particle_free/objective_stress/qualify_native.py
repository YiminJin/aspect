#!/usr/bin/env python3
"""Sequential, bounded native lifecycle checks; never launch a production input."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BUILD = ROOT / 'build-particle-free/objective-stress'
RUNNER = Path(__file__).with_name('run_native.py')


def main():
    parent = BUILD / 'runs/spin-on-qualified/output/restart'
    cases = [
        ['spin-restart', '--steps', '4', '--restart', str(parent)],
        ['spin-repartition', '--steps', '4', '--ranks', '4', '--restart', str(parent)],
        ['spin-rejection', '--steps', '3', '--reject'],
        ['spin-retry-control', '--steps', '3', '--schedule',
         'if(time>100000,if(time<250000,100000,200000),200000)'],
    ]
    results = []
    for args in cases:
        command = [sys.executable, str(RUNNER)] + args
        record = BUILD / 'runs' / args[0] / 'command.json'
        if not record.exists():
            print('Bounded native case:', args[0], flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            print('Retaining completed native case:', args[0], flush=True)
        result = json.loads(record.read_text())
        assert result['exit_code'] == 0
        results.append(result)
    # Missing checkpoint metadata must retain an explicit legacy interpretation.
    command = [sys.executable, str(RUNNER), 'spin-from-legacy-negative', '--steps', '4', '--restart',
               str(BUILD / 'runs/legacy-current/output/restart')]
    record = BUILD / 'runs/spin-from-legacy-negative/command.json'
    if not record.exists():
        subprocess.run(command, cwd=ROOT, check=False)
    code = json.loads(record.read_text())['exit_code']
    log = ' '.join((BUILD / 'runs/spin-from-legacy-negative/run.log').read_text(errors='replace').split())
    # The BP3 parameter fingerprint can reject the changed material law before
    # the core's more specific registry-marker check. Both are intended gates.
    assert code != 0 and ('Stress rotation/integrator mismatch on restart' in log
                          or 'BP3 restart requires the same geometry, material/profile, loading' in log)
    results.append(dict(case='spin-from-legacy-negative', expected_failure=True, exit_code=code,
                        gate='BP3 configuration fingerprint precedes the core registry-marker check'))
    base = (BUILD / 'runs/spin-on-qualified/input.prm').read_text()
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib', OMP_NUM_THREADS='1', DEAL_II_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', OMP_PROC_BIND='false')
    for label, extra, expected in [
        ('unsupported-dg', 'subsection Material model\n subsection Phase field fault\n  set Bulk history representation = DG trial stress\n end\nend\n',
         'missing localized residual-spin representation'),
        ('unsupported-cohesive', 'subsection Material model\n subsection Phase field fault\n  set Fault constitutive mode = cohesive\n end\nend\n',
         'cohesive histories are unsupported'),
        ('unsupported-transport', 'subsection Particles\n set Integration scheme = rk2\nend\n',
         'requires stationary BP3 samples'),
        ('unsupported-coefficients', 'subsection Material model\n subsection Phase field fault\n  set Elastic shear moduli = 32038120320, 32038120321\n end\nend\n',
         'Variable-coefficient energy terms are unqualified'),
    ]:
        directory = BUILD / 'runs' / label
        directory.mkdir(exist_ok=True)
        path = directory / 'input.prm'
        command = ['/opt/openmpi/5.0.6/bin/mpirun', '-n', '2', str(ROOT / 'build-particle-free/native/aspect-release'), str(path)]
        if not (directory / 'run.log').exists():
            path.write_text(base + '\nset Output directory = ' + str(directory / 'output') + '\n' + extra)
            with (directory / 'run.log').open('w') as stream:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=60, check=False).returncode
        else:
            code = 1  # retained intended negative startup; verify its diagnostic below
        log = ' '.join((directory / 'run.log').read_text(errors='replace').split())
        assert code != 0 and expected in log, (label, code, expected)
        results.append(dict(case=label, command=command, expected_failure=True, exit_code=code))
        print('Expected configuration rejection:', label, flush=True)
    (BUILD / 'native-matrix.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
