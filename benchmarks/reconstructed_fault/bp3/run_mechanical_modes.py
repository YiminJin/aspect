"""Fresh, saved-clock prefix followed by a noncommitting response probe.

No server checkpoint is transplanted across incompatible deal.II builds.
This driver preserves evidence and does not retry a failure or timeout.
"""
import argparse
import csv
import hashlib
import json
import os
import resource
import shlex
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def parameters(text):
    result, sections = {}, []
    for line in text.splitlines():
        line = line.split('#')[0].strip()
        if line.startswith('subsection '):
            sections.append(line[11:])
        elif line == 'end':
            sections.pop()
        elif line.startswith('set '):
            key, value = line[4:].split('=', 1)
            result[tuple(sections + [key.strip()])] = value.strip()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--probe-step', type=int, default=11)
    parser.add_argument('--probe-newton', type=int, default=6)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.run_prepared:
        record = json.loads((out/'mechanical_launch.json').read_text())
        assert not (out/'probe.log').exists(), 'Refuse to overwrite/retry evidence'
        for name, digest in record['hashes'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        env = {k: v for k, v in os.environ.items() if not k.startswith('ASPECT_')}
        env.update(record['environment'])
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
        env['LD_BIND_NOW'] = '1'
        command = ['timeout', str(record['hard_wall_cap_s'])] + record['command']
        start = time.monotonic()
        with (out/'probe.log').open('w') as log:
            result = subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
        passed = 'MECHANICAL MODES VERIFIED' in (out/'probe.log').read_text()
        execution = dict(returncode=result.returncode, seconds=time.monotonic()-start,
                         expected_noncommitting_stop=passed,
                         peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        (out/'probe_execution.json').write_text(json.dumps(execution, indent=2)+'\n')
        print(execution)
        assert passed, 'Preserved failure; no automatic retry'
        return
    subprocess.run(['python3', str(HERE/'run_long.py'), '--output', str(out),
                    '--purpose', 'recurrence', '--end-years', '40', '--wall-hours', '0.5',
                    '--velocity-preconditioner', 'amg', '--verify-through-step', '11'], check=True)
    plugin = REPO/'benchmarks/reconstructed_fault/performance/build-gmg/libfault_mechanical_modes.release.so'
    rows = {}
    with (HERE/'first_long_run/output/accepted_steps.csv').open() as stream:
        for row in csv.DictReader(stream):
            k = int(row['step'])
            if k <= 12:
                rows[k] = row
    with (out/'clock.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'time', 'dt'])
        writer.writerows((k, rows[k]['time'], rows[k]['dt']) for k in sorted(rows))
    prm = out/'fresh.prm'
    text = prm.read_text()
    # Record all differences from the resolved server input before overrides.
    local = parameters(text)
    server = parameters((HERE/'first_long_run/output/parameters.prm').read_text())
    differences = { '/'.join(key): [server.get(key), value]
                    for key, value in local.items() if server.get(key) != value }
    (out/'resolved_parameter_differences.json').write_text(json.dumps(differences, indent=2)+'\n')
    text += f'''
set Additional shared libraries = {HERE/'build/libbp3.release.so'}, {plugin}
subsection Time stepping
  set List of model names = convection time step, reconstructed fault time step, mechanical probe clock
end
subsection Postprocess
  subsection BP3
    set Profile time interval = 1
  end
end
subsection Checkpointing
  set Steps between checkpoint = 10
  set Time between checkpoint = 0
end
'''
    prm.write_text(text)
    record = json.loads((out/'launch.json').read_text())
    record['environment']['ASPECT_BP3_TIMESTEP_SEQUENCE'] = str(out/'clock.csv')
    record['environment']['ASPECT_MECHANICAL_PROBE_STEP'] = str(args.probe_step)
    record['environment']['ASPECT_MECHANICAL_PROBE_NEWTON'] = str(args.probe_newton)
    record['hashes'][str(plugin)] = hashlib.sha256(plugin.read_bytes()).hexdigest()
    record['hashes'][str(prm)] = hashlib.sha256(prm.read_bytes()).hexdigest()
    record['hard_wall_cap_s'] = 1800
    (out/'mechanical_launch.json').write_text(json.dumps(record, indent=2)+'\n')
    (out/'launch.sh').write_text('#!/bin/sh\nexec python3 '+shlex.quote(str(Path(__file__).resolve()))
                                +' --output '+shlex.quote(str(out))+' --run-prepared\n')
    print('Prepared only; inspect resolved_parameter_differences.json before --run-prepared.')


if __name__ == '__main__':
    main()
