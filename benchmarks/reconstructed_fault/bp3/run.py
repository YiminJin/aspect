"""One bounded BP3 Release run; preserve all attempts and record provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parameter', type=Path)
    parser.add_argument('--cap', type=float, default=900)
    parser.add_argument('--ranks', type=int, choices=(1,2), default=1)
    parser.add_argument('--no-normal-diagnostic', action='store_true',
                        help='For the inherited adiabatic-pressure unit fixture only')
    args = parser.parse_args()
    parameter = args.parameter.resolve()
    log = parameter.with_suffix('.log')
    if log.exists():
        raise SystemExit('Preserve the existing attempt; select a new wrapper.')
    binary = ROOT/'build-pf-cpdi/aspect-release'
    plugin = HERE/'build/libbp3.release.so'
    paths = [binary, plugin, parameter, HERE/'bp3.cc', HERE/'bp3_model.h',
             HERE/'bp3_smoke.prm', HERE/'bp3_pilot.prm', HERE/'fault.txt']
    if (HERE/'build/libbp3_coupling_checks.release.so').exists():
        paths += [HERE/'build/libbp3_coupling_checks.release.so',
                  ROOT/'tests/phase_field_fault_surface_system.cc']
    info = dict(command=[str(binary), str(parameter)], cap_seconds=args.cap,
                hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    if args.ranks != 1:
        info['command'] = ['mpirun', '-np', str(args.ranks)] + info['command']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', ASPECT_FAULT_PERFORMANCE='1',
               ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC='1',
               ASPECT_FAULT_NONLINEAR_DIAGNOSTIC='1')
    if args.no_normal_diagnostic:
        env.pop('ASPECT_FAULT_NORMAL_STRESS_DIAGNOSTIC', None)
    start = time.monotonic()
    with log.open('x') as out:
        try:
            status = subprocess.run(info['command'], cwd=ROOT, env=env, stdout=out,
                                    stderr=subprocess.STDOUT, timeout=args.cap).returncode
        except subprocess.TimeoutExpired:
            status = 124
    info.update(status=status, wall_seconds=time.monotonic()-start,
                peak_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    parameter.with_suffix('.resources.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info), flush=True)
    raise SystemExit(status)
