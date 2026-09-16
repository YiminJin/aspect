"""One non-retrying Release comparison, with elapsed time, peak RSS and provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parameter', type=Path)
    parser.add_argument('--cap', type=float, default=300)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--env', action='append', default=[])
    parser.add_argument('--no-performance', action='store_true',
                        help='Do not enable opt-in profiling for a production-like verification run.')
    args = parser.parse_args()
    parameter = args.parameter.resolve()
    log = parameter.with_suffix('.log')
    if log.exists():
        raise SystemExit('Preserve existing evidence: choose a new wrapper, no retry.')
    binary = ROOT/'build-pf-cpdi/aspect-release'
    command = [str(binary), str(parameter)]
    if args.ranks > 1:
        command = ['mpirun', '-np', str(args.ranks)] + command
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if not args.no_performance:
        env['ASPECT_FAULT_PERFORMANCE']='1'
    else:
        env.pop('ASPECT_FAULT_PERFORMANCE',None)
    env.update(value.split('=', 1) for value in args.env)
    record = dict(command=command, cap_seconds=args.cap, overrides=args.env,
                  profiling=not args.no_performance,
                  sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (binary, parameter, ROOT/'source/material_model/phase_field_fault.cc')})
    start = time.monotonic()
    with log.open('x') as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=args.cap)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            status = 124
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(status=status, wall_seconds=time.monotonic()-start,
                  peak_rss_KiB=usage.ru_maxrss, user_seconds=usage.ru_utime,
                  system_seconds=usage.ru_stime)
    parameter.with_suffix('.resources.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    raise SystemExit(status)
