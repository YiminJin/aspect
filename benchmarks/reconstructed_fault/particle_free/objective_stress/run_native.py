#!/usr/bin/env python3
"""One bounded native spin/lifecycle fixture; no production continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'benchmarks/reconstructed_fault/particle_free'))
from run_pf6_mature import flatten


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--spin', choices=['legacy', 'Jaumann lagged Lie'], default='Jaumann lagged Lie')
    parser.add_argument('--ranks', type=int, choices=[1, 2, 4], default=2)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seconds', type=int, default=240)
    parser.add_argument('--restart', type=Path)
    parser.add_argument('--reject', action='store_true')
    parser.add_argument('--schedule', default='200000')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--baseline', action='store_true', help='Protected pre-change binary and its matching original runtime plugin')
    args = parser.parse_args()
    build = ROOT / 'build-particle-free/objective-stress'
    out = build / 'runs' / args.label
    out.mkdir(parents=True, exist_ok=False)
    binary = ROOT / 'build-particle-free/native/aspect-release'
    plugins = [build / 'bp3-plugin/libbp3_restore_150x50.release.so',
               build / 'checks/libobjective_stress_checks.release.so']
    if args.baseline:
        assert args.spin == 'legacy' and not args.restart and not args.reject and not args.audit
        binary = build / 'baseline/0-aspect-release'
        plugins = [ROOT / 'build-particle-free/bp3-optimization/plugins/native/libbp3_restore_150x50.release.so']
    text = flatten(ROOT / 'benchmarks/reconstructed_fault/bp3_native/section7b-common.prm', ROOT)
    # The isolated bottom-test parameter is absent in the ordinary runtime plugin.
    lines, skip = [], 0
    for line in text.splitlines():
        if 'subsection BP3 serial coupled checks' in line:
            skip = 1
            continue
        if skip:
            skip += int(line.strip().startswith('subsection '))-int(line.strip()=='end')
            continue
        if 'set Local state disturbance' not in line and 'set List of postprocessors' not in line:
            lines.append(line)
    text = '\n'.join(lines)
    text += f'''
set Additional shared libraries = {', '.join(map(str, plugins))}
set Output directory = {out / 'output'}
set End time = 2000000
subsection Material model
 subsection Phase field fault
  set Bulk history representation = particle stress
  set Particle stress strain convention = three dimensional deviator
  {'' if args.baseline else 'set Stress rotation = ' + args.spin}
 end
end
subsection Mesh refinement
 set Initial global refinement = 5
end
subsection Postprocess
 set List of postprocessors = reconstructed fault BP3, particles, BP3 output complete, BP3 restored monitor{'' if args.baseline else ', objective stress checks'}
 subsection BP3
  set Last accepted step = {args.steps}
  set Profile time interval = 1
  set Audit full state every step = true
  set Graceful wall seconds = 1000000
 end
end
subsection Time stepping
 set List of model names = function{', objective rejection' if args.reject else ''}
 subsection Function
  set Variable names = time
  set Function expression = {args.schedule}
 end
end
subsection Termination criteria
 set Checkpoint on termination = true
end
'''
    if args.restart:
        shutil.copytree(args.restart.resolve(), out / 'output/restart')
        text += '\nset Resume computation = true\n'
    path = out / 'input.prm'; path.write_text(text)
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH='/opt/openmpi/5.0.6/lib', OMP_NUM_THREADS='1', DEAL_II_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', OMP_PROC_BIND='false', OMPI_MCA_mpi_yield_when_idle='1')
    if args.audit:
        env['ASPECT_K1_FLOOR_AUDIT'] = '1'
    command = ['/opt/openmpi/5.0.6/bin/mpirun', '-n', str(args.ranks), str(binary), str(path)]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = dict(arguments=vars(args) | {'restart': str(args.restart)}, command=command,
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  binary_sha256=sha(binary), plugin_sha256={str(p): sha(p) for p in plugins}, input_sha256=sha(path))
    start = time.monotonic()
    with (out / 'run.log').open('w') as stream:
        try:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=args.seconds, check=False).returncode
        except subprocess.TimeoutExpired:
            code = 124
    record.update(exit_code=code, seconds=time.monotonic()-start)
    (out / 'command.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(label=args.label, exit_code=code, seconds=record['seconds'])))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
