#!/usr/bin/env python3
"""Run one selected bounded PF6a case with explicit fixed-particle provenance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--divisor', type=int, choices=[1, 2, 4], default=1)
    p.add_argument('--normal-cells', type=int, choices=[32, 64], default=32)
    p.add_argument('--quadrature', type=int, choices=[6, 12], default=6)
    p.add_argument('--end-time', type=float, choices=[.25, .53125, .5625], default=.5625)
    p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--seconds', type=int, choices=[180, 240], default=180)
    p.add_argument('--checkpoint', action='store_true')
    p.add_argument('--restart-from', type=Path)
    p.add_argument('--schedule', choices=['constant', 'cutback', 'matched'], default='constant',
                   help='Use the native cutback plugin or its accepted-schedule control.')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    target = root/'build-particle-free/pf6a/inputs'/(a.label+'.prm')
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), 'Refusing to overwrite input'
    dt = .03125/a.divisor
    assert a.schedule == 'constant' or (a.divisor == 1 and not a.restart_from)
    models = 'function, repeat on cutback' if a.schedule == 'cutback' else 'function'
    expression = str(dt) if a.schedule == 'constant' else (
        'if(time<0.5,0.03125,0.015625)' if a.schedule == 'cutback' else
        'if(time<0.46875,0.03125,0.015625)')
    text = f'''include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_free/pf6a-common.prm
set End time = {a.end_time}
set Resume computation = {'true' if a.restart_from else 'false'}
set Maximum time step = {dt}
set Maximum first time step = {dt}
subsection Geometry model
 subsection Box
  set Y repetitions = {a.normal_cells//2}
 end
end
subsection Time stepping
 set List of model names = {models}
 subsection Repeat on cutback
  set Relative repeat threshold = 0.75
  set Cut back amount = 0.5
 end
 subsection Function
  set Function expression = {expression}
 end
end
subsection Material model
 subsection Phase field fault
  set FE bulk normal quadrature points = {a.quadrature}
 end
end
subsection Boundary velocity model
 subsection Function
  set Function expression = 0.256*max(0,t-{.5*dt})*(y-0.5); 0
 end
end
subsection Postprocess
 subsection PF6a checks
  set Operator and rejection checks = {'true' if a.diagnostics else 'false'}
 end
end
subsection Termination criteria
 set Checkpoint on termination = {'true' if a.checkpoint else 'false'}
end
'''
    if a.schedule != 'constant':
        text += '''subsection Boundary velocity model
 set Prescribed velocity boundary indicators = left y:PF6a interval ramp, right y:PF6a interval ramp, bottom:PF6a interval ramp, top:PF6a interval ramp
end
'''
    target.write_text(text)
    command = [sys.executable, str(root/'benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py'),
               a.label, '--input', str(target), '--plugin', str(root/'build-particle-free/plugin/libpf6a_checks.release.so'),
               '--postprocessor', 'PF6a cohesive endpoint', '--seconds', str(a.seconds)]
    if a.restart_from:
        command += ['--restart-from', str(a.restart_from.resolve())]
    env = dict(os.environ, ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', DEAL_II_NUM_THREADS='1')
    code = subprocess.run(command, cwd=root, env=env, check=False).returncode
    record = root/'build-particle-free/runs'/a.label/'record.json'
    if record.exists():
        r = json.loads(record.read_text())
        r.update(PF6a_diagnostic_environment={'ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION': '1'},
                 PF6a_case={'normal_cells': a.normal_cells, 'divisor': a.divisor,
                           'quadrature': a.quadrature, 'diagnostics': a.diagnostics,
                           'schedule': a.schedule,
                           'end_time': a.end_time, 'restart_from': str(a.restart_from) if a.restart_from else None})
        record.write_text(json.dumps(r, indent=2)+'\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
