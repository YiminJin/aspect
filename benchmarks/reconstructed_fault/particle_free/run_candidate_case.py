#!/usr/bin/env python3
"""Run one bounded candidate-construction diagnostic; never overwrite a label."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--construction', choices=['shift', 'cls'], default='shift')
    p.add_argument('--detector', choices=['unlimited', 'kxrcf', 'fu_shu'], default='fu_shu')
    p.add_argument('--case', choices=['signed_jump', 'finite_width', 'pulse_translation'], default='signed_jump')
    p.add_argument('--divisor', type=int, choices=[1, 2, 4], default=1)
    p.add_argument('--mesh', type=int, choices=[4, 16, 32], default=32)
    p.add_argument('--local', action='store_true')
    p.add_argument('--no-trace', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    inputs = root/'build-particle-free/weno_cls/inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    target = inputs/(args.label+'.prm')
    assert not target.exists(), 'Refusing to overwrite '+str(target)
    construction = 'constrained least squares' if args.construction == 'cls' else 'mean shift'
    target.write_text(f'''# Controlled candidate-construction comparison, serial prescribed transport.
include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/particle_free/transport_{args.mesh}.prm
subsection Postprocess
 subsection PF4 stabilization checks
  set Time divisor = {args.divisor}
  set Candidate construction = {construction}
  set Transport case = {args.case}
  set Limiter path = {args.detector}
  set Trace reconstruction = {'false' if args.no_trace else 'true'}
  set Local only = {'true' if args.local else 'false'}
  set Local candidate checks = {'true' if args.local else 'false'}
 end
end
''')
    command = [sys.executable, str(root/'benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py'),
               args.label, '--input', str(target), '--plugin', str(root/'build-particle-free/plugin/libstabilization_checks.release.so'),
               '--postprocessor', 'PF4 stabilization checks', '--seconds', '180']
    raise SystemExit(subprocess.run(command, cwd=root, check=False).returncode)


if __name__ == '__main__':
    main()
