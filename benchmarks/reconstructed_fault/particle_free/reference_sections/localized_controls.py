"""Run short two-rank localized-selection controls in separate native output trees."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from fixture import fixture, vector

root = Path(__file__).resolve().parents[4]
work = root/'build-particle-free/localized-sections'
runs = root/'build-particle-free/bulk-tip/runs'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('label')
parser.add_argument('--executable',type=Path,default=Path('/tmp/aspect-reference-sections-native/aspect-release'))
parser.add_argument('--checker',type=Path,default=work/'checks/libbulk_tip_checks.release.so')
parser.add_argument('--degrees', type=float, default=0)
parser.add_argument('--rotate-mesh', action='store_true')
parser.add_argument('--single-tip', action='store_true', help='Extend the formed seed to the incoming boundary with its original forward taper')
parser.add_argument('--nx', type=int, default=16)
parser.add_argument('--step', type=int, default=3)
parser.add_argument('--radius', type=float, default=2.)
parser.add_argument('--coulomb', action='store_true')
parser.add_argument('--strip-subdivisions', type=int, default=1)
parser.add_argument('--exact', action='store_true')
parser.add_argument('--global-score', action='store_true')
parser.add_argument('--probes', action='store_true')
parser.add_argument('--front-check', action='store_true')
parser.add_argument('--quiescent', action='store_true')
parser.add_argument('--remote-check', action='store_true')
parser.add_argument('--detached', action='store_true')
parser.add_argument('--save-seed', action='store_true')
parser.add_argument('--match-seed')
parser.add_argument('--parent')
parser.add_argument('--snapshot', type=int)
parser.add_argument('--prescribe')
args = parser.parse_args()
mangle = args.degrees if args.rotate_mesh else 0.
c, s = vector(args.degrees)
m, n = vector(mangle)
tip = (m-.5*n+.5*c, n+.5*m+.5*s)
text = fixture(mangle, args.degrees, args.rotate_mesh, .003, args.single_tip)
text = text.replace('X repetitions = 8', f'X repetitions = {args.nx}').replace('Y repetitions = 4', f'Y repetitions = {args.nx//2}')
if args.coulomb:
    text = text.replace('closure = material strips', 'closure = material strips Coulomb formation')
if args.quiescent:
    base, boundary = text.split('subsection Boundary velocity model',1)
    lines = boundary.splitlines()
    index = next(i for i,line in enumerate(lines) if 'set Function expression =' in line)
    lines[index] = '  set Function expression = 0;0'
    text = base+'subsection Boundary velocity model'+'\n'.join(lines)+'\n'
if args.detached:
    lines = text.splitlines()
    index = next(i for i,line in enumerate(lines) if 'Function expression = 0;0;0;' in line)
    lines[index] += '+1e-15*max(0,1-((x-1.94)/.05)^2)*max(0,1-((y-.5)/.1)^2)'
    text = '\n'.join(lines)+'\n'
if args.prescribe:
    assert args.step==1 or args.parent, 'Each diagnostic replay requires its own parent.'
    text += '\nsubsection Termination criteria\n set Checkpoint on termination = false\nend\n'
text += f"""
subsection Material model
 subsection Phase field fault
  set Localize prospective tip direction = {str(not args.global_score).lower()}
  set Initial direction selection tip = {tip[0]:.17g},{tip[1]:.17g}
  set Direction selection radius in phase lengths = {args.radius}
  set Bulk tip oblique integration = {'exact lines' if args.exact else 'local strips'}
 end
end
"""
if args.strip_subdivisions != 1:
    if args.exact or args.strip_subdivisions < 1:
        parser.error("Strip subdivisions require local strips and a positive integer.")
    text = text.replace('set Bulk tip oblique integration = local strips',
                        f'set Bulk tip oblique integration = local strips\n  set Bulk tip strip subdivisions = {args.strip_subdivisions}')
name = 'localized-'+args.label
prm = work/(name+'.prm')
work.mkdir(parents=True,exist_ok=True)
with prm.open('x') as f:
    f.write(text)
cmd = [sys.executable, Path(__file__).parents[1]/'bulk_tip/run.py', name,
       '--executable', args.executable,
       '--checker', args.checker, '--fixture', prm,
       '--dt', '.005', '--end', '.05', '--end-step', args.step, '--ranks', 2, '--bind-cores',
       '--checkpoint-steps', 1000000 if args.prescribe else 1, '--seconds', 360, '--time-update', 'midpoint']
env = os.environ.copy()
def output(label):
    return runs/('localized-'+label)/'output'
if args.parent:
    cmd += ['--resume-from',output(args.parent),'--copy-selected-snapshot-only']
    if args.snapshot: cmd += ['--snapshot-id',args.snapshot]
if args.prescribe:
    env['ASPECT_TIP_PRESCRIBED_REPLAY'] = str(output(args.prescribe)/f'tip-planes-{args.step:04}.csv')
if args.probes:
    cmd += ['--rejection','--rejection-step',2,'--tangent','--tangent-step',1]
if args.match_seed:
    env['ASPECT_SECTION_SEED_FROM'] = str(output(args.match_seed))
if args.remote_check:
    env['ASPECT_TIP_REMOTE_DIRECTION_CHECK'] = '1'
if args.front_check:
    env['ASPECT_TIP_FRONT_CHECK'] = '1'
if args.save_seed:
    env['ASPECT_SECTION_SAVE_SEED'] = '1'
sys.exit(subprocess.run(list(map(str,cmd)),env=env).returncode)
