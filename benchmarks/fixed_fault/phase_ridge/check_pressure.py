"""Measure normal compression at the actual Q8 section rows of a dipping trace."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import shutil

import numpy as np


def section(name, contents):
    return f"\nsubsection {name}\n{contents}\nend\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aspect', 'plugin', 'destination'):
        parser.add_argument(name, type=lambda s: Path(s).resolve())
    parser.add_argument('--ranks', choices=[1, 2], type=int, default=1)
    parser.add_argument('--mpiexec', default='mpirun')
    parser.add_argument('--serial', type=Path)
    parser.add_argument('--lifecycle', action='store_true')
    args = parser.parse_args()
    args.destination.mkdir(parents=True)
    base = Path(__file__).resolve().parents[1].joinpath('fixed_fault.prm').read_text()
    adiabatic = section('Adiabatic conditions model', 'set Model name = function\n' + section(
        'Function', 'set Variable names = depth\nset Function expression = 293;200+50*depth;1000'))
    material = section('Phase field fault', '''set Background compression =
set Background shear traction = 128
set Fault geometry = fixed interior
set Fault vertices = .2,.2; .35,.35; .5,.5; .65,.65; .8,.8
set Fit trace to phase = false
set Direct effect = .1
set Radiation damping = 100''')
    common = (base + section('Mesh refinement', 'set Initial global refinement = 2') + section('Material model', 'set Model name = observed ridge\n' + material)
              + section('Compositional fields', 'set Types of fields = stress, stress, stress, phase field')
              + section('Postprocess', 'set List of postprocessors = fault friction pressure samples')
              + f'\nset Additional shared libraries = {args.plugin}\nset End time = .0125\n')
    selected = section('Material model', section('Phase field fault', 'set Friction normal stress = adiabatic pressure'))
    closed = (section('Boundary traction model', 'set Prescribed traction boundary indicators =')
              + section('Boundary velocity model', 'set Prescribed velocity boundary indicators = left:function, right:function, bottom:function, top:function')
              + '\nset Pressure normalization = volume\n')
    moving = (closed + section('Boundary velocity model', section('Function', 'set Function expression = 0;1'))
              + '\nset Nonlinear solver scheme = single Advection, iterated Stokes\nset Nonlinear solver tolerance = 1e-8\n'
              + section('Compositional fields', 'set Types of fields = stress, stress, stress, phase field\n'
                        'set Compositional field methods = static, static, static, explicit DG SSPRK3')
              + section('Boundary composition model', 'set Fixed composition boundary indicators = left,right,bottom,top\n'
                        'set List of model names = function' + section('Function','set Function expression = 0;0;0;.35'))
              + section('Material model', section('Phase field fault','set Fault geometry = advected interior\nset Advect physical stress = true')))
    hydrostatic = (closed + section('Gravity model', section('Vertical', 'set Magnitude = .05'))
                   + section('Material model', section('Simple model', 'set Reference density = 1000\nset Thermal expansion coefficient = 0'))
                   + '\nset Surface pressure = 200\nset Nonlinear solver tolerance = 1e-8\n')
    resolved = section('Material model', section('Phase field fault','set Background compression = 225'))
    def case(name, settings, resume=None, diagnostic=None):
        run = args.destination / name
        run.mkdir()
        if resume:
            shutil.copytree(resume/'output/restart/02', run/'output/restart/02')
            settings += '\nset Resume computation = true\n'+section('Checkpointing','set Resume checkpoint = 2')
        (run/'input.prm').write_text(common + settings + '\nset Output directory = output\n')
        launch = [] if args.ranks == 1 else [args.mpiexec, '-np', str(args.ranks)]
        with (run/'log').open('w') as log:
            proc = subprocess.run([*launch, args.aspect, 'input.prm'], cwd=run, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DEAL_II_NUM_THREADS='1'))
        if diagnostic:
            if not proc.returncode or diagnostic not in ' '.join((run/'log').read_text().split()):
                raise ValueError(f'{name}: expected {diagnostic}')
            print(name,'diagnosed',diagnostic)
        else:
            proc.check_returncode()
        return run
    if args.lifecycle:
        save = '\nset End time = .025\n'+section('Checkpointing','set Steps between checkpoint = 1\nset Number of checkpoints to keep = 4')
        clean = case('clean', selected+adiabatic+save)
        resumed = case('resume', selected+adiabatic+save, resume=clean)
        for f in (clean/'output').glob('trace-pressure-2-*.csv'):
            if f.read_bytes() != (resumed/'output'/f.name).read_bytes():
                raise ValueError('Mature resumed projection differs')
        if not list((resumed/'output').glob('trace-pressure-2-*.csv')):
            raise ValueError('Missing resumed step')
        case('slope',selected+adiabatic.replace('50*depth','49*depth')+save, resume=clean,
             diagnostic='pressure source or profile differs from the checkpoint')
        case('closure',resolved+save,resume=clean,diagnostic='pressure source or profile differs from the checkpoint')
        case('negative',selected+adiabatic.replace('200+50*depth','if(depth<.4,-1,200)'),
             diagnostic='Invalid adiabatic fault compression -1 Pa at trace position')
        case('conflict',selected+adiabatic+resolved,diagnostic='omit Background compression')
        return
    for name, settings in [('affine', selected+adiabatic), ('resolved', resolved), ('computed', selected+hydrostatic),
                           ('moving', selected+adiabatic+moving)]:
        case(name,settings)
        for step in (0, 1):
            def records(root):
                table = {}
                files=sorted((root/name/'output').glob(f'trace-pressure-{step}-r*.csv'))
                if root==args.destination and len(files)!=args.ranks:
                    raise ValueError('Missing rank output')
                for path in files:
                    count=len(table)
                    with path.open() as f:
                        for row in csv.DictReader(f):
                            key = row.pop('cell'), int(row.pop('q'))
                            if key in table:
                                raise ValueError(f'duplicate {key}')
                            table[key] = {k: float(v) for k, v in row.items()}
                    if len(table)==count:
                        raise ValueError(f'No owned section contribution in {path}')
                if not table:
                    raise ValueError('missing section rows')
                return table
            data = records(args.destination)
            if args.serial:
                serial = records(args.serial)
                if data.keys() != serial.keys():
                    raise ValueError('Different owned Q8 rows')
                for k, v in data.items():
                    np.testing.assert_allclose(list(v.values()),list(serial[k].values()),rtol=1e-10,atol=1e-10)
            # The first physical interval uses the 10 s accepted parent at every
            # material vertex. Each Q1 virtual slip multiplies the *point* RSF.
            residual = np.zeros(5)
            scale = np.zeros(5)
            for row in data.values():
                speed = row['V']
                friction = row['N']*.1*math.asinh(speed/.002*math.exp((.6+.015*math.log(10*.001/.008))/.1))+100*speed
                defect = row['shear']-row['stiffness']*row['chi']*speed-friction
                left = int(row['left'])
                w = row['volume']*row['chi']*np.array([1-row['xi'],row['xi']])
                residual[left:left+2] += w*defect
                scale[left:left+2] += w*abs(row['shear'])
            relative = max(abs(residual)/scale)
            if not relative<1e-8:
                raise ValueError(f'{name}: published rates violate Q1 section balance by {relative}')
            print(name,step,'independent Q1 residual',relative)
            if name == 'resolved':
                for r in data.values():
                    np.testing.assert_allclose([r['delta_pressure'],r['delta_normal']], [1,-1],atol=1e-12)
                continue
            errors = []
            groups = {}
            for r in data.values():
                # Orthogonal projection on the physical material segment, independent of FE transfer.
                a = np.array([r['ax'],r['ay']])
                b = np.array([r['bx'],r['by']])
                x = np.array([r['x'],r['y']])
                foot = a + np.clip((x-a)@(b-a)/sum((b-a)**2),0.,1.)*(b-a)
                np.testing.assert_allclose(foot,[r['foot_x'],r['foot_y']],atol=2e-13)
                errors.append(abs(r['N']-(200+50*(1-foot[1]))))
                np.testing.assert_equal([r['delta_pressure'],r['delta_normal']], [0,0])
                groups.setdefault(round(foot[1],11), []).append(r)
            if max(errors)>1e-9:
                raise ValueError(f'{name} pressure error {max(errors)}')
            if name=='moving' and step and min(r['ay'] for r in data.values())<.21:
                raise ValueError('The material seed did not rise')
            if name=='affine' and step:
                pair=next(g for g in groups.values() if max(r['y'] for r in g)-min(r['y'] for r in g)>.1)
                print('Same station, distinct bulk depths:', [(1-r['y'], r['N']) for r in sorted(pair,key=lambda r:r['y'])[::len(pair)-1]])
            print(name,step,len(data),'Q8 rows, maximum pressure error',max(errors),'Pa')
        print(name,'passed', flush=True)


if __name__ == '__main__':
    main()
