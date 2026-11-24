"""Run the 4C review reproducers using ordinary ASPECT inputs in fresh directories."""
import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from check import Frame, basis, gauss, read, require, same

here = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('aspect', 'plugin', 'destination'):
        parser.add_argument(arg, type=lambda x: Path(x).resolve())
    args = parser.parse_args()
    root = args.destination
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ | dict(ASPECT_SOURCE_DIR=str(here.parents[1]), DEAL_II_NUM_THREADS='1',
                            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def run(name, parameters, diagnostic=None):
        directory = root/name
        directory.mkdir()
        out = directory/'output'
        prm = directory/'input.prm'
        prm.write_text(parameters + f'\nset Additional shared libraries = {args.plugin}\nset Output directory = {out}\n')
        with (directory/'log').open('w') as log:
            result = subprocess.run([args.aspect, prm], stdout=log, stderr=subprocess.STDOUT, env=env)
        if diagnostic:
            require(result.returncode != 0 and diagnostic in (directory/'log').read_text(), name)
        else:
            result.check_returncode()
        return out

    def fault(key, val):
        return f'\nsubsection Material model\n subsection Phase field fault\n  set {key} = {val}\n end\nend\n'

    base = (here/'layered.prm').read_text()
    principal = run('principal', base)
    gate = run('traction-step', (here/'traction_step.prm').read_text())
    log = (gate.parent/'log').read_text().split('Timestep 1:')[-1]
    changes = [float(x) for x in re.findall('Relative change in returned fault stress projection: (\\S+)',log)]
    require(changes and changes[0]>.4 and 'Coupled Stokes unknowns require another Picard iteration' in log,
            'The traction step failed to veto the first returned projection')
    residuals = [float(x) for x in re.findall(r'after nonlinear iteration \d+: (\S+)',log)]
    require(residuals[0]<.001 and len(residuals)>=2,'The Stokes-only stopping test must pass first')
    frame = Frame(gate,1)
    cell, actual = (gate/'load-1.txt').read_text().split()
    data = frame.rows[frame.rows['cell']==cell]
    w = .25*np.outer(gauss(3)[1],gauss(3)[1]).ravel()
    k = -data['eta']*np.expm1(-.125/data['eta'])
    shear = 1.4+data['x']
    implicit = w@(2*k*(data['y']*(data['Lxx']-(data['Lxx']+data['Lyy'])/3) +
                       .5*shear*(data['Lxy']+data['Lyx'])))
    published = -w@(data['y']*data['tauxx']+shear*data['tauxy'])
    work_error = same(float(actual)-implicit, published, .001, 'traction-step weak work at publication')
    print(f'traction step: Stokes {residuals[0]}, returned change {changes[0]}, '
          f'{len(residuals)} assemblies, final cell work error {work_error:.3g}', flush=True)

    horizontal = (here/'translated_horizontal.prm').read_text()
    for name, width in [('full',0.), ('finite',2.1)]:
        out = run(name,horizontal+fault('Profile half width',width))
        labels = read(out/'material-1-surface-0.csv')
        same(labels['y'], .0125, 2e-8, 'translated y')
        d = 50*.35*1.35/.65**2
        same(labels['I_m']/d,4.,1e-9,'entire in-box normal length')
        same(labels['width_m'],width,2e-8,'transport of finite width or full-ray convention')
        print(f'{name} translated y {labels["y"]}, normal length {labels["I_m"]/d}, '
              f'width {labels["width_m"]}',flush=True)
    run('cropped',horizontal+fault('Profile half width',2.),'omitted fraction 0.003125')

    # For a deforming finite strip compute F^{-T} n independently from the
    # two one-sided Q2 velocity gradients at the seed's horizontal cell face.
    out = run('deformed',horizontal.replace('0;1\n','-.1*x;1+.1*y\n')+fault('Profile half width',2.1))
    before, after = (read(out/f'material-{step}-surface-0.csv') for step in (0,1))
    bulk = read(out/'material-0-rank0.csv')
    for j,x in enumerate(before['x']):
        gradients = []
        for cell in np.unique(bulk['cell']):
            rows = np.sort(bulk[bulk['cell']==cell],order='q')
            center = np.array([rows['x'].mean(),rows['y'].mean()])
            unit = (np.array([x,0.])-center)/.5+.5
            if np.all(unit>=-1e-12) and np.all(unit<=1+1e-12):
                weights = np.outer(basis(unit[1]),basis(unit[0])).ravel()
                gradients.append([weights@rows[k] for k in ('Lxx','Lxy','Lyx','Lyy')])
        require(len(gradients)==2,'Missing one-sided velocity gradient')
        F = np.eye(2)+.0125*np.mean(gradients,axis=0).reshape(2,2)
        expected = 2.1/np.linalg.norm(np.linalg.solve(F.T,[0,1]))
        same(after['width_m'][j],expected,2e-12,'deformation of normal radius')
        require(abs(expected-2.1)>1e-4,'The material width must measurably deform')
    print('deformed widths',after['width_m'],flush=True)

    frozen = run('frozen',base+fault('Fault geometry','fixed interior'))
    initial, final = Frame(frozen,0), Frame(frozen,2)
    same(initial.X,final.X,0.,'frozen vertices')
    initial_load = lambda path: float((path/'load-0.txt').read_text().split()[1])
    same(initial_load(frozen),initial_load(principal),1e-14,'zero-duration weak action')
    same(initial.rows['offset_xy'],Frame(principal,0).rows['offset_xy'],1e-12,'initial projected shear')
    # A uniform polynomial gives the same bulk iterate in both nodal and Bernstein bases.
    static = horizontal.replace('include $ASPECT_SOURCE_DIR/benchmarks/layered_moving_fault/layered.prm',base)
    static += fault('Fault geometry','fixed interior')
    with_transport = run('constant-frozen',static)
    no_transport = static.replace('set SSPRK3 fields = phase_field','set SSPRK3 fields =') \
                         .replace('static, static, static, field','static, static, static, static') \
                         .replace('single Advection, iterated Stokes','no Advection, iterated Stokes')
    without_transport = run('static-frozen',no_transport)
    same(initial_load(with_transport),initial_load(without_transport),1e-12,'independent mechanical selection')
    run('invalid-selection', no_transport+fault('Fault geometry','advected interior'),
        'requires bounded SSPRK3 transport')
    print('initial assembled weak action, fixed and advected',initial_load(frozen),initial_load(principal),
          'constant, selected and static',initial_load(with_transport),initial_load(without_transport),flush=True)

    # Swap phase and xx, including their roles, methods and nonuniform initial data.
    ordered = base.replace('fault_stress_xx, fault_stress_yy, fault_stress_xy, phase_field',
                           'phase_field, fault_stress_yy, fault_stress_xy, fault_stress_xx')
    ordered = ordered.replace('stress, stress, stress, phase field','phase field, stress, stress, stress')
    ordered = ordered.replace('static, static, static, field','field, static, static, static')
    ordered, count = re.subn(r'0;0;0;(\.35\*[^\n]+)',r'\1;0;0;0',ordered)
    require(count==1,'Missing nonuniform initial data')
    permuted = run('permuted',ordered)
    worst = 0.
    for step in (0,1,2):
        a,b = Frame(principal,step),Frame(permuted,step)
        np.testing.assert_array_equal(a.rows['cell'],b.rows['cell'])
        for key in a.rows.dtype.names:
            if key!='cell':
                worst = max(worst,same(a.rows[key],b.rows[key],1e-10,key))
        for key in a.surface.dtype.names:
            worst = max(worst,same(a.surface[key],b.surface[key],1e-10,key))
    print(f'permuted named bulk and surface columns, maximum difference {worst:.4g}')


if __name__ == '__main__':
    main()
