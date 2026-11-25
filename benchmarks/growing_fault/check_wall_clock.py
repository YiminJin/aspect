"""Compare midpoint wall displacement, native stress and work for a ramp and constant controls."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from check_repairs import subsection, function
from compare import continuation
sys.path.insert(0, str(Path(__file__).resolve().parent/'timestep'))
from audit import ORDER, shapes, tensor
from frames import read


class Frame:
    def __init__(self, folder, step):
        tag = f'{step:04}'
        self.mesh = read(folder.glob(f'mesh-{tag}-r*.csv'), ('cell','component','local'))
        self.point = read(folder.glob(f'points-{tag}-r*.csv'), ('cell','q'))
        self.book = {k:v.item() for k,v in read([folder/f'balance-{tag}.csv']).items()} if step else {}
        self.time = self.book.get('time',0.)



def compute_wall_action(output, step, factor):
    old, new = (Frame(output, i) for i in (step-1, step))
    dt = new.time-old.time
    mesh, points = new.mesh, new.point
    np.testing.assert_array_equal(old.point['cell'], points['cell'])
    np.testing.assert_array_equal(old.point['q'], points['q'])
    stress = (tensor(old.point, 'tau') + tensor(points, 'tau'))/2
    nodes = np.array([(i%3, i//3) for i in ORDER])*.5
    tangent = np.array([np.sqrt(3)/2, .5])
    normal = np.array([-.5, np.sqrt(3)/2])
    if not all(np.isfinite(array).all() for data in (mesh,new.point,old.point)
               for name,array in data.items() if name!='cell'):
        raise ValueError('Nonfinite accepted field in wall work check')
    work = displacement_error = 0.
    for cell in np.unique(mesh['cell']):
        m = {k:v[mesh['cell']==cell] for k,v in mesh.items()}
        q = points['cell']==cell
        lo, hi = (np.array([m[k][0] for k in axes]) for axes in [('x0','y0'),('x1','y1')])
        pos = lo + nodes*(hi-lo)
        on_wall = np.any((pos==[0.,0.]) | (pos==[2.,1.]), axis=1)
        if not on_wall.any():
            continue
        s, n = (pos-[1.,.5])@tangent, (pos-[1.,.5])@normal
        lift = factor*(np.outer(.003*s+.02*n,tangent)-np.outer(.003*n,normal))
        velocity = np.stack([m['value'][m['component']==d] for d in (0,1)],axis=1)
        displacement_error = max(displacement_error, dt/2*np.max(abs(velocity[on_wall]-lift[on_wall])))
        lift[~on_wall] = 0.
        x, y = ((np.stack([points['x'][q],points['y'][q]],axis=1)-lo)/(hi-lo)).T
        _, dx, dy = shapes(x,y,hi-lo,True)
        pressure = np.stack([(1-x)*(1-y),x*(1-y),(1-x)*y,x*y],axis=1)@m['value'][m['component']==2]
        xx, yy, _, xy = stress[q].T
        work += dt * (points['weight'][q] @ ((xx-pressure)*(dx@lift[:,0]) +
                       (yy-pressure)*(dy@lift[:,1]) + xy*(dx@lift[:,1]+dy@lift[:,0])))
    return new, displacement_error, abs(work-new.book['Wext'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aspect','plugin','destination'):
        parser.add_argument(name,type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    root = args.destination
    root.mkdir(parents=True,exist_ok=False)
    base = Path(__file__).with_name('onecut.prm').read_text()
    wall = base.split('subsection Boundary velocity model')[1].split('set Function expression = ')[1].splitlines()[0].split(';')
    observer = args.plugin.parent/'timestep/libmidpoint_fields.release.so'
    env = os.environ | dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DEAL_II_NUM_THREADS='1')
    def run(name, factor, extra='', steps=1, diagnostic=None):
        d = root/name
        d.mkdir()
        expr = ';'.join(f'({u})*({factor})' for u in wall)
        prm = base + subsection('Boundary velocity model',function(expr)) + extra
        prm += f'\nset Output directory = {d/"output"}\nset Additional shared libraries = {args.plugin}, {observer}\n'
        prm += subsection('Postprocess','set List of postprocessors = midpoint publication, midpoint mesh fields')
        prm += subsection('Termination criteria',f'set Termination criteria = end step\nset End step = {steps}')
        (d/'input.prm').write_text(prm)
        with (d/'log').open('w') as log:
            result = subprocess.run([args.aspect,d/'input.prm'],env=env,stdout=log,stderr=subprocess.STDOUT)
        if diagnostic:
            if result.returncode==0 or diagnostic not in ' '.join((d/'log').read_text().split()):
                raise ValueError(f'{name}: missing diagnostic {diagnostic}')
        else:
            result.check_returncode()
        return d/'output'
    outputs = {name:run(name, factor) for name,factor in [('full','1'),('half','0.5'),('ramp','min(t/.005,1)')]}
    year = 60*60*24*365.2425
    duration = .005/year
    settings = (f'\nset Use years instead of seconds = true\nset Maximum first time step = {duration:.17g}'
                f'\nset Maximum time step = {duration:.17g}\nset End time = {2*duration:.17g}\n')
    outputs['years'] = run('years',f'{year:.17g}*min(t/{duration:.17g},1)',settings)
    # The time manager rejects a *solved* 10 ms attempt at step two,
    # cuts it to 5 ms, then repeats that duration from the same parent.
    stepper = subsection('Time stepping','set List of model names = retry growing fault')
    for name, extra in [('clean',''),('cut',stepper+'\nset Maximum time step = .01\n')]:
        outputs[name] = run(name,'min(t/.015,1)',extra+'\nset End time = .05\n',steps=2)
    if (outputs['cut'].parent/'log').read_text().count('Repeating the current time step')!=2:
        raise ValueError('Expected two solved and rejected attempts')
    continuation(outputs['clean'],outputs['cut'],[1,2])
    for name,step,factor in [('full',1,1.),('half',1,.5),('ramp',1,.5),('years',1,.5),('clean',2,.5),('cut',2,.5)]:
        frame, dx, dw = compute_wall_action(outputs[name],step,factor)
        print(f'{name}: t={frame.time} s, Ebulk={frame.book["bulk"]:.13g} J/m, '
              f'Wext={frame.book["Wext"]:.13g} J/m, displacement error={dx:.3g} m, work error={dw:.3g} J/m',flush=True)
        if abs(frame.time-.005*step)>1e-16 or not np.isfinite(dw) or dx>1e-17 or dw>1e-12:
            raise ValueError(f'{name}: prescribed displacement or independent wall work failed')
    continuation(outputs['half'],outputs['ramp'],[1])
    full, half = (Frame(outputs[k],1) for k in ('full','half'))
    if full.book['bulk'] < 3.9*half.book['bulk'] or np.max(abs(full.point['tauxy']-half.point['tauxy']))<1.:
        raise ValueError('The full load must give a distinguishable stress and bulk energy')
    run('incomplete-mask','1',subsection('Boundary velocity model',
        'set Prescribed velocity boundary indicators = left x:function, right:function, top:function, bottom:function'),
        diagnostic='requires fully prescribed wall velocity')
    run('unsupported','1',subsection('Boundary velocity model',
        'set Prescribed velocity boundary indicators = left:ascii data, right:function, top:function, bottom:function'),
        diagnostic="cannot evaluate wall velocity plugin 'ascii data' at the attempted midpoint")


if __name__=='__main__':
    main()
