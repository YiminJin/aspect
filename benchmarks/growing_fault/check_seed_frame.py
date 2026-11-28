"""Check the growing seed, physical tip, and applied affine velocity in one frame."""
import ast
from dataclasses import dataclass
import json
import math
import operator
from pathlib import Path

import numpy as np

from parameter_file import composition_names, strip_material, resolved_function, subsection, wall_components
from compare import rows


@dataclass(frozen=True)
class SeedFrame:
    center: tuple
    degrees: float

    @property
    def tangent(self):
        angle = math.radians(self.degrees)
        return np.array([math.cos(angle), math.sin(angle)])

    def function_constants(self, output, model):
        original = resolved_function(output, model)['Function constants']['value']
        constants = dict((k.strip(),v.strip()) for k,v in (item.split('=') for item in original.split(',')))
        for name, value in zip(('xc','yc','tx','ty'), (*self.center,*self.tangent)):
            if name not in constants:
                raise ValueError(f'{model} lacks the named constant {name}')
            constants[name] = format(value, '.17g')
        return constants

    def settings(self, output):
        settings = ''
        for model in ('Initial composition model', 'Boundary velocity model'):
            constants = self.function_constants(output, model)
            entries = ', '.join(f'{name}={value}' for name,value in constants.items())
            settings += subsection(model, subsection('Function', 'set Function constants = ' + entries))
        length = float(self.function_constants(output,'Initial composition model')['length'])
        tip = self.center + length*self.tangent
        return settings + subsection('Material model', subsection('Phase field fault',
                     f'set Seed tangent angle = {self.degrees}\nset Forward seed tip = {tip[0]:.17g},{tip[1]:.17g}'))


def parsed_velocity(output, points):
    """Evaluate the two affine arithmetic expressions recorded by ASPECT, including the interior origin."""
    function = resolved_function(output,'Boundary velocity model')
    constants = dict(item.split('=') for item in function['Function constants']['value'].split(','))
    variables = {k.strip():float(v) for k,v in constants.items()}
    expressions = [ast.parse(expr, mode='eval').body for expr in wall_components(output)]
    binary = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul, ast.Div:operator.truediv}

    def value(node):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](value(node.left), value(node.right))
        raise ValueError(f'Unexpected operator in the resolved affine wall function: {ast.dump(node)}')

    velocities = []
    x_name, y_name, time_name = (n.strip() for n in function['Variable names']['value'].split(','))
    for x, y in points:
        variables.update({x_name:x, y_name:y, time_name:.0025})
        velocities.append([value(expr) for expr in expressions])
    return np.array(velocities)


def check_frame(output, frame):
    # Build the independent laboratory velocity gradient from the two seed-frame rates.
    x, y = frame.center
    t = frame.tangent
    rotation = np.array([[t[0],-t[1]],[t[1],t[0]]])
    gradient = rotation @ [[.003,.020],[0.,-.003]] @ rotation.T
    parameters = json.loads((output/'parameters.json').read_text())
    actual_tip = np.fromstring(strip_material(parameters)['Forward seed tip']['value'], sep=',')
    np.testing.assert_allclose(actual_tip, frame.center + .5*t, atol=2e-12, rtol=0)
    # At the first solve friction samples the straight accepted parent. Use its
    # physical pressure locations, even when the prescribed pressure is constant.
    sections = rows([output/'sections-0000.csv'], ('section',))
    stations = rows([output/'friction-pressure-0001.csv'], ('section',))
    if len(stations) < 2 or stations.keys() != sections.keys():
        raise ValueError('Incomplete initial material pressure stations')
    positions = np.array([[float(stations[k][d]) for d in ('x', 'y')] for k in sorted(stations)])
    along, normal = ((positions - frame.center) @ rotation).T
    if not np.isfinite(positions).all() or not np.all(np.diff(along) > 0):
        raise ValueError('Invalid or unordered material pressure stations')
    dx, dy = positions[-1] - positions[0]
    measured_angle = math.degrees(math.atan2(dy, dx))
    offset = max(abs(normal))
    if offset > 2e-12:
        raise ValueError(f'Material pressure stations lie at {measured_angle:.9g} degrees, '
                         f'expected {frame.degrees:g}; maximum normal offset {offset:.9g} m')
    mesh = rows(output.glob('mesh-0001-r*.csv'), ('cell','component','local'))
    initial = rows(output.glob('mesh-0000-r*.csv'), ('cell','component','local'))
    names, phase_name = composition_names(output)
    phase_index = 4 + names.index(phase_name)
    walls = {}
    phase_error = 0.
    # deal.II Q2 velocity local order: vertices, edge midpoints, center.
    sites = np.array([[0,0],[1,0],[0,1],[1,1],[0,.5],[1,.5],[.5,0],[.5,1],[.5,.5]])
    weights = np.array([1,1,1,1,4,4,4,4,16])/36
    for cell in {key[0] for key in mesh}:
        row = mesh[cell,0,0]
        lo, hi = (np.array([float(row[a]),float(row[b])]) for a,b in [('x0','y0'),('x1','y1')])
        pos = lo + sites*(hi-lo)
        local = (pos-frame.center)@rotation
        seed = .2*np.maximum(0,1-(np.maximum(0,local[:,0])/.5)**2)*np.maximum(0,1-(local[:,1]/.25)**2)
        # Q2 Simpson integration and the mean of the tensor Bernstein coefficients
        # give the same cell mean, including cells limited at initialization.
        coefficients = [float(initial[cell,phase_index,i]['value']) for i in range(9)]
        if not np.isfinite(coefficients).all():
            raise ValueError(f'Nonfinite initial phase in {cell}')
        phase_error = max(phase_error, abs(np.mean(coefficients)-weights@seed))
        for i,(px,py) in enumerate(pos):
            if px in (x-1,x+1) or py in (y-.5,y+.5):
                walls[px,py] = np.array([float(mesh[cell,d,i]['value']) for d in (0,1)])
    if not walls or not np.isfinite(list(walls.values())).all() or not np.isfinite(phase_error):
        raise ValueError('Missing or nonfinite measured seed/velocity')
    if phase_error > 4e-12:
        raise ValueError(f'Initial {phase_name} mean differs from the prescribed seed by {phase_error:g}')
    offsets = np.array([(0.,0.),(1.,0.),(0.,.5)])
    velocities = parsed_velocity(output, offsets+frame.center)
    parsed_error = np.max(abs(velocities-offsets@gradient.T),axis=0)
    wall_error = np.max([abs(value-gradient@(p-np.array(frame.center))) for p,value in walls.items()],axis=0)
    print(f'{output.parent.name}: {len(stations)} pressure stations at {measured_angle:.9g} degrees, '
          f'tip {actual_tip} m, max cell phase mean error {phase_error:.3g}; '
          f'parsed velocities {velocities.tolist()} m/s; applied wall errors {wall_error} m/s')
    if not np.isfinite(velocities).all() or max(wall_error)>2e-12 or max(parsed_error)>2e-12:
        raise ValueError(f'Applied Cartesian wall velocity error {wall_error} m/s; parsed error {parsed_error} m/s')


def check_geometry(run, mesh_plugin):
    observer = subsection('Postprocess','set List of postprocessors = midpoint publication, midpoint mesh fields')
    base = Path(__file__).with_name('growing_seed.prm').read_text()
    def measure(name, extra='', end='.005'):
        return run(name, extra+observer, source=base, extra_plugin=mesh_plugin, end=end)
    frame = SeedFrame((1.,.5), 30.)
    clean = measure('origin', end='.015')
    check_frame(clean, frame)
    shifted = SeedFrame((10001.,.5), 30.)
    box = subsection('Geometry model', subsection('Box', 'set Box origin X coordinate = 10000'))
    settings = box + shifted.settings(clean)
    translated = measure('translated', settings, end='.015')
    check_frame(translated, shifted)
    p,q = (rows(folder.glob('points-0000-r*.csv'),('cell','q')) for folder in (clean,translated))
    if p.keys()!=q.keys() or len(p)!=4968 or any(p[k]['section']!=q[k]['section'] for k in p):
        raise ValueError('Translated geometry changed quadrature identities or strip membership')
    position=max(max(abs(float(q[k]['x'])-10000-float(p[k]['x'])),abs(float(q[k]['y'])-float(p[k]['y']))) for k in p)
    if position>4.e-12: raise ValueError(f'Translated physical quadrature positions differ by {position} m')
    for step in range(4):
        a,b=(rows([folder/f'sections-{step:04}.csv'],('section',)) for folder in (clean,translated))
        if any(a[k]['xi']!=b[k]['xi'] or a[k]['formed']!=b[k]['formed'] or
               (step==0 and a[k]['ds']!=b[k]['ds']) for k in a):
            raise ValueError(f'Translated stations, widths or formation differ at clock {step}')
    print('Translated native points',len(p),'maximum physical position defect',position,'m')
    second = SeedFrame((1.,.5), 20.)
    check_frame(measure('second-dip', second.settings(clean)), second)
    stale_angle = second.settings(clean) + subsection(
        'Material model', subsection('Phase field fault', 'set Seed tangent angle = 19'))
    stale_stations = measure('stale-material-angle', stale_angle)
    try:
        check_frame(stale_stations, second)
    except ValueError as exc:
        if 'Material pressure stations lie at' not in str(exc):
            raise
        print(f'Stale material angle rejected: {exc}')
    else:
        raise ValueError('A stale material angle passed the pressure-station check')
    old_wall = resolved_function(clean,'Boundary velocity model')['Function constants']['value']
    stale = measure('stale-wall-origin', settings+subsection('Boundary velocity model',
                     subsection('Function','set Function constants = '+old_wall)))
    try:
        check_frame(stale, shifted)
    except ValueError as exc:
        if 'Applied Cartesian wall velocity error' not in str(exc):
            raise
        print(f'Stale wall origin rejected: {exc}')
    else:
        raise ValueError('A stale wall origin passed the translated-velocity check')
