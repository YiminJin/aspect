"""Qualify 3-D composition transport through ordinary ASPECT inputs and accepted outputs."""
import argparse
from copy import copy
from itertools import product
from pathlib import Path
import shutil

import numpy as np

from check import compare, field_data, fields, fn, read_output, require, run, section, setting
from reference import basis, bern, nodes

REPORT = section('Postprocess', setting('List of postprocessors', 'DG coefficient samples'))
SLABS = Path(__file__).with_name('slabs.prm').read_text() + REPORT
TO_VALUES = np.kron(bern, np.kron(bern, bern))
TO_COEFF = np.linalg.inv(TO_VALUES)
SIMPSON = np.array([1., 4., 1.]) / 6
MEAN = np.kron(SIMPSON, np.kron(SIMPSON, SIMPSON))
Z, Y, X = np.meshgrid(nodes, nodes, nodes, indexing='ij')


def simulate(args, name, settings='', **options):
    return run(args, name, settings, template=SLABS, dim=3, **options)


def nodal(data, step, field):
    boxes, coefficients, owners = field_data(data, step, field)
    expected_basis = 'FE_DGQBernstein<3>(2)' if field == 'damage' else 'FE_DGQ<3>(2)'
    for (_, name), (_, fe, _, _) in data[step][1].items():
        if name == field:
            require(fe == expected_basis, f'{name} has basis {fe}')
    if field == 'damage':
        coefficients = coefficients @ TO_VALUES.T
    return boxes, coefficients.reshape(-1, 3, 3, 3), owners


# Interpolate the polynomial at the child's Lobatto points and invert its
# evaluation matrix. This enclosure is independent of the production split.
child_eval = [basis((nodes + shift) / 2) for shift in (0, 1)]
children = [TO_COEFF @ np.kron(z, np.kron(y, x)) @ TO_VALUES
            for z, y, x in product(child_eval, repeat=3)]


def hull(coeff, depth=4):
    lo, hi = coeff.min(), coeff.max()
    if depth == 0 or (lo >= 0 and hi <= 1):
        return lo, hi
    bounds = [hull(matrix @ coeff, depth - 1) for matrix in children]
    return min(b[0] for b in bounds), max(b[1] for b in bounds)


def limit(values):
    """Scale in physical nodal coordinates and independently check the mapped mean."""
    flat = values.reshape(-1)
    lo, hi = hull(TO_COEFF @ flat)
    mean = MEAN @ flat
    require(0 <= mean <= 1, 'inadmissible reference mean')
    scale = min(1., mean / (mean - lo) if lo < 0 else 1.,
                (1 - mean) / (hi - mean) if hi > 1 else 1.)
    result = mean + scale * (values - mean)
    mean_error = abs(MEAN @ result.ravel() - mean)
    lower, upper = hull(TO_COEFF @ result.ravel())
    require(lower >= -3e-14 and upper <= 1 + 3e-14 and mean_error < 3e-14,
            'whole-cell enclosure or scaled mean failed')
    return result, scale < 1 - 1e-12, mean_error


def columns(parent, boxes, inflow, h, *, velocity=lambda z: 1., transpose=False,
            bounded=False, intermediate=True, start=0.):
    """Strong P2 z derivative and exact incoming jump lift at nine physical (x,y) points."""
    transfers, scales, mean_errors = [], [], []

    def derivative(state, time):
        action = np.empty_like(state)
        incoming = inflow(X[0], Y[0], time)
        for k, (values, box) in enumerate(zip(state, boxes)):
            z0, z1 = box[2], box[5]
            bottom, middle, top = values
            # Reference-coordinate P2 derivative on each of the three z planes.
            grad = np.stack((-3*bottom + 4*middle - top, top - bottom,
                             bottom - 4*middle + 3*top))
            speed = velocity(z0 + Z * (z1-z0))
            jump = velocity(z0) * (incoming - bottom)
            action[k] = (-speed * grad + np.array([9., -1.5, 3.])[:, None, None]*jump) / (z1-z0)
            # The deliberately transposed upstream trace exchanges x and y.
            incoming = top.T if transpose else top
            if k == 0 and len(boxes) == 2:
                transfers.append(float(velocity(z1) * (SIMPSON @ top @ SIMPSON)))
        return action

    state = parent
    for fraction, clock in [(1., start), (.25, start+h), (2/3, start+h/2)]:
        state = parent + fraction*(state - parent + h*derivative(state, clock))
        if bounded and (intermediate or fraction == 2/3):
            for k, cell in enumerate(state):
                state[k], scaled, error = limit(cell)
                scales.append(scaled)
                mean_errors.append(error)
    flux = h * np.dot([1/6, 1/6, 2/3], transfers) if transfers else 0.
    return state, flux, sum(scales), max(mean_errors, default=0.)


def profile(initial, boundary):
    def three(expr):
        return f'({expr})-.5; {expr}; ({expr})-.5'
    return fn('Initial composition model', three(initial)) + fn('Boundary composition model', three(boundary))


def slabs(args):
    serial = simulate(args, 'slabs')
    parallel = simulate(args, 'slabs-mpi', ranks=2)
    require(serial[1][0] == (.01, .01) and len(serial) == 2, 'wrong step or extra endpoint')
    defect = compare(serial, parallel, 1e-12)
    for name, offset, expected in [('damage', 0., .18748775), ('tracer', -.5, -.06251225)]:
        boxes, _, owners = nodal(parallel, 0, name)
        require(boxes == [(0.,0.,0.,1.,1.,.5), (0.,0.,.5,1.,1.,1.)] and set(owners)=={0,1},
                'expected two adjoining z slabs with distinct MPI owners')
        phi = np.array([.25 + np.minimum(b[2]+.5*Z, .5)/4 + offset for b in boxes])
        reference, transfer, _, _ = columns(phi, boxes, lambda x,y,t: (1-t)/4 + offset, .01)
        _, observed, _ = nodal(parallel, 1, name)
        err = np.max(abs(reference-observed))
        mass = .5 * (MEAN @ observed[1].ravel())
        # With a frozen incoming parent the constant receiver cannot move.
        frozen_mass = .5*(MEAN @ phi[1].ravel())
        require(abs(mass-frozen_mass)>1.2e-5, 'missing stage-local partition transfer')
        require(abs(transfer) > .001 and err < 1e-12 and abs(mass-expected) < 1e-12,
                f'{name} upper mass {mass}, column defect {err}, face integral {transfer}')
        print(f'{name}: owners {owners}, upper integral {mass:.17g}, absolute error {abs(mass-expected):.3g}, '
              f'column defect {err:.3g}, stage-weighted partition transfer {transfer:.12g}', flush=True)

    default = simulate(args, 'implicit', fields(SSPRK3_fields=''))
    companion = field_data(serial, 1, 'companion')[1]
    implicit_error = np.max(abs(companion-field_data(default, 1, 'companion')[1]))
    separation = np.max(abs(companion-field_data(serial, 1, 'tracer')[1]))
    require(implicit_error < 1e-12 and separation > 1e-6, 'implicit dispatch failed to separate')
    print(f'one/two rank max coefficient difference {defect:.3g} (162 per clock); '
          f'implicit companion difference {implicit_error:.3g}, explicit/implicit separation {separation:.6g}', flush=True)


def individual_fields(args):
    together = read_output(args.directory/'slabs/output',1,3)
    default = read_output(args.directory/'implicit/output',1,3)
    for selected in ('damage','tracer'):
        single = simulate(args,'only-'+selected,fields(SSPRK3_fields=selected))
        error = 0.
        for step, (clock, rows) in single.items():
            require(clock==together[step][0] and rows.keys()==together[step][1].keys(),'field selection changed the frame')
            for key, (box, fe, values, owner) in rows.items():
                reference = together if key[1]==selected else default
                ref_box, ref_fe, expected, ref_owner = reference[step][1][key]
                require((box,fe,owner)==(ref_box,ref_fe,ref_owner),'field selection changed the wrong basis or owner')
                error = max(error,float(np.max(abs(values-expected))))
        require(error<1e-12,'independent field selection changed another trajectory')
        print(f'{selected} selected alone: max coefficient difference against per-field controls {error:.3g}',flush=True)


def solved_velocity(args):
    # Uniform translation is the discrete solution with constant Dirichlet
    # velocity on every wall and zero body force. Transport uses that accepted
    # solve from t=0 on its first physical step.
    walls = ', '.join(face+':function' for face in ('left','right','front','back','bottom','top'))
    settings = (setting('Nonlinear solver scheme','single Advection, single Stokes')
                +section('Prescribed Stokes solution', setting('Model name','unspecified'))
                +section('Boundary velocity model',setting('Prescribed velocity boundary indicators',walls)
                         +section('Function',setting('Function expression','0;0;1')))
                +section('Gravity model',section('Vertical',setting('Magnitude',0)))
                +section('Material model',section('Simple model',setting('Viscosity',1)))
                +section('Solver parameters',section('Stokes solver parameters',setting('Stokes solver type','block AMG'))))
    data = simulate(args,'solved-stokes',settings)
    prescribed = read_output(args.directory/'slabs/output',1,3)
    err = compare(data,prescribed,1e-12)
    print(f'solved Stokes velocity: all coefficients agree with prescribed translation to {err:.3g}',flush=True)


def asymmetric(args):
    a = '.30+.04*x+.025*y^2+.03*x*y'
    b = '(.14+.075*x-.035*y+.05*x*y)'
    def value(x, y, z):
        return .30+.04*x+.025*y*y+.03*x*y + (.14+.075*x-.035*y+.05*x*y)*z
    settings = profile(a+'+z*'+b, a+'+(z-t)*'+b)
    data = simulate(args, 'asymmetric', settings)
    mpi = simulate(args, 'asymmetric-mpi', settings, ranks=2)
    mpi_error = compare(data, mpi, 1e-12)
    years = simulate(args,'asymmetric-years',settings + setting('Use years instead of seconds','true'))
    require(abs(years[1][0][0]-.01*365.2425*86400)<1e-9, 'unexpected internal clock in seconds')
    for name, shift in [('damage',0.), ('tracer',-.5)]:
        boxes, observed, _ = nodal(data, 1, name)
        exact = np.array([value(X, Y, box[2]+.5*Z-.01)+shift for box in boxes])
        initial = np.array([value(X, Y, box[2]+.5*Z)+shift for box in boxes])
        inflow = lambda x,y,t: value(x, y, -t)+shift
        transposed = columns(initial, boxes, inflow, .01, transpose=True)[0]
        frozen = columns(initial, boxes, lambda x,y,t: inflow(x,y,0.), .01)[0]
        err = np.max(abs(observed-exact))
        year_error = np.max(abs(nodal(years,1,name)[1]-exact))
        face_error = np.max(abs(transposed[1]-observed[1]))
        clock_error = np.max(abs(frozen[0]-observed[0]))
        require(max(err, year_error) < 1e-12 and face_error > 1e-3 and clock_error > 1e-5, 'face or clock insensitive')
        print(f'asymmetric {name}: analytic defect {err:.3g}, transposed face {face_error:.6g}, '
              f'frozen inflow clock {clock_error:.6g}, year-unit defect {year_error:.3g}, MPI defect {mpi_error:.3g}', flush=True)


def diagonal(args):
    # Both quadratic factors are invariant along this velocity. The exact
    # material translation is therefore affine in time even at stage clocks.
    polynomial = '.3+.02*x+.045*y+.06*z+.025*(x+2*y)*(z+2*x)'
    settings = (profile(polynomial, polynomial+'+.06125*t')
                +section('Geometry model',section('Box',setting('X repetitions',2)+setting('Y repetitions',2)))
                +section('Boundary composition model',setting('Fixed composition boundary indicators',
                                                              'left, right, front, back, bottom, top'))
                +fn('Prescribed Stokes solution','.5;-.25;-1','Velocity function'))
    data = simulate(args,'diagonal',settings)
    for field, shift in [('damage',0.), ('tracer',-.5)]:
        boxes, measured, _ = nodal(data, 1, field)
        require(len(boxes)==8, 'expected interior faces in all three directions')
        exact = []
        for x0,y0,z0,x1,y1,z1 in boxes:
            x, y, z = x0+(x1-x0)*X, y0+(y1-y0)*Y, z0+(z1-z0)*Z
            exact.append(.3+.02*x+.045*y+.06*z+.025*(x+2*y)*(z+2*x)+.06125*.01+shift)
        err = np.max(abs(measured-exact))
        require(err<1e-12,'diagonal material translation')
        print(f'diagonal {field}: max analytic defect on eight cells {err:.3g}',flush=True)


def limiting(args):
    cube = section('Geometry model',section('Box',setting('Z repetitions',1)))
    stationary = fn('Prescribed Stokes solution','0;0;0','Velocity function')
    for name, peak, weights in [('admissible',.9,(.4,.2,.15)), ('overshoot',1.1,(.3,.2,.1))]:
        wx, wy, wz = weights
        expr = f'{peak}-{wx}*(2*x-1)^2-{wy}*(2*y-1)^2-{wz}*(2*z-1)^2'
        data = simulate(args, name, cube+stationary+profile(expr,'.5'))
        original = peak-wx*(2*X-1)**2-wy*(2*Y-1)**2-wz*(2*Z-1)**2
        expected, scaled, _ = limit(original)
        for k in data:
            coeff = field_data(data,k,'damage')[1][0]
            observed = TO_VALUES @ coeff
            lo, hi = hull(coeff)
            err = np.max(abs(observed-expected.ravel()))
            mean_error = abs(MEAN @ observed - (peak-sum(weights)/3))
            require(err<1e-12 and mean_error<1e-13 and lo>=-1e-13 and hi<=1+1e-13,
                    f'{name} enclosure or mean mismatch')
        require(scaled==(name=='overshoot') and (name=='overshoot' or coeff.max()>1.6), name)
        print(f'{name}: input peak {peak}, max input coefficient {max(TO_COEFF@original.ravel()):.6g}, '
              f'published enclosure [{lo:.8g}, {hi:.8g}], mean {MEAN@observed:.12g}, '
              f'mean defect {mean_error:.3g}, nodal defect {err:.3g}',flush=True)

    # A transverse overshoot drives new extrema when the bottom inflow enters.
    data = simulate(args, 'stage-limiter', profile('.6+2*x*(1-x)', '.15+.1*t')
                    +fn('Prescribed Stokes solution','0;0;1+.35*z','Velocity function'))
    boxes, measured, _ = nodal(data, 1, 'damage')
    initial = np.array([limit(.6+2*X*(1-X))[0] for _ in boxes])
    options = dict(velocity=lambda z: 1+.35*z, bounded=True)
    ref, _, count, mean_err = columns(initial, boxes, lambda x,y,t:.15+.1*t, .01, **options)
    wrong = columns(initial, boxes, lambda x,y,t:.15+.1*t, .01, intermediate=False, **options)[0]
    err, discriminator = np.max(abs(measured-ref)), np.max(abs(measured-wrong))
    require(count>0 and err<1e-12 and discriminator>1e-6, 'stage limiting trajectory')
    print(f'stage limiting: {count} scaled stages/cells, max reference mean defect {mean_err:.3g}, '
          f'endpoint defect {err:.3g}, endpoint-only defect {discriminator:.6g}',flush=True)


def cfl(args):
    cube = section('Geometry model',section('Box',setting('Z repetitions',1)))
    # Linear mode of the material equation gives the RK stability polynomial.
    # Interior-only: u=z; two outward faces: u=2z-1. Only the z contributions
    # separate the predicted substep counts in these one-cell problems.
    for name, velocity, slope, h, phase_steps, signed_steps in [
            ('interior','z',1,.15,2,2), ('mean-loss','2*z-1',2,.08,2,1)]:
        anchor = 0 if name=='interior' else .5
        expr = '.25+z/2'
        data = simulate(args,'cfl-'+name,cube+profile(expr,expr)
                        +fn('Prescribed Stokes solution','0;0;'+velocity,'Velocity function')
                        +setting('End time',h)+fn('Time stepping',str(h)))
        errors = []
        for field, shift, n in [('damage',0.,phase_steps), ('tracer',-.5,signed_steps)]:
            def prediction(steps):
                w = -slope*h/steps
                return .25+shift+anchor/2+(Z-anchor)/2*(1+w+w*w/2+w**3/6)**steps
            observed = nodal(data,1,field)[1][0]
            err = np.max(abs(observed-prediction(n)))
            wrong = np.max(abs(observed-prediction(3-n)))
            require(err<1e-12 and wrong>1e-6, f'{name} {field} CFL count')
            errors.append(f"{field} ({n} steps): {err:.3g}; alternate count: {wrong:.6g}")
        print(f'z CFL {name}: '+', '.join(errors),flush=True)


def lifecycle(args):
    clean_settings = setting('End time',.025) + fn('Time stepping','.0125')
    clean = simulate(args, 'clean', clean_settings)
    retry = section('Time stepping',setting('List of model names','function, reject long first fault step')+
                    section('Function',setting('Function expression','if(time==0,.025,.0125)')))
    repeated = simulate(args, 'retry', clean_settings+retry, retry=True)
    print('retry maximum coefficient difference',compare(clean, repeated, 0.),flush=True)

    write = section('Checkpointing',setting('Steps between checkpoint',1))
    resume = setting('Resume computation','true')
    finish = setting('End time',.03)
    implicit = fields(SSPRK3_fields='')
    for name, select, other in [('bernstein','',implicit), ('nodal',implicit,'')]:
        simulate(args, 'seed-'+name, select+setting('End time',.02)+write)
        seed = args.directory/('seed-'+name)/'output'
        # Read the mixed-basis checkpoint written with the legacy name list
        # using only the ordinary per-field methods.
        continuation_methods = (fields(SSPRK3_fields='', Compositional_field_methods='field, explicit DG SSPRK3, explicit DG SSPRK3')
                                if name == 'bernstein' else select)
        continued = simulate(args, 'continued-'+name, continuation_methods+finish+resume, checkpoint=seed)
        direct = simulate(args, 'direct-'+name, select+finish)
        err = compare(direct, continued, 1e-12)
        simulate(args, 'mismatch-'+name, other+finish+resume, checkpoint=seed,
                 fail='Checkpoint basis for field damage')
        print(f'{name} continuation max defect {err:.3g}, reverse basis rejected',flush=True)
    simulate(args, 'seed-partition', setting('End time',.02)+write, ranks=2)
    parallel = simulate(args, 'continued-partition', finish+resume,
                        checkpoint=args.directory/'seed-partition/output', ranks=2)
    require(set(nodal(parallel,max(parallel),'damage')[2])=={0,1}, 'resumed partition lost an owner')
    err = compare(read_output(args.directory/'direct-bernstein/output',1,3), parallel, 1e-12)
    print(f'two-rank restart against uninterrupted serial coefficients {err:.3g}',flush=True)
    def metadata(path):
        slot = int((path/'restart/last_good_checkpoint.txt').read_text())
        return path/'restart'/f'{slot:02d}'/'metadata.txt'
    damaged = args.directory/'missing-record'
    shutil.copytree(args.directory/'seed-bernstein/output', damaged)
    record = metadata(damaged)
    record.write_text('\n'.join(record.read_text().splitlines()[:2])+'\n')
    simulate(args,'unmarked-modern',implicit+finish+resume,checkpoint=damaged,
             fail='Missing 3-D checkpoint composition basis record')
    print('current DGQ2 checkpoint missing its basis record rejected',flush=True)
    if args.legacy_aspect is None:
        print('Legacy checkpoint check skipped; supply a pre-basis Release writer to exercise it.',flush=True)
        return
    producer = copy(args)
    # The old writer's plugin must match its build type.
    producer.aspect, producer.plugins, producer.mode = args.legacy_aspect, args.legacy_plugins or args.plugins, 'release'
    simulate(producer, 'legacy', implicit+setting('End time',.02)+write)
    parent = args.directory/'legacy/output'
    require('composition_bases' not in metadata(parent).read_text(), 'expected an actual pre-basis checkpoint')
    legacy = simulate(args, 'continue-legacy', implicit+finish+resume, checkpoint=parent)
    nodal_control = read_output(args.directory/'direct-nodal/output',1,3)
    err = compare(nodal_control, legacy, 1e-12)
    # A transport-method change that keeps a nodal basis is legal. Advance the
    # saved implicit tracer once using the strong column action at its saved clock.
    selected = simulate(args,'legacy-as-signed',fields(SSPRK3_fields='tracer')+finish+resume,checkpoint=parent)
    k = max(selected)
    t, h = selected[k][0]
    boxes, old, _ = nodal(selected,k-1,'tracer')
    predicted = columns(old, boxes, lambda x,y,t: (1-t)/4 - .5, h, start=t-h)[0]
    signed_error = np.max(abs(nodal(selected,k,'tracer')[1]-predicted))
    require(signed_error<1e-12,'legacy nodal coefficients misread by signed advance')
    print(f'legacy nodal to signed SSPRK3 column defect {signed_error:.3g}',flush=True)
    simulate(args,'legacy-as-bernstein',finish+resume,checkpoint=parent,fail='Legacy 3-D checkpoint has nodal basis for field damage')
    print(f'legacy 3-D nodal continuation max defect {err:.3g}; legacy-to-Bernstein rejected',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('aspect','plugins','directory'):
        parser.add_argument(arg, type=lambda p: Path(p).resolve())
    parser.add_argument('legacy_aspect', nargs='?', type=lambda p: Path(p).resolve(),
                        help='pre-basis Release executable for the legacy restart checks')
    parser.add_argument('legacy_plugins', nargs='?', type=lambda p: Path(p).resolve(),
                        help='directory containing the coefficient probe for the legacy writer (defaults to plugins)')
    parser.add_argument('--mpiexec',default='mpiexec')
    parser.add_argument('--slabs-only', action='store_true',
                        help='run the two-slab serial, partition and implicit controls with the installed coefficient observer')
    args = parser.parse_args()
    args.mode = args.aspect.name.split('-')[-1]
    slabs(args)
    if args.slabs_only:
        raise SystemExit(0)
    individual_fields(args)
    solved_velocity(args)
    asymmetric(args)
    diagonal(args)
    limiting(args)
    cfl(args)
    lifecycle(args)
