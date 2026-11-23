"""Check scalar SSPRK3 stability, ordinary field consumers and checkpoint compatibility."""
import argparse
import shutil
from copy import copy
from pathlib import Path

import numpy as np
from check import field_data, fields, fn, phase_settings, require, run, section, setting
from reference import mean_weights, nodes_y, to_values


one_cell = section('Geometry model', section('Box', setting('X repetitions', 1)))
phase_profile = '.9-.65*(2*y-1)^2'
stats = section('Postprocess', setting('List of postprocessors', 'DG coefficient samples, composition statistics'))


def statistics(output, name):
    lines = (output/'statistics').read_text().splitlines()
    names = [line.split(': ', 1)[1] for line in lines if line.startswith('#')]
    column = names.index(name)
    return np.array([float(line.split()[column]) for line in lines if line.strip() and not line.startswith('#')])


def vortex(args):
    # M = integral (1-2y) phi. Simpson integrates the Q2 x Q3 product exactly.
    moment_weights = mean_weights*(1-2*nodes_y)
    flow = 'x*(1-x)*(1-2*y); -(1-2*x)*y*(1-y)'
    profile = '.25+x/2'
    def moment(name, velocity, end, dt):
        data = run(args, name, one_cell+phase_settings(profile, profile, velocity, end=end, dt=dt))
        require(abs(data[max(data)][0][0]-end) < 1e-14, 'wrong vortex endpoint')
        return [float(moment_weights @ to_values @ field_data(data, k, 'damage')[1][0]) for k in (0, max(data))]

    # On a closed unit cell, phi_x=1/2, so M'(0)=-int x(1-x)(1-2y)^2 / 2=-1/36.
    defects = []
    for h in (.0125, .00625):
        start, finish = moment(f'vortex-{h}', flow, h, h)
        slope = (finish-start)/h
        defects.append(abs(slope+1/36))
        print(f'closed vortex h={h}: slope {slope:.12g}, expected {-1/36:.12g}', flush=True)
    require(defects[0] < 1e-7 and defects[1] < .3*defects[0], 'closed vortex moment derivative')

    fast = '1000*x*(1-x)*(1-2*y)+.001; -1000*(1-2*x)*y*(1-y)'
    moments = [moment(f'circulation-{dt}', fast, .0125, dt)[1] for dt in (.0125, .0001, .00005)]
    print('strong circulation moments:', moments, flush=True)
    # Freeze the review's refined moment. Halving the macrostep once more
    # estimates its remaining temporal error independently of the CFL count.
    require(abs(moments[1]-.080302600504167) < 1e-11, 'refined circulation trajectory')
    require(abs(moments[0]-moments[1]) < 1e-5 and abs(moments[1]-moments[2]) < 1e-6,
            'underresolved interior circulation')


    # Both x faces discharge. At dt=.08 the normalized face-loss rate is
    # 15/s and the interior rate is 10/s: only the former selects two steps.
    dt = .08
    data = run(args,'mean-loss',one_cell+phase_settings(profile,profile,'2*x-1;0',end=dt,dt=dt))
    raw = field_data(data,max(data),'damage')[1][0]
    nodes = to_values@raw
    def amplitude(h):
        z = -2*h
        return 1+z+z*z/2+z**3/6
    expected = .5-.25*amplitude(dt/2)**2
    single = .5-.25*amplitude(dt)
    require(abs(nodes[0]-expected)<1e-13 and abs(nodes[0]-single)>1e-6,'mean-loss CFL discarded')
    print(f'two-outflow mean bound: left {nodes[0]:.12g}, one-step prediction {single:.12g}',flush=True)


def prescriptions(args):
    initial = section('Prescribed solution', setting('List of model names','initial composition')
                      +section('Initial composition', section('Indicator function',setting('Function expression','if(x<.51,1,0)'))))
    run(args, 'prescribed-tracer', initial, fail='SSPRK3 cannot advance prescribed DoFs on field tracer')
    fixed = run(args, 'prescribed-implicit', initial+fields(SSPRK3_fields=''))
    left = field_data(fixed,1,'tracer')[1][0,0]
    require(abs(left+.04) < 1e-14, 'prescribed implicit DoF moved')

    base = one_cell+phase_settings('.25+x/2','.25+x/2','x*(1-x)*(1-2*y); -(1-2*x)*y*(1-y)',end=.0125)
    run(args,'prescribed-bernstein',base+initial,fail='Cannot prescribe point values on non-nodal component')
    temperature = section('Prescribed solution',setting('List of model names','temperature function')+
                          section('Temperature function',section('Indicator function',setting('Function expression','if(x<.51,1,0)'))+
                                  section('Function',setting('Function expression','100'))))
    reports = section('Postprocess',setting('List of postprocessors','DG coefficient samples, temperature statistics'))
    control = run(args,'phase-unconstrained',base)
    with_temp = run(args,'prescribed-temperature',base+temperature+reports)
    temp = statistics(args.directory/'prescribed-temperature/output','Maximal temperature (K)')[-1]
    require(abs(temp-100) < 1e-8,'temperature prescription missed its nodal field')

    joint = fields(Number_of_fields=2,Names_of_fields='damage, companion',Types_of_fields='phase field, generic',
                   SSPRK3_fields='damage')+fn('Initial composition model','.25+x/2; 0')+fn('Boundary composition model','.25+x/2; 0')
    with_comp = run(args,'prescribed-companion',base+joint+
                    section('Prescribed solution',setting('List of model names','prescribed companion')))
    coeff = field_data(with_comp,1,'companion')[1][0].reshape(3,3)
    np.testing.assert_allclose(coeff[:,:2],[[.25,.75]]*3,atol=1e-13,rtol=0)
    phase = field_data(control,1,'damage')[1]
    require(np.max(abs(phase-field_data(control,0,'damage')[1])) > .001,'phase did not advance')
    for data in (with_temp,with_comp):
        np.testing.assert_allclose(field_data(data,1,'damage')[1],phase,rtol=0,atol=1e-13)
    print(f'prescribed implicit left {left:g}, temperature max {temp:g} K, companion x=0,.5: {coeff[0,:2]}',flush=True)


def degrees_and_statistics(args):
    base = one_cell+phase_settings('.5','.5','x^2;0',end=.0125)
    data = run(args,'q2-constant',base)
    error = np.max(abs(field_data(data,1,'damage')[1]-.5))
    require(error<1e-13,'Q2 velocity changed constant phase')
    high = (section('Discretization',setting('Stokes velocity polynomial degree',7))+
            section('Solver parameters',section('Stokes solver parameters',setting('Stokes solver type','block AMG'))) +
            fn('Prescribed Stokes solution','x^7;0','Velocity function'))
    run(args,'q7-explicit',base+high,fail='qualified for Stokes velocity degree 2')
    data = run(args,'q7-implicit',base+high+fields(SSPRK3_fields=''))
    implicit_error = np.max(abs(field_data(data,1,'damage')[1]-.5))
    require(implicit_error<1e-12,'implicit Q7 velocity changed constant scalar')
    data = run(args,'physical-extrema',one_cell+phase_settings(phase_profile,phase_profile,'0;0')+stats)
    output = args.directory/'physical-extrema/output'
    for col,expected in [('Minimal value',.25),('Maximal value',.9)]:
        np.testing.assert_allclose(statistics(output,col+' for composition damage'),expected,rtol=0,atol=1e-12)
    for k in data:
        require(abs(field_data(data,k,'damage')[1].max()-1.55)<1e-14,'admissible coefficient changed')
    print(f'constant defects explicit Q2 {error:.3g}, implicit Q7 {implicit_error:.3g}; physical extrema .25/.9, coefficient 1.55',flush=True)


def checkpoint_checks(args):
    base = one_cell+phase_settings(phase_profile,phase_profile,'0;0')+stats
    write = section('Checkpointing',setting('Steps between checkpoint',1))
    resume = setting('Resume computation','true')+setting('End time',.05)
    nodal = fields(SSPRK3_fields='')
    for name, selection, other in [('bernstein','',nodal),('nodal',nodal,'')]:
        run(args,'seed-'+name,base+selection+write)
        parent = args.directory/('seed-'+name)/'output'
        data = run(args,'same-'+name,base+selection+resume,checkpoint=parent)
        require(data[max(data)][0][0]==.05,'continuation missed the final time')
        for step in data:
            np.testing.assert_allclose(field_data(data,step,'damage')[1], field_data(data,0,'damage')[1],rtol=0,atol=1e-12)
        mass = statistics(args.directory/('same-'+name)/'output','Global mass for composition damage')
        np.testing.assert_allclose(mass,41/60,rtol=0,atol=5e-9)  # 8-digit statistics
        run(args,'mismatch-'+name,base+other+resume,checkpoint=parent,fail='Checkpoint basis for field damage')
        print(f'{name} continued to .05, mass {mass[-1]:.9g}; reverse basis rejected',flush=True)

    pair = (base+fields(Number_of_fields=2,Names_of_fields='first, second',Types_of_fields='phase field, phase field',
                        SSPRK3_fields='first')+fn('Initial composition model',phase_profile+'; .5')+
            fn('Boundary composition model',phase_profile+'; .5'))
    run(args,'seed-pair',pair+write)
    run(args,'mismatch-pair',pair+fields(SSPRK3_fields='second')+resume,
        checkpoint=args.directory/'seed-pair/output',fail='Checkpoint basis for field first')
    print('two-field basis swap rejected at the first field, with the same number of Bernstein fields',flush=True)

    # Exercise the old two-line metadata, optionally using the actual old writer.
    # Preserve known nodal restarts, and fail closed on an unmarked 2-D DGQ2.
    producer = copy(args)
    if args.legacy_aspect:
        producer.aspect = args.legacy_aspect
    for name,disc in [('legacy-continuous','false'),('legacy-dg','true')]:
        settings = base+nodal+section('Discretization',setting('Use discontinuous composition discretization',disc))
        run(producer,name,settings+write)
        parent = args.directory/name/'output'
        slot = int((parent/'restart/last_good_checkpoint.txt').read_text())
        original = parent/'restart'/f'{slot:02d}'/'metadata.txt'
        legacy = args.directory/(name+'-copy')
        shutil.copytree(parent,legacy)
        meta = legacy/original.relative_to(parent)
        if args.legacy_aspect:
            require('composition_bases' not in meta.read_text(),'expected a pre-repair checkpoint')
        else:
            meta.write_text('\n'.join(meta.read_text().splitlines()[:2])+'\n')
        if disc=='true':
            for suffix,active in [('nodal',settings),('bernstein',base)]:
                run(args,'ambiguous-'+suffix,active+resume,checkpoint=legacy,fail='Ambiguous legacy DGQ2 checkpoint basis')
            # Here the producing input and the source FE are independently known.
            with meta.open('a') as out:
                out.write('composition_bases 1 "FE_DGQ<2>(2)"\n')
        data = run(args,'resume-'+name,settings+resume,checkpoint=legacy)
        require(data[max(data)][0][0]==.05,'legacy continuation missed endpoint')
        np.testing.assert_allclose(statistics(args.directory/('resume-'+name)/'output','Global mass for composition damage'),41/60,
                                   atol=5e-9,rtol=0)
    print('legacy continuous restarted; ambiguous DGQ2 rejected for both readers; identified nodal DGQ2 continued',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('aspect','plugins','directory'):
        parser.add_argument(name,type=lambda s:Path(s).resolve())
    parser.add_argument('--legacy-aspect',type=lambda s:Path(s).resolve(),help='pre-repair binary of the same build type')
    args=parser.parse_args()
    args.mode=args.aspect.name.split('-')[-1]
    vortex(args)
    prescriptions(args)
    degrees_and_statistics(args)
    checkpoint_checks(args)
