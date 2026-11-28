"""Exercise both public fault routes, their inputs and the two checkpoint formats."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from compare import rows, compare_values
from check_adiabatic_pressure import adiabatic_constants
from parameter_file import take_material_subsection
from check_repairs import subsection


def material(route, values):
    return subsection('Material model', subsection(route,values))


def compare_growing_frames(reference, target, with_age=True, steps=(0, 1)):
    """Compare native and strip history at each requested step and return the last step's errors."""
    for step in steps:
        tag = f'{step:04}'
        errors = {}
        a, b = (rows(d.glob(f'points-{tag}-r*.csv'), ('cell', 'q')) for d in (reference, target))
        if a.keys() != b.keys() or any(a[k]['section'] != b[k]['section'] for k in a):
            raise ValueError(f'Native identities or strip membership changed at frame {tag}')
        # Both decks use the same mesh and chord quadrature. Stresses are in
        # Pa; phi, cohesive strain and the two point shear histories have no units.
        for col in ('x', 'y', 'weight'):
            errors[col] = compare_values(a, b, col, 0., 0.)
        for col in ('tauxx', 'tauyy', 'tauzz', 'tauxy'):
            errors[col] = compare_values(a, b, col, 1e-12, 1e-12)
        for col in ('phi', 'Axx', 'Ayy', 'Azz', 'Axy'):
            errors[col] = compare_values(a, b, col, 1e-15, 1e-12)
        for col in ('signed_shear', 'absolute_shear'):
            errors[col] = compare_values(a, b, col, 0., 1e-12)

        a, b = (rows([d/f'sections-{tag}.csv'], ('section',)) for d in (reference, target))
        if a.keys() != b.keys() or any(a[k]['formed'] != b[k]['formed'] for k in a):
            raise ValueError(f'Strip identities or formation flags changed at frame {tag}')
        errors['xi'] = compare_values(a, b, 'xi', 0., 0.)                 # fixed strip coordinate (m)
        errors['ds'] = compare_values(a, b, 'ds', 1e-14, 1e-12)           # current arclength (m)
        errors['beta'] = compare_values(a, b, 'beta', 1e-14, 1e-12)         # direction (rad)
        if any(('age' in next(iter(table.values()))) != with_age for table in (a, b)):
            raise ValueError(f'Unexpected constitutive history at frame {tag}')
        if with_age:
            errors['age'] = compare_values(a, b, 'age', 1e-12, 1e-12)      # accepted RSF history (s)
        # Signed speed (m/s), integrated slip (m) and resistance (Pa) have
        # no absolute floor. Zero must remain zero, and either creep direction
        # must agree even when the rate is well below 1e-12 m/s.
        for col in ('V', 'slip', 'force'):
            errors[col] = compare_values(a, b, col, 0., 1e-12)
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('aspect','plugin','destination'):
        parser.add_argument(arg,type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    root=args.destination
    root.mkdir(parents=True,exist_ok=False)
    growing=Path(__file__).resolve().parent
    fixed=growing.parent/'fixed_fault'

    def run(name,base,settings='',extra_files=(),expected=None,checkpoint=None,plugin=False):
        work=root/name
        work.mkdir()
        for file in extra_files:
            shutil.copy(file,work)
        if checkpoint:
            shutil.copytree(checkpoint/'restart/02',work/'output/restart/02')
            settings+='\nset Resume computation = true\n'+subsection('Checkpointing','set Resume checkpoint = 2')
        if plugin:
            settings+=f'\nset Additional shared libraries = {args.plugin}\n'
        (work/'input.prm').write_text(base+'\nset Output directory = output\n'+settings)
        with (work/'log').open('w') as log:
            result=subprocess.run([args.aspect,'input.prm'],cwd=work,stdout=log,stderr=subprocess.STDOUT,
                                  env=os.environ|dict(DEAL_II_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),timeout=120)
        if expected:
            if not result.returncode or expected not in ' '.join((work/'log').read_text().split()):
                raise ValueError(f'{name}: missing diagnostic {expected}; see {work}/log')
            print(name,'rejected:',expected,flush=True)
        else:
            result.check_returncode()
        return work/'output'

    example=(growing/'growing_seed.prm').read_text()
    # The reference for this regression uses 200 Pa + (50 Pa/m) times depth.
    example += adiabatic_constants(200, 50)
    inputs=[growing/'onecut.prm',growing/'adiabatic_pressure.prm']
    seed=run('growth',example,extra_files=inputs)
    legacy=run('legacy-growth',(growing/'adiabatic_pressure.prm').read_text(),extra_files=inputs[:1])
    compare_growing_frames(seed,legacy)
    print('Canonical one-field and legacy four-field accepted native stress and strips match',flush=True)
    topography = subsection('Geometry model', subsection('Initial topography model',
        'set Model name = function\n' + subsection('Function',
        'set Function expression = 0.02\nset Maximum topography value = 0.02')))
    rejected = run('raised-top', example, topography,
                   expected='Set Geometry model / Initial topography model / Model name = zero topography')
    if any(rejected.glob('points-*.csv')) or any(rejected.glob('sections-*.csv')):
        raise ValueError('The growing material published a frame before rejecting initial topography')

    common=(fixed/'mature_trace.prm').read_text()
    data=(fixed/'mature_vertices.txt').read_text()
    fault=lambda x:material('Phase field fault',x)
    inline='''set Fault vertex file =
    set Fault vertices = .5,0;.5,.5;.5,1
    set Initial label ages = 10,11,12
    set Initial label shear tractions = 3.85,3.80,3.75'''
    table=run('mature',common,extra_files=[fixed/'fixed_fault.prm',fixed/'mature_vertices.txt'])
    old=run('inline',common,fault(inline)+subsection('Postprocess',
            'set List of postprocessors = fixed fault statistics, visualization'),extra_files=[fixed/'fixed_fault.prm'])
    def labels_equal(a,b):
        for step in range(3):
            f=f'fault-labels-{step:05}.txt'
            if (a/f).read_bytes()!=(b/f).read_bytes():
                raise ValueError(f'Changed material identities or history: {f}')
    labels_equal(table,old)
    for step in range(3):
        vtu=f'solution/solution-{step:05}.0000.vtu'
        trees=[ET.parse(directory/vtu) for directory in (table,old)]
        for name in ('fault_tau_xx','fault_tau_yy','fault_tau_xy'):
            path=f'.//PointData/DataArray[@Name="{name}"]'
            a,b=(tree.find(path) for tree in trees)
            if a is None or b is None or not a.text or (a.attrib,a.text)!=(b.attrib,b.text):
                raise ValueError(f'Changed accepted {name} at step {step}')
    print('Canonical table and inline labels and physical stress arrays identical at all three clocks',flush=True)
    def vertex_case(name,content,settings='',error=None,plugin=False):
        file=root/(name+'.txt')
        file.write_text(content)
        return run(name,common,fault(f'set Fault vertex file = {file}')+settings,[fixed/'fixed_fault.prm'],error,plugin=plugin)
    bent_table = data.replace('1 0.5 0.5', '1 0.7 0.5')
    prescribed = vertex_case('prescribed-bend', bent_table, fault('set Fit trace to phase = false'))
    for step in range(3):
        positions = np.loadtxt(prescribed/f'fault-labels-{step:05}.txt')[:, :3]
        np.testing.assert_array_equal(positions, [[0, .5, 0.], [1, .7, .5], [2, .5, 1.]])
    for name,text,setting,message in [
        ('donor-format','0.0 0.0 0.35\n','', 'anonymous donor core-phase triples'),
        ('out-of-order',data.replace('1 0.5 0.5','2 0.5 0.5'),'','consecutive material IDs'),
        ('duplicate-column',data.replace('background_shear_Pa','initial_age_s'),'','duplicate vertex column'),
        ('nonfinite',data.replace('11 3.80','11 nan'),'','Invalid vertex value'),
        ('fit-bent-seed',bent_table,fault('set Fit trace to phase = true'),
         'Set Fit trace to phase = false to use a prescribed curved trace'),
        ('duplicate-vertices',data,fault('set Fault vertices = 0,0; 0,1'),'Choose Fault vertex file'),
        ('duplicate-age',data,fault('set Initial label ages = 1,2,3'),'Duplicate inline and file source'),
        ('age-on-dashpot',data,fault('set Resistance law = linear dashpot'),'does not accept initial ages'),
    ]:
        vertex_case(name,text,setting,message,plugin=name=='age-on-dashpot')
    # The initial DG tensor vanishes. Both rates respond to the same table
    # shear at each endpoint, despite an intentionally different scalar.
    stateless=fault('set Background shear traction = 17\n set Resistance law = linear dashpot')
    # Use resolved traction for the analytic wall-traction control.
    homogeneous=(fixed/'fixed_fault.prm').read_text()
    file=root/'shear.txt'
    file.write_text('id x_m y_m background_shear_Pa\n0 0 0 3.85\n1 0 1 3.85\n')
    result=run('persistent',homogeneous,stateless+fault('set Fault vertex file = '+str(file)),plugin=True)
    for step in (1,2):
        label_file=result/f'fault-labels-{step:05}.txt'
        if 'age(s)' in label_file.read_text().splitlines()[0]:
            raise ValueError('Stateless mature law published a constitutive age')
        actual=np.loadtxt(label_file)[:,-1]
        expected=(3.85-(.02+.0125*step))/20
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-12)
        print('Persistent shear, endpoint',step,'expected/measured',expected,actual,flush=True)
    alias=run('legacy-dashpot-trace',homogeneous,subsection('Material model','set Model name = linear trace resistance')
              +fault('set Background shear traction = 17\nset Fault vertex file = '+str(file)),plugin=True)
    labels_equal(alias,result)
    alias=run('legacy-dashpot-strip','include adiabatic_pressure.prm\n'+
              subsection('Material model','set Model name = linear strip resistance')
              + material('Simple model', 'set Viscosity = 1e6'), extra_files=inputs, plugin=True)
    explicit=run('dashpot-strip',example,fault('set Resistance law = linear dashpot'),inputs,plugin=True)
    if (alias/'sections-0001.csv').read_bytes()!=(explicit/'sections-0001.csv').read_bytes():
        raise ValueError('Legacy dashpot strip selector changed the law')
    run('strip-selector-on-trace', homogeneous,
        subsection('Material model', 'set Model name = linear strip resistance'),
        expected='Additional information: The linear strip resistance selector requires midpoint strip mechanics.', plugin=True)
    run('trace-selector-on-strips', example,
        subsection('Material model', 'set Model name = linear trace resistance')
        + fault('set Resistance law = linear dashpot'), inputs,
        expected='Additional information: The linear trace resistance selector requires mature trace mechanics.', plugin=True)
    def check_dashpot(output, impedance):
        strips=rows([output/'sections-0001.csv'],('section',))
        for row in strips.values():
            if 'age' in row:
                raise ValueError('Stateless growing law published a constitutive age')
            np.testing.assert_allclose(float(row['force']),impedance*float(row['V']),atol=0,rtol=1e-12,equal_nan=False)
        print(f'Strip 8, impedance {impedance} Pa s/m: V={strips[(8,)]["V"]} m/s, force={strips[(8,)]["force"]} Pa',flush=True)
    check_dashpot(explicit,1000.)
    alternate='set Resistance law = linear dashpot\nset Dashpot impedance = 750'
    public_impedance=run('public-impedance',example,fault(alternate),inputs,plugin=True)
    legacy_impedance=run('legacy-impedance','include adiabatic_pressure.prm',
                         material('Growing fault',alternate),inputs,plugin=True)
    compare_growing_frames(public_impedance,legacy_impedance,with_age=False)
    check_dashpot(public_impedance,750.)
    run('legacy-selector-conflict',example,subsection('Material model','set Model name = linear strip resistance')+
        fault('set Resistance law = rate and state'),inputs,'Legacy dashpot selector conflicts',plugin=True)

    run('wrong-statistics',example,subsection('Postprocess','set List of postprocessors = mature fault statistics'),inputs,'require a mature material trace')
    run('wrong-history',homogeneous,subsection('Postprocess','set List of postprocessors = growing fault history'),
        expected='requires the two-dimensional phase field fault material with the midpoint fault phase composition method')
    run('route-conflict',example,fault('set Fault geometry = fixed interior'),inputs,'requires mature trace mechanics')

    # Both checkpoint writers use the same field names, degrees and Bernstein
    # phase basis. Each reader therefore reaches the material history boundary.
    parent=(growing/'onecut.prm').read_text()
    parent+=subsection('Compositional fields',
                       'set Names of fields = fault_stress_xx, fault_stress_yy, fault_stress_xy, phase_field\n'
                       'set Types of fields = stress, stress, stress, phase field')
    parent+='\nset Surface pressure =\n'+subsection('Material model','set Model name = phase field fault')
    parent+=fault('set Friction normal stress = adiabatic pressure')
    parent+=subsection('Adiabatic conditions model','set Model name = function\n'+subsection('Function','set Function expression = 293;200;1'))
    save=subsection('Checkpointing','set Steps between checkpoint = 1\nset Number of checkpoints to keep = 4')
    grow_parent=run('growing-parent',parent,'\nset End time = .005\n'+save)
    # Keep the checkpoint-compatible mesh and FE slots and remove the growth
    # physics from this mature input. A mature checkpoint must reach the
    # growing material's archive reader to exercise its wrong-route guard.
    mature, _ = take_material_subsection(parent, 'Growing fault')
    mature+='\nset Nonlinear solver scheme = single Advection, iterated Stokes\n'
    mature+=subsection('Compositional fields','set Compositional field methods = static, static, static, explicit DG SSPRK3')
    mature+=fault('set Fault geometry = fixed interior\nset Fit trace to phase = false\nset Fault vertices = .5,.2; 1,.8\nset Background shear traction = 128\nset Shear modulus = 10\nset Radiation damping = 100\nset Reference speed = .001\nset Direct effect = .1')
    mature+=fault('set Viscosity = 10')
    mature+=subsection('Solver parameters',subsection('Stokes solver parameters','set Linear solver tolerance = 1e-13'))
    mature+=subsection('Boundary velocity model','set Prescribed velocity boundary indicators = left: function, right: function, bottom: function\n'+subsection('Function','set Function expression = 0;0'))
    mature+=subsection('Initial composition model',subsection('Function','set Function expression = 0;0;0;.35'))
    mature+=subsection('Boundary composition model','set Fixed composition boundary indicators = left,right,bottom,top\nset List of model names = function\n'+subsection('Function','set Function expression = 0;0;0;.35'))
    run('growth-to-mature',mature,checkpoint=grow_parent,expected='Cannot resume a growing fault checkpoint')
    trace_parent=run('mature-parent',mature,'\nset End time = .005\n'+save)
    run('mature-to-growth',parent,checkpoint=trace_parent,expected='no native growing fault history')
    # Change only the law while restoring each RSF parent. The strip reader
    # checks the law's physical parameter signature before its archived name.
    dashpot=fault('set Resistance law = linear dashpot')
    run('growth-changed-law',parent,dashpot,plugin=True,checkpoint=grow_parent,
        expected='Growing fault physical parameters differ from the checkpoint.')
    run('trace-changed-law',mature,dashpot,plugin=True,checkpoint=trace_parent,
        expected='Mature fault resistance law differs from the checkpoint')


if __name__=='__main__':
    main()
