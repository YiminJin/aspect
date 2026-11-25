"""Exercise both public fault routes, their inputs and the two checkpoint formats."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from compare import rows, compare_values
from check_repairs import subsection


def material(route, values):
    return subsection('Material model', subsection(route,values))


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
    inputs=[growing/'onecut.prm',growing/'adiabatic_pressure.prm']
    seed=run('growth',example,extra_files=inputs)
    legacy=run('legacy-growth',(growing/'adiabatic_pressure.prm').read_text(),extra_files=inputs[:1])
    for tag in ('0000','0001'):
        for files,keys in [('points-'+tag+'-r*.csv',('cell','q')),('sections-'+tag+'.csv',('section',))]:
            a,b=(rows(d.glob(files),keys) for d in (seed,legacy))
            for col in set(next(iter(a.values())))-set(keys):
                compare_values(a,b,col,1e-12,1e-12)
    print('Canonical one-field and legacy four-field accepted native stress and strips match',flush=True)

    common=(fixed/'mature_trace.prm').read_text()
    data=(fixed/'mature_vertices.txt').read_text()
    inline='''set Fault vertices = .5,0;.5,.5;.5,1
    set Initial label ages = 10,11,12
    set Initial label shear tractions = 3.85,3.80,3.75'''
    table=run('mature',common,extra_files=[fixed/'fixed_fault.prm',fixed/'mature_vertices.txt'])
    old=run('inline',common.replace('set Fault vertex file = mature_vertices.txt',inline)
            .replace('mature fault statistics','fixed fault statistics'),extra_files=[fixed/'fixed_fault.prm'])
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
        return run(name,common.replace('mature_vertices.txt',str(file)),settings,[fixed/'fixed_fault.prm'],error,plugin=plugin)
    fault=lambda x:material('Phase field fault',x)
    for name,text,setting,message in [
        ('donor-format','0.0 0.0 0.35\n','', 'anonymous donor core-phase triples'),
        ('out-of-order',data.replace('1 0.5 0.5','2 0.5 0.5'),'','consecutive material IDs'),
        ('duplicate-column',data.replace('background_shear_Pa','initial_age_s'),'','duplicate vertex column'),
        ('nonfinite',data.replace('11 3.80','11 nan'),'','Invalid vertex value'),
        ('nonstraight',data.replace('1 0.5 0.5','1 0.7 0.5'),'','straight and strictly ordered'),
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
        actual=np.loadtxt(result/f'fault-labels-{step:05}.txt')[:,-1]
        expected=(3.85-(.02+.0125*step))/20
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-12)
        print('Persistent shear, endpoint',step,'expected/measured',expected,actual,flush=True)
    alias=run('legacy-dashpot-trace',homogeneous.replace('Model name = phase field fault','Model name = linear trace resistance'),
              fault('set Background shear traction = 17\nset Fault vertex file = '+str(file)),plugin=True)
    labels_equal(alias,result)
    alias=run('legacy-dashpot-strip',example.replace('Model name = phase field fault','Model name = linear strip resistance'),extra_files=inputs,plugin=True)
    explicit=run('dashpot-strip',example,material('Growing fault','set Resistance law = linear dashpot'),inputs,plugin=True)
    if (alias/'sections-0001.csv').read_bytes()!=(explicit/'sections-0001.csv').read_bytes():
        raise ValueError('Legacy dashpot strip selector changed the law')

    run('wrong-statistics',example,subsection('Postprocess','set List of postprocessors = mature fault statistics'),inputs,'require a mature material trace')
    run('wrong-history',homogeneous,subsection('Postprocess','set List of postprocessors = growing fault history'),expected='requires the two-dimensional growing fault material')
    run('route-conflict',example,fault('set Fault geometry = fixed interior'),inputs,'cannot accept mature trace geometry')

    # Both checkpoint writers use the same field names, degrees and Bernstein
    # phase basis. Each reader therefore reaches the material history boundary.
    parent=(growing/'onecut.prm').read_text()
    parent=parent.replace('tau_xx, tau_yy, tau_xy','fault_stress_xx, fault_stress_yy, fault_stress_xy')
    parent=parent.replace('generic, generic, generic, phase field','stress, stress, stress, phase field')
    parent=parent.replace('set Model name = growing fault','set Model name = phase field fault')
    parent+='\nset Surface pressure =\n'+subsection('Material model','set Friction normal stress = adiabatic pressure')
    parent+=subsection('Adiabatic conditions model','set Model name = function\n'+subsection('Function','set Function expression = 293;200;1'))
    save=subsection('Checkpointing','set Steps between checkpoint = 1\nset Number of checkpoints to keep = 4')
    grow_parent=run('growing-parent',parent,'\nset End time = .005\n'+save)
    mature=parent.replace('single Advection, iterated Newton Stokes','single Advection, iterated Stokes').replace('midpoint fault phase','explicit DG SSPRK3')
    mature+=material('Growing fault','set Seed tangent angle = 0\nset Forward seed tip = 0,0')
    mature+=fault('set Fault geometry = fixed interior\nset Fit trace to phase = false\nset Fault vertices = .5,.2; 1,.8\nset Background shear traction = 128\nset Shear modulus = 10\nset Radiation damping = 100\nset Reference speed = .001\nset Direct effect = .1')
    mature+=subsection('Material model',subsection('Simple model','set Viscosity = 10'))
    mature+=subsection('Solver parameters',subsection('Stokes solver parameters','set Linear solver tolerance = 1e-13'))
    mature+=subsection('Boundary velocity model','set Prescribed velocity boundary indicators = left: function, right: function, bottom: function\n'+subsection('Function','set Function expression = 0;0'))
    mature+=subsection('Initial composition model',subsection('Function','set Function expression = 0;0;0;.35'))
    mature+=subsection('Boundary composition model','set Fixed composition boundary indicators = left,right,bottom,top\nset List of model names = function\n'+subsection('Function','set Function expression = 0;0;0;.35'))
    run('growth-to-mature',mature,checkpoint=grow_parent,expected='Cannot resume a growing fault checkpoint')
    trace_parent=run('mature-parent',mature,'\nset End time = .005\n'+save)
    run('mature-to-growth',parent,checkpoint=trace_parent,expected='no native growing fault history')


if __name__=='__main__':
    main()
