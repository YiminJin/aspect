"""One true-normal-stress initialization with resolved surface Ih RHS quadrature."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
import run_steady_large_step as runner
from startup_30km import ROOT, HERE, BP3, BIN, STUDY, FIXTURE, parameters, render, digest
from run_mechanical_width import VirtualProfile, NORMAL
from run_normal_control import BASE

OUT=STUDY/'surface-quadrature'
PANELS=8
CASE='panels8-initial'
runner.OUT=OUT


def prepare():
    path=OUT/CASE;path.mkdir()
    reference=json.loads((BASE/'launch.json').read_text())
    # The reference executable predates the separately verified normal-control
    # guard change. Keep its saved copy and all new inputs recoverable.
    for name in ('fault.txt','target_cells.txt','completion.txt'):
        p=FIXTURE/name
        assert digest(p)==reference['hashes'][str(p)],p
    profile=VirtualProfile(100.,24.4140625)
    fault=np.loadtxt(FIXTURE/'fault.txt')[:,:2]
    points=np.concatenate([(k+(leggauss(3)[0]+1)/2)/PANELS for k in range(PANELS)])
    rows=[];error=0.
    for a,b in zip(fault[:-1],fault[1:]):
        for z in points:
            p=(1-z)*a+z*b;e=profile.extent
            intervals=[(-e,min(-p[1]/NORMAL[1],e)),(max((100000-p[1])/NORMAL[1],-e),e)]
            value=sum(profile.integrate(p,lo,hi) for lo,hi in intervals)
            check=sum(profile.integrate(p,lo,hi,16) for lo,hi in intervals) if value else 0.
            error=max(error,abs(value-check));rows.append([len(rows),*p,value])
    assert error<1e-6,error
    completion=OUT/f'completion{PANELS}.txt'
    np.savetxt(completion,rows,fmt=['%d','%.17g','%.17g','%.17g'],header=str(len(rows)),comments='')
    values=parameters((BASE/'run.prm').read_text());original=dict(values)
    values['Output directory',]=str(path)
    values['Postprocess','BP3','Last accepted step']='0'
    values['Postprocess','BP3','Bottom normalization completion file']=str(completion)
    values['Material model','Phase field fault','I h surface quadrature subdivisions']=str(PANELS)
    assert values['Material model','Phase field fault','Use adiabatic pressure in fault friction']=='false'
    assert values['Resume computation',]=='false'
    (path/'run.prm').write_text(render(values))
    sources=[ROOT/'source/material_model/phase_field_fault.cc',ROOT/'include/aspect/material_model/phase_field_fault.h',
             BP3/'bp3.cc',BP3/'work_replay.h',HERE/'steady_initialization.h',Path(__file__).resolve()]
    for source in sources:
        dest=path/'source-tested'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    inputs=[BIN,*map(Path,values['Additional shared libraries',].split(', ')),path/'run.prm',completion,
            FIXTURE/'fault.txt',FIXTURE/'target_cells.txt',FIXTURE/'prestress.txt',*sources]
    env=dict(reference['environment']);env['ASPECT_FAULT_PERFORMANCE']='1'
    record=dict(command=[*reference['command'][:-1],str(path/'run.prm')],environment=env,
                cap_seconds=1200,reference=str(BASE),source_revision=reference['source_revision'],
                hashes={str(p):digest(p) for p in inputs},completion_check_absolute=error,
                completion_profiles=len(rows),
                parameter_changes={' / '.join(k):[original.get(k,'<default 1>'),v] for k,v in values.items() if original.get(k)!=v})
    (path/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record['parameter_changes'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else runner.run(CASE)
