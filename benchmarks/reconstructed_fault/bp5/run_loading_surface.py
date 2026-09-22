"""Three adaptive loading steps and checkpoint subdivision with eight Ih panels."""
import argparse
import json
import shutil
from pathlib import Path
import run_steady_large_step as runner
import analyze_loading_startup as analysis
from startup_30km import ROOT, STUDY, parameters, render, digest
from run_surface_quadrature import OUT as IH_OUT, CASE

OUT=STUDY/'loading-surface8'
runner.OUT=OUT
analysis.OUT=OUT


def prepare(case):
    if case=='startup':
        source=IH_OUT/CASE
        assert json.loads((source/'execution.json').read_text())['passed']
        path=OUT/case;path.mkdir(parents=True)
        record=json.loads((source/'launch.json').read_text())
        for name,sha in record['hashes'].items():assert digest(Path(name))==sha,name
        values=parameters((source/'run.prm').read_text())
        values['Output directory',]=str(path)
        values['Postprocess','BP3','Last accepted step']='3'
        assert values['Material model','Phase field fault','I h surface quadrature subdivisions']=='8'
        assert values['Postprocess','BP3','Weakening region length']=='30000'
        assert values['Checkpointing','Steps between checkpoint']=='2'
        assert values['Time stepping','BP5 state startup','Maximum logarithmic state change']=='0.02'
        (path/'run.prm').write_text(render(values))
        record['command'][-1]=str(path/'run.prm')
        record['hashes'].pop(str(source/'run.prm'))
        # Keep normal coarse timing; detailed profiling is not needed for this
        # history check and does not affect the equations or accepted trajectory.
        record['environment'].pop('ASPECT_FAULT_PERFORMANCE',None)
        shutil.copytree(source/'source-tested',path/'source-tested')
        record['reference_initialization']=str(source)
    else:
        runner.prepare('half',1e7)
        path=OUT/case
        record=json.loads((path/'launch.json').read_text())
    record['hashes'][str(path/'run.prm')]=digest(path/'run.prm')
    record['hashes'][str(Path(__file__).resolve())]=digest(Path(__file__))
    (path/'launch.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','check','plot'])
    parser.add_argument('case',choices=['startup','half'],nargs='?',default='startup')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.case)
    elif args.action=='run':runner.run(args.case)
    elif args.action=='plot':analysis.plot()
    else:
        analysis.check(args.case)
        if args.case=='half':analysis.compare()
