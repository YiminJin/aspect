"""Only R_VW=0.8: three adaptive steps and one checkpoint-based half comparison."""
import argparse
import json
from pathlib import Path

import run_steady_large_step as runner
from startup_30km import STUDY, parameters, render, digest

OUT = STUDY / 'loading-startup'
runner.OUT = OUT


def prepare(case):
    runner.prepare(case, 1e7)
    path = OUT/case
    values = parameters((path/'run.prm').read_text())
    if case == 'startup':
        values['Maximum first time step',] = '1e6'
        values['Maximum time step',] = '1e7'
        values['End time',] = '3e7'
        values['CFL number',] = '0.5'
        values['Material model','Phase field fault','Initial time step'] = '1e6'
        values['Postprocess','BP3','Last accepted step'] = '3'
        values['Postprocess','BP3','Weakening initial state ratio'] = '0.8'
    values['Time stepping','BP5 state startup','Record timestep selection'] = 'true'
    assert values['Time stepping','BP5 state startup','Maximum logarithmic state change'] == '0.02'
    assert values['Use years instead of seconds',] == 'false'
    assert values['Resume computation',] == ('true' if case=='half' else 'false')
    (path/'run.prm').write_text(render(values))
    record=json.loads((path/'launch.json').read_text())
    record['initialization']='projected-mixture R=0.8**(1-f); native weak background; no seed'
    record['hashes'][str(path/'run.prm')]=digest(path/'run.prm')
    record['hashes'][str(Path(__file__).resolve())]=digest(Path(__file__))
    (path/'launch.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('case',choices=['startup','half'])
    args=p.parse_args()
    (prepare if args.action=='prepare' else runner.run)(args.case)
