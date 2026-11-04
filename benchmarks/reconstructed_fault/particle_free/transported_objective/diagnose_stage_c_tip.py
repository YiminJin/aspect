#!/usr/bin/env python3
"""Reduce bounded signed-solver traces without changing a solve or its loading."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[4]
RUNS=ROOT/'build-particle-free/transported-objective/stage-c/runs'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled'

def main():
    cases=[]
    for name in ['tip-trace-cut','tip-trace-legacy','tip-trace-half']:
        p=RUNS/name;command=json.loads((p/'command.json').read_text())
        trace=list(csv.DictReader((p/'signed-trace.csv').open()))
        log=(p/'run.log').read_text().replace('\0','')
        assert command['exit_code']!=0 and 'Native signed line search failed;' in log
        last={k:float(v) for k,v in trace[-1].items()}
        match=re.search(r'iteration=(\d+), entry\s+L2=([0-9.e+-]+)',log);assert match
        iteration=int(match[1]);entry=float(match[2])
        failed=[{k:float(v) for k,v in row.items()} for row in trace if int(row['iteration'])==iteration]
        assert len(failed)==35 and last['minimum_contact_Pa']>0
        assert last['complementarity_Linf']<2e-13
        cases.append(dict(case=name,command=command,first_failed_condition='signed force residual merit decrease after 35 backtracks',
          failed_iteration=iteration,entry_residual_L2=entry,last=last,
          force_residual_Pa=3000*last['force_Linf'],force_acceptance_Pa=2e-9,
          full_step_residual_L2=failed[0]['residual_L2'],
          minimum_backtrack_residual_L2=min(row['residual_L2'] for row in failed),
          linear_relative_defect=last['linear_step_defect_L2']/entry,
          trace_sha256=hashlib.sha256((p/'signed-trace.csv').read_bytes()).hexdigest(),
          log_sha256=hashlib.sha256((p/'run.log').read_bytes()).hexdigest(),
          checkpoint_published=False,admission_reached=False))
        (OUT/(name+'.csv')).write_bytes((p/'signed-trace.csv').read_bytes())
    same=(RUNS/'tip-trace-cut/signed-trace.csv').read_bytes()==(RUNS/'tip-trace-legacy/signed-trace.csv').read_bytes()
    assert same
    result=dict(cases=cases,legacy_objective_traces_byte_identical=same,
      interpretation='Force-residual stagnation, not loss of contact, complementarity failure, or inaccurate linear Newton solve. Negative linearized merit slope does not certify the evaluated nonlinear residual at this scale.',
      next_diagnostic='Replay the stalled local residual with compensated or higher-precision arithmetic, including jump-reference subtraction and active branches, before any loading or solver change.',
      limitations='No tip admission, event, propagation or physical qualification follows from this failed solve.')
    (OUT/'tip-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(legacy_objective_identical=same,failed_conditions=[c['first_failed_condition'] for c in cases])))

if __name__=='__main__':main()
