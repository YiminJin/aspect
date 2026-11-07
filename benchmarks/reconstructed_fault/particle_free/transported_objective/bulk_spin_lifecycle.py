#!/usr/bin/env python3
"""Sequential bounded MPI, retry, same-law and incompatible-law checks."""
from pathlib import Path
import sys,subprocess,json
from bulk_spin_matrix import ROOT,RUNNER,BASE,CASES
RUNS=ROOT/'build-particle-free/transported-objective/stage-c/material-runs'

def run(label,args,expect=0):
    result=subprocess.run([sys.executable,str(RUNNER),label,*BASE,*args],cwd=ROOT)
    assert (result.returncode==0)==(expect==0),(label,result.returncode)
    if expect:
        log=(RUNS/label/'run.log').read_text(errors='replace')
        assert 'DG checkpoint metadata mismatch' in log or 'Incompatible accepted DG history' in log or 'checkpoint' in log.lower(),label
    return dict(case=label,exit_code=result.returncode,expected_failure=bool(expect))
if __name__=='__main__':
    group=sys.argv[1] if len(sys.argv)>1 else 'all';records=[]
    if group in ['all','controls']:
        for label in ['bulk-cd-mpi','bulk-cd-reject','bulk-cd-retry-control']:records.append(run(label,CASES[label]))
    if group in ['all','restart']:
        for label,law,source,expected in [
          ('bulk-cd-restart','bulk slip subtracted','bulk-cd3',0),
          ('total-cd-restart','total material','bulk-old-total-control',0),
          ('bulk-to-total-refused','total material','bulk-cd3',1),
          ('total-to-bulk-refused','bulk slip subtracted','total-cd4',1)]:
            records.append(run(label,['--constitutive-spin',law,'--representation','direct physical stress','--dt','.025','--ranks','4','--restart',str(RUNS/source),'--checkpoint-slot','oldest'],expected))
    out=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bulk_spin'
    (out/f'lifecycle-{group}.json').write_text(json.dumps(records,indent=2)+'\n')
