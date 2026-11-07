#!/usr/bin/env python3
"""Short native DG transaction and cross-rank checkpoint comparisons, sequential."""
from pathlib import Path
import os,sys,json,csv,subprocess
root=Path(__file__).resolve().parents[4]
here=Path(__file__).resolve().parents[1]/'transported_objective'
runs=root/'build-particle-free/transported-objective/stage-c/material-runs'
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_phase_field/lifecycle'
out.mkdir(exist_ok=False)
env=os.environ.copy();env['LAYERED_EVIDENCE']=str(out)
common=['--candidate',str(root/'build-particle-free/dgq2-phase/candidate'),'--spin','bulk slip subtracted',
        '--phase-discretization','DGQ2','--background-shear','3.7','--translation','2','--halfwidth','6',
        '--level','4','--dt','.0125','--end','.05','--checkpoint-steps','2','--seconds','600']
def run(script,*args):
 command=[sys.executable,str(here/script),*args]
 print(json.dumps(dict(command=command)),flush=True)
 subprocess.run(command,cwd=root,env=env,check=True)
control='dgphase-life-control01';restart='dgphase-life-restart01'
run('run_layered_campaign.py',control,*common,'--ranks','4')
run('run_layered_campaign.py',restart,*common,'--ranks','2','--resume-from','layered-'+control,'--checkpoint-slot','oldest')
run('check_layered_lifecycle.py','layered-'+control,'layered-'+restart,'--restart')
rejected='dgphase-life-reject01';retried='dgphase-life-retry-control01'
run('run_layered_campaign.py',rejected,*common,'--ranks','4','--reject')
with (runs/('layered-'+rejected)/'output/material-summary.csv').open() as stream:rows=list(csv.DictReader(stream))
assert int(rows[-1]['retries'])==1
steps=rows[1:];expression=format(float(steps[-1]['dt']),'.17g')
for i in range(len(steps)-2,-1,-1):
 threshold=.5*(float(rows[i]['time'])+float(rows[i+1]['time']))
 expression=f"if(time<{threshold:.17g},{float(steps[i]['dt']):.17g},{expression})"
(out/'accepted-schedule.json').write_text(json.dumps(dict(rows=rows,function=expression),indent=2)+'\n')
run('run_layered_campaign.py',retried,*common,'--ranks','4','--function',expression)
run('check_layered_lifecycle.py','layered-'+rejected,'layered-'+retried)
for name in (control,rejected,retried):run('check_layered_reference.py','layered-'+name)
