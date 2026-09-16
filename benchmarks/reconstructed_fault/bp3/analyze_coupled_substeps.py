"""Final-time comparison of the one-, two- and four-step coupled-state tests."""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/'coupled-substeps-50-local4'
SAVED=HERE/'work-replay-50-local4'
read=lambda p:np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
old=read(SAVED/'fault_9.csv')
single=json.loads((HERE/'within-step-50-local4/B/analysis.json').read_text())
nodes={r['node']:r for r in single['nodes']}
rows=[dict(substeps=1,dt_s=single['dt'],V39900=nodes[797]['V'],V39950=nodes[796]['V'],
    deficit=single['deficit'],contrast=single['contrast'],gradient_increment=single['gradient_increment'],
    final_gradient=single['predicted_gradient'],Theta39950=nodes[796]['candidate_Theta'],
    slip39950=nodes[796]['predicted_slip'],lower_active=single['lower_active'])]
verification={}
for count in (2,4):
    directory=ROOT/f'substeps{count}'
    execution=json.loads((directory/'execution.json').read_text())
    assert execution['status']==0 and execution['candidate_commits']==count
    assert execution['fresh_linear_passed'] and execution['completed'] and execution['source_unchanged']
    accepted=read(directory/'accepted_steps.csv')
    assert list(accepted['step'].astype(int))==list(range(10,10+count))
    theta=old['Theta'].copy();slip=old['slip'].copy();maximum_theta=0.;maximum_slip=0.;gradient=0.
    final_residuals=[]
    log=(directory/'run.log').read_text()
    blocks=re.split(r'\*\*\* Timestep \d+:',log)[1:]
    assert len(blocks)==count
    for block,record in zip(blocks,accepted):
        step=int(record['step']);dt=record['dt'];current=read(directory/f'fault_{step}.csv')
        x=np.longdouble(dt)*current['V'].astype(np.longdouble)/np.longdouble('.008')
        theta=theta*np.exp(-x)-np.longdouble('.008')/current['V']*np.expm1(-x)
        slip=slip+dt*current['V']
        maximum_theta=max(maximum_theta,float(np.max(np.abs(current['Theta']/theta-1))))
        maximum_slip=max(maximum_slip,float(np.max(np.abs(current['slip']-slip))))
        assert maximum_theta<1e-12 and maximum_slip<1e-12
        assert np.array_equal(current['x'],old['x']) and np.array_equal(current['y'],old['y'])
        assert np.array_equal(current['Ih'],old['Ih']) and np.all(current['C']==0)
        assert np.all(current['V'][current['prescribed']==1]==1e-9)
        assert abs(current['time'][0]-record['time'])<1e-6
        assert abs(dt-single['dt']/count)<1e-6
        candidate=read(directory/f'candidate_commit_{step}.csv')[0]
        assert candidate['max_relative_error']<1e-12 and candidate['qp_samples']>0
        r=re.findall(r'Relative nonlinear residuals \(bulk, fault\) after nonlinear iteration \d+: ([^,]+), ([^\n]+)',block)
        assert r and max(map(float,r[-1]))<1e-8
        final_residuals.append([step,*map(float,r[-1])])
        gradient+=dt*(current['V'][795]-current['V'][796])/50.
    assert abs(accepted[-1]['time']-single['time'])<1e-6
    final=(current['slip'][795]-current['slip'][796])/50.
    assert abs(final-single['old_gradient']-gradient)<1e-14
    rows.append(dict(substeps=count,dt_s=dt,V39900=float(current['V'][797]),V39950=float(current['V'][796]),
        deficit=float(1-current['V'][796]/1e-9),contrast=float((current['V'][797]-current['V'][796])/1e-9),
        gradient_increment=gradient,final_gradient=final,Theta39950=float(current['Theta'][796]),
        slip39950=float(current['slip'][796]),lower_active=int(accepted[-1]['lower_active'])))
    verification[str(count)]=dict(**execution,independent_theta_error=maximum_theta,
        independent_slip_error_m=maximum_slip,final_nonlinear_residuals=final_residuals)
    assert np.all(accepted['lower_active']==0)
    if count==2:
        failed=ROOT/'substeps2-failed-audit-order'
        identical=all(hashlib.sha256((failed/f'state_qp_rank{rank}.csv').read_bytes()).digest()
                      ==hashlib.sha256((directory/f'state_qp_step10_rank{rank}.csv').read_bytes()).digest()
                      for rank in range(4))
        assert identical,'Audit-order repair changed the converged mechanical samples.'
        verification[str(count)]['discarded_attempt_qp_identical']=identical
metrics={}
for key in ('deficit','contrast','gradient_increment','final_gradient'):
    values=[r[key] for r in rows];d12=values[1]-values[0];d24=values[2]-values[1]
    metrics[key]=dict(values=values,change_1_to_2=d12,change_2_to_4=d24,
        difference_ratio=abs(d24/d12),relative_change_1_to_2=d12/values[0],
        relative_change_2_to_4=d24/values[1],decreasing_changes=bool(abs(d24)<abs(d12)))
with (ROOT/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(metrics=metrics,verification=verification)
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
