"""Summarize the accepted startup prefix separately from a failed pending solve."""
import json
import re
import numpy as np
from startup_30km import STUDY
from check_startup_30km import table


def analyze():
    path=STUDY/'startup';clock=table(path/'accepted_steps.csv')
    initial=table(path/'weak_initialization.csv')
    zero=table(path/'state_work_0.csv');one=table(path/'state_work_1.csv')
    report=dict(accepted_times=clock['time'].tolist(),
        initial_projected_nodal_weak_error_Pa=float(max(abs(initial['nodal_friction_excess_Pa']))),
        initial_final_weak_error_Pa=float(max(abs(initial['weak_friction_excess_Pa']))),
        initialization_V_over_Vp=[float(min(zero['V']/1e-9)),float(max(zero['V']/1e-9))],
        first_update_theta_relative_error=float(clock['Theta_relative_error'][-1]),regions={})
    for name,lo,hi in [('weakening',1000,29000),('transition',30000,33000),('strengthening',34000,110000)]:
        use=(zero['xd']>=lo)&(zero['xd']<=hi)
        report['regions'][name]=dict(
            Theta0_range= [float(min(zero['Theta_in'][use])),float(max(zero['Theta_in'][use]))],
            Theta1_range= [float(min(one['Theta_out'][use])),float(max(one['Theta_out'][use]))],
            Theta1_over_Theta0=[float(min(one['Theta_out'][use]/zero['Theta_in'][use])),float(max(one['Theta_out'][use]/zero['Theta_in'][use]))],
            V0_over_Vp=[float(min(zero['V'][use]/1e-9)),float(max(zero['V'][use]/1e-9))])
    chunks=re.split(r'\*\*\* Timestep (\d+):',(path/'run.log').read_text())
    nonlinear=[]
    for i in range(1,len(chunks),2):
        step=int(chunks[i]);body=chunks[i+1]
        rows=re.findall(r'after nonlinear iteration\s+(\d+): ([^,\n]+), ([^\n]+)',body)
        alphas=re.findall(r'alpha=([^\.\n]*(?:\.[^\n]*)?)\.\n',body)
        nonlinear.append(dict(step=step,accepted=bool(step in clock['step']),
            residuals=[dict(iteration=int(k),bulk=float(b),fault=float(f)) for k,b,f in rows],
            accepted_alpha=[float(a) for a in alphas]))
    report['nonlinear']=nonlinear
    (STUDY/'startup_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='nonlinear'},indent=2))


if __name__=='__main__':analyze()
