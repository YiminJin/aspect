"""Summarize the resumed K3 run without resetting any reference history."""
import hashlib
import json
from pathlib import Path

import numpy as np

here=Path(__file__).resolve().parent
parent=here.parent
name='spatial0375_n256_f32_periodic_floor'
run=parent/name
comparison=json.loads((parent/f'{name}-comparison.json').read_text())
assert comparison['complete_smoke'] and len(comparison['states'])==9
phases=[]
for step in range(1,9):
    entry=np.genfromtxt(run/f'phase_probe_{step}.csv',delimiter=',',names=True)
    exit=np.genfromtxt(run/f'phase_probe_exit_{step}.csv',delimiter=',',names=True)
    target=max(1e-8*float(entry['R_old_phi_Hprevious']),float(entry['roundoff']))
    assert target==exit['absolute_target'] and exit['R_new_phi_Hprevious']<=target
    phases.append(dict(step=step,initial=float(entry['R_old_phi_Hprevious']),
        absolute=float(exit['R_new_phi_Hprevious']),relative=float(exit['relative']),
        allowance=float(entry['roundoff']),target=target))
differences={}
for step in (0,1):
    for field in ('particles','phase','surface'):
        old=np.atleast_1d(np.genfromtxt(parent/f'spatial0375_n256_f32_periodic/{field}_{step}.csv',delimiter=',',names=True))
        new=np.atleast_1d(np.genfromtxt(run/f'{field}_{step}.csv',delimiter=',',names=True))
        if field=='particles': old=np.sort(old,order='id');new=np.sort(new,order='id')
        keys={'particles':['H','tau_xx','tau_xy','tau_yy'],'phase':['phi'],'surface':['x','y','V','Theta','C','Ih']}[field]
        differences[f'{field}_{step}']={key:float(np.max(abs(new[key]-old[key]))) for key in keys}
coarse=json.loads((parent/'spatial0375_n128_f32_periodic-comparison.json').read_text())
spatial=[]
for n,result,directory in ((128,coarse,parent/'spatial0375_n128_f32_periodic'),(256,comparison,run)):
    final=result['states'][-1]
    phi=np.genfromtxt(directory/'comparison_phase_8.csv',delimiter=',',names=True)
    phi0=np.genfromtxt(directory/'comparison_phase_0.csv',delimiter=',',names=True)
    H=np.genfromtxt(directory/'comparison_H_8.csv',delimiter=',',names=True)
    H0=np.genfromtxt(directory/'comparison_H_0.csv',delimiter=',',names=True)
    spatial.append(dict(normal=n,final=final,
        total_feedback=dict(phi=float(np.max(abs(phi['production_mean_phi']-phi0['production_mean_phi']))),
            H=float(np.max(H['production_mean_H']-H0['production_mean_H'])),
            Ih=final['surface_means']['Ih']-result['states'][0]['surface_means']['Ih']),
        cumulative_feedback_error=dict(phi=float(np.max(abs(phi['difference']-phi0['difference']))),
            H=float(np.max(abs(H['difference']-H0['difference']))),
            Ih=final['surface_means']['Ih']-result['states'][0]['surface_means']['Ih']
                 -(final['reference']['Ih']-result['states'][0]['reference']['Ih']))))
report=dict(passed=True,resources=json.loads((parent/f'{name}.resources.json').read_text()),
    phases=phases,states_passing=len(comparison['states']),
    fresh_linear_checks=len(comparison['fresh_linear_checks']),
    max_fresh_over_target=max(r['fresh']/r['target'] for r in comparison['fresh_linear_checks']),
    max_containment=max(r['guard']['max_omitted_fraction'] for r in comparison['states']),
    max_normalization=max(r['guard']['max_supported_normalization_error'] for r in comparison['states']),
    max_surface_balance=max(r['surface_balance_rms_Pa'] for r in comparison['states']),
    max_theta_update_error=max(r.get('theta_update_max_error_s',0) for r in comparison['states']),
    old_prefix_changes=differences,spatial=spatial)
(here/'replay-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='spatial'},indent=2))
