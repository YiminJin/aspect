"""Accepted-state matched mesh/history comparison; no constitutive recomputation."""
import csv
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('--tag',default='qualified');args=parser.parse_args()
here=Path(__file__).resolve().parent;pair=here/('junction-matched'+('-'+args.tag if args.tag else '')+'-local4')
old=here/'normal-stress-complete-local4'
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def col(data,key):return np.array([float(r[key]) for r in data])
def xd(r):return (.5e5*(1+1/np.sqrt(3))-float(r['x']))*.5+(1e5-float(r['y']))*np.sqrt(3)/2
def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def profile(root,k):
    folder=root/f'analysis/step{k}'
    if not folder.exists():
        subprocess.run([sys.executable,str(here/'analyze_normal_stress.py'),str(root),
                       '--step',str(k),'--output',str(folder)],check=True,stdout=subprocess.DEVNULL)
    return rows(folder/'projected_full.csv')

summary={'cases':{},'spatial':{},'temporal_coarse_vs_old':{}}
states={};profiles={};weak={};mesh={}
for label in ['coarse','refined']:
    root=pair/label;log=(root/'run.log').read_text()
    assert json.loads((root/'execution.json').read_text())['status']==0
    states[label]=rows(root/'accepted_steps.csv');last=int(states[label][-1]['step'])
    assert abs(float(states[label][-1]['time'])-2232176379.2516127)<1e-3
    assert len(states[label])==last+1 and all(re.search(r'BP3 accepted state\s+'+str(i)+r'\b',log) for i in range(last+1))
    assert 'BP3 target mesh verified before mechanics' in log
    checks=[]
    for line in log.splitlines():
        if 'Fault linear solve:' in line:
            get=lambda key:float(re.search(r'\b'+key+r'=([^,\s]+)',line)[1])
            checks.append({'iterations':int(get('iterations')),'fresh':get('fresh'),'target':get('target')})
    assert checks and all(r['fresh']<=r['target'] for r in checks)
    histories=[rows(root/f'history_{i}.csv')[0] for i in range(last+1)]
    assert max(col(histories,'Theta_reference_relative_error'))<1e-12
    profiles[label]=profile(root,last);profile(root,0)
    subprocess.run([sys.executable,str(here/'analyze_history_mechanics.py'),str(root)],check=True,stdout=subprocess.DEVNULL)
    weak[label]=rows(root/'surface_comparison.csv')
    fault=rows(root/f'fault_{last}.csv')
    error=max(abs(col(weak[label],'particle_R')-col(fault,'weak_residual')))
    assert error/max(abs(col(weak[label],'particle_q')))<1e-12
    mesh[label]=[r for f in sorted(root.glob('initial_mesh_*.csv')) for r in rows(f)]
    assert len({r['cell'] for r in mesh[label]})==len(mesh[label])
    assert abs(sum(float(r['h'])**2 for r in mesh[label])-1e10)<1e-3
    summary['cases'][label]={'steps':last,'time':float(states[label][-1]['time']),
      'seconds':json.loads((root/'execution.json').read_text())['seconds'],'cells':len(mesh[label]),
      'linear_returns':len(checks),'krylov':sum(r['iterations'] for r in checks),
      'worst_fresh_target':max(r['fresh']/r['target'] for r in checks),
      'Theta_audit_max':max(col(histories,'Theta_reference_relative_error')),
      'weak_R_reproduction_error':error,'final_free':int(states[label][-1]['free']),
      'final_active':int(states[label][-1]['lower_active'])}

assert len(states['coarse'])==len(states['refined'])
for key in ['time','dt']:
    np.testing.assert_allclose(col(states['coarse'],key),col(states['refined'],key),rtol=1e-13,atol=1e-8)
for k in range(len(states['coarse'])-1):
    limits=[]
    for label in ['coarse','refined']:
        time,limit=map(float,(pair/'clock'/f'{label}_{k}').read_text().split());limits.append(limit)
        assert abs(time-float(states[label][k]['time']))<1e-5
    assert float(states['coarse'][k+1]['dt'])<=.95*min(limits)*(1+1e-13)

old_mesh={r['cell']:r for f in old.glob('initial_mesh_*.csv') for r in rows(f)}
assert {r['cell'] for r in mesh['coarse']}==set(old_mesh)
for r in mesh['coarse']:
    assert all(r[k]==old_mesh[r['cell']][k] for k in ['x','y','h'])
fine_ids={r['cell'] for r in mesh['refined']}
control=[r for r in mesh['coarse'] if 24000<=xd(r)<=26000 and float(r['distance'])<1500]
bottom=[r for r in mesh['coarse'] if xd(r)>113000 and float(r['distance'])<1500]
assert all(r['cell'] in fine_ids for r in control+bottom)
summary['mesh_controls']={'exact_coarse':True,'unchanged_control_cells':len(control),'unchanged_bottom_cells':len(bottom)}

x=col(profiles['coarse'],'xd')
np.testing.assert_allclose(x,col(profiles['refined'],'xd'),rtol=0,atol=1e-8)
initial={l:rows(pair/l/'fault_0.csv') for l in ['coarse','refined']}
summary['initial_projection']={key:{'spatial_max_difference':float(max(abs(col(initial['refined'],key)-col(initial['coarse'],key)))),
  'coarse_vs_old_max_difference':float(max(abs(col(initial['coarse'],key)-col(rows(old/'fault_0.csv'),key))))}
  for key in ['C','Ih','tau_bg','Theta','V']}
for label in ['coarse','refined']:
    for k in range(len(states[label])):
        f=rows(pair/label/f'fault_{k}.csv')
        np.testing.assert_allclose(col(f,'xd'),col(initial[label],'xd'),rtol=0,atol=1e-8)
        np.testing.assert_allclose(col(f,'tau_bg'),col(initial[label],'tau_bg'),rtol=0,atol=0)
        np.testing.assert_allclose(col(f,'Ih'),col(initial[label],'Ih'),rtol=0,atol=0)

previous=rows(old/'analysis/step12/projected_full.csv')
fields=['delta_p','minus_delta_tau_N','sigma_n','q','V','Theta_committed','slip']
for name,(lo,hi) in {'junction':(37000,43000),'control':(24000,26000)}.items():
    mask=(x>=lo-1e-7)&(x<=hi+1e-7)
    summary['spatial'][name]={};summary['temporal_coarse_vs_old'][name]={}
    for key in fields:
        b=col(profiles['coarse'],key)[mask];f=col(profiles['refined'],key)[mask];o=col(previous,key)[mask]
        summary['spatial'][name][key]={'coarse_ptp':float(np.ptp(b)),'refined_ptp':float(np.ptp(f)),
          'difference_rms':float(np.sqrt(np.mean((f-b)**2))),'difference_max':float(max(abs(f-b)))}
        summary['temporal_coarse_vs_old'][name][key]={'difference_rms':float(np.sqrt(np.mean((b-o)**2))),
          'difference_max':float(max(abs(b-o)))}

summary['nodes']=[]
for i in [754,755,756,757,758,905]:
    r={'node':i,'xd':x[i]}
    for label in ['coarse','refined']:
        w=weak[label][i];p=profiles[label][i]
        r[label]={k:float(w[k]) for k in ['V','weight','particle_q_density','particle_C_density',
          'particle_friction_density','particle_damping_density','particle_R_density',
          'particle_Kdiag_density','fe_Kdiag_density','delta_q_density','delta_sigma_density',
          'delta_friction_density','delta_R_density','particle_tensile_weight','fe_tensile_weight']}
        r[label]['active']=int(p['lower_active'])
        r[label]['reaction']= -float(w['particle_R_density']) if int(p['lower_active']) else 0.
    summary['nodes'].append(r)

trajectory=[]
for k in range(len(states['coarse'])):
    for label in ['coarse','refined']:
        root=pair/label
        parts=[rows(root/f'history_surface_step{k}_rank{r}.csv') for r in range(4)]
        accepted=rows(root/f'stress_projected_{k}.csv')
        for i in [756,757,758,905]:
            w=sum(float(p[i]['weight']) for p in parts)
            terms={name:sum(float(p[i][name]) for p in parts)/w for name in
              ['particle_q','particle_C','particle_friction','particle_R','fe_R','particle_sigma','fe_sigma']}
            trajectory.append(dict(case=label,step=k,time=float(states[label][k]['time']),node=i,
              V=float(parts[0][i]['V']),active=int(accepted[i]['lower_active']),
              delta_R=terms['fe_R']-terms['particle_R'],**terms))
with (pair/'node_history.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(trajectory[0]));writer.writeheader();writer.writerows(trajectory)

summary['raw_and_weights']={};summary['common_weak_tests']={}
for label in ['coarse','refined']:
    root=pair/label;last=int(states[label][-1]['step'])
    raw=rows(root/f'analysis/step{last}/raw_selected.csv')
    w=weak[label];d={}
    for name,(lo,hi) in {'junction':(37000,43000),'bottom':(113000,116000)}.items():
        selected=[r for r in raw if lo<=float(r['xd'])<=hi]
        d[name]={'selected_min':min(selected,key=lambda r:float(r['sigma_n'])),
                 'selected_max':max(selected,key=lambda r:float(r['sigma_n'])),
                 'selection_caveat':'Raw selection is by rank/support extrema, not every sample.',
                 **{mode+'_tensile_weight':sum(float(r[mode+'_tensile_weight']) for r in w if lo<=float(r['xd'])<=hi)
                    for mode in ['particle','fe']}}
    summary['raw_and_weights'][label]=d
    tests=rows(root/'common_history_tests.csv');comp=[]
    for r in tests:
        if r['order']!='6':continue
        low=next(v for v in tests if v['order']=='4' and v['center']==r['center'] and v['direction']==r['direction'])
        p=float(r['particle_load']);f=float(r['fe_load']);unc=abs(p-float(low['particle_load']))+abs(f-float(low['fe_load']))
        comp.append(dict(center=float(r['center']),direction=r['direction'],particle=p,fe=f,difference=f-p,
                         quadrature_change_bound=unc))
    summary['common_weak_tests'][label]=comp

fig,axes=plt.subplots(4,1,figsize=(9,11),sharex=True);mask=(x>=37000-1e-7)&(x<=43000+1e-7)
for ax,key in zip(axes,['delta_p','minus_delta_tau_N','sigma_n','V']):
    for label,data,style in [('old coarse',previous,':'),('matched coarse',profiles['coarse'],'-'),('matched refined',profiles['refined'],'--')]:
        ax.plot(x[mask]/1000,col(data,key)[mask]/(1 if key=='V' else 1e6),style,label=label)
    ax.set_ylabel(key+(' (m/s)' if key=='V' else ' (MPa)'));ax.grid(alpha=.3);ax.axvline(40,color='gray',ls=':')
axes[0].legend();axes[-1].set_xlabel('down-dip distance (km)');fig.tight_layout();fig.savefig(pair/'junction_profiles.png',dpi=160)
with (pair/'final_profiles.csv').open('w',newline='') as f:
    data=[dict(xd=x[i],**{label+'_'+key:float(r[i][key]) for label,r in profiles.items() for key in fields}) for i in range(len(x))]
    writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
dump(pair/'comparison.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k in ['cases','nodes','mesh_controls','initial_projection']},indent=2))
