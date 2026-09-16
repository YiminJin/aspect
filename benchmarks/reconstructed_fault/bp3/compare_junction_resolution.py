"""Compare only accepted, common-time states; never label a stopped run step 12."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
base=here/'normal-stress-complete-local4'
fine=here/'normal-stress-junction-refined-local4'
out=fine/'resolution-analysis'
out.mkdir(exist_ok=True)
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def column(data,key):return np.array([float(r[key]) for r in data])
def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
regions={'junction':(37000,43000),'control':(24000,26000)}
summary={'mesh':{},'accepted_states':[]}
mesh_cells={}
for label,root in [('coarse',base),('refined',fine)]:
    cells=[r for p in sorted(root.glob('initial_mesh_*.csv')) for r in rows(p)]
    mesh_cells[label]=cells
    assert len({r['cell'] for r in cells})==len(cells)
    xd=(.5e5*(1+1/np.sqrt(3))-column(cells,'x'))*.5+(1e5-column(cells,'y'))*np.sqrt(3)/2
    near=column(cells,'distance')<800
    summary['mesh'][label]={'cells':len(cells)}
    for name,(lo,hi) in regions.items():
        hs=column(cells,'h')[near&(xd>=lo)&(xd<=hi)]
        summary['mesh'][label][name]={'h_min':float(min(hs)),'h_max':float(max(hs)),'cells':len(hs)}
        on=column(cells,'h')[(column(cells,'distance')<50)&(xd>=lo)&(xd<=hi)]
        summary['mesh'][label][name]['centerline_h_min']=float(min(on))
        summary['mesh'][label][name]['centerline_h_max']=float(max(on))

# Compare actual ancestors, not only the requested minimum-refinement formula:
# another adaptive pass can also finish pending refinements away from the patch.
def key(r):return (r['cell'].split('_')[0],r['cell'].split(':')[1])
coarse_cells={key(r):r for r in mesh_cells['coarse']}
changed={}
fine_keys={key(r) for r in mesh_cells['refined']}
coarsened=[]
for r in mesh_cells['coarse']:
    root,path=key(r)
    original=path
    while path and (root,path) not in fine_keys:path=path[:-1]
    if (root,path) in fine_keys and path!=original:coarsened.append(r)
for r in mesh_cells['refined']:
    root,path=key(r)
    while path and (root,path) not in coarse_cells:path=path[:-1]
    if (root,path) not in coarse_cells:continue
    old=coarse_cells[root,path]
    if float(r['h'])<float(old['h']):changed[key(old)]=old
xd=(.5e5*(1+1/np.sqrt(3))-column(list(changed.values()),'x'))*.5+(1e5-column(list(changed.values()),'y'))*np.sqrt(3)/2
summary['mesh']['changed_coarse_cells']=len(changed)
summary['mesh']['changed_coarse_cells_outside_36_44_km']=int(np.count_nonzero((xd<36000)|(xd>44000)))
summary['mesh']['coarsened_coarse_cells']=len(coarsened)

old_target=rows(base/'initial_traction_target.csv');new_target=rows(fine/'initial_traction_target.csv')
np.testing.assert_allclose(column(old_target,'xd'),column(new_target,'xd'),rtol=0,atol=1e-8)
summary['initial_target_max_difference']={k:float(max(abs(column(new_target,k)-column(old_target,k))))
  for k in ['q_target','sigma_target','Theta','a','C_eval','Q1_friction_correction']}
summary['initial_target_weak_error']=max(column(new_target,'weak_relative_error'))
summary['side_velocity_error']=max(column(rows(fine/'velocity_constraints.csv'),'max_actual_error'))
initial=rows(fine/'history_surface_rank0.csv')
top=[r for r in initial if (.5e5*(1+1/np.sqrt(3))-float(r['x']))*.5+(1e5-float(r['y']))*np.sqrt(3)/2<15000]
fast=max(top,key=lambda r:float(r['V']))
summary['initial_VW_controller']={'node':int(fast['node']),'V':float(fast['V']),
  'split_RSF_dt':.5*.010*.008/(.015*float(fast['V']))}

log=(fine/'run.log').read_text()
checks=[]
for line in log.splitlines():
    if 'Fault linear solve:' in line:
        get=lambda k:float(re.search(r'\b'+k+r'=([^,\s]+)',line)[1])
        checks.append({'iterations':int(get('iterations')),'fresh':get('fresh'),'target':get('target')})
assert checks and all(r['fresh']<=r['target'] for r in checks)
summary['linear']={'returns':len(checks),'iterations':sum(r['iterations'] for r in checks),
                   'worst_fresh_target':max(r['fresh']/r['target'] for r in checks)}
summary['matched_timestep_guard_stopped']=('BP3 controller requires a smaller timestep' in log)
summary['matched_timestep_messages']=[line.strip() for line in log.splitlines() if 'BP3 matched timestep:' in line]
accepted=rows(fine/'accepted_steps.csv') if (fine/'accepted_steps.csv').exists() else []
coarse_accepted=rows(base/'accepted_steps.csv')
for a in accepted:
    k=int(a['step'])
    # No exit-zero inference: the BP3 postprocessor requires actual nonlinear
    # convergence; also demand its completed output message and common time/dt.
    assert re.search(r'BP3 accepted state\s+'+str(k)+r'\b',log)
    for key in ['time','dt']:
        assert abs(float(a[key])-float(coarse_accepted[k][key]))<=1e-12*max(1.,abs(float(a[key])))
    folder=out/f'step{k}'
    if not folder.exists():
        subprocess.run([sys.executable,str(here/'analyze_normal_stress.py'),str(fine),
                        '--step',str(k),'--output',str(folder)],check=True,stdout=subprocess.DEVNULL)
    old=rows(base/f'analysis/step{k}/projected_full.csv')
    new=rows(folder/'projected_full.csv')
    x=column(new,'xd')
    np.testing.assert_allclose(x,column(old,'xd'),rtol=0,atol=1e-8)
    state={'step':k,'time':float(a['time']),'regions':{},'nodes':[]}
    for name,(lo,hi) in regions.items():
        take=(x>=lo-1e-7)&(x<=hi+1e-7)
        state['regions'][name]={}
        for key in ['delta_p','minus_delta_tau_N','sigma_n','q','V','Theta_committed','slip']:
            b=column(old,key)[take];f=column(new,key)[take]
            state['regions'][name][key]={'coarse_ptp':float(np.ptp(b)),'refined_ptp':float(np.ptp(f)),
              'difference_rms':float(np.sqrt(np.mean((f-b)**2))), 'difference_max':float(max(abs(f-b)))}
    for i in [754,755,756,757,758,905]:
        state['nodes'].append({'node':i,'xd':x[i],**{label+'_'+key:float(source[i][key])
          for label,source in [('coarse',old),('refined',new)] for key in ['V','sigma_n','lower_active','q','Theta_committed']}})
    initial_old=rows(base/'fault_0.csv');initial_new=rows(fine/'fault_0.csv')
    state['initial_projection_max_difference']={key:float(max(abs(column(initial_new,key)-column(initial_old,key))))
      for key in ['Ih','C','tau_bg','Theta','V']}
    summary['accepted_states'].append(state)
    fig,axes=plt.subplots(4,1,figsize=(9,10),sharex=True)
    take=(x>=37000-1e-7)&(x<=43000+1e-7)
    for ax,key in zip(axes,['delta_p','minus_delta_tau_N','sigma_n','V']):
        scale=1 if key=='V' else 1e6
        for label,source in [('coarse',old),('refined',new)]:
            ax.plot(x[take]/1000,column(source,key)[take]/scale,label=label)
        ax.axvline(40,color='grey',ls=':');ax.grid(alpha=.3);ax.set_ylabel(key+(' (m/s)' if key=='V' else ' (MPa)'))
    axes[0].legend();axes[-1].set_xlabel('down-dip distance (km)')
    fig.suptitle(f'Accepted step {k}, t={float(a["time"]):.9g} s; consistent Q1 profiles')
    fig.tight_layout();fig.savefig(out/f'profiles_step{k}.png',dpi=140);plt.close(fig)
summary['step12_comparison_complete']=any(r['step']==12 for r in summary['accepted_states'])
dump(out/'comparison.json',summary)
print(json.dumps(summary,indent=2))
